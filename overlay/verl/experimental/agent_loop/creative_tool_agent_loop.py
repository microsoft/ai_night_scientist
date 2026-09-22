# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import logging
import os
import copy
import random
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.proposal_gen_utils import action_dict as proposal_action_dict
from verl.interactions.utils.proposal_gen_utils import LOGPROBS_NUM as PROPOSAL_LOGPROBS_NUM
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        video_data: list[tuple[torch.Tensor, dict[str, Any]]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.video_data = video_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        self.routed_experts = None

        # Extra fields for dynamic addition, e.g., tool session data
        self.extra_fields: dict[str, Any] = {}


@register("creative_tool_agent")
class CreativeToolAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize tools from config file
        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.max_parallel_calls = self.rollout_config.multi_turn.max_parallel_calls
        self.max_tool_response_length = self.rollout_config.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = self.rollout_config.multi_turn.tool_response_truncate_side
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)
        self.tool_parser_name = self.rollout_config.multi_turn.format
        self.swap_action = self.rollout_config.multi_turn.swap_action

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        # Initialize interactions from config file
        self.interaction_config_file = self.rollout_config.multi_turn.interaction_config_path
        if self.interaction_config_file:
            self.interaction_map: dict[str, BaseInteraction] = self._initialize_interactions(
                self.interaction_config_file
            )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            # Deep-copy so that mutable fields (action_history, action_jsons, logprobs)
            # are not shared across concurrent rollouts generated from the same data row.
            interaction_kwargs = copy.deepcopy(kwargs["extra_info"]["interaction_kwargs"])
            # Prefer per-sample swap probability from trainer for step-wise decay.
            interaction_kwargs["swap_action"] = kwargs.get("swap_action", self.swap_action)
            interaction_kwargs["swap_history"] = []
            interaction_kwargs["action_output_history"] = []
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)
        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=agent_data.routed_experts,
            extra_fields={},
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        output.extra_fields.update(
            {
                "request_id": agent_data.request_id,
                "logprobs": list(agent_data.interaction_kwargs.get("logprobs", [])),
                "action_history": list(agent_data.interaction_kwargs.get("action_history", [])),
                "swap_history": agent_data.interaction_kwargs.get("swap_history", []),
                "action_jsons": list(agent_data.interaction_kwargs.get("action_jsons", [])),
                "action_output_history": list(agent_data.interaction_kwargs.get("action_output_history", []))
            }
        )
        return output

    def _get_action_kwargs(self, agent_data: AgentData) -> dict[str, Any]:
        """Return action-specific generation kwargs based on current interaction state."""
        action_history = agent_data.interaction_kwargs.get("action_history", [])
        if not action_history:
            return {}

        action_name = action_history[-1][0]
        return proposal_action_dict.get(action_name, {}).get("kwargs", {}) or {}

    async def _convert_json_tool_calls(self, content: str, action: str, level: str) -> list[FunctionCall]:
        """Convert action JSON output into normalized tool calls for parser-agnostic execution."""
        if isinstance(content, dict):
            action_json = content
        else:
            try:
                action_json = json.loads(content)
            except (TypeError, json.JSONDecodeError):
                logger.error("Failed to parse content as JSON: %s", content)
                return []

        query_list = None
        if action == "search":
            queries = action_json.get("search_queries")
            if not isinstance(queries, list):
                return []

            queries = [query.strip() for query in queries if isinstance(query, str) and query.strip()]
            if not queries:
                return []

            if self.rollout_config.task == "code_gen":
                query_list = [(query, level) for query in queries]
            else:
                query_list = queries
        elif action == "debate_setup":
            all_participant_keywords: list[str] = []
            for participant in action_json.get("debate_participants", []):
                expertise = participant.get("expertise")
                if isinstance(expertise, str) and expertise.strip():
                    all_participant_keywords.append(expertise.strip())
            query_list = all_participant_keywords

        if not query_list:
            return []

        return [
            FunctionCall(
                name="search",
                arguments=json.dumps({"query_list": query_list, "topk": 1, "return_scores": False}, ensure_ascii=False),
            )
        ]

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=self.tool_schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.prompt_ids = prompt_ids
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []

        max_new_tokens = min(8192, self.rollout_config.max_model_len - len(agent_data.prompt_ids) - 1)
        kwargs = sampling_params.copy()
        kwargs["max_new_tokens"] = max_new_tokens
        # kwargs["n"] = self.rollout_config.n
        action_history = agent_data.interaction_kwargs.get("action_history", [])

        # If this is the first action turn, then we need to randomly swap the action based on the configured probability to encourage diversity in multi-turn interactions.
        should_swap = False
        # (agent_data.assistant_turns == 0)
        if (action_history[-1][0] == 'select_action') and self.interaction_config_file:
            swap_action_prob = agent_data.interaction_kwargs.get("swap_action", 0.0)
            should_swap = (swap_action_prob > 0) \
                        and (self.max_assistant_turns and agent_data.assistant_turns < (self.max_assistant_turns - 3)) \
                        and (random.random() < swap_action_prob)
            agent_data.interaction_kwargs["swap_history"].append(should_swap)

        if should_swap:
            action_options = ["search", "debate", "spark", "write"]
            next_action = random.choice(action_options)
            max_level = proposal_action_dict[next_action]['max_level'] if 'max_level' in proposal_action_dict[next_action] else 1
            next_noise = random.choice([opt for opt in range(1, max_level + 1)])
            logger.info(f"Swapping action ({next_action, next_noise}) for request {agent_data.request_id} with probability {swap_action_prob}")

            # first time to set num_preempted
            if agent_data.metrics.get("num_preempted") is None:
                agent_data.metrics["num_preempted"] = -1
            # then add num_preempted to the metrics
            else:
                agent_data.metrics["num_preempted"] += 0

            agent_data.assistant_turns += 1
            agent_data.interaction_kwargs.setdefault("logprobs", [])

            # Check termination conditions
            if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
                return AgentState.TERMINATED
            if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
                return AgentState.TERMINATED
            if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
                return AgentState.TERMINATED

            assistant_message = json.dumps({"action": next_action, "noise": next_noise})
            # Tokenize the synthetic response so prompt_ids stays in sync with messages.
            # Use mask=0 (not trained) since this response was injected, not model-generated.
            synthetic_response_ids = await self.apply_chat_template(
                [{"role": "assistant", "content": assistant_message}],
                remove_system_prompt=True,
            )
            agent_data.prompt_ids += synthetic_response_ids
            agent_data.response_mask += [0] * len(synthetic_response_ids)
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * len(synthetic_response_ids)

        else:

            # Use action-specific generation knobs from interaction action dict.
            extra_kwargs = self._get_action_kwargs(agent_data)
            if extra_kwargs:
                kwargs.update(extra_kwargs)
            if ("json_schema" not in kwargs) or (kwargs["json_schema"] is None):
                kwargs["json_schema"] = json.dumps({"response": {"type": "string"}})
            if kwargs.get("logprobs", False):
                kwargs["top_logprobs_num"] = PROPOSAL_LOGPROBS_NUM

            with simple_timer("generate_sequences", agent_data.metrics):
                output = await self.server_manager.generate(
                    request_id=agent_data.request_id,
                    prompt_ids=agent_data.prompt_ids,
                    sampling_params=kwargs,
                    image_data=agent_data.image_data,
                    video_data=agent_data.video_data,
                )

            # first time to set num_preempted
            if agent_data.metrics.get("num_preempted") is None:
                agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
            # then add num_preempted to the metrics
            else:
                agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

            agent_data.assistant_turns += 1
            agent_data.response_ids = output.token_ids
            agent_data.prompt_ids += agent_data.response_ids
            agent_data.response_mask += [1] * len(agent_data.response_ids)
            if output.log_probs:
                agent_data.response_logprobs += output.log_probs

            if output.routed_experts is not None:
                agent_data.routed_experts = output.routed_experts

            if output.top_logprobs is not None:
                agent_data.interaction_kwargs.setdefault("logprobs", []).append(output.top_logprobs)

            # Check termination conditions
            if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
                return AgentState.TERMINATED
            if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
                return AgentState.TERMINATED
            if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
                return AgentState.TERMINATED

            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )

        # First, support the legacy action-JSON => tool-call conversion path.
        converted_tool_calls: list[FunctionCall] = []
        agent_data.tool_calls = [] # No tool-enabled action taken

        if action_history:
            curr_action, curr_level = action_history[-1]
            if proposal_action_dict.get(curr_action, {}).get("tool_enabled", False):
                converted_tool_calls = await self._convert_json_tool_calls(assistant_message, curr_action, curr_level)
                # Fall back to standard parser extraction if no converted tool calls are produced.
                if converted_tool_calls:
                    agent_data.tool_calls = converted_tool_calls
                else:
                    _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

        # Handle interaction if needed
        if self.interaction_config_file:
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)

        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        for tool_response, tool_reward, _ in responses:
            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            # Handle image data
            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        agent_data.messages.extend(add_messages)

        if self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            # Note that we have to pass None to the images and videos if there are no new images / videos
            # to stay compatible with downstream image processing logic!
            images = new_images_this_turn if new_images_this_turn else None
            videos = None
            response_ids = await self.apply_chat_template(
                add_messages,
                images=images,
                videos=videos,
                remove_system_prompt=True,
            )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.INTERACTING if self.interaction_config_file else AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        # If the token budget is running low, tighten the effective max_turns so that the must-write
        # condition in generate_response fires this turn rather than being skipped when GENERATING
        # later terminates due to response_length exhaustion.
        tokens_remaining = self.response_length - len(agent_data.response_mask)
        write_budget_needed = proposal_action_dict.get("write", {}).get("kwargs", {}).get("max_new_tokens", 5120) + 256
        if self.max_assistant_turns is not None and tokens_remaining < write_budget_needed:
            logger.info(f"Low token budget ({tokens_remaining} tokens remaining), tightening effective max_assistant_turns for request {agent_data.request_id}")
            agent_data.assistant_turns = self.max_assistant_turns - 2

        agent_data.interaction_kwargs["turns"] = (agent_data.assistant_turns, self.max_assistant_turns)

        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )
        agent_data.user_turns += 1

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]
        agent_data.messages.extend(add_messages)

        if reward is not None:
            agent_data.turn_scores.append(reward)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        response_ids = await self.apply_chat_template(
            add_messages,
            remove_system_prompt=True,
        )

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # double check prompt
        # Check termination condition
        if should_terminate_sequence:
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, tool_reward, res = await tool.execute(
                instance_id, tool_args, agent_data=agent_data
            )
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return (
                ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                0.0,
                {},
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res

    def _initialize_interactions(self, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        return interaction_map