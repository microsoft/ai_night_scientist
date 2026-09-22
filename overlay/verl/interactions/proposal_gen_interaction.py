# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
import json
import json_repair
import random

from verl.interactions.utils.proposal_gen_utils import action_dict

from .base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ProposalGenInteraction(BaseInteraction):
    """An interaction for proposal generation.

    - `start_interaction`: start a interaction instance for a trajectory.
    - `generate_response`: generate the response of the user.
    - `calculate_score`: calculate the score of the interaction.
    - `finalize_interaction`: finalize the interaction instance.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}

    @staticmethod
    def _append_action_history_once(action_history: list, action_name: str, noise: int | str) -> None:
        """Append action only when it differs from the latest action entry."""
        action_tuple = (action_name, str(noise))
        action_history.append(action_tuple)

    @staticmethod
    def _tool_message_has_reference_paper(message: dict[str, Any]) -> bool:
        """Handle both text-only and multimodal tool message formats."""
        if message.get("role") != "tool":
            return False

        content = message.get("content")
        if isinstance(content, str):
            return ": " in content

        if isinstance(content, list):
            for chunk in content:
                if isinstance(chunk, dict) and ": " in str(chunk.get("text", "")):
                    return True
        return False

    @staticmethod
    def _safe_load_action_json(content: Any) -> tuple[Any, bool]:
        if isinstance(content, dict):
            return content, True

        if not isinstance(content, str):
            return content, False

        try:
            return json.loads(content), True
        except json.JSONDecodeError:
            pass

        try:
            return json_repair.loads(content), True
        except (json.JSONDecodeError, RecursionError, TypeError, ValueError):
            return content, False

    async def start_interaction(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "action": None,
            "noise": -1,
            "problem": kwargs.get("problem", ""),
            "logprobs": [],
            "reward": 0.0,
        }
        return instance_id

    async def validate_action(self, action_name: str, messages: list, action_json: dict, kwargs: dict) -> bool:
        """
        Validate the action and its parameters.

        Args:
            action: The action to validate
            action_json: The JSON object containing the action parameters
        Returns:
            bool: True if the action is valid, False otherwise
        """

        if action_name == 'select_action':
            # Check selected action and noise
            invalid_response = f"""You did not format your selected action correctly. Output your selected action in the following JSON format:
{{
    "action": "string (one of the following: 'search', 'debate', 'spark', 'write', 'complete')",
    "noise": int (an integer between 1 and 5, inclusive, representing the noise parameter for the action; 1 means no noise, 5 means maximum noise)"
}}
"""
            if not isinstance(action_json, dict):
                return False, invalid_response
            if 'action' not in action_json or 'noise' not in action_json:
                return False, invalid_response
            if not isinstance(action_json['noise'], int):
                return False, invalid_response
            if action_json['action'] not in action_dict.keys():
                return False, invalid_response
            if not (1 <= action_json['noise'] <= 5):
                return False, invalid_response
            return True, None

        elif action_name == 'search':
            # Check search queries
            invalid_response = f"""You did not select a valid search query. You should output your noise-aware, diverse queries in the following JSON format:
{{
    "search_queries": [list of five, brief 1-3 word string queries]
}}
"""
            if type(action_json) != dict or 'search_queries' not in action_json:
                return False, invalid_response
            if not isinstance(action_json.get('search_queries'), list):
                return False, invalid_response
            if not all(isinstance(query, str) for query in action_json['search_queries']):
                return False, invalid_response
            # Check if the last message is a tool response that contains reference papers.
            if not self._tool_message_has_reference_paper(messages[-1]):
                last_msg = messages[-1]
                if last_msg.get("role") == "tool":
                    # Tool ran but failed (e.g., retrieval service error) — not a format error
                    return False, "The search could not be completed due to a retrieval error. Please try again with different queries."
                print(f"No Reference Paper in search: {messages[-1]}")
                return False, invalid_response

            kwargs["action_output_history"].append({
                "action_name": action_name,
                "action_output": {
                    "search_queries": action_json.get("search_queries", []),
                    "search_results": messages[-1].get("content", [])
                }
            })

            return True, None

        elif action_name == 'debate_setup':
            # Check debate participants
            invalid_response = f"""You did not format your debate participants correctly. Output your debate participants in the following JSON format:
{{
    "debate_participants": [
        {{
            "name": "You",
            "job": "string (your job title or role, including current domain)",
            "expertise": "string (your specific area of expertise or background, including 1-2 sentences on your current work or research interests)"
        }},
        ...
        {{
            "name": "string (name of the participant)",
            "job": "string (job title or role of the participant in the debate, including their current domain)",
            "expertise": "string (specific area of expertise or background of the participant; 1-2 sentences about their current work or research interests)"
        }}
    ],
    "debate_topics": ["string (specific topic or question to be debated; 1-2 sentences)", 
                        "string (another topic or question to be debated)",
                        ...],
    "debate_structure": "string (description of how to ; 1-2 sentences about the structure of the debate, e.g., open-ended discussion, single-turn debate, etc.)
}}
"""
            if 'debate_participants' not in action_json or 'debate_topics' not in action_json or 'debate_structure' not in action_json:
                return False, invalid_response
            if not isinstance(action_json.get('debate_participants'), list):
                return False, invalid_response
            if not (any('You' in participant['name'] for participant in action_json['debate_participants']) and all(isinstance(participant, dict) for participant in action_json['debate_participants'])):
                return False, invalid_response
            # Check debate topics
            if not isinstance(action_json.get('debate_topics'), list):
                return False, invalid_response
            if not all(isinstance(topic, str) for topic in action_json['debate_topics']):
                return False, invalid_response
            # Check debate structure
            if not isinstance(action_json.get('debate_structure'), str):
                return False, invalid_response
            # Check if the last message is a tool response that contains reference papers.
            if not self._tool_message_has_reference_paper(messages[-1]):
                last_msg = messages[-1]
                if last_msg.get("role") == "tool":
                    # Tool ran but failed (e.g., retrieval service error) — not a format error
                    print("Papers associated with the debate participants could not be found. Please try again with different participant keywords.")
                    return True, None
                return False, invalid_response

            kwargs["action_output_history"].append({
                "action_name": action_name,
                "action_output": {
                    "debate_setup": action_json,
                    "debate_papers": messages[-1].get("content", [])
                }
            })

            return True, None

        elif action_name == 'debate':
            # Check debate conversation
            invalid_response = f"""I did not execute a valid debate conversation. I should output the debate conversation history in the following JSON format:
{{
    "conversation_history": [
        {{
            "speaker_name": "string (name of the participant)",
            "speaker_response": "string (message sent by the participant in the debate; 1-2 sentences)"
        }},
        ...
    ]
}}
"""
            if 'conversation_history' not in action_json:
                return False, invalid_response
            if not isinstance(action_json.get('conversation_history'), list):
                return False, invalid_response
            if not all(isinstance(conversation, dict) for conversation in action_json['conversation_history']):
                return False, invalid_response
            if not all('speaker_name' in conversation and 'speaker_response' in conversation for conversation in action_json['conversation_history']):
                return False, invalid_response

            if kwargs["action_output_history"][-1]["action_name"] == "debate_setup":
                kwargs["action_output_history"][-1]["action_output"]["debate"] = action_json
            else:
                kwargs["action_output_history"].append({
                    "action_name": action_name,
                    "action_output": action_json
                })

            return True, None

        elif action_name == 'spark':
            # Check spark parameters
            invalid_response = f"""You did not format your spark correctly. Output your spark in the following JSON format:
{{
    "bit": "string (provide 2-3 sentences that clearly state the status quo or conventional approach. Highlight the limitation or problem it creates. Include enough detail so it is self-contained and does not rely on additional context from elsewhere)",
    "flip": "string (Provide at least two sentences describing the novel approach or perspective. Explain the method or technique that enables this change. Include enough detail so that the Flip is understandable on its own. It should be a clear departure from the Bit, showing how it challenges the status quo.)",
    "spark": "string (A concise 4-6 word phrase capturing the core idea.)"
}}
"""
            if 'bit' not in action_json or 'flip' not in action_json or 'spark' not in action_json:
                return False, invalid_response
            if not isinstance(action_json.get('bit'), str):
                return False, invalid_response
            if not isinstance(action_json.get('flip'), str):
                return False, invalid_response
            if not isinstance(action_json.get('spark'), str):
                return False, invalid_response

            kwargs["action_output_history"].append({
                "action_name": action_name,
                "action_output": action_json
            })

            return True, None

        elif action_name == 'write':
            # Check write parameters
            invalid_response = f"""I did not write a valid proposal. I should output my proposal in the following JSON format:
{{
"proposal_title": "string",
"proposal_summary": "string: include a concise statement of exactly what you want to do. why should the work be done?  what specific problems will it resolve? include motivation, objectives, responsibilities, training, urgency, etc.",
"background_and_significance": "string: a historical review of existing work; what has been done?  What remains to be done? Discuss how what you propose is related to what has been done, and how it is different. Include background/review of relevant literature, how you will build on it, and why what you propose is much more novel.",
"research_plan": [
    {{
        "phase": "what is the phase of the research proposal",
        "idea": "string: detailed description of the idea for this research phase. what is the central hypothesis or concept being tested in this phase? details behind your core contribution during this phase (e.g., novel algorithms, models, or techniques you are developing).",
        "experimental_plan": "string: detailed experimental plan for the research phase, including how you specifically will conduct the research and what exact methods you will use. Be precise and specific about the methods you will use, including any specific techniques, tools/resources, or approaches you will employ. Briefly mention fallback plans if this does not work."
    }},
    ...
]
}}
"""

            if not isinstance(action_json, dict):
                print(f"Action JSON is not a dict: {action_json}")
                return False, invalid_response
            if 'proposal_title' not in action_json or 'proposal_summary' not in action_json or 'background_and_significance' not in action_json or 'research_plan' not in action_json:
                return False, invalid_response  
            if not isinstance(action_json.get('proposal_title'), str):
                return False, invalid_response
            if not isinstance(action_json.get('proposal_summary'), str):
                return False, invalid_response
            if not isinstance(action_json.get('background_and_significance'), str):
                return False, invalid_response
            if not isinstance(action_json.get('research_plan'), list):
                return False, invalid_response
            if not all((isinstance(plan, dict) and ('phase' in plan) and ('idea' in plan) and ('experimental_plan' in plan)) for plan in action_json['research_plan']):
                return False, invalid_response

            kwargs["action_output_history"].append({
                "action_name": action_name,
                "action_output": action_json
            })

            return True, None

        elif action_name == 'complete':
            return True, None

        else:
            # If the action is not recognized, return False
            return False, "You did not select a valid action. Please select one of the following actions: 'search', 'debate', 'spark', 'write', or 'complete'."

    async def generate_response(
        self, instance_id: str, messages: List[Dict[str, Any]], **kwargs
    ) -> Tuple[bool, str, float, dict]:
        content = ""
        # Get the current action
        curr_action, curr_noise = kwargs.get("action_history")[-1]
        curr_noise = int(curr_noise)
        current_turn, max_turns = kwargs.get("turns")

        problem = self._instance_dict[instance_id]["problem"]
        should_terminate_sequence = False
        self._instance_dict[instance_id]["logprobs"].append(kwargs.get("logprobs", []))

        # We get the last user message
        for i in range(len(messages) - 1, -1, -1):
            item = messages[i]
            if item.get("role") == "assistant":
                content = item['content']
                break

        # The action should be a JSON
        action_json, parsed_ok = self._safe_load_action_json(content)
        if not parsed_ok:
            logger.error(f"Failed to parse action ({curr_action}, {curr_noise}) JSON: {content}")
            self._append_action_history_once(kwargs["action_history"], curr_action, curr_noise)
            kwargs["action_jsons"].append(content)
            invalid_response = "You did not format your action correctly. Output your action in a JSON format."
            self._instance_dict[instance_id]["response"] = invalid_response
            logger.error(f"Invalid action format: {invalid_response}")

            return should_terminate_sequence, invalid_response, 0.0, {}

        # Validate the action and noise
        valid_action, invalid_response = await self.validate_action(curr_action, messages, action_json, kwargs)
        if not valid_action:
            self._append_action_history_once(kwargs["action_history"], curr_action, curr_noise)
            kwargs["action_jsons"].append(action_json)
            self._instance_dict[instance_id]["response"] = invalid_response
            logger.error(f"Invalid action: {invalid_response}\n\nInvalid content for {curr_action}: {action_json}")
            return should_terminate_sequence, invalid_response, 0.0, {}

        kwargs["action_jsons"].append(action_json)

        # Ensure that the last action is a write action
        if (current_turn >= (max_turns-2)) and (curr_action != "complete"):
            must_write_response = f"You have executed the '{curr_action}' action with a noise of {curr_noise}. However, you have reached the end of your writing process and now must write the final draft of your proposal.\n\n" + action_dict['write']['prompt'](problem)
            kwargs["action_history"].append(("write", "1"))
            self._instance_dict[instance_id]["response"] = must_write_response
            self._instance_dict[instance_id]["action"] = "write"
            self._instance_dict[instance_id]["noise"] = 1

            return should_terminate_sequence, must_write_response, 1.0, {}

        #################### Handling 'select_action' ####################
        if curr_action == "select_action":

            # Store the action and noise in the instance dictionary
            self._instance_dict[instance_id]["action_json"] = action_json
            should_terminate_sequence, response = await self.execute_select_action(instance_id, **kwargs)
            self._instance_dict[instance_id]["response"] = response

        #################### Handling 'search' ####################
        elif curr_action == "search":
            # Prompt model for the next action OR choose a random action and noise between 1 and the action's max level:
            next_action, next_noise, prompt = await self.sample_next_action(kwargs, problem)

            response = f"You have executed the 'search' action with a noise of {curr_noise}.\n\n" + prompt
            self._instance_dict[instance_id]["response"] = response
            self._instance_dict[instance_id]["action"] = next_action
            self._instance_dict[instance_id]["noise"] = next_noise

        ##################### Handling 'debate_setup' ####################
        elif curr_action == "debate_setup":
            # Prompt the model to generate the debate conversation
            kwargs["action_history"].append(('debate', str(curr_noise)))

            response = f"You have set up the debate with the following participants, topics, and structure:\n{action_json}\n\nYou have also identified the above related papers based on each of the participants' expertise and interests. Now we can proceed with the debate conversation.\n\n" + action_dict['debate']['prompt'](action_json)
            self._instance_dict[instance_id]["response"] = response
            self._instance_dict[instance_id]["action"] = "debate"
            self._instance_dict[instance_id]["noise"] = curr_noise

        ##################### Handling 'debate' ####################
        elif curr_action == "debate":
            # Prompt the model for the next action
            # Prompt model for the next action OR choose a random action and noise between 1 and the action's max level:
            next_action, next_noise, prompt = await self.sample_next_action(kwargs, problem)

            response = f"You have executed the 'debate' action with a noise of {curr_noise}.\n\n" + prompt
            self._instance_dict[instance_id]["response"] = response
            self._instance_dict[instance_id]["action"] = next_action
            self._instance_dict[instance_id]["noise"] = next_noise

        ##################### Handling 'spark' ####################
        elif curr_action == "spark":
            # Prompt the model for the next action
            # Prompt model for the next action OR choose a random action and noise between 1 and the action's max level:
            next_action, next_noise, prompt = await self.sample_next_action(kwargs, problem)

            response = f"You have executed the 'spark' action with a noise of {curr_noise}.\n\n" + prompt
            self._instance_dict[instance_id]["response"] = response
            self._instance_dict[instance_id]["action"] = next_action
            self._instance_dict[instance_id]["noise"] = next_noise

        ##################### Handling 'write' ####################
        elif curr_action == "write":
            # Prompt the model for the next action
            # Prompt model for the next action OR choose a random action and noise between 1 and the action's max level:
            next_action, next_noise, prompt = await self.sample_next_action(kwargs, problem)

            response = f"You have written the latest version of the proposal.\n\n" + prompt
            self._instance_dict[instance_id]["response"] = response
            self._instance_dict[instance_id]["action"] = next_action
            self._instance_dict[instance_id]["noise"] = next_noise

        ##################### Handling 'complete' ####################
        elif curr_action == "complete":
            # Mark the interaction as complete
            should_terminate_sequence = True

            response = "You have already completed the proposal generation process!"
            self._instance_dict[instance_id]["response"] = response
            self._instance_dict[instance_id]["action"] = "complete"
            self._instance_dict[instance_id]["noise"] = -1

        return should_terminate_sequence, response, 1.0, {}

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        return 1.0

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

    async def execute_select_action(self, instance_id: str, **kwargs) -> str:
        """Execute the selected action and return the response."""

        action_json = self._instance_dict[instance_id]["action_json"]
        problem = self._instance_dict[instance_id]["problem"]
        terminate_sequence = False

        action = action_json["action"]
        noise = action_json["noise"]

        self._instance_dict[instance_id]["action"] = action
        self._instance_dict[instance_id]["noise"] = noise

        if action == 'search':
            response = f"\n-------------------------------\n\nYou have selected the 'search' action with a noise of {noise}:\n\n" + action_dict['search']['prompt'](problem, noise)
            kwargs["action_history"].append((action, str(noise)))
        elif action == 'debate':
            response = f"\n-------------------------------\n\nYou have selected the 'debate' action with a noise of {noise}:\n\n" + action_dict['debate_setup']['prompt'](problem, noise)
            kwargs["action_history"].append(('debate_setup', str(noise)))
        elif action == 'spark':
            response = f"\n-------------------------------\n\nYou have selected the 'spark' action with a noise of {noise}:\n\n" + action_dict['spark']['prompt'](problem, noise)
            kwargs["action_history"].append((action, str(noise)))
        elif action == 'write':
            response = f"\n-------------------------------\n\nYou have selected the 'write' action:\n\n" + action_dict['write']['prompt'](problem)
            kwargs["action_history"].append((action, "1"))
        elif action == 'complete':
            response = f"\n-------------------------------\n\nYou have completed writing the proposal.\n"
            kwargs["action_history"].append((action, "1"))
            terminate_sequence = True
        else:
            response = "Invalid action selected. Please select a valid action."
            terminate_sequence = True

        return terminate_sequence, response

    async def sample_next_action(self, kwargs, problem):
        action_options = ["search", "debate_setup", "spark", "write"]
        should_swap = False # (likelihood_of_swap > 0) and (random.random() < likelihood_of_swap)

        if should_swap:
            next_action = random.choice(action_options)
            max_level = action_dict[next_action]['max_level'] if 'max_level' in action_dict[next_action] else 1
            next_noise = random.choice([opt for opt in range(1, max_level + 1)])
            if max_level == 1:
                prompt = action_dict[next_action]['prompt'](problem)
            else:
                prompt = action_dict[next_action]['prompt'](problem, noise=next_noise)

        else:
            next_action = "select_action"
            next_noise = -1
            prompt = action_dict[next_action]['prompt'](problem)

        kwargs["action_history"].append((next_action, str(next_noise)))

        return next_action, next_noise, prompt