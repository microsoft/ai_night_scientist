# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from collections import defaultdict
import os

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.tools.utils.arxiv_utils import call_search_api
from verl.workers.reward_manager.creative_api import CreativeAzureBatchClient
from sentence_transformers import SentenceTransformer
import requests
from typing import List
import json_repair
import json


def _safe_parse_json_like(value):
    if isinstance(value, (dict, list, int, float, bool)) or value is None:
        return value

    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass

    try:
        return json_repair.loads(value)
    except (json.JSONDecodeError, RecursionError, TypeError, ValueError):
        return value

class Client:
    """A client to interact with the Azure OpenAI API."""

    # Raised when a judge call is attempted with nothing configured, so the
    # failure names the missing setting instead of surfacing as a network error.
    _NO_JUDGE = (
        "No LLM judge is configured. Set CREATIVE_AZURE_ENDPOINTS (see .env.example), "
        "or pass reward.reward_kwargs.azure_endpoints in the training config, or set "
        "CREATIVE_LLM_SINGLE_URL / CREATIVE_LLM_BATCH_URL to an OpenAI-compatible proxy."
    )

    def __init__(self, retriever_url=None, single_url=None, batch_url=None, azure_llm_config=None):
        self.retriever_url = retriever_url or os.getenv("CREATIVE_RETRIEVER_URL") or "http://127.0.0.1:8000/retrieve"
        # No default: these are deployment-specific and must be supplied explicitly.
        self.single_url = single_url or os.getenv("CREATIVE_LLM_SINGLE_URL")
        self.batch_url = batch_url or os.getenv("CREATIVE_LLM_BATCH_URL")

        # Prefer config-supplied Azure settings, with an env fallback.
        self.azure_llm_client = None
        if azure_llm_config and azure_llm_config.get("endpoints"):
            self.azure_llm_client = CreativeAzureBatchClient.from_config(azure_llm_config)
        elif os.getenv("CREATIVE_AZURE_ENDPOINTS", "").strip():
            self.azure_llm_client = CreativeAzureBatchClient.from_env()
    
    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        if not queries:
            return []

        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries, top_k=1):
        """
        Internal batch search function.
        Args:
            queries: queries to call the search engine
        Returns:
            search results
        """
        if not queries:
            return {"result": []}

        api_response, error_msg = call_search_api(
                    retrieval_service_url=self.retriever_url,
                    query_list=queries,
                    topk=top_k,
                    return_scores=True,
                    timeout=30,
                )

        if api_response is None:
            error_msg = error_msg or "Unknown error during search API call."
            print(f"Batch search API error: {error_msg}")
            return {"result": []}
        
        return api_response
    
    def call_api(self, prompt, max_num_tokens=1024):
        if self.azure_llm_client is not None:
            return self.azure_llm_client.call_api(prompt, max_num_tokens=max_num_tokens)

        if not self.single_url:
            raise RuntimeError(self._NO_JUDGE)

        # Make an HTTP request using self.single_url; we do this using the requests library (post)
        headers = {"Content-Type": "application/json"}

        params = {
            "deployment": "gpt-4.1",
            "mode": "user",
            "prompts": [prompt] if isinstance(prompt, str) else prompt,
            "tokens": max_num_tokens
            }

        responses = requests.post(self.single_url, json=params, headers=headers)

        if responses.status_code != 200:
            e = ValueError(f"API call failed with status code {responses.status_code}: {responses.text}")
            print(e)
            return []

        return _safe_parse_json_like(responses.json())

    def call_api_batch(self, prompts, max_num_tokens=1024):
        if self.azure_llm_client is not None:
            return self.azure_llm_client.call_api_batch(prompts, max_num_tokens=max_num_tokens)

        if not self.batch_url:
            raise RuntimeError(self._NO_JUDGE)

        # Make an HTTP request using self.batch_url; we do this using the requests library (post)
        headers = {"Content-Type": "application/json"}

        set_params = lambda p: {
            'deployment': 'gpt-4.1',
            'mode': 'user',
            'prompts': p,
            'tokens': max_num_tokens
            }

        params = {"requests": [set_params([str(prompt)]) for prompt in prompts]}

        responses = requests.post(self.batch_url, json=params, headers=headers)

        if responses.status_code != 200:
            e = ValueError(f"API call failed with status code {responses.status_code}: {responses.text}")
            print(e)
            return []

        responses = [_safe_parse_json_like(r) for r in responses.json()]

        return responses
    
    def _passages2string(self, retrieval_result):
        """
        Convert the retrieval result into a formatted string.
        Args:
            retrieval_result: the retrieval result from the search engine
        Returns:
            A formatted string containing the titles and contents of the retrieved documents.
        """
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n\t".join(content.split("\n")[1:])
            format_reference += f"Retrieved Paper {idx+1} ({title})\n{text}\n\n"

        return format_reference

@register("creative")
class CreativeRewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        retriever_url=None,
        llm_single_url=None,
        llm_batch_url=None,
        azure_endpoints=None,
        azure_models="gpt-4.1",
        azure_endpoint_ratios=None,
        azure_requests_per_minute=1500,
        azure_api_keys=None,
        azure_api_key_env_vars=None,
        azure_api_version="2024-12-01-preview",
        azure_temperature=0.0,
        azure_top_p=1.0,
        **kwargs,
    ) -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.sentence_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

        azure_llm_config = {
            "endpoints": azure_endpoints,
            "models": azure_models,
            "endpoint_ratios": azure_endpoint_ratios,
            "requests_per_minute": azure_requests_per_minute,
            "api_keys": azure_api_keys,
            "api_key_env_vars": azure_api_key_env_vars,
            "api_version": azure_api_version,
            "temperature": azure_temperature,
            "top_p": azure_top_p,
        }
        self.client = Client(
            retriever_url=retriever_url,
            single_url=llm_single_url,
            batch_url=llm_batch_url,
            azure_llm_config=azure_llm_config,
        )

    def __call__(self, data: DataProto, return_dict=False):
        """Score a batch of rollouts.

        Returns the reward tensor, or {"reward_tensor", "reward_extra_info"} if
        return_dict.
        """
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        data_sources = []
        response_strs = []
        ground_truths = []
        valid_lengths = []
        extra_infos = {"problems": [], "request_ids": [], "logprobs": {}, "action_history": {}, 
                   "swap_action_history": {}, "action_jsons": {}, "action_output_history": {}}

        for i in range(len(data)):
            data_item = data[i]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            data_sources.append(data_source)

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_lengths.append(valid_response_length)
            valid_response_ids = response_ids[:valid_response_length]
            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            response_strs.append(response_str)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            ground_truths.append(ground_truth)

            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", {})

            print(tool_extra_fields.keys())
            print("action_history:\n", tool_extra_fields.get("action_history", []))
            print("swap_history:\n", tool_extra_fields.get("swap_history", []))
            # print("action_jsons:\n", tool_extra_fields.get("action_jsons", []))
            print("action_output_history:\n", tool_extra_fields.get("action_output_history", []))
            
            raw_request_id = tool_extra_fields.get("request_id", None)
            if raw_request_id is None:
                request_id = f"req_{i}"
            else:
                request_id = str(raw_request_id)
            problem = extra_info["interaction_kwargs"]["problem"]
            extra_infos["problems"].append(problem)  # Keep problems for backward compatibility
            extra_infos["request_ids"].append(request_id)

            # Agent-loop path: these fields are emitted as non-tensor extra fields.
            if "logprobs" in tool_extra_fields:
                extra_infos["logprobs"][request_id] = tool_extra_fields.get("logprobs", [])
                extra_infos["action_history"][request_id] = tool_extra_fields.get("action_history", [])
                extra_infos["swap_action_history"][request_id] = tool_extra_fields.get("swap_history", [])
                extra_infos["action_output_history"][request_id] = tool_extra_fields.get("action_output_history", [])
                extra_infos["action_jsons"][request_id] = tool_extra_fields.get("action_jsons", [])
            # Legacy path: preserved for backward compatibility with old rollout output format.
            elif request_id in data_item.meta_info:
                extra_infos["logprobs"][request_id] = data_item.meta_info[request_id].get("logprobs", [])
                extra_infos["swap_action_history"][request_id] = data_item.meta_info[request_id].get("swap_history", [])
                extra_infos["action_history"][request_id] = data_item.meta_info[request_id].get("action_history", [])
                extra_infos["action_jsons"][request_id] = data_item.meta_info[request_id].get("action_jsons", [])
                extra_infos["action_output_history"][request_id] = data_item.meta_info[request_id].get("action_output_history", [])
            else:
                extra_infos["logprobs"][request_id] = []
                extra_infos["action_history"][request_id] = []
                extra_infos["swap_action_history"][request_id] = []
                extra_infos["action_jsons"][request_id] = []
                extra_infos["action_output_history"][request_id] = []

        scores = self.compute_score(
                data_source=data_sources,
                solution_strs=response_strs,
                ground_truth=ground_truths,
                model=self.sentence_model,
                client=self.client,
                extra_info=extra_infos
            )

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        for i, request_id in enumerate(extra_infos["request_ids"]):
            valid_response_length = valid_lengths[i]
            reward_tensor[i, valid_response_length - 1] = scores[request_id]['final_score']
            reward_extra_info["action_level_entropy"].append(scores[request_id]["action_level"].get("entropy"))
            reward_extra_info["action_level_similarity"].append(scores[request_id]["action_level"].get("similarity"))
            reward_extra_info["action_level_score"].append(scores[request_id]["action_level"].get("score"))
            reward_extra_info["process_level_num_intermediate"].append(
                scores[request_id]["process_level"].get("num_intermediate")
            )
            reward_extra_info["process_level_score"].append(scores[request_id]["process_level"].get("score"))
            reward_extra_info["outcome_level_feasibility"].append(
                scores[request_id]["outcome_level"].get("feasibility")
            )
            reward_extra_info["outcome_level_score"].append(scores[request_id]["outcome_level"].get("score"))
            reward_extra_info["final_score"].append(scores[request_id].get("final_score"))
            reward_extra_info["reward_log_json"].append(scores[request_id].get("final_str", ""))

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor