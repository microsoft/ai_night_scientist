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

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager.creative import CreativeRewardManager as LegacyCreativeRewardManager


@register("creative")
class CreativeRewardManager(RewardManagerBase):
    """Experimental reward-loop adapter for legacy creative reward manager.

    The experimental reward loop expects RewardManagerBase.run_single() API,
    while the legacy creative manager exposes __call__(DataProto)->reward tensor.
    This adapter bridges the two interfaces.
    """

    def __init__(
        self,
        config,
        tokenizer,
        compute_score,
        reward_router_address=None,
        reward_model_tokenizer=None,
        num_examine: int = 0,
        reward_fn_key: str = "data_source",
        **kwargs,
    ):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.legacy_manager = LegacyCreativeRewardManager(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=self.compute_score,
            reward_fn_key=reward_fn_key,
            **kwargs,
        )

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"

        # Legacy manager is sync; run in executor to keep async contract.
        output = await self.loop.run_in_executor(None, lambda: self.legacy_manager(data, return_dict=True))
        reward_tensor = output["reward_tensor"]
        reward_extra_info = output.get("reward_extra_info", {})

        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()

        reward_score = float(reward_tensor[0, valid_response_length - 1].item())
        return {"reward_score": reward_score, "reward_extra_info": reward_extra_info}
