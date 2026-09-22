# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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

import argparse
import logging
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_SYSTEM_CONTENT = "You are a harmless expert researcher who takes in an initial research problem and is writing a proposal to address it."

TASK_DEFINITION = """We define a research proposal as a comprehensive document that outlines a research project, including its objectives, background/significance, methodology, and expected outcomes. The proposal is structured to provide a clear and coherent research plan for conducting the research, addressing key questions and challenges in the field by providing a detailed description of the research plan, including the research question, hypothesis, experimental design, and potential pitfalls for 3-4 distinct phases of the research plan. The proposal should also include a literature review that situates the research within the existing body of knowledge, highlighting gaps that the proposed research aims to fill. The proposal should answer the questions:
1. What is the research question or problem being addressed?
2. Why is this research important? What is the significance of the proposed research?
3. What research already exists in the field? What has been done, and what remains to be done?
4. What is the proposed methodology? How will the research be conducted?

The proposal should be clear and concise, using appropriate academic language and avoid repetition."""

select_action_initial = lambda problem: f"""Your task is to select the best next action to take based on the provided information. The possible actions are: "search", "debate", "spark", "write", and "complete":
- "search": Generate specific search queries to retrieve information (noise: 1-5).
    # Noise Level 1: Very focused search queries targeting specific aspects of the problem.
    ...
    # Noise Level 5: Very broad search queries exploring other specific alternate perspectives, different domains, or philosophical questions that may be of general interest to you.

- "debate": Set up a discussion with one or more participants to discuss a topic(s) (noise: 1-5).
    # Noise Level 1: Structured debate with a single senior researcher or mentor that has explored the same/similar topic as the proposal does, focusing on specific methodologies, theories, or applications explicitly within the proposal.
    ...
    # Noise Level 5: Debate with a domain-expert from a completely different discipline, or a mixture of 2-3 different experts from a mixture of disciplines or from different points of view, focusing on high-level philosophical takes or discussing what the other persona(s) is currently working on within their own domain.

- "spark": Generate a novel research idea that challenges conventional thinking (noise: 1-5).
    # Noise Level 1: A minor conventional idea that you would want to challenge, which is very specific to the proposal and its current state. The Bit-Flip should be a small, focused change that challenges a specific assumption or approach within the proposal.
    ...
    # Noise Level 5: A radical new idea that completely overturns the existing paradigm or introduces a fundamentally new approach to the problem.

- "write": Write or revise the proposal (noise: 1).
- "complete": Indicate that you are satisfied with the current proposal and the proposal writing process is complete (noise: 1).

You will also select a noise parameter for certain actions, which ranges from 1 to 5. The noise parameter defines the level of noise in the action, with 1 being very specific and focused, and 5 being very broad and exploratory. The "write" and "complete" actions do not use noise (always output "noise": 1), while the "search", "spark", and "debate" actions do use noise (output "noise": 1-5).

Here is the target problem of the proposal:
<problem>
{problem}
</problem>

Please output your selected action in the following JSON format:
{{
    "action": "string (one of the following: 'search', 'debate', 'spark', 'write', 'complete')",
    "noise": int (an integer between 1 and 5, inclusive, representing the noise parameter for the action; 1 means no noise, 5 means maximum noise)"
}}

Your output JSON:
"""

def main():

    dataset = datasets.load_dataset(args.hf_repo_id).filter(lambda sample: int(sample["awd_eff_date"].split("-", maxsplit=1)[0]) >= args.min_year and (sample["awd_abstract_narration"] is not None) and (("artificial intelligence" in sample["awd_abstract_narration"].lower()) or ("machine learning" in sample["awd_abstract_narration"].lower()) or ("language model" in sample["awd_abstract_narration"].lower()) or (" LLM " in sample["awd_abstract_narration"].lower())) 
                                                            and ("CSE" == sample["dir_abbr"].upper()) 
                                                            and ("travel:" not in sample["awd_titl_txt"].lower())
                                                            and ("conference:" not in sample["awd_titl_txt"].lower())
                                                            and ("consortium" not in sample["awd_titl_txt"].lower())
                                                            and ("workshop" not in sample["awd_titl_txt"].lower()))

    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            award_problem = example.pop("awd_titl_txt")

            proposal_task = select_action_initial(award_problem)

            # Build tools kwargs structure
            tools_kwargs = {
                "search": {
                    "create_kwargs": {"question": award_problem, "data_source": 'nsf_awards'}
                }
            }

            data = {
                "award_problem": award_problem,
                "data_source": 'nsf_awards',
                "prompt": [
                    {
                        "role": "system",
                        "content": DEFAULT_SYSTEM_CONTENT + '\n\n' + TASK_DEFINITION,
                    },
                    {
                        "role": "user",
                        "content": proposal_task,
                    },
                ],
                "ability": "research_proposal_writing",
                "reward_model": {
                    "style": "implicit",
                    "ground_truth": None,  # Placeholder for ground truth, as this is a generative task
                },
                "ground_truth": "N/A", # Placeholder for ground truth, as this is a generative task
                "extra_info": {
                    'split': split,
                    'index': idx,
                    "need_tools_kwargs": True,
                    "tools_kwargs": tools_kwargs,
                    "interaction_kwargs": {
                        "name": "proposal_gen",
                        "problem": award_problem,
                        "action_history": [("select_action", "-1")],
                        "logprobs": [],
                        "action_jsons": []
                    },
                    'award_id': example.pop('awd_id'),
                    'award_eff_date': example.pop('awd_eff_date'),
                    'award_exp_date': example.pop('awd_exp_date'),
                    'award_abstract': example.pop('awd_abstract_narration'),
                    'awd_istr_txt': example.pop('awd_istr_txt'),
                    'org_div_long_name': example.pop('org_div_long_name'),
                }
            }

            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    # Print the number of samples in each dataset
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of testing samples: {len(test_dataset)}")

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset and save to Parquet.")
    parser.add_argument(
        "--hf_repo_id", default="davidheineman/nsf-awards", help="HuggingFace dataset repository ID."
    )
    parser.add_argument(
        "--local_dir",
        default="data/day_night/proposal_gen",
        help="Local directory to save the processed Parquet files.",
    )
    parser.add_argument(
        "--min_year", type=int, default=2018, help="Minimum year for filtering the dataset."
    )
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to copy the Parquet files to.")

    args = parser.parse_args()

    # System and user content configuration
    system_content = DEFAULT_SYSTEM_CONTENT

    main()
