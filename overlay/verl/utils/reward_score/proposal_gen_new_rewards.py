import socket
import os
import tempfile
import json
import numpy as np
import time
from sentence_transformers import util
from verl.interactions.utils.proposal_gen_utils import (
    action_dict,
    relevance_prompt,
    feasibility_prompt,
    novelty_prompt,
    action_contribution_prompt,
    process_exploration_prompt,
)

_LOG_RUN_SUFFIX = os.environ.get("PROPOSAL_GEN_RUN_TIMESTAMP") or time.strftime("%Y%m%d_%H%M%S", time.gmtime())


############################# UTILS #############################

def log_local_worker_results(output, fname="log"):
    """Append worker-local reward logs to a temp file to avoid shared mount contention."""
    worker_id = f"{socket.gethostname()}_{os.getpid()}"
    dir_path = os.path.join(tempfile.gettempdir(), "proposal_gen_reward_logs")
    os.makedirs(dir_path, exist_ok=True)
    log_fname = f"{fname}_{_LOG_RUN_SUFFIX}_{worker_id}"
    with open(f'{dir_path}/{log_fname}.txt', 'a+', encoding='utf-8') as f:
        f.write(output)
        f.write("\n")

def compute_semantic_similarity(model, outputs, references):
    reference_embs = model.encode(references, convert_to_tensor=True)
    output_embs = model.encode(outputs, convert_to_tensor=True)

    # Compute cosine similarity
    scores = util.pytorch_cos_sim(output_embs, reference_embs).cpu().numpy()
    return scores

# Converting logprobs to entropy
def logprobs_to_entropy(logprobs):
    # num of tokens, num of logprobs, tuple of (logprob, token_id, None)
    # convert to num of tokens x num of logprobs x 1
    logprobs = [[logprob[0] for logprob in token] for token in logprobs]

    avg_logprobs = np.mean(logprobs, axis=0)  
    # Convert logprobs to probabilities
    probs = np.exp(avg_logprobs)
    # Compute entropy
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1)  # Adding a small constant to avoid log(0)
    return entropy.tolist()

def _build_final_score_payload(scores, request_id, problem, action_histories, swap_action_histories, action_jsons, action_output_histories, final_score):
    return {
        "request_id": request_id,
        "problem": problem,
        "action_history": action_histories[request_id],
        "swap_action_history": swap_action_histories[request_id],
        "action_json": action_jsons[request_id],
        "action_output_history": action_output_histories[request_id],
        "action_level_raw_entropies": scores[request_id]["action_level"]["raw_entropies"],
        "action_level_raw_similarities": scores[request_id]["action_level"]["raw_similarities"],
        "action_level_entropy": scores[request_id]["action_level"]["entropy"],
        "action_level_similarity": scores[request_id]["action_level"]["similarity"],
        "action_level_score": scores[request_id]["action_level"]["score"],
        "process_level_exploration": scores[request_id]["process_level"]["exploration"],
        "process_level_contribution": scores[request_id]["process_level"]["contribution"],
        "process_level_contribution_components": scores[request_id]["process_level"]["contribution_components"],
        "process_level_num_intermediate": scores[request_id]["process_level"]["num_intermediate"],
        "process_level_score": scores[request_id]["process_level"]["score"],
        "outcome_level_novelty": scores[request_id]["outcome_level"]["novelty"],
        "outcome_level_relevance": scores[request_id]["outcome_level"]["relevance"],
        "outcome_level_feasibility": scores[request_id]["outcome_level"]["feasibility"],
        "outcome_level_score": scores[request_id]["outcome_level"]["score"],
        "all_writes_novelty": scores[request_id]["all_writes"]["novelty"],
        "all_writes_relevance": scores[request_id]["all_writes"]["relevance"],
        "final_score": final_score,
    }

def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")

def scale_conversion(values, lower_bound=1, upper_bound=5):
    """
    Convert a list of numeric values from [lower_bound, upper_bound] to [-1, 1].

    Args:
        values (list of numbers): The values to be scaled.
        lower_bound (float, optional): The lower bound of the input range. Defaults to 1.
        upper_bound (float, optional): The upper bound of the input range. Defaults to 5.

    Returns:
        list of float: Scaled values in the range [-1, 1]. If `values` is empty, returns [-1].
    """
    if values is None or len(values) == 0:
        return [-1]
    return [(value - (upper_bound + lower_bound) / 2) / ((upper_bound - lower_bound) / 2) for value in values]  # Normalize to [-1, 1]

############################## ACTION-LEVEL REWARD ##############################

# compute the similarity between action and the last version of the written proposal (if version does not exist, compute similarity with the proposal problem)
def action_proposal_sim(model, action_attrs, problem, count_write=False):
    actions = []
    references = []

    last_proposal = problem

    for action in action_attrs:

        if (action_dict[action['action_type']]['level_enabled']) or (count_write and (action['action_type'] == 'write')):
            actions.append(str(action['action']))
            references.append(str(last_proposal))

        if action['action_type'] == 'write':
            last_proposal = action['action']

    # Compute the semantic similarity between actions and their corresponding last proposal
    similarities = compute_semantic_similarity(model, actions, references).diagonal().flatten()

    return similarities

# Computing a score based on the correlation between entropy and noise (0 means no correlation, 1 means perfect positive correlation, -1 means perfect negative correlation)
## Ignore if the action is not noise-enabled
def entropy_noise_corr(action_attrs):
    entropies = []
    noises = []
    for attr in action_attrs:
        if (action_dict[attr['action_type']]['level_enabled']):
            entropies.append(attr['entropy'])
            noises.append(attr['noise'])

    entropies = np.array(entropies)
    noises = np.array(noises)

    if len(entropies) == 0 or len(noises) == 0:
        return -1, entropies  # Return -1 if no noise-enabled actions are found
    if len(noises) == 1:
        return -0.9, entropies  # Not -1 since at least some noise-enabled actions are present

    # Compute correlation
    try:
        correlation = np.corrcoef(entropies, noises)[0, 1]
    except Exception as e:
        print("Error computing correlation between entropies and noises.", e)
        return -0.9, entropies  # Not -1 since at least some noise-enabled actions are present
    if np.isnan(correlation):
        return -0.9, entropies # Not -1 since at least some noise-enabled actions are present

    return correlation, entropies

def similarity_noise_corr(model, action_attrs, problem, count_write=False):

    noises = np.array([1 - (attr['noise']/action_dict[attr['action_type']]['max_level']) for attr in action_attrs if action_dict[attr['action_type']]['level_enabled']])

    if len(noises) == 0:
        return -1, np.array([])  # Return -1 if no noise-enabled actions are found
    if len(noises) == 1:
        return -0.9, np.array([])  # Not -1 since at least some noise-enabled actions are present

    similarities = action_proposal_sim(model, action_attrs, problem, count_write)

    # Compute correlation
    try:
        correlation = np.corrcoef(similarities, noises)[0, 1] # (higher noise should lead to lower similarity)
    except Exception as e:
        print("Error computing correlation between similarities and noises.", e)
        return -0.9, similarities  # Not -1 since at least some noise-enabled actions are present
    if np.isnan(correlation):
        return -0.9, similarities # Not -1 since at least some noise-enabled actions are present

    return correlation, similarities

# Enforce relationship between noise and generated action attributes (when we increase the noise, we should tangibly see a difference in the attributes generated):
## Entropy(a_i) - Entropy(a_{i-1}) ∝ Noise(a_i) - Noise(a_{i-1}) ∝ Sim(a_i, p_i)
def action_noise_score(model, action_attrs, problem, count_write=False):
    entropy_corr, entropies = entropy_noise_corr(action_attrs)

    similarity_corr, similarities = similarity_noise_corr(model, action_attrs, problem, count_write)

    # Combine the two correlations
    score = (similarity_corr + entropy_corr) / 2

    return entropy_corr, similarity_corr, score, entropies, similarities

############################### PROCESS-LEVEL REWARD ###############################
# Given a full process, collect all of the actions before the final write
## How much did the action contribute to the final write?
### Given the final proposal and action/action output -> how much did the action contribute towards the novelty, feasibility, and overall quality of the final proposal?

def compute_action_contributions(api_client, problem, action_outputs, final_write):
    """Score each intermediate action by its contribution to the final write.

    Returns per-action contribution scores in [-1, 1] together with raw component scores.
    """
    if not action_outputs:
        return [], []

    if not isinstance(final_write, str):
        final_write = json.dumps(final_write, ensure_ascii=False)

    def _to_text(value):
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)

    prompts = []
    previous_actions = []

    for idx, attr in enumerate(action_outputs):
        action_type = attr.get("action_name", "unknown")
        action_output = _to_text(attr.get("action_output", "No output."))
        if previous_actions:
           previous_actions_text = _to_text(previous_actions)
        else:
            previous_actions_text = "No previous actions taken."

        prompts.append(
            action_contribution_prompt(
                problem,
                final_write,
                previous_actions_text,
                action_type,
                action_output,
            )
        )

        previous_actions.append({
                "action_type": action_type,
                "action_output": action_output
            })

    responses = api_client.call_api_batch(prompts)

    per_action_scores = {idx: {"novelty": 0, "feasibility": 0, "quality": 0} for idx in range(len(action_outputs))}

    for idx, resp in enumerate(responses):
        try:
            parsed = json.loads(resp) if not isinstance(resp, dict) else resp
            novelty_score = int(parsed.get("novelty_score", 1))
            feasibility_score = int(parsed.get("feasibility_score", 1))
            quality_score = int(parsed.get("quality_score", 1))
            novelty_score = max(1, min(5, novelty_score))
            feasibility_score = max(1, min(5, feasibility_score))
            quality_score = max(1, min(5, quality_score))
        except Exception:
            novelty_score = 1
            feasibility_score = 1
            quality_score = 1
        per_action_scores[idx]["novelty"] = novelty_score
        per_action_scores[idx]["feasibility"] = feasibility_score
        per_action_scores[idx]["quality"] = quality_score

    contribution_components = []
    contributions = []
    for idx in range(len(action_outputs)):
        raw_scores = [
            per_action_scores[idx]["novelty"],
            per_action_scores[idx]["feasibility"],
            per_action_scores[idx]["quality"],
        ]
        contribution_components.append({
            "novelty": per_action_scores[idx]["novelty"],
            "feasibility": per_action_scores[idx]["feasibility"],
            "quality": per_action_scores[idx]["quality"],
        })
        valid_scores = [s for s in raw_scores if 1 <= s <= 5]
        if not valid_scores:
            contributions.append(-1.0)
            continue
        contributions.append(float(np.mean(scale_conversion(valid_scores))))

    return contributions, contribution_components

def compute_action_exploration(api_client, problem, action_outputs):
    """Score each intermediate action by its exploration value (novelty vs prior actions).

    Returns a list of per-action exploration scores in [-1, 1].
    """
    if not action_outputs:
        return []

    def _to_text(value):
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)

    prompts = []
    action_index = []
    previous_actions = []

    for idx, attr in enumerate(action_outputs):
        action_type = attr.get("action_name", attr.get("action_type", "unknown"))
        action_output = _to_text(attr.get("action_output", ""))
        if previous_actions:
           previous_actions_text = _to_text(previous_actions)
        else:
            previous_actions_text = "No previous actions taken."

        prompts.append(
            process_exploration_prompt(
                problem,
                action_type,
                action_output,
                previous_actions_text,
            )
        )
        action_index.append(idx)
        previous_actions.append({
            "action_type": action_type,
            "action_output": action_output
        })

    responses = api_client.call_api_batch(prompts)

    per_action_exploration = {idx: 0 for idx in range(len(action_outputs))}

    for resp, idx in zip(responses, action_index):
        try:
            parsed = json.loads(resp) if not isinstance(resp, dict) else resp
            score = int(parsed.get("score", 1))
            score = max(1, min(5, score))
        except Exception:
            score = 1
        per_action_exploration[idx] = score

    explorations = []
    for idx in range(len(action_outputs)):
        score = per_action_exploration[idx]
        if 1 <= score <= 5:
            explorations.append(float(scale_conversion([score])[0]))
        else:
            explorations.append(-1.0)

    return explorations


def _max_intermediate_actions(max_assistant_turns):
    """Compute max intermediate actions before final write under current loop design.

    For ProposalGenInteraction, assistant turns alternate between selecting and executing.
    With forced final write near the end, max executable actions is floor((T - 1) / 2),
    and max intermediate (excluding final write) is one less than that.
    """
    if max_assistant_turns is None:
        raise ValueError("max_assistant_turns must be provided by config for process-level reward computation")
    max_assistant_turns = int(max_assistant_turns)
    return max(0, ((max_assistant_turns - 1) // 2) - 1)

def process_level_reward(api_client, problems, request_ids, action_output_history, max_assistant_turns):
    """Compute the process-level reward for each action in the proposal generation process.

    Combines two dimensions:
    - Contribution: how much the action contributes to the final proposal quality
    - Exploration: how novel/different the action is from prior actions

    Both are scored [-1, 1] and aggregated equally into final process-level score.
    """
    contribution_scores = {request_id: [] for request_id in request_ids}
    contribution_components = {request_id: [] for request_id in request_ids}
    exploration_scores = {request_id: [] for request_id in request_ids}

    # Last action of type 'write' for each request; if does not exist, keep as None but set index to last
    final_writes = {request_id: (len(action_output_history[request_id]), None) for request_id in request_ids}

    for p_id, (problem, request_id) in enumerate(zip(problems, request_ids)):
        for idx, action_output in enumerate(reversed(action_output_history[request_id])):
            if action_output['action_name'] == 'write':
                final_writes[request_id] = (len(action_output_history[request_id]) - idx - 1, action_output['action_output'])
                break

        focus_attrs = action_output_history[request_id][:final_writes[request_id][0]]

        # Compute exploration scores (how novel each action is vs prior actions)
        explorations = compute_action_exploration(api_client, problem, focus_attrs)
        exploration_scores[request_id] = explorations

        # If there is no final write action, skip contribution computation
        if final_writes[request_id][1] is None:
            continue

        # Compute contribution scores (how much each action contributed to the final proposal)
        contributions, raw_components = compute_action_contributions(api_client, problem, focus_attrs, final_writes[request_id][1])
        contribution_scores[request_id] = contributions
        contribution_components[request_id] = raw_components

    # Aggregate contribution and exploration scores equally
    process_level_scores = {request_id: [] for request_id in request_ids}
    max_available_actions = _max_intermediate_actions(max_assistant_turns)
    for request_id in request_ids:
        if len(contribution_scores[request_id]) > 0 and len(exploration_scores[request_id]) > 0:
            # Combine contribution and exploration scores equally
            combined_scores = [
                (contrib + explore) / 2.0
                for contrib, explore in zip(contribution_scores[request_id], exploration_scores[request_id])
            ]
            process_level_scores[request_id] = combined_scores
        elif len(contribution_scores[request_id]) > 0:
            # Consider exploration as 0
            combined_scores = [
                (contrib + 0.0) / 2.0
                for contrib in contribution_scores[request_id]
            ]
            process_level_scores[request_id] = combined_scores
        elif len(exploration_scores[request_id]) > 0:
            # Consider contribution as 0
            combined_scores = [
                (0.0 + explore) / 2.0
                for explore in exploration_scores[request_id]
            ]
            process_level_scores[request_id] = combined_scores

        missing_actions = max(0, max_available_actions - len(process_level_scores[request_id]))
        if missing_actions > 0:
            process_level_scores[request_id].extend([-1.0] * missing_actions)

    # Compute the overall process-level reward
    overall_rewards = {request_id: 0.0 for request_id in request_ids}
    for request_id in request_ids:
        if len(process_level_scores[request_id]) > 0:
            overall_rewards[request_id] = float(np.mean(process_level_scores[request_id]))
        else:
            overall_rewards[request_id] = -1

    return contribution_scores, contribution_components, exploration_scores, overall_rewards


############################### OUTCOME-LEVEL REWARD ###############################
def decompose_proposal_batch(api_client, problems, request_ids, action_attrs, should_log=False):
    """Decompose each written proposal into atomic ideas."""
    # Compute the process-level scores
    core_ideas = {request_id: [] for request_id in request_ids}

    for p_id, (problem, request_id) in enumerate(zip(problems, request_ids)):
        version_id = 0
        for attr in action_attrs[request_id]:
            if attr['action_type'] == 'write':

                action_json = attr['action']
                core_ideas[request_id].append([])

                if type(action_json) is not dict:
                    print(f"Error decoding action JSON for request_id {request_id}, version {version_id}: {action_json}")
                    version_id += 1
                    continue
                if 'research_plan' not in action_json or not isinstance(action_json['research_plan'], list):
                    print(f"Invalid research plan for request_id {request_id}, version {version_id}: {action_json}")
                    version_id += 1
                    continue
                if not all((isinstance(plan, dict) and ('phase' in plan) and ('idea' in plan) and ('experimental_plan' in plan)) for plan in action_json['research_plan']):
                    print(f"Invalid research plan for request_id {request_id}, version {version_id}: {action_json}")
                    version_id += 1
                    continue

                for idea_id, idea in enumerate(action_json['research_plan']):
                    phase = idea['phase']
                    idea_text = idea['idea']
                    experimental_plan = idea["experimental_plan"]

                    what_text = f"-Phase: {phase}\n\n-Idea: {idea_text}\n\n"
                    how_text = f"-Experimental Plan: {experimental_plan}\n\n"

                    core_ideas[request_id][version_id].append({
                        "what": what_text,
                        "what_keyword": idea_text,
                        "how": how_text,
                        "how_keyword": experimental_plan
                    })
                version_id += 1

    return core_ideas


def proposal_novelty_batch_score(api_client, problems, request_ids, core_ideas):
    """Compute the proposal novelty score relative to retrieved papers."""

    # retrieve related papers from the API
    refs = [(problem, request_id, v_id, idea["what"], idea["what_keyword"]) for problem, request_id in zip(problems, request_ids) for v_id, version in enumerate(core_ideas[request_id]) for idea in version]
    retrieved_papers = api_client.batch_search([f'{i[-1]}' for i in refs])

    prompts = [novelty_prompt(problem, idea_long, related_paper) for (problem, _, _, idea_long, _), related_paper in zip(refs, retrieved_papers)]
    responses = api_client.call_api_batch(prompts)

    novelty_scores = {request_id: [[] for v_id in range(len(core_ideas[request_id]))] for request_id in request_ids}
    log_str = "Novelty responses:\n\n"
    for r, (problem, request_id, v_id, idea_long, idea_keyword) in zip(responses, refs):
        # log_str += f"{problem} (Version {v_id}):\n\n"
        # log_str += f"Idea is {idea_long}:\n{r}\n\n"
        try:
            if not isinstance(r, dict):
                response = json.loads(r)
            else:
                response = r
            novelty_scores[request_id][v_id].append(int(response['score']))
        except Exception as e:
            print(e)
            print(f"Error decoding NOVELTY JSON response: {r}")
            novelty_scores[request_id][v_id].append(0)

    # log_results(log_str)

    # Normalize scores to [-1, 1]
    for request_id in request_ids:
        for version_id in range(len(core_ideas[request_id])):
            if type(novelty_scores[request_id][version_id]) is not list:
                print(f"NON-UNIQUE PROBLEMS?: {len(problems)}; {len(set(problems))}")
                print("NOVELTY SCORE NOT LIST:", novelty_scores[request_id][version_id])
                novelty_scores[request_id][version_id] = [novelty_scores[request_id][version_id]]

            if len(novelty_scores[request_id][version_id]) == 0:
                novelty_scores[request_id][version_id] = -1
            else:
                novelty_scores[request_id][version_id] = max(-1, min(1, np.mean(scale_conversion(novelty_scores[request_id][version_id]))))

    return novelty_scores

def proposal_relevance_batch_score(api_client, problems, request_ids, core_ideas):
    """Compute the proposal relevance score based on the actions taken."""

    refs = [(problem, request_id, v_id, idea["what"]) for problem, request_id in zip(problems, request_ids) for v_id, version in enumerate(core_ideas[request_id]) for idea in version]

    prompts = [relevance_prompt(problem, idea) for (problem, _, _, idea) in refs]
    responses = api_client.call_api_batch(prompts)

    relevance_scores = {request_id: [[] for _ in range(len(core_ideas[request_id]))] for request_id in request_ids}
    for r, (problem, request_id, v_id, idea) in zip(responses, refs):
        # log_results(f"Relevance response for {problem}:\n{r}")
        try:
            if not isinstance(r, dict):
                response = json.loads(r)
            else:
                response = r
            relevance_scores[request_id][v_id].append(int(response['score']))
        except json.JSONDecodeError:
            print(f"Error decoding RELEVANCE JSON response: {r}")
            relevance_scores[request_id][v_id].append(0)

    # Normalize scores to [-1, 1]
    for request_id in request_ids:
        for version_id in range(len(core_ideas[request_id])):
            if type(relevance_scores[request_id][version_id]) is not list:
                print("RELEVANCE SCORE NOT LIST:", relevance_scores[request_id][version_id])
                relevance_scores[request_id][version_id] = [relevance_scores[request_id][version_id]]

            if len(relevance_scores[request_id][version_id]) == 0:
                relevance_scores[request_id][version_id] = -1
            else:
                relevance_scores[request_id][version_id] = max(-1, min(1, np.mean(scale_conversion(relevance_scores[request_id][version_id]))))

    return relevance_scores

def feasibility_batch_score(api_client, problems, request_ids, core_ideas):
    """Compute the feasibility score based on the actions taken."""

    refs = [(problem, request_id, (idea["how"], idea["how_keyword"])) for problem, request_id in zip(problems, request_ids) if len(core_ideas[request_id]) > 0 for idea in core_ideas[request_id][-1]]
    feasibility_scores = {request_id: -1 for request_id in request_ids}

    if not refs:
        return feasibility_scores

    # retrieve related papers from the API
    queries = [idea_keyword for (problem, _, (idea_long, idea_keyword)) in refs]
    if not queries:
        return feasibility_scores

    retrieved_papers = api_client.batch_search(queries)

    prompts = [feasibility_prompt(problem, idea_long, related_paper) for (problem, _, (idea_long, idea_keyword)), related_paper in zip(refs, retrieved_papers)]
    if not prompts:
        return feasibility_scores

    responses = api_client.call_api_batch(prompts)
    # log_results(f"Feasibility responses:\n{responses}")

    feasibility_scores = {request_id: -1 for request_id in request_ids}

    for r, (problem, request_id, (idea_long, idea_keyword)) in zip(responses, refs):
        try:
            if not isinstance(r, dict):
                response = json.loads(r)
            else:
                response = r
            feasibility_scores[request_id] = max(-1, np.min(scale_conversion([int(response['score'])])))
        except json.JSONDecodeError:
            print(f"Error decoding FEASIBILITY JSON response: {r}")
            feasibility_scores[request_id] = -1

    return feasibility_scores

############################### MAIN FUNCTIONS ###############################

def construct_scores(client, model, request_ids, problems, logprobs, action_histories, action_jsons, action_output_history, max_assistant_turns):
    entropies = {}
    for request_id in request_ids:
        entropies[request_id] = []
        for logprob in logprobs[request_id]:
            entropies[request_id].append(logprobs_to_entropy(logprob))

    action_attrs = {request_id:[{'problem': problem, 
                                 'action_type': action_name, 
                                 'noise': int(noise), 
                                 'action': action_json, 
                                 'entropy': entropy} 
                                for (action_name, noise), action_json, entropy in zip(action_histories[request_id], action_jsons[request_id], entropies[request_id])] 
                                for problem, request_id in zip(problems, request_ids)}

    scores = {
        request_id: {
            'decomposed': [],
            'action_level': {
                'raw_entropies': None,
                'raw_similarities': None,
                'entropy': None, 
                'similarity': None, 
                'score': None},
            'process_level': {
                'exploration': None,
                'contribution': None,
                'contribution_components': None,
                'num_intermediate': None,
                'score': None},
            'outcome_level': {
                'novelty': None,
                'relevance': None,
                'feasibility': None,
                'score': None # feasibility score
            },
            'all_writes': {
                'novelty': None,
                'relevance': None
            },
            'final_score': None
        } for request_id in request_ids}

    core_ideas = decompose_proposal_batch(api_client=client, problems=problems, request_ids=request_ids, action_attrs=action_attrs, should_log=False)

    # Compute action-level rewards
    for request_id, problem in zip(request_ids, problems):
        print("Action Histories", action_histories[request_id])
        # log_results(f"Action JSONs:\n\n{action_jsons[request_id]}")

        entropy_corr, similarity_corr, noise_action_score, entropies_out, similarities_out = action_noise_score(model, action_attrs[request_id], problem, count_write=False)
        scores[request_id]['action_level']['raw_entropies'] = entropies_out
        scores[request_id]['action_level']['raw_similarities'] = similarities_out
        scores[request_id]['action_level']['entropy'] = entropy_corr
        scores[request_id]['action_level']['similarity'] = similarity_corr
        scores[request_id]['action_level']['score'] = noise_action_score

    # Compute process-level rewards
    contribution_scores, contribution_components, exploration_scores, process_level_scores = process_level_reward(api_client=client, problems=problems,
                                                                                                                  request_ids=request_ids, action_output_history=action_output_history,
                                                                                                                  max_assistant_turns=max_assistant_turns)

    # Compute outcome-level rewards
    novelty_scores = proposal_novelty_batch_score(client, problems, request_ids, core_ideas)
    relevance_scores = proposal_relevance_batch_score(client, problems, request_ids, core_ideas)
    feasibility_score = feasibility_batch_score(client, problems, request_ids, core_ideas)

    for request_id, problem in zip(request_ids, problems):
        scores[request_id]['process_level']['exploration'] = exploration_scores[request_id]
        scores[request_id]['process_level']['contribution'] = contribution_scores[request_id]
        scores[request_id]['process_level']['contribution_components'] = contribution_components[request_id]
        scores[request_id]['process_level']['num_intermediate'] = len(action_output_history[request_id])
        scores[request_id]['process_level']['score'] = process_level_scores[request_id]

        scores[request_id]['outcome_level']['novelty'] = novelty_scores[request_id][-1] if len(novelty_scores[request_id]) > 0 else -1
        scores[request_id]['outcome_level']['relevance'] = relevance_scores[request_id][-1] if len(relevance_scores[request_id]) > 0 else -1
        scores[request_id]['outcome_level']['feasibility'] = feasibility_score[request_id]

        outcome_level_score = ((novelty_scores[request_id][-1] if len(novelty_scores[request_id]) > 0 else -1) 
                               + (relevance_scores[request_id][-1] if len(relevance_scores[request_id]) > 0 else -1) 
                               + feasibility_score[request_id]) / 3

        scores[request_id]['outcome_level']['score'] = outcome_level_score

        scores[request_id]['all_writes']['novelty'] = novelty_scores[request_id]
        scores[request_id]['all_writes']['relevance'] = relevance_scores[request_id]

    return scores


def compute_score(model, client, extra_info=None, reward_type="action_process_outcome", max_assistant_turns=None):

    problems = extra_info["problems"]
    request_ids = extra_info["request_ids"]
    action_histories = extra_info["action_history"]
    swap_action_histories = extra_info["swap_action_history"]
    action_output_history = extra_info["action_output_history"]
    logprobs = extra_info["logprobs"]
    action_jsons = extra_info["action_jsons"]
    scores = construct_scores(client, model, request_ids, problems, logprobs, action_histories, action_jsons, action_output_history, max_assistant_turns=max_assistant_turns)

    all_strs = []
    all_num_actions = []

    for request_id, problem in zip(request_ids, problems):
        # Add num of actions
        all_num_actions.append(len(action_histories[request_id]))

        # Compute the final score
        if reward_type == "action_process_outcome":
            # Weighted final score (less weight on action level)
            final_score = (0.2 * scores[request_id]['action_level']['score'] + 0.4 * scores[request_id]['process_level']['score'] + 0.4 * scores[request_id]['outcome_level']['score'])
        elif reward_type == "outcome":
            final_score = scores[request_id]['outcome_level']['score']
        elif reward_type == "process":
            final_score = scores[request_id]['process_level']['score']
        elif reward_type == "process_outcome":
            final_score = (scores[request_id]['process_level']['score'] + scores[request_id]['outcome_level']['score']) / 2
        else:
            raise ValueError(f"Unknown reward_type: {reward_type}")
        scores[request_id]['final_score'] = final_score

        final_payload = _build_final_score_payload(
            scores,
            request_id,
            problem,
            action_histories,
            swap_action_histories,
            action_jsons,
            action_output_history,
            final_score,
        )
        final_str = json.dumps(final_payload, indent=4, default=_json_default)
        scores[request_id]["final_str"] = final_str

        all_strs.append(final_str)

        print(f"Final Score for request {request_id} (Problem {problem}): {final_score}")

    # num_actions = (max(all_num_actions) - 1) // 2
    num_actions = 11

    if all_strs:
        log_local_worker_results("New Step:\n\n", f"worker_complete_{num_actions}")
        log_local_worker_results('\n\n'.join(all_strs), f"worker_complete_{num_actions}")

    return scores

def compute_action_process_outcome_score(data_source, solution_strs, ground_truth, model, client, extra_info=None, max_assistant_turns=None):
    scores = compute_score(model, client, extra_info=extra_info, reward_type="action_process_outcome", max_assistant_turns=max_assistant_turns)
    return scores

def compute_outcome_score(data_source, solution_strs, ground_truth, model, client, extra_info=None, max_assistant_turns=None):
    scores = compute_score(model, client, extra_info=extra_info, reward_type="outcome", max_assistant_turns=max_assistant_turns)
    return scores

def compute_process_score(data_source, solution_strs, ground_truth, model, client, extra_info=None, max_assistant_turns=None):
    scores = compute_score(model, client, extra_info=extra_info, reward_type="process", max_assistant_turns=max_assistant_turns)
    return scores

def compute_process_outcome_score(data_source, solution_strs, ground_truth, model, client, extra_info=None, max_assistant_turns=None):
    scores = compute_score(model, client, extra_info=extra_info, reward_type="process_outcome", max_assistant_turns=max_assistant_turns)
    return scores