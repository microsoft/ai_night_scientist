from pydantic import BaseModel
from annotated_types import Len
from typing_extensions import Annotated
import json
from pydantic import BaseModel, Field

# This module defines the schemas and prompts for various actions in the research proposal writing process.

LOGPROBS_NUM = 10

################################### SCHEMAS FOR ACTIONS ###################################

class action_schema(BaseModel):
    action: str = Field(..., description="one of the following: 'query', 'debate_setup', 'spark', 'write'")
    noise: int = Field(..., ge=1, le=5, description="noise parameter for the action, ranging from 1 to 5")

class retrieval_schema(BaseModel):
    search_queries: Annotated[list[str], Len(min_length=1, max_length=5)] = Field(..., description="list of search queries")

class Participant(BaseModel):
    name: str = Field(..., description="name of the participant in the debate")
    job: str = Field(..., description="job title or role of the participant in the debate, including their affiliation or organization")
    expertise: str = Field(..., description="area of expertise or background of the participant, including their current work or research interests")

class debate_setup_schema(BaseModel):
    debate_participants: Annotated[list[Participant], Len(min_length=1, max_length=10)] = Field(..., description="list of debate participants")
    debate_topics: Annotated[list[str], Len(min_length=1, max_length=5)] = Field(..., description="list of debate topics")
    debate_structure: str = Field(..., description="description of the debate structure")

class convo_history_schema(BaseModel):
    speaker_name: str = Field(..., description="name of the participant in the debate")
    speaker_response: str = Field(..., description="message sent by the participant in the debate")

class debate_schema(BaseModel):
    conversation_history: Annotated[list[convo_history_schema], Len(min_length=1, max_length=10)] = Field(..., description="conversation history of the debate")

class spark_schema(BaseModel):
    bit: str = Field(..., description="prevailing belief or assumption in the research domain that the paper aims to challenge")
    flip: str = Field(..., description="novel approach or counterargument that the paper introduces to advance the field")
    spark: str = Field(..., description="concise phrase capturing the core idea of the reformulated research idea")

class plan_schema(BaseModel):
    phase: str = Field(..., description="name or identifier of the research phase")
    idea: str = Field(..., description="detailed description of the idea for this research phase")
    experimental_plan: str = Field(..., description="detailed description of the experimental plan for this research phase")

class proposal_schema(BaseModel):
    proposal_title: str = Field(..., description="title of the proposal")
    proposal_summary: str = Field(..., description="concise statement of what the proposal aims to do, why it is important, and what specific problems it will resolve")
    background_and_significance: str = Field(..., description="historical review of the field, including what has been done, what remains to be done, and how the proposal builds on existing knowledge")
    research_plan: Annotated[list[plan_schema], Len(min_length=1, max_length=5)] = Field(..., description="list of research phases within the proposal")

################################### TASK DEFINITION ###################################

task_definition = """We define a research proposal as a comprehensive document that outlines a research project, including its objectives, background/significance, methodology, and expected outcomes. The proposal is structured to provide a clear and coherent research plan for conducting the research, addressing key questions and challenges in the field by providing a detailed description of the research plan, including the research question, hypothesis, experimental design, and potential pitfalls for 3-4 distinct phases of the research plan. The proposal should also include a literature review that situates the research within the existing body of knowledge, highlighting gaps that the proposed research aims to fill. The proposal should answer the questions:
1. What is the research question or problem being addressed?
2. Why is this research important? What is the significance of the proposed research?
3. What research already exists in the field? What has been done, and what remains to be done?
4. What is the proposed methodology? How will the research be conducted?
The proposal should be clear and concise, using appropriate academic language and avoid repetition."""

################################### QUERY ACTION PROMPTS ###################################

# Query generation prompt for searching background information and related works
## This prompt is designed to help generate search queries based on a research proposal's title and abstract.

# Noise parameter description for query generation
query_noise_description = {
    1: "Extremely relevant to the proposal idea, focusing on discovering background information and related works based on terms extracted from the proposal/proposal title.",
    2: "Very relevant to the proposal idea, focusing on background information and related works that are closely related to the proposal idea.",
    3: "Somewhat relevant to the proposal idea, focusing on background information and related works that are tangentially related to the proposal idea.",
    4: "Not very relevant to the proposal idea, exploring background information and related works that are loosely related to the proposal idea. This can include broader topics or concepts that may not be directly related to the proposal but still provide useful context.",
    5: "Very distantly relevant to the proposal, exploring specific alternate perspectives, domains, or philosophical questions that may be of general interest."
}

# Query generation with noise parameter
def generate_queries_noise(problem, noise, current_proposal=None, context=None): 
    
    prompt = f"""You are a researcher that is starting your literature and background review for writing a research proposal. You take in a research proposal title and suggest a set of specific, unique search queries. We note that the relevancy of your queries to the proposal should depend on the input parameter, noise, which ranges from 1 to 5:
- Noise = 1: "Extremely relevant to the proposal idea, focusing on discovering background information and related works based on terms extracted from the proposal/proposal title.",
- Noise = 2: "Very relevant to the proposal idea, focusing on background information and related works that are closely related to the proposal idea.",
- Noise = 3: "Somewhat relevant to the proposal idea, focusing on background information and related works that are tangentially related to the proposal idea.",
- Noise = 4: "Not very relevant to the proposal idea, exploring background information and related works that are loosely related to the proposal idea. This can include broader topics or concepts that may not be directly related to the proposal but still provide useful context.",
- Noise = 5: "Very distantly relevant to the proposal (for example, exploring other specific alternate perspectives, different domains, or philosophical questions that may be of general interest to you)."

Here is the target problem of the proposal:
{problem}

Here is the noise parameter:
Noise: {noise} ({query_noise_description[noise]})

"""
    if current_proposal is not None:
        prompt += f"""
Here is the current proposal as reference:
{current_proposal}
"""
    if context is not None:
        prompt += f"""
Here is the context of the proposal writing process:
{context}
"""
    prompt += f"""
Please output your diverse, unique search queries in the following JSON format:
{{
"search_queries": [list of five string queries to search for]
}}

Your output JSON:

"""
    return prompt

################################### DEBATE ACTION PROMPTS ###################################
# These are the prompts for setting up the setting of the debate:
## Who to debate with?
### Less noise (1): Someone in the same field as the proposal, ideally an expert in the specific topic of the proposal (e.g., a senior researcher or mentor)
### Intermediate noise (3): Someone in the same field as the proposal, but not necessarily an expert in the specific topic (e.g., a colleague or peer)
### Maximum noise (5): Someone from a completely different domain or even multiple domain experts-- a literal mixture of experts (e.g., a domain-expert from a completely different discipline, 2-3 different experts with different points of view)?

## What to debate on?
### Less noise (1): Ideas explicitly within the proposal (e.g., specific methodologies, theories, or applications)
### Intermediate noise (3): Ideas related to the proposal, but not explicitly within it (e.g., related methodologies, theories, or motivations)
### Maximum noise (5): Not random topics, but maybe some high-level philosophical takes or what the other persona is currently “working” on within their domain (e.g., a discussion on the future of the field, or a debate on the implications of the proposal's ideas in a different field)

## How to debate?
### Less noise (1): Structured debate with clear rules and a very specific focus on the topic (e.g., a formal single-turn debate that does not deviate from the chosen topic)
### Intermediate noise (3): Open-ended, structured discussion with a general focus on the topic (e.g., a multi-turn debate that allows for some deviation from the topic and unstructured discussion)
### Maximum noise (3): Open-ended, unstructured discussion with a focus on exploring different perspectives and ideas (e.g., a multi-turn debate that allows for significant deviation from the topic and encourages exploration of different ideas)

# The description of the debate setup noise parameter (there are five levels). It should include all three aspects of the debate: who to debate with, what to debate on, and how to debate. It should be very detailed and specific, so that the user can understand what to expect from the debate:
debate_setup_noise_description = {
    1: "Setup a discussion with a colleague or peer that has explored the same/similar topic as the proposal does, focusing on specific methodologies, theories, or applications explicitly within the proposal. The topics should be specific to aspects of the proposal which the colleague/peer should give explicit feedback on. The debate should be structured with clear rules and a focus on the proposal, allowing for a formal single-turn debate that does not deviate from the chosen topic.",
    2: "Setup a discussion with a colleague or peer that has explored the related topics as the proposal does, either focusing on specific methodologies, theories, or applications explicitly within the proposal or the proposal as a whole. The debate should be structured with clear rules and a focus on the proposal, allowing for a formal single-turn debate that does not deviate from the chosen topic.",
    3: "Setup a debate with a colleague or peer that has explored the different topic from the proposal but is in the same high-level field, focusing on related methodologies, theories, or motivations that are not explicitly within the proposal. The debate should be open-ended, structured discussion with a general focus on the topic, allowing for a multi-turn debate that allows for some deviation from the topic and unstructured discussion.",
    4: "Setup a debate with a domain-expert from a different discipline, or a mixture of 2-3 different experts from the same discipline, focusing on broader topics or concepts that may not be directly related to the proposal but still provide useful context. The debate should be open-ended, structured discussion with a general focus on the topic, allowing for a multi-turn debate that allows for some deviation from the topic and unstructured discussion.",
    5: "Setup a debate with a domain-expert from a completely different discipline, or a mixture of 2-3 different experts from a mixture of disciplines or from different points of view, focusing on high-level philosophical takes or discussing what the other persona(s) is currently working on within their own domain. The debate should be open-ended, unstructured discussion with a focus on exploring different perspectives and ideas, allowing for a multi-turn debate that allows for significant deviation from the topic and encourages exploration of different ideas."
}

# Debate setup prompt for generating the debate setup based on the proposal title and noise parameter
def generate_debate_setup(problem, noise, current_proposal=None, context=None):
    
    prompt = f"""You are a researcher who takes in a research proposal title and a noise parameter, which ranges from 1 to 5. The noise parameter defines the level of noise in the debate setup, including who to debate with, what to debate on, and how to debate. The noise parameter should be interpreted as follows:
- Noise = 1: "Setup a discussion with a single senior researcher or mentor that has explored the same/similar topic as the proposal does, focusing on specific methodologies, theories, or applications explicitly within the proposal. The debate should be structured with clear rules and a focus on the proposal, allowing for a formal single-turn debate that does not deviate from the chosen topic.",
- Noise = 2: "Setup a discussion with a single colleague or peer that has explored the related topics as the proposal does, either focusing on specific methodologies, theories, or applications explicitly within the proposal or the proposal as a whole. The debate should be structured with clear rules and a focus on the proposal, allowing for a formal single-turn debate that does not deviate from the chosen topic.",
- Noise = 3: "Setup a debate with a single colleague or peer that has explored the different topic from the proposal but is in the same high-level field, focusing on related methodologies, theories, or motivations that are not explicitly within the proposal. The debate should be open-ended, structured discussion with a general focus on the topic, allowing for a multi-turn debate that allows for some deviation from the topic and unstructured discussion.",
- Noise = 4: "Setup a debate with a domain-expert from a different discipline, or a mixture of 2-3 different experts from the same discipline, focusing on broader topics or concepts that may not be directly related to the proposal but still provide useful context. The debate should be open-ended, structured discussion with a general focus on the topic, allowing for a multi-turn debate that allows for some deviation from the topic and unstructured discussion.",
- Noise = 5: "Setup a debate with a domain-expert from a completely different discipline, or a mixture of 2-3 different experts from a mixture of disciplines or from different points of view, focusing on high-level philosophical takes or discussing what the other persona(s) is currently working on within their own domain. The debate should be open-ended, unstructured discussion with a focus on exploring different perspectives and ideas, allowing for a multi-turn debate that allows for significant deviation from the topic and encourages exploration of different ideas."

Your task is to generate a detailed and specific debate setup based on the provided title and noise. The goal is to create a coherent and comprehensive debate setup that effectively communicates the debate's participants, topics, and structure.

Here is the target problem of the proposal:
{problem}

Here is the noise parameter:
Noise: {noise} ({debate_setup_noise_description[noise]})
"""
    if current_proposal is not None:
        prompt += f"""
Here is the current proposal as reference:
{current_proposal}
"""
    if context is not None:
        prompt += f"""
Here is the context of the proposal writing process:
{context}
"""
    prompt += f"""
Please output your debate setup in the following JSON format:
{{
    "debate_participants": [
        {{
            "name": "You",
            "job": "string (your job title or role, including current domain)",
            "expertise": "string (your specific area of expertise or background, including 1-2 sentences on your current work or research interests)",
        }},
        {{
            "name": "string (name of the participant)",
            "job": "string (job title or role of the participant in the debate, including their current domain",
            "expertise": "string (specific area of expertise or background of the participant; 1-2 sentences about their current work or research interests)"
        }}
    ],
    "debate_topics": ["string (specific topic or question to be debated; 1-2 sentences)", 
                     "string (another topic or question to be debated)",
                     ...],
    "debate_structure": "string (description of how to ; 1-2 sentences about the structure of the debate, e.g., open-ended discussion, single-turn debate, etc.)"
}}

ONLY output your final JSON:
"""
    return prompt

def generate_debate_conversation(debate_setup, current_proposal=None, context=None): 
    
    prompt = f"""You are a researcher who is debating with one or more participants based on the provided debate setup. Your task is to generate a conversation history of the debate, where each participant takes turns discussing the topics and questions outlined in the debate setup. The conversation should be coherent, engaging, and reflect the participants' expertise and perspectives. Each participant response should include specific details from their respective papers and have concrete thoughts, as retrieved in the above history. Nothing should be surface level.
"""
    if current_proposal is not None:
        prompt += f"""
Here is the current proposal as reference:
{current_proposal}
"""
    if context is not None:
        prompt += f"""
Here is the context of the proposal writing process:
{context}
"""
    prompt += f"""Here is the debate setup:
{debate_setup}

Please output the conversation history between yourself and the other debate participants in the following JSON format:
{{
    "conversation_history": [
        {{
            "speaker_name": "string (name of the participant)",
            "speaker_response": "string (message sent by the participant in the debate; 1-4 sentences)"
        }},
        ...
    ]
}}

Your output JSON:
"""
    return prompt

spark_noise_description = {
    1: "A minor conventional idea that you would want to challenge, which is very specific to the proposal and its current state. The Bit-Flip should be a small, focused change that challenges a specific assumption or approach within the proposal.",
    2: "A minor conventional idea that you would want to challenge, which is somewhat broad but still focused on the proposal. The Bit-Flip should be a meaningful change that questions existing assumptions or approaches within the proposal.",
    3: "A moderate conventional idea that you would want to challenge, which is somewhat broad but still focused on the proposal. The Bit-Flip should be a meaningful change that questions existing assumptions or approaches within the proposal.",
    4: "A significant conventional idea that you would want to challenge, which is broad and exploratory. The Bit-Flip should be a substantial departure from the status quo, introducing a new perspective or approach that has the potential to impact the field.",
    5: "A major conventional idea that you would want to challenge, which is very broad and exploratory. The Bit-Flip should be a significant departure from the status quo, introducing a new perspective or approach that has the potential to revolutionize the field."
}

def generate_spark(problem, noise, current_proposal=None, context=None):

    prompt = f"""You are a researcher who is proposing a novel, innovative research idea and you have suddenly thought of something that dramatically breaks conventional thinking and your own assumptions! This idea introduces a new perspective or approach to a specific field. Your task is to rethink the conventional wisdom within the problem you are workng on (including breaking away from your own assumptions and ideas throughout the proposal writing process) into a unique and innovative research idea. We call this Bit-Flip: inverting a commonly held assumption, questioning existing constraints or reapplying techniques to new domains/scales. The "Bit" is the prevailing belief, and the "Flip" is the counterargument. The noise parameter for the Bit-Flip ranges from 1 to 5, where:
- Noise = 1: "A minor conventional idea that you would want to challenge, which is very specific to the proposal and its current state. The Bit-Flip should be a small, focused change that challenges a specific assumption or approach within the proposal.",
- Noise = 2: "A minor conventional idea that you would want to challenge, which is somewhat broad but still focused on the proposal. The Bit-Flip should be a meaningful change that questions existing assumptions or approaches within the proposal.",
- Noise = 3: "A moderate conventional idea that you would want to challenge, which is somewhat broad but still focused on the proposal. The Bit-Flip should be a meaningful change that questions existing assumptions or approaches within the proposal.",
- Noise = 4: "A significant conventional idea that you would want to challenge, which is broad and exploratory. The Bit-Flip should be a substantial departure from the status quo, introducing a new perspective or approach that has the potential to impact the field.",
- Noise = 5: "A major conventional idea that you would want to challenge, which is very broad and exploratory. The Bit-Flip should be a significant departure from the status quo, introducing a new perspective or approach that has the potential to revolutionize the field."

You must structure your spark in the following manner:
1. **Bit:** A statement which identifies the prevailing belief or assumption in the research domain that the paper aims to challenge.
2. **Flip:** A statement which articulates the novel approach or counterargument that the paper introduces to advance the field.
3. **Spark:** A statement which contains the "essence of an idea", formalized as a conceptual leap.

Here is the target problem of the proposal:
{problem}

Here is the noise parameter:
Noise: {noise} ({spark_noise_description[noise]})

You want to break away from your own assumptions and ideas throughout the proposal writing process, so you should not rely on the current proposal to generate your Bit-Flip. However, you can use it as a reference to ensure that your Bit-Flip is relevant to the proposal.

"""
    if current_proposal is not None:
        prompt += f"""
Here is the current proposal as reference:
{current_proposal}
"""
    if context is not None:
        prompt += f"""
Here is the context of the proposal writing process:
{context}
"""
    prompt += f"""

Please output your problem's idea spark in the following JSON format:
{{
"bit": "string (provide 2-3 sentences that clearly state the status quo or conventional approach for your problem. Highlight the limitation or problem it creates. Include enough detail so it is self-contained and does not rely on additional context from elsewhere)",
"flip": "string (Provide at least two sentences describing your novel approach or perspective for the problem. Explain the method or technique that enables this change. Include enough detail so that the Flip is understandable on its own. It should be a clear departure from the Bit, showing how it challenges the status quo.)",
"spark": "string (A phrase capturing the core idea.)"
}}

Your output JSON:
"""
    return prompt

################################### NON-NOISE ACTION PROMPTS ###################################

# Selecting an action based on the proposal title, current version, and context history

select_action = lambda problem: f"""You are a researcher whose task is to select the best next action to take based on your existing proposal writing process. The possible actions are: "search", "debate", "spark", "write", and "complete":
- "search": Generate specific search queries to retrieve information (noise: 1-5).
- "debate": Set up a discussion with one or more participants to discuss a topic(s) (noise: 1-5).
- "spark": Generate a novel research idea that challenges conventional thinking (noise: 1-5).
- "write": Write or revise the proposal (noise: 1).
- "complete": Indicate that you are satisfied with the current proposal and the proposal writing process is complete (noise: 1).

You will also select a noise parameter for the action, which ranges from 1 to 5. The noise parameter defines the level of noise in the action, with 1 being very specific and focused, and 5 being very broad and exploratory. A low noise parameter (1-2) indicates that the action should be very relevant to the proposal, while a high noise parameter (4-5) indicates that the action can be more exploratory or tangentially related, incentivizing creativity and exploration of different ideas.
The "write", and "complete" actions do not use noise (always output "noise": 1), while the "spark", "search" and "debate" actions do use noise (output "noise": 1-5).

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

def generate_write_free_proposal(problem): 
    prompt = f"""You are a researcher who takes in all of the information and ideas that you have gathered throughout your writing process. Your task is to generate a proposal without writing it directly. Instead, focus on generating the key components and structure of the proposal based on the provided recent information. The goal is to create a coherent and comprehensive research proposal outline that effectively communicates the project's objectives, methodology, and expected outcomes. The proposal should include a title, an abstract summarizing the entire proposal, a motivation explaining why the research is important, a research question guiding the research, and a detailed description of each phase within the proposal.

Here is the target problem of the proposal:
{problem}
</problem>
"""
    
    prompt += f"""
Please output your proposal outline in the following JSON format:
{{
"proposal_title": "string",
"proposal_summary": "string: include a concise statement of exactly what you want to do. why should the work be done?  what specific problems will it resolve? include motivation, objectives, responsibilities, training, urgency, etc.",
"background_and_significance": "string: a historical review of existing work; what has been done?  What remains to be done? Discuss how what you propose is related to what has been done, and how it is different. Include background/review of relevant literature, how you will build on it, and why what you propose is much more novel.",
"research_plan": [
    {{
        "phase": "what is this phase of the research proposal",
        "idea": "string: detailed description of the idea for this research phase. what is the central hypothesis or concept being tested in this phase? details behind your core contribution during this phase (e.g., novel algorithms, models, or techniques you are developing).",
        "experimental_plan": "string: detailed experimental plan for the research phase, including how you specifically will conduct the research and what exact methods you will use. Be precise and specific about the methods you will use, including any specific techniques, tools/resources, or approaches you will employ. Briefly mention fallback plans if this does not work."
    }}
    , ...
]
}}
"""
    return prompt


################################### REWARD FUNCTION PROMPTS ###################################

relevance_prompt = lambda problem, idea: f"""You are a reviewer for NSF proposals who has been a professor in your field for over 20 years. You are tasked with evaluating the relevance of a proposal to a given problem statement.

Problem statement: "{problem}"

Given the problem statement, evaluate the following idea proposed by the proposal for its relevance. By relevance, we mean how well the idea addresses the problem statement, aligns with its objectives, and fits within the proposed methods. Consider whether the idea is directly aligned with the problem statement and whether it offers a clear approach to addressing it.

Idea: {idea}

Please provide a score from 1 to 5, where:
1 = Not relevant at all. The idea does not address the problem statement or is completely off-topic.
2 = Slightly relevant. The idea has some connection to the problem statement but lacks depth or clarity.
3 = Moderately relevant. The idea addresses the problem statement but may not fully align with its objectives or methods.
4 = Very relevant. The idea is closely aligned with the problem statement and provides a clear approach to addressing it.
5 = Highly relevant. The idea is directly aligned with the problem statement and offers a comprehensive solution.

Provide a brief explanation for your score. Your output should be in the following JSON format:
{{
    "score": <integer between 1 and 5>,
    "explanation": "<brief explanation of the score with specific references to the proposal's objectives, methods, and expected outcomes>"
}}

Please ensure your response is concise and directly addresses the relevance of the proposal to the problem statement.

Your output JSON:
"""

feasibility_prompt = lambda problem, idea, related_paper: f"""You are a reviewer for NSF proposals who has been a professor in your field for over 20 years. Your task is to evaluate the feasibility of a given core idea from a proposal related to a specific problem statement. Each idea should be scored based on its practicality, resources required, and potential challenges. We also provide a related paper to help you assess the feasibility of the idea in the context of existing work.

Problem statement: "{problem}"

Related paper:
{related_paper}

Given the problem statement, evaluate the following implementation idea for its feasibility. By feasibility, we mean how practical the idea is to implement, considering the resources available, the complexity of the idea, the ethical implications of the idea, and any potential challenges that may arise during implementation. Furthermore, if an idea is too broad or vague, it may be infeasible to implement effectively, so please consider the specificity of the idea as well. Use the related paper to help you assess the feasibility of the idea in the context of existing work.

Note that a feasible idea should not simply mention different techniques or tools without explaining how they will be applied to the specific problem at hand. A highly feasible proposal includes specific technical details wherever applicable.

Implementation Idea: {idea}

Please provide a score from 1 to 5, where:
1 = Not feasible at all. The idea is impractical, requires resources that are not available, poses significant ethical challenges, or is too vague to implement effectively.
2 = Slightly feasible. The idea has some practical elements but may require significant resources, faces ethical challenges, or is too broad to implement effectively.
3 = Moderately feasible. The idea is practical and can be implemented with available resources, but may face some challenges or ethical considerations.
4 = Very feasible. The idea is practical, requires reasonable resources, and has manageable challenges or ethical considerations.
5 = Highly feasible. The idea is practical, requires minimal resources, poses no significant ethical challenges, and is specific enough to be implemented effectively.

Provide a brief explanation for your score. Your output should be in the following JSON format:
{{
    "score": <integer between 1 and 5>,
    "explanation": "<brief explanation of the score with specific references to the idea's practicality, resources required, potential challenges, and ethical implications>"
}}

Please ensure your response is concise and directly addresses the feasibility of the idea.

Your output JSON:
"""

novelty_prompt = lambda problem, idea, related_paper: f"""You are a reviewer for NSF proposals who has been a professor in your field for over 20 years. Your task is to evaluate the novelty of a proposal's core idea in relation to a specific problem statement and a related paper. Each idea should be scored based on its originality, contribution to the field, and how it compares to existing work.

Problem statement: "{problem}"

Related paper:
{related_paper}

Given the problem statement and the related paper, evaluate the following core idea for its novelty. Consider how the idea introduces new concepts, methodologies, or perspectives that differentiate it from the related paper OR generally what has already been accomplished within the field. Assess whether the idea challenges existing paradigms or practices and whether it offers a unique contribution to the field.

Idea: {idea}

Please provide a score from 1 to 5, where:
1 = Not novel at all. The idea is a rehash of existing work, offers no new insights, and does not challenge existing assumptions.
2 = Slightly novel. The idea has some original elements but is largely derivative and does not significantly advance the field.
3 = Moderately novel. The idea introduces some new concepts or approaches but may still rely heavily on existing work.
4 = Very novel. The idea is original, offers new insights, and has the potential to significantly advance the field.
5 = Highly novel. The idea is groundbreaking, challenges existing paradigms, and offers a unique and valuable contribution to the field.

Provide a brief explanation for your score. Your output should be in the following JSON format:
{{
    "score": <integer between 1 and 5>,
    "explanation": "<brief explanation of the score with specific references to the idea's originality, contribution to the field, and comparison to the related paper>"
}}

Please ensure your response is concise and directly addresses the novelty of the idea.

Your output JSON:
"""

action_contribution_prompt = lambda problem, final_proposal, previous_actions, action_type, action_output: f"""You are a reviewer for NSF proposals who has been a professor in your field for over 20 years. Evaluate how much a single intermediate action contributed to the final proposal across three dimensions: novelty, feasibility, and overall quality.

Problem statement: "{problem}"

Final proposal:
{final_proposal}

Prior actions before this one (ordered):
{previous_actions}

Intermediate action type: {action_type}

Intermediate action output:
{action_output}

When scoring, condition on prior actions. If this action mostly repeats prior actions with similar outputs (if prior actions have been taken), assign low contribution scores.

Scoring guidance:
- Novelty contribution:
    1 = No contribution or harmful to novelty.
    2 = Slight contribution.
    3 = Moderate contribution.
    4 = Strong contribution.
    5 = Essential contribution to novelty.
- Feasibility contribution:
    1 = No contribution or harmful to feasibility.
    2 = Slight contribution.
    3 = Moderate contribution.
    4 = Strong contribution.
    5 = Essential contribution to feasibility.
- Overall quality contribution:
    1 = No contribution or harmful.
    2 = Slight contribution.
    3 = Moderate contribution.
    4 = Strong contribution.
    5 = Essential contribution.

Overall quality includes clarity, coherence, rigor, and alignment with the problem.

Output JSON format:
{{
        "novelty_score": <integer between 1 and 5>,
        "novelty_explanation": "<brief explanation of how this action impacted novelty in the final proposal>",
        "feasibility_score": <integer between 1 and 5>,
        "feasibility_explanation": "<brief explanation of how this action impacted feasibility in the final proposal>",
        "quality_score": <integer between 1 and 5>,
        "quality_explanation": "<brief explanation of how this action impacted overall quality in the final proposal>"
}}

Your output JSON:
"""

process_exploration_prompt = lambda problem, action_type, action_output, previous_actions: f"""You are a reviewer for NSF proposals who has been a professor in your field for over 20 years. Evaluate whether the current action is sufficiently different and novel compared to previous actions in the research process.

Problem statement: "{problem}"

Current action type: {action_type}

Current action output:
{action_output}

Prior actions taken (ordered):
{previous_actions}

Your task is to assess how meaningfully different the current action is from the previous actions. Consider:
- Does the current action explore a new direction or perspective not previously covered?
- Would the current action provide new information or insights that are substantially different from what prior actions generated?
- If this is a search action, are the queries fundamentally different from prior search queries or debate topics?
- If this is a debate action, are the topics/participants substantially different from prior debates?
- If this is a spark action, does it challenge different assumptions than prior spark actions?

If no prior actions have been taken, consider the current action as highly exploratory by default.

Assign a low exploration score if the current action would likely generate outputs similar to or redundant with prior actions. Assign a high exploration score if the current action would explore a genuinely new area or angle.

Score the action's exploration value from 1 to 5:
1 = Not exploratory at all. The action would generate outputs highly similar to prior actions; very redundant.
2 = Slightly exploratory. Minimal new ground covered; mostly overlaps with prior actions.
3 = Moderately exploratory. Some new perspectives or information, but with limited novelty.
4 = Very exploratory. Covers substantially new ground; good diversity from prior actions.
5 = Highly exploratory. Explores a genuinely novel direction with minimal overlap to prior actions.

Output JSON format:
{{
    "score": <integer between 1 and 5>,
    "explanation": "<brief explanation of whether this action provides sufficient exploration value relative to prior actions>"
}}

Your output JSON:
"""

################################### ACTION DICTIONARY ###################################
# This dictionary defines the actions available in the proposal writing process, including their schemas, prompts, and noise parameters.

action_dict = {
    "select_action": {
        "prompt": select_action,
        "prompt_args": ["title", "current_proposal", "context"],
        "level_enabled": False,  # This action does not use noise
        "tool_enabled": False,
        "kwargs": {"top_p": 0.99,
                   "max_new_tokens": 100,
                   "json_schema": json.dumps(action_schema.model_json_schema())}
    },
    "search": {
        "prompt": generate_queries_noise,
        "prompt_args": ["problem", "noise", "current_proposal", "context"],
        "level_enabled": True,  # This action uses noise to define the level of specificity in queries
        "level_description": query_noise_description,  # This action has a noise description to help the user understand the noise parameter
        "tool_enabled": True,
        "kwargs": {"top_p": 0.99,
                   "max_new_tokens": 256,
                   "json_schema": json.dumps(retrieval_schema.model_json_schema())},
        "max_level": max(query_noise_description.keys())
    },
    "debate_setup": {
        "prompt": generate_debate_setup,
        "prompt_args": ["problem", "noise", "current_proposal", "context"],
        "level_enabled": True,  # This action uses noise to define the level of specificity in the debate setup
        "level_description": debate_setup_noise_description,  # This action has a noise description to help the user understand the noise parameter
        "tool_enabled": True,
        "kwargs": {"top_p": 0.99,
                   "max_new_tokens": 1024,
                   "json_schema": json.dumps(debate_setup_schema.model_json_schema())},
        "max_level": max(debate_setup_noise_description.keys())
    },
    "debate": {
        "prompt": generate_debate_conversation,
        "prompt_args": ["debate_setup", "current_proposal", "context"],
        "level_enabled": True,  # This action does not use noise, but it generates a conversation based on the debate setup
        "level_description": debate_setup_noise_description,  # This action uses the same noise description as the debate setup action
        "tool_enabled": False,
        "kwargs": {"top_p": 0.99,
                   "max_new_tokens": 1024,
                   "json_schema": json.dumps(debate_schema.model_json_schema())},
        "max_level": max(debate_setup_noise_description.keys())
    },
    "spark": {
        "prompt": generate_spark,
        "prompt_args": ["problem", "noise", "current_proposal", "context"],
        "level_enabled": True,  
        "level_description": spark_noise_description,  # This action has a noise description to help the user understand the noise parameter
        "tool_enabled": False,
        "kwargs": {"top_p": 0.99,
                   "max_new_tokens": 512,
                   "json_schema": json.dumps(spark_schema.model_json_schema())},
        "max_level": max(spark_noise_description.keys())  # The maximum noise level for the search action is the maximum noise level defined in the query_noise_description
    },
    "write": {
        "prompt": generate_write_free_proposal,  
        "prompt_args": ["title", "current_proposal", "context"],
        "level_enabled": False,  # This action does not use noise, but it generates or revises the proposal based on the project idea and search results
        "tool_enabled": False,
        "kwargs": {"top_p": 0.99,
                   "max_new_tokens": 5120,
                   "json_schema": json.dumps(proposal_schema.model_json_schema())}
    },
    "complete": {
        "schema": None,
        "prompt": None,
        "prompt_args": [],
        "level_enabled": False, # This action does not use noise, but it indicates that the proposal writing process is complete
        "tool_enabled": False,
        "kwargs": {}
    }
}