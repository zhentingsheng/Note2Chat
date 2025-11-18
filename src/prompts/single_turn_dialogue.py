

ADD_SUMMARY = '''\
Rewrite the following doctor-patient conversation into a third-person, concise statement describing the patient's condition. Accurately describe the situation without omitting any details or adding any information not present in the original conversation.
Conversation:\n {conversation}
'''


def get_add_summary_prompt(conversation):
    prompt = ADD_SUMMARY.format(conversation=conversation)
    return prompt

ADD_QUESTION_PLANNING = '''\
Based on the current conversation summary and the doctor’s next action, write the doctor’s internal reasoning from a first-person perspective. The explanation should sound natural and concise, focusing on the medical diagnostic rationale—specifically, which potential conditions this action could help rule in or rule out.
conversation summary: {conversation_summary}
next action: {next_action}
Your internal reasoning:
'''

def get_add_question_planning_prompt(conversation_summary, next_action):
    prompt = ADD_QUESTION_PLANNING.format(conversation_summary=conversation_summary, next_action=next_action)
    return prompt

ADD_DDX_PLANNING = '''\
Based on the current conversation summary and the doctor’s next action (preliminary diagnoses), write the doctor’s internal reasoning from a first-person perspective. The explanation should be natural and concise, clearly state that these are preliminary diagnoses, and explain the medical diagnostic rationale—why these diseases are being considered and why some are more likely than others at this stage.
conversation summary: {conversation_summary}
next action: {next_action}
Your internal reasoning:
'''

def get_add_ddx_planning_prompt(conversation_summary, next_action):
    prompt = ADD_DDX_PLANNING.format(conversation_summary=conversation_summary, next_action=next_action)
    return prompt