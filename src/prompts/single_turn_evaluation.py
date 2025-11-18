

EXIST_IN_HISTORY_PROMPT = ''' \
Input:
- Conversation History: {conversation}
- Question to Evaluate: {question}
Answer Yes if the information asked for in the question appears in the Conversation History. Answer No if it does not. And give the short reason.
'''


EXIST_IN_NOTE_PROMPT = '''
Input:
- Case Vignette: {case_vignette}
- Question to Evaluate: {question}

Instruction:
Decide ONLY whether the Case Vignette contains enough information to answer the Question.

Rules:
1. Do not break character and do not reveal that you are describing symptoms from the Case Vignette.
2. Do not generate any new symptoms, conditions, or knowledge that are not explicitly stated in the Case Vignette; otherwise you will be penalized.
3. Do not reveal more information than what the question asks.

Output:
- If there is enough information: Answer “Yes.”
- If there is NOT enough information: Answer “No.”
'''


def get_exist_in_history_prompt(conversation, question):
    prompt = EXIST_IN_HISTORY_PROMPT.format(conversation=conversation, question=question)
    return prompt

def get_exist_in_note_prompt(case_vignette, question):
    prompt = EXIST_IN_NOTE_PROMPT.format(case_vignette=case_vignette, question=question)
    return prompt