
COMPARE_SENTENCES_WITH_CONVERSATION = '''\
You are given two inputs:
A list of sentences from a patient's medical note.
A multi-turn conversation between a patient and a doctor.

Your goal is:
For each sentence in the medical note:
Identify the latest turn in the conversation (whether spoken by the doctor or the patient) that mentions the information in that sentence, based on meaning.
If a matching statement appears, output the turn number where it first appears.
If no statement in the conversation corresponds to the sentence, output -1.

Important rules:
Match based on meaning, not necessarily exact wording.
Consider both doctor and patient utterances.
Pick the latest (last) turn that matches.
Be strict: if the information was not mentioned, output -1.

Input:
sentences of medical note: 
{note_sentences}
conversation: 
{conversation}
Please respond strictly in the following JSON format and do not include any other text:
[
  {{
    "index": 0,
    "sentence": "<medical note sentence>",
    "turn": <turn number> // use -1 if not asked
  }},
  ...
]
'''

def get_compare_sentences_with_conversation_prompt(note_sentences, conversation):
    prompt = COMPARE_SENTENCES_WITH_CONVERSATION.format(
        note_sentences=note_sentences,
        conversation=conversation
    )
    return prompt




MATCH_DIAGNOSIS_PROMPT = """
You are given a ground truth diagnosis and a list of candidate diseases.
Your task is to determine the index (starting from 0) of the first disease in the list that is a valid match for the ground truth diagnosis based on **medical meaning**.
A candidate disease is considered a match if:
- It exactly matches the ground truth diagnosis, OR
- It is a **more specific subtype** of the ground truth diagnosis — that is, the ground truth is a **broader category** that includes the candidate disease.
Do not match based on text similarity alone. Use your medical knowledge to judge whether the candidate disease is a specific instance of the broader ground truth diagnosis.
ground truth diagnosis: {diagnosis}
candidate diseases: {ddx_list}
Return your result in the following JSON format:
```json
{{
  "match_index": INDEX
}}
"""

def get_match_diagnosis_prompt(diagnosis, ddx_list):
    prompt = MATCH_DIAGNOSIS_PROMPT.format(diagnosis=diagnosis, ddx_list=ddx_list)
    return prompt
