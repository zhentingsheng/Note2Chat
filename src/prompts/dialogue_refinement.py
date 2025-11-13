


DIALOGUE_CRITIC_PROMPT = '''\
You are criticizing a dialogue from an AI doctor agent asking a patient about their symptoms over \
an online chat interface (because it is virtual, the clinician cannot do physical exams like in a clinic). The\
patient is suffering from a particular medical problem, and the doctor hopes to understand their chief\
complaint, history of present illness in order to best determine what is going on and their likely diagnoses. \
The doctor must dig deep(potentially asking followup questions) into the particular symptoms the patient is \
complaining about and anything clinically significant. 

Make suggestions for the doctor to better meet the following criteria:
- The doctor agent avoids asking too many questions, focusing on a maximum of one or two per response.
- The responses should not reveal that the doctor agent is an AI chatbot. They should flow naturally, maintain factual accuracy, and facilitate further engagement from the patient.
- The doctor should only learn medical information from what the patient says during the conversation. They should not reference any lab results, diagnoses, hospitalizations, or medications unless the patient has brought them up.

Make suggestions for the patient to better meet the following criteria:
- The patient may respond only with facts from the Medical Note—no guessing or assumptions—using simple, layman-friendly language without medical jargon.

### Your subtasks
1. Missing facts  
For each item in missing_facts, assume that the doctor has no prior knowledge of this information — not even a hint. The doctor must not mention or imply any part of the fact directly. Instead, the doctor should ask a natural, general, and open-ended question to give the patient an opportunity to bring up the information themselves.
If the missing fact relates to past medical history, the doctor should ask a single, general question (e.g., “Do you have any past medical conditions?”), and the patient should respond using simple, layman-friendly language without going into overly specific medical details. Demographic information can be ignored.

2. Logical inconsistencies  
For each turn in the conversation, detect instances where the doctor references facts that the patient has not mentioned, or where the patient volunteers unsolicited information. Fix these by using either an add_turn (to insert a new turn after a specific one) or a revise_turn (to replace the problematic turn). Make sure to evaluate every turn in the dialogue for these issues.

### Input (provided to you)
Medical Note:
- Chief Complaint: {chief_complaint}
- HPI: {hpi}
- Final Diagnosis: {diagnosis}

Missing Facts (not mentioned in the conversation):
{missing_facts}

Conversation:
{conversation}

Please respond strictly in the following JSON format and do not include any other text:
{{
  "critic_res": [
    {{
      "action": "add_turn",
      "location": <even turn number>,  // Insert immediately AFTER a patient turn
      "doctor":   "<revised or new doctor utterance>",
      "patient":  "<corresponding patient reply>",
      "comment":  "Why this exchange is needed."
    }},
    {{
      "action": "revise_turn",
      "location": <turn number>,          // turn to be replaced
      "content":  "<full replacement text>",
      "comment":  "What was wrong and how this fixes it."
    }},
    ...
  ]
}}
'''

def get_critic_prompt(chief_complaint, hpi, diagnosis, conversation, missing_sentences):
    prompt = DIALOGUE_CRITIC_PROMPT.format(chief_complaint=chief_complaint, hpi=hpi, diagnosis=diagnosis, conversation=conversation, missing_facts=missing_sentences)
    return prompt