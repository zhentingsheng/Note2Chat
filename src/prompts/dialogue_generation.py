DECISION_TREE_PROMPT = '''
Based on the provided medical note below, generate a structured decision tree for differential diagnosis. Terminate branches with potential diagnoses, including the final diagnosis as the confirmed condition. Prioritize clinical relevance and logical progression. Ensure the decision tree incorporates all symptoms mentioned in the history of present illness (HPI) to the greatest extent possible.
Structure the output in JSON format with the following keys:
tree: A nested object where each node contains:
criteria: Short clinical question or finding (e.g., 'Fever present?').
branches: Sub-nodes for 'yes'/'no' responses (if applicable).
diagnoses: List of potential diagnoses (if terminal node), each with:
condition: Diagnosis name.
confidence: Likelihood (e.g., 'high', 'moderate', 'low').
is_final: Boolean indicating if it matches the final diagnosis.

Medical Note:
Chief Complaint: {chief_complaint}
HPI: {hpi}
Final Diagnosis: {diagnosis}

Provide only the JSON output, without additional text.
Example JSON Structure (for clarity):
{{  
  "tree": {{
    "criteria": "Chief complaint: Chest pain",  
    "branches": {{
      "yes": {{
        "criteria": "Radiation to left arm?",  
        "branches": {{  
          "yes": {{
            "diagnoses": [  
              {{  
                "condition": "Acute coronary syndrome",  
                "confidence": "high",  
                "is_final": true  
              }}  
            ]  
          }},  
          "no": {{  
            "criteria": "Pleuritic pain?",  
            "branches": {{  
              "yes": {{  
                "diagnoses": [  
                  {{  
                    "condition": "Pulmonary embolism",  
                    "confidence": "moderate",  
                    "is_final": false  
                  }}  
                ]  
              }}  
            }}  
          }}  
        }}  
      }}  
    }}  
  }}  
}}
'''

def get_generate_decision_tree_prompt(chief_complaint, hpi, diagnosis):
  prompt = DECISION_TREE_PROMPT.format(chief_complaint=chief_complaint, hpi=hpi, diagnosis=diagnosis)
  return prompt


DIALOGUE_GENERATE_PROMPT = '''
generate a history-taking dialogue between a doctor and a patient based on the decision tree and the medical note. The conversation opens with a natural and concise initial statement from the patient, describing clear symptoms from Patient's chief complaint or only one symptom History of Present Illness, and concludes with five preliminary differential diagnoses—ranked by clinical likelihood—provided by the doctor.
To make the conversation clinically convincing, the doctor knows nothing about the patient in advance and the doctor should only learn information through the patient's responses, ask appropriate follow-up questions in plain, patient-friendly language, and actively compare, rule in, and rule out the potential diseases without using unexplained medical jargon.
The patient should answer exclusively based on the provided medical note, using layperson terms. Avoid volunteering unsolicited information; answers should be direct and relevant to the doctor’s questions, and strictly adhere to the facts in the medical note without fabricating or altering any information.

Input
Decision Tree: {decision_tree}
Medical Note:
Chief Complaint: {chief_complaint}
HPI: {hpi}
Final Diagnosis: {diagnosis}

### Final Output Format (JSON):
{{
  "conversation": [
    {{"role": "patient", "content": "..."}},
    {{"role": "doctor", "content": "..."}},
    ...
    {{"role": "patient", "content": "..."}},
    {{"role": "doctor", "content": "diagnosis..."}}
  ],
  "preliminary_diagnoses": [
    {{
      "disease": "Disease A",
      "reason": "Explanation of why this disease is suspected, based on the conversation."
    }},
    {{
      "disease": "Disease B",
      "reason": "Explanation of why this disease is also considered."
    }}
  ]
}}
'''

def get_generate_dialogue_prompt(chief_complaint, hpi, diagnosis,decision_tree):
  prompt = DIALOGUE_GENERATE_PROMPT.format(chief_complaint=chief_complaint, hpi=hpi, diagnosis=diagnosis, decision_tree=decision_tree)
  return prompt
