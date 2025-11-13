
CLINICAL_CASE_SPLIT = '''\
Given the following clinical case: {medical_note}, please divide the information into two distinct sections by finding a logical midpoint. Keep the original wording and sequence intact, do not modify the original text:
1. Before Medical Intervention (Pre-Treatment):  
   This section should include details about the patient's initial complaints, medical history, symptom onset, and the patient's communication with the physician, including any self-treatment attempts or prior actions. It should focus on the period before any medical interventions were applied.
2. After Medical Intervention (Post-Treatment):  
   This section should include the patient's vital signs, lab results, diagnostic measures, treatments administered, and the progression of symptoms following medical intervention. Focus on how the patient’s condition changed after the interventions.
Please output in JSON format with the following structure, and do not include any other content:
{{
  "pre_treatment": "",
  "post_treatment": ""
}}
'''

HPI_SENTENCE_SPLIT = '''\
1. Split the medical note into as many individual sentences as possible based on periods (".").
2. Expand all medical abbreviations to their full form.
3. Remove all specific measurements or technical terms from test results. Only describe what the result shows in layman's language.
4. Remove any sentence that is unclear, ambiguous.
Medical note:
{hpi}
Please respond strictly in the following JSON format and do not include any other text:
[
   "sentence1",
   "sentence2",
   ...
]
'''

FACT_PMH_TREATMENT_CLASSIFY = '''\
Determine whether the following fact describes a past medical history or prior treatment. 
Label as "yes" if the fact refers to:
- A known chronic disease or pre-existing condition
- A treatment or medication the patient has received in the past (including current medications for chronic conditions)
- A treatment or medication the patient received before the current hospital visit (including recent urgent care or outpatient visits)
Label as "no" if the fact only describes:
- Current symptoms, current physical findings, or diagnostic impressions
- Denials of symptoms
Respond with "yes" or "no" only.
Fact: {fact}
Category:
'''

FACT_SITE_CLASSIFY = '''\
Determine whether the following fact describes the exact location of a symptom or issue. 
Label as "yes" if the fact refers to:
- A specific anatomical site or region (e.g., left chest, right lower quadrant, behind the eye)
- A localized area where symptoms are experienced

Label as "no" if the fact refers only to:
- The nature of the symptom (e.g., sharp, burning)
- Timing, severity, or other non-location details

Respond with "yes" or "no" only.
Fact: {fact}
Category:
'''


FACT_ONSET_CLASSIFY = '''\
Determine whether the following fact describes the onset of a symptom or condition.
Label as "yes" if the fact refers to:
- When the symptom or issue began
- The manner of onset, such as sudden or gradual

Label as "no" if the fact refers to:
- Duration, frequency, or severity
- Site or nature of the symptom

Respond with "yes" or "no" only.
Fact: {fact}
Category:
'''


FACT_CHARACTER_CLASSIFY = '''\
Determine whether the following fact describes the character or quality of a symptom.
Label as "yes" if the fact refers to:
- The nature or sensation of the symptom (e.g., sharp, dull, burning, throbbing)

Label as "no" if the fact refers to:
- Location, timing, or severity
- Associated symptoms or modifying factors

Respond with "yes" or "no" only.
Fact: {fact}
Category:
'''

FACT_RADIATION_CLASSIFY = '''\
Determine whether the following fact describes symptom radiation (whether the symptom spreads elsewhere).
Label as "yes" if the fact refers to:
- The symptom moving or radiating to another location (e.g., pain radiates to the jaw or arm)

Label as "no" if the fact only describes:
- The primary site of the symptom
- Character or severity of the symptom

Respond with "yes" or "no" only.
Fact: {fact}
Category:
'''


FACT_TIMING_CLASSIFY = '''\
Determine whether the following fact describes the timing of a symptom or condition.
Label as "yes" if the fact refers to:
- Duration, frequency, or temporal pattern (e.g., intermittent, constant, started two days ago)

Label as "no" if the fact only describes:
- Onset, location, or severity
- Modifying factors or character

Respond with "yes" or "no" only.
Fact: {fact}
Category:
'''

FACT_MODIFYING_FACTORS_CLASSIFY = '''\
Determine whether the following fact describes factors that worsen or relieve the symptom.
Label as "yes" if the fact refers to:
- Activities, medications, or situations that make the symptom better or worse (e.g., worsens with exertion, relieved by rest)

Label as "no" if the fact only describes:
- The symptom itself, location, or severity
- Associated symptoms or timing

Respond with "yes" or "no" only.
Fact: {fact}
Category:
'''

FACT_SEVERITY_CLASSIFY = '''\
Determine whether the following fact describes the severity of a symptom.
Label as "yes" if the fact refers to:
- The intensity or degree of the symptom (e.g., pain rated 8/10, mild headache, severe discomfort)

Label as "no" if the fact only describes:
- Timing, character, or associated symptoms
- Location or radiation

Respond with "yes" or "no" only.
Fact: {fact}
Category:
'''

FACT_ASSOCIATED_SYMPTOMS_CLASSIFY = '''\
Determine whether the following fact describes symptoms associated with the main complaint.
Label as "yes" if the fact refers to:
- Other symptoms that occur together with or secondary to the main issue (e.g., nausea with chest pain)

Label as "no" if the fact only describes:
- The primary symptom itself
- Modifying factors, timing, or severity

Respond with "yes" or "no" only.
Fact: {fact}
Category:
'''


def get_clinal_case_split_prompt(medical_note):
  prompt = CLINICAL_CASE_SPLIT.format(medical_note=medical_note)
  return prompt


def get_hpi_sentence_split_prompt(hpi):
  prompt = HPI_SENTENCE_SPLIT.format(hpi=hpi)
  return prompt


def get_fact_pmh_treatment_classify_prompt(fact):
  prompt = FACT_PMH_TREATMENT_CLASSIFY.format(fact=fact)
  return prompt


def get_fact_site_classify_prompt(fact):
  prompt = FACT_SITE_CLASSIFY.format(fact=fact)
  return prompt

def get_fact_onset_classify_prompt(fact):
  prompt = FACT_ONSET_CLASSIFY.format(fact=fact)
  return prompt

def get_fact_character_classify_prompt(fact):
  prompt = FACT_CHARACTER_CLASSIFY.format(fact=fact)
  return prompt

def get_fact_radiation_classify_prompt(fact):
  prompt = FACT_RADIATION_CLASSIFY.format(fact=fact)
  return prompt

def get_fact_timing_classify_prompt(fact):
  prompt = FACT_TIMING_CLASSIFY.format(fact=fact)
  return prompt

def get_fact_modifying_factors_classify_prompt(fact):
  prompt = FACT_MODIFYING_FACTORS_CLASSIFY.format(fact=fact)
  return prompt

def get_fact_severity_classify_prompt(fact):
  prompt = FACT_SEVERITY_CLASSIFY.format(fact=fact)
  return prompt


def get_fact_associated_symptoms_classify_prompt(fact):
  prompt = FACT_ASSOCIATED_SYMPTOMS_CLASSIFY.format(fact=fact)
  return prompt