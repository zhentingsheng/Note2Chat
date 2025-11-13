import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from prompts.preprocessing import (
    get_fact_pmh_treatment_classify_prompt,
    get_fact_site_classify_prompt,
    get_fact_onset_classify_prompt,
    get_fact_character_classify_prompt,
    get_fact_radiation_classify_prompt,
    get_fact_timing_classify_prompt,
    get_fact_modifying_factors_classify_prompt,
    get_fact_severity_classify_prompt,
    get_fact_associated_symptoms_classify_prompt
)
from utils.models import call_open_llm_api

lock = Lock()

CLASSIFY_TASKS = {
    "pmh_treatment": get_fact_pmh_treatment_classify_prompt,
    "site": get_fact_site_classify_prompt,
    "onset": get_fact_onset_classify_prompt,
    "character": get_fact_character_classify_prompt,
    "radiation": get_fact_radiation_classify_prompt,
    "timing": get_fact_timing_classify_prompt,
    "modifying_factors": get_fact_modifying_factors_classify_prompt,
    "severity": get_fact_severity_classify_prompt,
    "associated_symptoms": get_fact_associated_symptoms_classify_prompt,
}

def classify_sentence(note_id, sentence, model_id):
    results = {}
    for task_name, prompt_func in CLASSIFY_TASKS.items():
        prompt = prompt_func(fact=sentence)
        convo = [{'role': 'user', 'content': prompt}]
        try:
            raw_output = call_open_llm_api(convo=convo, model=model_id).strip().lower()
            results[task_name] = raw_output.startswith("yes")
        except Exception as e:
            results[task_name] = None
    return note_id, sentence, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multithreaded multi-category classifier using LLM")
    parser.add_argument("--model_id", type=str, required=True, help="LLM model ID")
    parser.add_argument("--hpi_sentences_path", type=str, required=True, help="Input HPI sentences file")
    parser.add_argument("--hpi_sentence_categories_path", type=str, required=True, help="Output classification file")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads to use")

    args = parser.parse_args()

    model_id = args.model_id
    hpi_sentences_path = args.hpi_sentences_path
    hpi_sentence_categories_path = args.hpi_sentence_categories_path
    num_threads = args.num_threads

    note_sentences = {}
    with open(hpi_sentences_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            note_id = data['note_id']
            sentences = data['sentences']
            note_sentences[note_id] = sentences

    hpi_sentence_categories_dir = os.path.dirname(hpi_sentence_categories_path)
    if not os.path.exists(hpi_sentence_categories_dir):
        os.makedirs(hpi_sentence_categories_dir)

    note_sentence_categories = {}

    tasks = []
    for note_id, sentences in note_sentences.items():
        if note_id not in note_sentence_categories:
            note_sentence_categories[note_id] = {}
        for sentence in sentences:
            if sentence not in note_sentence_categories[note_id]:
                tasks.append((note_id, sentence))
    # tasks = tasks[:100]
    print(f"Processing {len(tasks)} sentences across {num_threads} threads with {len(CLASSIFY_TASKS)} categories.")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(classify_sentence, nid, sent, model_id) for nid, sent in tasks]
        for idx, future in enumerate(as_completed(futures)):
            try:
                note_id, sentence, result = future.result()
                with lock:
                    note_sentence_categories[note_id][sentence] = result
                print(f"[{idx+1}/{len(tasks)}] Done: {sentence[:50]}...")
                
                if idx % 1000 == 0:
                    with open(hpi_sentence_categories_path, 'w', encoding='utf-8') as f:
                        json.dump(note_sentence_categories, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Error: {str(e)}")

    with open(hpi_sentence_categories_path, 'w', encoding='utf-8') as f:
        json.dump(note_sentence_categories, f, ensure_ascii=False, indent=2)
