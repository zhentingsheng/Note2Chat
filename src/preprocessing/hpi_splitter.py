import pandas as pd
from prompts.preprocessing import get_clinal_case_split_prompt
from utils.models import batch_call_open_llm_api
from utils.extractor import TextExtractor
import argparse
import json
import os

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Split HPI section from discharge notes using LLM")
    parser.add_argument("--model_id", type=str, help="model_id")
    parser.add_argument("--note_path", type=str, help="note_path")
    parser.add_argument("--note_hpi_path", type=str, help="note_hpi_path")

    args = parser.parse_args()

    model_id=args.model_id
    note_path=args.note_path
    note_hpi_path=args.note_hpi_path

    text_extractor = TextExtractor()

    discharge_note_df = pd.read_csv(note_path)

    convos = []
    note_ids = []

    for _, note in discharge_note_df.iterrows():
        note_id = note['note_id']
        hpi = note["history_of_present_illness"]

        prompt = get_clinal_case_split_prompt(hpi)
        convos.append([{'role':'user', 'content': prompt}])
        note_ids.append(note_id)

    raw_texts = batch_call_open_llm_api(convos, model=model_id, max_tokens=2048, depth_limit=1)

    note_hpi = {}

    for raw_text, note_id in zip(raw_texts, note_ids):
        result = text_extractor.extract_first_json_object(raw_text)
        pre_treatment = result['pre_treatment']
        note_hpi[note_id] = pre_treatment

    note_hpi_dir = os.path.dirname(note_hpi_path)
    if not os.path.exists(note_hpi_dir):
        os.makedirs(note_hpi_dir)

    with open(note_hpi_path, 'w', encoding='utf-8') as f:
        for note_id, hpi in note_hpi.items():
            o = {
                "note_id": note_id,
                "hpi": hpi
            }
            line = json.dumps(o, ensure_ascii=False)
            f.write(line + '\n')
