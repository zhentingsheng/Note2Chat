import argparse
import pandas as pd
import time
import json
import traceback
from prompts.dialogue_generation import get_generate_decision_tree_prompt, get_generate_dialogue_prompt
from utils.models import call_gpt_server
from utils.extractor import TextExtractor
import os
import csv


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--note_path',type=str, required=True, help="note path")
    parser.add_argument('--note_hpi_path',type=str, required=True, help="note_hpi_path")
    parser.add_argument('--sample_dir',type=str, required=True, help="sample_dir")
    parser.add_argument('--model_id',type=str, required=True, help="model_id")

    args = parser.parse_args()
    note_path = args.note_path
    note_hpi_path = args.note_hpi_path
    sample_dir = args.sample_dir
    model_id = args.model_id

    extractor = TextExtractor()
    
    
    note_hpi = {}
    with open(note_hpi_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            note_id = sample.get('note_id')
            hpi = sample.get('hpi')
            if note_id is None or hpi is None:
                continue
            note_hpi[note_id] = hpi

    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    start_time = time.time()
    count = 0
    discharge_note_df = pd.read_csv(note_path)
    total = len(discharge_note_df)
    for _, note in discharge_note_df.iterrows():
        note_id, chief_complaint, history_of_present_illness, diagnosis = note[['note_id', "chief_complaint", "history_of_present_illness", 'final_diagnosis']]
        
        chief_complaint = chief_complaint.strip()
        
        hpi = note_hpi.get(note_id)
        if hpi is None:
            continue
        
        sample = {
            "note_id": note_id,
            "chief_complaint": chief_complaint,
            "hpi": hpi,
            "diagnosis": diagnosis,
        }
        output_path = f'{sample_dir}/{note_id}.json'
        try:
            # step 1: generate decision tree
            prompt = get_generate_decision_tree_prompt(chief_complaint=chief_complaint, hpi=hpi, diagnosis=diagnosis)

            convo = [{'role': 'user', 'content': prompt}]
            
            res_text = call_gpt_server(convo, model_id, max_tokens=8192)

            decision_tree = extractor.extract_first_json_object(res_text)
            sample['decision_tree'] = decision_tree
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(sample, f)
            
            # step 2: generate dialogue
            prompt = get_generate_dialogue_prompt(chief_complaint=chief_complaint, hpi=hpi, diagnosis=diagnosis,decision_tree=decision_tree)
            convo = [{'role': 'user', 'content': prompt}]

            res_text = call_gpt_server(convo, model_id, max_tokens=8192)
  
            conv = extractor.extract_first_json_object(res_text)

            preliminary_diagnosis = conv['preliminary_diagnoses']

            ddx = 'preliminary_diagnoses:'

            for i, item in enumerate(preliminary_diagnosis):
                disease = item['disease']
                ddx += f'\n{i + 1}. {disease}'

            conv['conversation'][-1]['content'] = ddx

            sample['conv'] = conv
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(sample, f)

        except Exception:
            error_str = traceback.format_exc()
            print("[ERROR]", error_str)
        count += 1
        elapsed_time = time.time() - start_time
        print(f"Generate Dialogue | Processed {count} / {total} notes in {elapsed_time:.2f} seconds.")
        # break