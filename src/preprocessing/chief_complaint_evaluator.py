import os
import json
import argparse
from evaluation.sentence_evaluator import SentenceEvaluator


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="evaluate chief complaint")
    parser.add_argument("--model_id", type=str, help="model_id")
    parser.add_argument("--gpt_dialogues_dir", type=str, help="gpt_dialogues_dir")
    parser.add_argument("--chief_complaint_hit_sentences_dir", type=str, help="chief_complaint_hit_sentences_dir")
    
    args = parser.parse_args()
    gpt_dialogues_dir = args.gpt_dialogues_dir
    model_id = args.model_id
    chief_complaint_hit_sentences_dir = args.chief_complaint_hit_sentences_dir

    if not os.path.exists(chief_complaint_hit_sentences_dir):
        os.makedirs(chief_complaint_hit_sentences_dir)

    sentence_evaluator = SentenceEvaluator(model_id=model_id)

    note_ids = []
    conversations = []
    for root, _, files in os.walk(gpt_dialogues_dir):
        for file in files:
            full_path = os.path.join(root, file)
            with open(full_path, 'r', encoding='utf-8') as f:
                sample = json.load(f)
            note_id = sample['note_id']
            conversation = sample['conv']['conversation'][:1]

            
            max_retries = 3

            while max_retries:
                try:
                    supported_sentences, _ = sentence_evaluator.evaluate(note_id, conversation, evaluate_chief_complaint=True)
                    break
                except Exception as e:
                    print(f"Attempt {attempt} failed: {e}")
                    max_retries -= 1
                    continue

            data = {
                "note_id": note_id,
                "conv":conversation,
                "supported_sentences": supported_sentences
            }
            output_path = os.path.join(chief_complaint_hit_sentences_dir, f'{note_id}.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
