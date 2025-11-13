from prompts.preprocessing import get_hpi_sentence_split_prompt
from utils.models import batch_call_open_llm_api
from utils.extractor import TextExtractor
import argparse
import json
import os

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Split HPI sentences using LLM")
    parser.add_argument("--model_id", type=str, help="model_id")
    parser.add_argument("--note_hpi_path", type=str, help="note_hpi_path")
    parser.add_argument("--hpi_sentences_path", type=str, help="hpi_sentences_path")

    args = parser.parse_args()

    model_id=args.model_id
    note_hpi_path=args.note_hpi_path
    hpi_sentences_path=args.hpi_sentences_path

    text_extractor = TextExtractor()

    convos = []
    note_ids = []

    with open(note_hpi_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            hpi = data['hpi']
            note_id = data['note_id']

            prompt = get_hpi_sentence_split_prompt(hpi)
            convos.append([{'role': 'user', 'content': prompt}])
            note_ids.append(note_id)

    raw_texts = batch_call_open_llm_api(convos, model=model_id, max_tokens=2048)

    note_id_hpi_sentences = {}

    for raw_text, note_id in zip(raw_texts, note_ids):
        result = text_extractor.extract_first_json_array(raw_text)
        note_id_hpi_sentences[note_id] = result
       

    hpi_sentence_dir = os.path.dirname(hpi_sentences_path)
    if not os.path.exists(hpi_sentence_dir):
        os.makedirs(hpi_sentence_dir)

    with open(hpi_sentences_path, 'w', encoding='utf-8') as f:
        for note_id, hpi_sentences in note_id_hpi_sentences.items():
            o = {
                "note_id": note_id,
                "sentences": hpi_sentences
            }
            line = json.dumps(o, ensure_ascii=False)
            f.write(line + '\n')
