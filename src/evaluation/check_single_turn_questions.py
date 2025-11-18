import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.file_readers import read_json
from prompts.single_turn_evaluation import get_exist_in_history_prompt, get_exist_in_note_prompt
from utils.models import call_open_llm_api


def process_file(full_path, model_id):
    try:
        sample = read_json(full_path)
        outputs = sample['outputs']
        note = sample['note']
        prompt = sample['prompt']
        history = []

        for turn in prompt[1:]:
            if turn['role'] == 'assistant':
                turn['content'] = turn['content'].split('</think>')[-1]
            history.append(turn)

        for i, output in enumerate(outputs):
            question = output['output'].split('</think>')[-1]

            if question.startswith('disease:'):
                continue

            exist_in_history_prompt = get_exist_in_history_prompt(conversation=history, question=question)
            convo_history = [{'role': 'user', 'content': exist_in_history_prompt}]
            in_history = call_open_llm_api(convo_history, model=model_id, host="http://localhost:8002")

            exist_in_note_prompt = get_exist_in_note_prompt(case_vignette=note, question=question)
            convo_note = [{'role': 'user', 'content': exist_in_note_prompt}]
            in_note = call_open_llm_api(convo_note, model=model_id, host="http://localhost:8002")

            outputs[i]['in_history'] = in_history
            outputs[i]['in_note'] = in_note

        sample['outputs'] = outputs
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)

        return f"Processed: {os.path.basename(full_path)}"

    except Exception as e:
        return f"Error processing {full_path}: {str(e)}"


def main(args):
    input_dir = args.single_turn_sample_dir
    model_id = args.model_id
    num_threads = args.num_threads

    files_to_process = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            files_to_process.append(os.path.join(root, file))

    print(f"Found {len(files_to_process)} files to process using {num_threads} threads.")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(process_file, file_path, model_id): file_path
            for file_path in files_to_process
        }

        for idx, future in enumerate(as_completed(futures)):
            result = future.result()
            print(f"[{idx + 1}/{len(files_to_process)}] {result}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single-turn question checking script")
    parser.add_argument('--single_turn_sample_dir', type=str, help="Path to single-turn sample directory")
    parser.add_argument('--model_id', type=str, help="Model path or ID to use")
    parser.add_argument('--num_threads', type=int, default=10, help="Number of threads")

    args = parser.parse_args()
    main(args)
