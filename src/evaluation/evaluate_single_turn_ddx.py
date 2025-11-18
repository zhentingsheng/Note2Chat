import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.file_readers import read_json
from evaluation.diagnosis_evaluator import DiagnosisEvaluator

diagnosis_evaluator = None

def process_file(full_path):
    try:
        sample = read_json(full_path)
        diagnosis = sample['diagnosis']
        outputs = sample['outputs']
        prompt = sample['prompt']
        history = []

        for turn in prompt[1:]:
            if turn['role'] == 'assistant':
                turn['content'] = turn['content'].split('</think>')[-1]
            history.append(turn)

        processed = False
        for i, output in enumerate(outputs):
            preliminary_diagnoses = output['output'].split('</think>')[-1]

            if 'diagnoses' not in preliminary_diagnoses:
                continue
            match_index = diagnosis_evaluator.run(diagnosis, preliminary_diagnoses)
            processed = True
            outputs[i]['ddx_match_index'] = match_index
        if processed:
            sample['outputs'] = outputs
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(sample, f, ensure_ascii=False, indent=2)

            return f"Processed: {os.path.basename(full_path)}"
        else:
            return "Processed: No ddx"

    except Exception as e:
        return f"Error processing {full_path}: {str(e)}"


def main(args):
    input_dir = args.single_turn_sample_dir
    model_id = args.model_id
    num_threads = args.num_threads
    global diagnosis_evaluator

    diagnosis_evaluator = DiagnosisEvaluator(
        model_id=model_id
    )

    files_to_process = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            files_to_process.append(os.path.join(root, file))

    print(f"Found {len(files_to_process)} files to process using {num_threads} threads.")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(process_file, file_path): file_path
            for file_path in files_to_process
        }

        for idx, future in enumerate(as_completed(futures)):
            result = future.result()
            print(f"[{idx + 1}/{len(files_to_process)}] {result}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single-turn question checking script")
    # parser.add_argument('--single_turn_sample_dir', type=str, required=True, help="Path to single-turn sample directory")
    # parser.add_argument('--model_id', type=str, required=True, help="Model path or ID to use")
    # parser.add_argument('--num_threads', type=int, default=4, help="Number of threads")

    parser.add_argument('--single_turn_sample_dir', type=str, help="Path to single-turn sample directory")
    parser.add_argument('--model_id', type=str, help="Model path or ID to use")
    parser.add_argument('--num_threads', type=int, default=1, help="Number of threads")

    args = parser.parse_args()
    main(args)
