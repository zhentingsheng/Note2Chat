import json
import os
import argparse
import demjson3
from concurrent.futures import ThreadPoolExecutor, as_completed

from prompts.single_turn_dialogue import (
    get_add_summary_prompt,
    get_add_question_planning_prompt,
    get_add_ddx_planning_prompt
)
from utils.models import call_open_llm_api


def format_conversation(conversation):
    for i in range(len(conversation)):
        if 'turn' in conversation[i]:
            del conversation[i]['turn']
        if 'thought' in conversation[i]:
            del conversation[i]['thought']
        if conversation[i]['role'] == 'user':
            conversation[i]['role'] = 'patient'
        if conversation[i]['role'] == 'assistant':
            conversation[i]['role'] = 'doctor'
    return conversation


def process_file(full_path, output_dir, model_id):
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            sample = json.load(f)
        conv_key = 'conv_inference' if 'conv_inference' in sample else 'conv_revised'
        conversation = sample[conv_key]['conversation']
        preliminary_diagnoses = sample[conv_key]['preliminary_diagnoses']
        try:
            while isinstance(preliminary_diagnoses, str):
                preliminary_diagnoses = demjson3.decode(preliminary_diagnoses)
            last_turn = ''
            if isinstance(preliminary_diagnoses, dict):
                preliminary_diagnoses = preliminary_diagnoses['preliminary_diagnoses']
            for item in preliminary_diagnoses:
                last_turn += f'disease: {item["disease"]}, reason: {item["reason"]}, '
            conversation[-1]['content'] = last_turn
        except Exception:
            pass

        conversation = format_conversation(conversation)

        for turn in range(0, len(conversation), 2):
            start = max(turn - 1, 0)
            end = turn + 1

            single_turn = conversation[start:end]
            conv_text = '\n'.join([
                f'turn {i}: {single_turn[i]["role"]}: {single_turn[i]["content"]}'
                for i in range(len(single_turn))
            ])

            prompt = get_add_summary_prompt(conv_text)
            convo = [{'role': 'user', 'content': prompt}]
            single_turn_summary = call_open_llm_api(convo, model=model_id)

            conversation[turn]['single_turn_summary'] = single_turn_summary

        summary = ''
        paired_turn = -1
        for turn in range(1, len(conversation), 2):
            paired_turn += 1
            single_turn_summary = conversation[turn - 1]['single_turn_summary']
            summary += f'\nturn {paired_turn}: {single_turn_summary}'
            summary = summary.strip()
            conversation[turn]['summary'] = summary

            next_action = conversation[turn]['content']

            if turn < len(conversation) - 1:
                prompt = get_add_question_planning_prompt(
                    conversation_summary=summary,
                    next_action=next_action
                )
            else:
                prompt = get_add_ddx_planning_prompt(
                    conversation_summary=summary,
                    next_action=next_action
                )

            convo = [{'role': 'user', 'content': prompt}]
            planning = call_open_llm_api(convo, model=model_id)
            conversation[turn]['planning'] = planning

        sample[conv_key]['conversation'] = conversation

        output_path = os.path.join(output_dir, os.path.basename(full_path))
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)

        return f'Processed: {os.path.basename(full_path)}'

    except Exception as e:
        return f'Error processing {full_path}: {str(e)}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-threaded conversation processor")
    parser.add_argument('--multi_turn_sample_dir', type=str, required=True, help="multi_turn_sample_dir")
    parser.add_argument('--model_id', type=str, required=True, help="model_id")
    parser.add_argument('--single_turn_sample_dir', type=str, required=True, help="single_turn_sample_dir")
    parser.add_argument('--num_threads', type=int, default=4, help="Number of threads to use")

    args = parser.parse_args()

    input_dir = args.multi_turn_sample_dir
    output_dir = args.single_turn_sample_dir
    model_id = args.model_id
    num_threads = args.num_threads

    files_to_process = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            output_path = os.path.join(output_dir, file)
            if os.path.exists(output_path):
                continue
            files_to_process.append(os.path.join(root, file))
    
    files_to_process = files_to_process

    print(f"Found {len(files_to_process)} files to process using {num_threads} threads.")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_file = {
            executor.submit(process_file, file_path, output_dir, model_id): file_path
            for file_path in files_to_process
        }

        for idx, future in enumerate(as_completed(future_to_file)):
            result = future.result()
            print(f"[{idx+1}/{len(files_to_process)}] {result}")
