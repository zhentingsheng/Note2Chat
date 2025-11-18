import os
import json
import re

from simulator.doctor_agent import DoctorAgent
from prompts.doctor import get_doctor_system_prompt
import argparse

def get_turn_number(conversation):
    conversation_0 = conversation[0]
    if len(conversation) == 1:
        turn_number = 0
    else:
        turn_number = re.findall(r'(turn \d+:.*?)(?=turn \d+:|$)', conversation_0['summary'], flags=re.DOTALL)[-1].split(':')[0][len('turn '):]
        turn_number = int(turn_number) + 1
    return turn_number

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help="model_id")
    parser.add_argument('--lora_path', type=str, required=True, help="lora_path")
    parser.add_argument('--dataset_dir', type=str, required=True, help="dataset_dir")
    parser.add_argument('--output_dir', type=str, required=True, help="output_dir")
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)

    args = parser.parse_args()
    model_id = args.model_id
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    lora_path = args.lora_path

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    doctor_system_prompt = get_doctor_system_prompt(version='v1')

    max_turn = 26
    sampling = 10

    filenames = []
    note_ids = []
    inputs = []
    turn_numbers = []
    notes = []
    recalls = []
    diagnoses = []
    output_paths = []

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            filename = file[:-len('.json')]
            full_path = os.path.join(root, file)
            with open(full_path, 'r', encoding='utf-8') as f:
                sample = json.load(f)
            note_id = sample['note_id']
            hpi = sample['hpi']
            diagnosis = sample['diagnosis']

            conv_key = 'conv_inference' if 'conv_inference' in sample else 'conv_revised'
            eval_key = 'eval_inference' if 'eval_inference' in sample else 'eval_revised'
            conversation = sample[conv_key]['conversation'][:max_turn]
            eval = sample[eval_key]
            ground_truth_total = eval.get('symptom', {}).get('ground_truth_total')
            supported_sentences = eval.get('supported_sentences')
            if not ground_truth_total or not supported_sentences:
                continue

            format_error = False
            for i in range(len(conversation)):
                if conversation[i]['role'] in ['doctor', 'assistant']:
                    conversation[i]['role'] = 'assistant'
                    summary = conversation[i]['summary']
                    planning = conversation[i]['planning']
                    content = conversation[i]['content']
                    conversation[i]['content'] = f'<think>conversation history summary:{summary}\nPlan: \n{planning}</think>{content}'
                    del conversation[i]['summary']
                    del conversation[i]['planning']
                if conversation[i]['role'] == 'patient':
                    conversation[i]['role'] = 'user'
                    if 'single_turn_summary' in conversation[i]:
                        del conversation[i]['single_turn_summary']
                
                if i % 2 == 0 and conversation[i]['role'] != 'user':
                    format_error = True
                    break
                if i % 2 == 1 and conversation[i]['role'] != 'assistant':
                    format_error = True
                    break
            if format_error:
                continue
            
            turn_number = -1
            for i in range(len(conversation)):
                if conversation[i]['role'] != 'user':
                    continue
                turn_number += 1
                input_conversation = [{'role': 'system', 'content': doctor_system_prompt}] + conversation[:i+1]
                support_total = 0
                for item in supported_sentences:
                    if item.get('turn') and item.get('turn') <= i and item.get('turn') > 0:
                        support_total += 1

                output_path = os.path.join(output_dir, f"{filename}_{turn_number}.json")
                if os.path.exists(output_path):
                    continue
                
                filenames.append(filename)
                note_ids.append(note_id)
                inputs.append(input_conversation)
                turn_numbers.append(turn_number)
                notes.append(hpi)
                recalls.append(support_total / ground_truth_total)
                diagnoses.append(diagnosis)
                output_paths.append(output_path)

    # for i in range(len(output_paths)):
    #     prompt = inputs[i]
    #     sample_path = output_paths[i]
    #     with open(sample_path, 'r', encoding='utf-8') as f:
    #         sample = json.load(f)
    #     sample['prompt'] = prompt
    #     new_path = sample_path.replace('single_turn/sampling', 'single_turn/new_sampling')
    #     with open(new_path, 'w', encoding='utf-8') as f:
    #         json.dump(sample, f)

    # Only keep the jobs for this rank
    my_indices = [i for i in range(len(inputs)) if i % args.world_size == args.rank]
    inputs = [inputs[i] for i in my_indices]
    filenames = [filenames[i] for i in my_indices]
    note_ids = [note_ids[i] for i in my_indices]
    turn_numbers = [turn_numbers[i] for i in my_indices]
    notes = [notes[i] for i in my_indices]
    recalls = [recalls[i] for i in my_indices]
    diagnoses = [diagnoses[i] for i in my_indices]
    output_paths = [output_paths[i] for i in my_indices]

    doctor_agent = DoctorAgent(base_model_path=model_id, lora_path=lora_path, max_turn=26, sampling=sampling)

    offset, size, total = 0, 64, len(inputs)

    while offset < total:
        temp_filenames = filenames[offset: offset + size]
        temp_note_ids = note_ids[offset: offset + size]
        temp_inputs = inputs[offset: offset + size]
        temp_turn_numbers = turn_numbers[offset: offset + size]
        temp_notes = notes[offset: offset + size]
        temp_recalls = recalls[offset: offset + size]
        temp_diagnoses = diagnoses[offset: offset + size]
        temp_output_paths = output_paths[offset: offset + size]
        offset += size

        all_outputs = doctor_agent.generate_responses_in_batches(temp_inputs, size, turn_mode='single')

        i = 0
        while i < len(all_outputs):
            outputs = all_outputs[i: i + sampling]
            output_objs = [{'output': output} for output in outputs]
            filename = temp_filenames[i // sampling]
            note_id = temp_note_ids[i // sampling]
            input = temp_inputs[i // sampling] # Bug
            turn_number = temp_turn_numbers[i // sampling]
            note = temp_notes[i // sampling]
            recall = temp_recalls[i // sampling]
            diagnosis = temp_diagnoses[i // sampling]
            output_path = temp_output_paths[i // sampling]
            i += sampling
            
            sample = {
                "note_id": note_id,
                "note": note,
                'recall': recall,
                'diagnosis': diagnosis,
                "turn_number": turn_number,
                "prompt": input,
                "outputs": output_objs
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(sample, f)