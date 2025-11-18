import json
import os
import argparse
from prompts.doctor import get_doctor_system_prompt

doctor_system_prompt = get_doctor_system_prompt("v1")

def build_input(conversation, turn_number):
    input_text = f'\nCurrent turn ({turn_number}):\n'
    for turn in conversation:
        role = turn['role']
        content = turn['content']
        if role in ['assistant', 'doctor']:
            role = 'assistant'
            summary = 'conversation history summary: ' + turn['summary']
            plan = turn['planning']
            
            sentence = f'{role}: <think>{summary}\nPlan: {plan}</think>{content}\n'

            input_text += sentence

        if role in ['user', 'patient']:
            role = 'user'
            input_text += f'{role}: {content}'
    return input_text


def build_output(turn):
    history = 'conversation history summary: ' + turn['summary']
    plan = turn['planning']
    content = turn['content']
    output_text = f'<think>{history}\nPlan: {plan}\n</think>{content}\n'
    return output_text


def create_single_turn_sample(conversation):
    trainset = []
    turn_number = -1
    for i in range(1, len(conversation)):
        if conversation[i]['role'] not in ['assistant', 'doctor'] or conversation[i - 1]['role'] not in ['user', 'patient']:
            continue
        turn_number += 1
        if i - 2 >= 0 and conversation[i - 2]['role'] not in ['assistant', 'doctor']:
            continue

        if i - 2 >= 0:
            if not conversation[i - 2].get('summary'):
                continue
            second_to_last_plan = conversation[i - 2].get('planning')
            if not second_to_last_plan:
                continue
    
        summary = conversation[i].get('summary')
        last_plan = conversation[i].get('planning')
        if not summary or not last_plan:
            continue

        if i - 2 >= 0:
            input = build_input(conversation[i-2:i], turn_number).strip()
        else: 
            input = build_input(conversation[i-1:i], turn_number).strip()
        output = build_output(conversation[i]).strip()
        

        input = input.strip()
        sample = {
            "instruction": doctor_system_prompt,
            "input": input,
            "output": output
        }
        # output = output.strip()
        # if '<think>' not in output:
        #     print(f"Error in output: {output}")
        #     return None


        trainset.append(sample)
    return trainset


def create_single_turn_sample_for_sampling(conversation):
    trainset = []
    turn_number = -1
    for i in range(1, len(conversation)):
        if conversation[i]['role'] not in ['assistant', 'doctor'] or conversation[i - 1]['role'] not in ['user', 'patient']:
            continue
        turn_number += 1
        if i - 2 >= 0 and conversation[i - 2]['role'] not in ['assistant', 'doctor']:
            continue
        if i - 2 >= 0:
            input = build_input(conversation[i-2:i], turn_number, True, True)
        else:
            input = build_input(conversation[i-1:i], turn_number, True, True)
        output = build_output(conversation[i], True, True).strip()

        input = input.strip()

        sample = {
            "instruction": doctor_system_prompt,
            "input": input,
            "output": output
        }
        # output = output.strip()
        # if '<think>' not in output:
        #     print(f"Error in output: {output}")
        #     return None

        trainset.append(sample)
    return trainset
        


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--dataset_dir',type=str, required=True, help="dataset_dir")
    parser.add_argument('--trainset_path',type=str, required=True, help="trainset_path")
    parser.add_argument('--with_sampling',type=str, required=True, help="with_sampling")

    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    trainset_path = args.trainset_path
    with_sampling = args.with_sampling


    dataset = []

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if not with_sampling and 'best_rollout' in file:
                continue
            full_path = os.path.join(root, file)
            with open(full_path, 'r', encoding='utf-8') as f:
                sample = json.load(f)
            conv_key = 'conv_inference' if 'conv_inference' in sample else 'conv_revised'
            eval_key = 'eval_inference' if 'eval_inference' in sample else 'eval_revised'

            conversation = sample[conv_key]['conversation']
            temp_dataset = create_single_turn_sample(conversation)
            dataset.extend(temp_dataset)


    with open(trainset_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f)
