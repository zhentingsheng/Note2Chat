import json
import os
import argparse
import copy
from prompts.dialogue_refinement import get_critic_prompt
from utils.models import call_gpt_server
from utils.extractor import TextExtractor



def get_critic_res(chief_complaint, hpi, diagnosis, conversation, eval_supported_sentences):
    missing_sentences = []
    for s in eval_supported_sentences:
        if s['turn'] == -1:
            missing_sentences.append(s['sentence'])

    for i in range(len(conversation)):
        conversation[i]['turn'] = i
    conv_texts = [f"Turn {idx} ({c['role']}): {c['content']}" for idx, c in enumerate(conversation)]

    # prompt
    prompt = get_critic_prompt(
        chief_complaint=chief_complaint,
        hpi=hpi,
        diagnosis=diagnosis,
        conversation='\n'.join(conv_texts),
        missing_sentences='\n'.join(missing_sentences)
    )

    convo = [{'role':'user', 'content': prompt}]

    result_raw = call_gpt_server(convo, model_id, max_tokens=2048)
    result = extractor.extract_first_json_object(result_raw)
    critic_res = result['critic_res']

    return critic_res



def create_new_turns(doctor_content, patient_content):
    turns = [
        {"role": "doctor","content": doctor_content},
        {"role": "patient","content": patient_content}
    ]
    return turns

def revise_conversation(conversation, critic_res):
    for i in range(len(conversation)):
        conversation[i]['turn'] = i
    
    for c in critic_res:
        action = c.get('action', '')
        if action != 'revise_turn':
            continue
        location = c.get('location')
        content = c.get('content')

        if location is None or content is None:
            continue
        if location >= len(conversation):
            continue
        conversation[location]['content'] = content
    
    add_turn_actions = []

    for c in critic_res:
        action = c.get('action', '')
        if action != "add_turn":
            continue
        location = c.get('location')
        doctor_content = c.get('doctor')
        patient_content = c.get('patient')
        if location is None or doctor_content is None or patient_content is None:
            continue
        if location % 2 == 1:
            continue
        if location >= len(conversation) - 1:
            continue
        add_turn_actions.append({
            'location': location,
            'doctor': doctor_content,
            'patient': patient_content
        })
    
    add_turn_actions = sorted(add_turn_actions, key=lambda x: x['location'], reverse=True)

    for item in add_turn_actions:
        location = item['location']
        doctor_content = item['doctor']
        patient_content = item['patient']

        new_turns = create_new_turns(doctor_content, patient_content)
        index = -1
        for i in range(len(conversation)):
            turn = conversation[i].get('turn')
            if turn == location:
                index = i
                break
        if index == -1:
            continue
        conversation = conversation[:index + 1] + new_turns + conversation[index + 1:]

    for i in range(len(conversation)):
        conversation[i]['turn'] = i
    
    return conversation



if __name__=='__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--sample_dir',type=str, required=True, help="sample_dir")
    parser.add_argument('--model_id',type=str, required=True, help="model_id")

    args = parser.parse_args()
    sample_dir = args.sample_dir
    model_id = args.model_id

    extractor = TextExtractor()
    # train_dir = 'data/dataset/train_raw'

    for root, _, files in os.walk(sample_dir):
        for file in files:
            full_path = os.path.join(root, file)
            with open(full_path, 'r', encoding='utf-8') as f:
                sample = json.load(f)

            chief_complaint = sample['chief_complaint']
            hpi = sample['hpi']
            diagnosis = sample['diagnosis']
            eval_supported_sentences = sample['eval']['supported_sentences']
            conversation = sample['conv']['conversation'][:-1]

            critic_res = get_critic_res(chief_complaint, hpi, diagnosis, conversation, eval_supported_sentences)
            sample['critic_res'] = critic_res

            conversation = copy.deepcopy(sample['conv']['conversation'])
            ddx = {"preliminary_diagnoses": sample['conv']['preliminary_diagnoses']}
            last_turn = conversation[-1]['content']

            conversation_revised = revise_conversation(conversation, critic_res)

            conversation_revised[-1]['content'] = last_turn
            sample['conv_revised'] = {
                "conversation": conversation_revised,
                "preliminary_diagnoses": sample['conv']['preliminary_diagnoses']
            }

            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(sample, f)
