import os
import json
from collections import defaultdict
import shutil
from pipelines.experiment_tools import Experiment, run_experiment, sentence_evaluator, diagnosis_evaluator

note2chat_dir = '/home'
exp = Experiment(
    name = "multi_turn_sft_sampling",
    model_path = f'{note2chat_dir}/Note2Chat/models/Qwen2.5-7B-Instruct',
    lora_path = f"{note2chat_dir}/Note2Chat/adapters/my_multi_turn_qwen2.5_7b_sft",
    turn_mode = "multi",
    dataset_dir = "data/gpt_dialogues",
)

for i in range(15):
    output_dir = f'data/sampling/{i}'
    exp.output_dir = output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    exp.output_dir = output_dir

    run_experiment(exp)

    sentence_evaluator.run_on_data_batch_from_dir(exp.output_dir, max_workers=10)
    diagnosis_evaluator.run_on_data_batch_from_dir(exp.output_dir, max_workers=10)



merged_dialogues_dir = 'data/gpt_dialogues_and_sampling_best_rollout'
if not os.path.exists(merged_dialogues_dir):
    os.makedirs(merged_dialogues_dir)

gpt_dialogues_dir = "data/gpt_dialogues"
for root, _, files in os.walk(gpt_dialogues_dir):
    for file in files:
        full_path = os.path.join(root, file)
        target_path = os.path.join(merged_dialogues_dir, file)
        shutil.copy(full_path, target_path)

note_id_samples = defaultdict(list)

sampling_dir = 'data/sampling'

for root, sub_dirnames, _ in os.walk(sampling_dir):
    for sub_dirname in sub_dirnames:
        print(f'Processing sub-directory: {sub_dirname}')
        sub_dir = os.path.join(root, sub_dirname)
        for _, _, files in os.walk(sub_dir):
            for file in files:
                full_path = os.path.join(sub_dir, file)
                with open(full_path, 'r', encoding='utf-8') as f:
                    sample = json.load(f)
                note_id = sample['note_id']
                recall = sample.get('eval_inference', {}).get('symptom', {}).get('recall', 0)
                conversation = sample.get('conv_inference', {}).get('conversation', None)
                if conversation is None:
                    continue
                conv_len = len(conversation)
                repeated = 0
                
                for turn in conversation:
                    if turn['role'] in ['patient', 'user'] and turn['content'] == "Sorry, you've already asked this question.":
                        repeated += 1
                ddx = sample['eval_inference'].get('diagnosis_rank', -1)
                score = recall
                if ddx == -1:
                    score -= 1

                note_id_samples[note_id].append([score, full_path])


for note_id, samples in note_id_samples.items():
    samples.sort(key=lambda x: x[0], reverse=True)
    full_path = samples[0][1]
    file_name = os.path.basename(full_path)[:-len('.json')] + "_best_rollout.json"
    target_path = os.path.join(merged_dialogues_dir, file_name)
    shutil.move(full_path, target_path)

