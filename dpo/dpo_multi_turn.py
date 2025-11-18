import os
import json
from peft import LoraConfig
import argparse
import torch
from datasets import Dataset
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import DPOConfig, DPOTrainer
from collections import defaultdict
import random

DOCTOR_PROMPT_V1 = '''\
You are a doctor. Your task is to gather the patient's symptoms without giving explanations or sharing \
impressions, simulating real-world medical interviews. In conducting the history-taking, you should use a \
decision tree framework for differential diagnosis. To make the conversation \
clinically convincing, you should ask appropriate follow-up questions in plain, patient-friendly language, \
and actively compare, rule in, and rule out the potential diseases without using unexplained medical jargon. \

For each symptom that arises, consider—when clinically relevant—asking about:
1. Site – exact location  
2. Onset – when it began (sudden vs. gradual)  
3. Character – quality or nature (e.g., sharp, dull, burning)  
4. Radiation – whether it spreads elsewhere  
5. Associated symptoms – what occurs alongside it  
6. Timing – duration, frequency, pattern  
7. Exacerbating / Relieving factors – what worsens or improves it  
8. Severity – intensity on an easy-to-understand scale

After gathering sufficient information, state five final diagnoses.\
'''


def get_system_instruct():
    return {
        "role": "system",
        "content": DOCTOR_PROMPT_V1
    }

def calculate_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def sample_without_shuffle(prompts, chosens, rejecteds, margins, max_sample):
    indices = list(range(len(prompts)))
    selected_indices = random.sample(indices, min(max_sample, len(prompts)))
    selected_indices.sort() 

    sampled_prompts = [prompts[i] for i in selected_indices]
    sampled_chosens = [chosens[i] for i in selected_indices]
    sampled_rejecteds = [rejecteds[i] for i in selected_indices]
    sampled_margins = [margins[i] for i in selected_indices]

    return sampled_prompts, sampled_chosens, sampled_rejecteds, sampled_margins


def prepare_data_for_multi_turn(dataset_dir, each_note_max_samples, config_dir):
    note_id_trajectories = defaultdict(list)
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            full_path = os.path.join(root, file)
            with open(full_path, 'r', encoding='utf-8') as f:
                sample = json.load(f)

            eval_key = 'eval_inference' if 'eval_inference' in sample else 'eval_revised'
            conv_key = 'conv_inference' if 'conv_inference' in sample else 'conv_revised'

            recall = sample.get(eval_key, {}).get('recall', {}).get('recall', None)
            conversation = sample.get(conv_key, {}).get('conversation', None)
            ddx_rank = sample.get(eval_key, {}).get('diagnosis_rank', None)
            support_total = sample.get(eval_key, {}).get('recall', {}).get('support_total', None)

            if recall is None or conversation is None or ddx_rank is None or support_total is None:
                continue

            if len(conversation) <= 2:
                continue
            
            conv_len = len(conversation) - 2
            precision = support_total / (conv_len / 2)
            if precision > 1.0:
                precision = 1.0
            # f1 = calculate_f1(precision, recall)
            
            note_id = sample['note_id']
            conv_len = len(conversation)
            conversation = [get_system_instruct()] + conversation
            repeated = 0
            for i in range(len(conversation)):
                if 'thought' in conversation[i]:
                    del conversation[i]['thought']
                if 'turn' in conversation[i]:
                    del conversation[i]['turn']
                if conversation[i]['role'] == 'doctor':
                    conversation[i]['role'] = 'assistant'
                if conversation[i]['role'] == 'patient':
                    conversation[i]['role'] = 'user'
                
                if conversation[i]['role'] == 'user' and conversation[i]['content'] == "Sorry, you've already asked this question.":
                    repeated += 1
                
                if i != 0 and i % 2 == 1 and conversation[i]['role'] != 'user':
                    print(full_path, i, conversation[i]['role'])
                if i != 0 and i % 2 == 0 and conversation[i]['role'] != 'assistant':
                    print(full_path, i, conversation[i]['role'])
                    

            prompt = conversation[:2]
            response = conversation[2:]
            score = (5 - ddx_rank) / 5 * recall
            if ddx_rank < 0:
                score = 0

            note_id_trajectories[note_id].append({'prompt': prompt, 'response': response, 'score': score})

    print(len(note_id_trajectories.keys()))

    prompts = []
    chosens = []
    rejecteds = []
    margins = []
    for note_id, trajectories in note_id_trajectories.items():
        if len(trajectories) < 2:
            continue
        trajectories = sorted(trajectories, key=lambda x:x['score'],reverse=True)
        temp_prompts = []
        temp_chosens = []
        temp_rejecteds = []
        temp_margins = []

        for i in range(len(trajectories)):
            for j in range(i + 1, len(trajectories)):
                prompt = trajectories[i]['prompt']
                chosen = trajectories[i]['response']
                rejected = trajectories[j]['response']
                margin = trajectories[i]['score'] - trajectories[j]['score']
                temp_prompts.append(prompt)
                temp_chosens.append(chosen)
                temp_rejecteds.append(rejected)
                temp_margins.append(margin)
        sampled_prompts, sampled_chosens, sampled_rejecteds, sampled_margins = sample_without_shuffle(temp_prompts, temp_chosens, temp_rejecteds, temp_margins, each_note_max_samples)

        for prompt, chosen, rejected, margin in zip(sampled_prompts, sampled_chosens, sampled_rejecteds, sampled_margins):
            prompts.append(prompt)
            chosens.append(chosen)
            rejecteds.append(rejected)
            margins.append(margin)

    print(f"Loaded {len(prompts)} prompts")
    print(f"Loaded {len(chosens)} chosen responses")
    print(f"Loaded {len(rejecteds)} rejected responses")
    # print(f"Loaded {len(margins)} margins")

    print("==========example===========")
    print("prompt: ", prompts[0])
    print("chosen: ", chosens[0])
    print("rejected: ", rejecteds[0])
    # print("margin: ", margins[0])

    dataset = Dataset.from_dict({"prompt": prompts, "chosen": chosens, "rejected": rejecteds, "margin": margins})

    dataset.to_json(f"{config_dir}/trainset.json", orient="records", lines=True)

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="multi_turn_dpo")
    parser.add_argument('--model_name',type=str, required=True, help="model_name")
    parser.add_argument('--trainset_name',type=str, required=True, help="trainset_name")
    parser.add_argument('--each_note_max_samples',type=int, required=True, help="each_note_max_samples")
    parser.add_argument('--output_dir',type=str, required=True, help="output_dir")
    parser.add_argument('--dataset_dir',type=str, required=True, help="dataset_dir")

    args = parser.parse_args()

    model_name = args.model_name
    trainset_name = args.trainset_name
    each_note_max_samples = args.each_note_max_samples
    output_dir = args.output_dir
    dataset_dir = args.dataset_dir


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = "dpo_" + trainset_name + model_name.split("/")[-1].lower() + '_' + timestamp

    config_dir = f'{output_dir}/config'

    config_path = os.path.join(config_dir, 'run_config.json')
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    output_dir = f'{output_dir}/output'

    train_dataset = prepare_data_for_multi_turn(dataset_dir, each_note_max_samples, config_dir)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.bos_token = tokenizer.eos_token
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Number of training samples: {len(train_dataset)}")

    training_args = DPOConfig(
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_strategy="steps",
        logging_steps=1,
        save_steps=500,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        learning_rate=2e-5,
        output_dir=output_dir,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=15,
        bf16=True,
        remove_unused_columns=False,
        loss_type = "sigmoid",
        run_name=run_name,
        beta=0.1,
        # max_target_length=2048,
        max_completion_length=2048,
        max_prompt_length=512,
        eval_strategy="no",
        report_to="wandb",
    )

    # 5. initialize the DPO trainer
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )

    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    print("begin to train")

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(output_dir)

    # 7. save
    output_dir = os.path.join(output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)

    print("done")
