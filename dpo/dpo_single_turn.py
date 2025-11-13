import os
from peft import LoraConfig
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import DPOConfig
# from dpo_trainer import MyDPOTrainer
from trl import DPOTrainer
from datetime import datetime
import json
from collections import defaultdict
import re
import argparse
import random
from statistics import mean, stdev
from typing import List, Dict

DEFAULT_HISTORY_TAKING_PROMPT = '''\
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

def build_single_turn_input(conversation):
    conversation = conversation[1:]
    turn_number = int(len(conversation) / 2)
    if len(conversation) >= 2:
        conversation = conversation[-2:]
    
    input_text = f'Current turn ({turn_number}):\n'

    for turn in conversation:
        role = turn['role']
        content = turn['content']
        sentence = f'{role}: {content}\n'
        input_text += sentence
    input_text = input_text.strip()
    return input_text



def calc_score(in_history, in_note, ddx_match_index, recall, gpt_recall):
    if in_history is None and in_note is None and ddx_match_index is None:
        return None
    if ddx_match_index is not None:
        if ddx_match_index == -1:
            return 0
        # recall *= 2
        # if recall > 1:
        #     recall = 1
        recall_weight = recall / gpt_recall
        return (5 - ddx_match_index) / 5 * recall_weight
    
    if in_history is not None and in_note is not None:
        # if in_history:
        #     return -1
        # if in_note:
        #     return 1
        # return 0
        if 'no' in in_history.lower() and 'yes' in in_note.lower():
            return 1
        else:
            return 0
    return None

def construct_dpo_pairs(
        trajectories,
        margin: float = 0.20,
        max_neg_per_pos: int = 3,
        tie_eps: float = 0.05,
        seed: int | None = None
) -> List[Dict[str, str]]:
    """
    Build preference pairs for Direct Preference Optimisation (DPO).

    Parameters
    ----------
    samples            : list of answer strings (one prompt, N ≥ 2)
    rewards            : list of floats, same length and order as `samples`
    margin             : minimum reward gap required between a pair
    max_neg_per_pos    : how many negative examples to pair with each positive
    tie_eps            : rewards whose absolute difference < tie_eps are skipped
    seed               : optional RNG seed for reproducibility

    Returns
    -------
    List[dict] with keys {"chosen", "rejected"}
    """

    assert len(trajectories) == len(trajectories) >= 2, "Lists must have the same length ≥ 2"

    if seed is not None:
        random.seed(seed)

    # --- 1. Sort all (sample, reward) pairs descending by reward ---
    # ranked = sorted(zip(samples, rewards), key=lambda x: x[1], reverse=True)

    ranked = [(traj['prompt'], traj['response'], traj['score']) for traj in trajectories]
    # prompts = [traj['prompt'] for traj in trajectories]
    # response = [traj['response'] for traj in trajectories]
    rewards = [traj['score'] for traj in trajectories]

    # --- 2. Compute μ and σ to define “bands” -----------------------
    μ  = mean(rewards)
    σ  = stdev(rewards) if len(rewards) > 1 else 0.0
    high_band = [(q, a, r) for q, a, r in ranked if r >= μ + σ]
    mid_band  = [(q, a, r) for q, a, r in ranked if μ - σ < r < μ + σ]
    low_band  = [(q, a, r) for q, a, r in ranked if r <= μ - σ]

    # --- 3. Construct pairs ----------------------------------------
    pairs = []
    def add_pairs(pos_list, neg_list):
        """Pair each positive with up to `max_neg_per_pos` negatives."""
        for pos in pos_list:
            # filter negatives that are far enough away
            eligible = [
                neg for neg in neg_list
                if abs(pos[-1] - neg[-1]) >= margin + tie_eps
            ]
            if not eligible:
                continue
            # sample at most `max_neg_per_pos` to limit class imbalance
            for neg in random.sample(
                    eligible, k=min(len(eligible), max_neg_per_pos)):
                pairs.append({"chosen": pos, "rejected": neg})

    # Strong signal: high‑band vs low‑band
    add_pairs(high_band, low_band)

    # # Medium signal (optional): mid‑band vs low‑band
    # add_pairs(mid_band, low_band)

    return pairs    

def prepare_data_for_single_turn(max_samples, config_dir):
    # single_turn_dir = '/home/zhouyang/history_taking/data/dataset/single_turn/sampling'
    data_dir = '/home/zhouyang/history_taking/data/dataset/single_turn/'
    prompts = []
    chosens = []
    rejecteds = []
    margins = []
    full_paths = []
    for root, _, files in os.walk(os.path.join(data_dir, "new_sampling")):
        for file in files:
            full_path = os.path.join(root, file)
            # if full_path != '/mnt/data/zy/zhenting/final_version/history_taking/data/dataset/single_turn/sampling/10165522-DS-9_sampling_2.json':
            #     continue

            # Extract the base filename and remove the trailing _number
            base = os.path.basename(full_path)  # '11459358-DS-12_0.json'
            note_name = base.split('_')[0] + '.json'  # '11459358-DS-12.json'
            # Construct the gpt_sample path
            gpt_full_path = os.path.join(data_dir, "train_raw", note_name)
            with open(gpt_full_path, 'r', encoding='utf-8') as f:
                gpt_sample = json.load(f)
            gpt_recall = gpt_sample['eval_revised']['recall']['recall']

            with open(full_path, 'r', encoding='utf-8') as f:
                sample = json.load(f)
            history = sample['prompt']
            recall = sample['recall']
            turn_number = sample['turn_number']
            input = build_single_turn_input(history)

            # if turn_number > 0:
            #     continue

            # if '<think>' not in input:
            #     continue

            if turn_number > 0 and '<think>' not in input:
                continue
            # print(input)
            # print(full_path)
            prompt = [
                {'role': 'system', 'content': DEFAULT_HISTORY_TAKING_PROMPT},
                {'role': 'user', 'content': input}
            ]
            outputs = sample['outputs']

            outputs_filtered = []
            for i in range(len(outputs)):
                if outputs[i].get('repeated'):
                    continue
                in_history = outputs[i].get('in_history')
                in_note = outputs[i].get('in_note')
                ddx_match_index = outputs[i].get('ddx_match_index')
                score = calc_score(in_history, in_note, ddx_match_index, recall, gpt_recall)
                if score is None:
                    continue
                outputs[i]['score'] = score
                outputs_filtered.append(outputs[i])
            if len(outputs_filtered) == 0:
                continue
            outputs_filtered.sort(key=lambda x: x['score'], reverse=True)
            # if outputs_filtered[0]['score'] < 0.8:
            #     continue

            # for i in range(len(outputs_filtered)):
            #     chosen = outputs_filtered[i]['output']
            #     score_i = outputs_filtered[i]['score']
            #     for j in range(i + 1, len(outputs_filtered)):
            #         rejected = outputs_filtered[j]['output']
            #         score_j = outputs_filtered[j]['score']
            #         if score_i <= score_j:
            #             continue
            #         prompts.append(prompt)
            #         chosens.append(chosen)
            #         rejecteds.append(rejected)
            #         margins.append(score_i - score_j)
            #         full_paths.append(full_path)

            # best_ddx = outputs_filtered[0].get('ddx_match_index')
            # if best_ddx is None and outputs_filtered[0]['score'] == 0:
            #     continue
            # if best_ddx is not None and outputs_filtered[0]['score'] == 0:
            #     continue

            if outputs_filtered[0]['score'] == 0:
                continue

            chosen = outputs_filtered[0]['output']
            score_i = outputs_filtered[0]['score']

            last_with_ddx = next(
                (item for item in reversed(outputs_filtered) 
                if item.get("ddx_match_index") is not None),
                None,
            )

            last_without_ddx = next(
                (item for item in reversed(outputs_filtered) 
                if item.get("ddx_match_index") is None),
                None,
            )

            if last_with_ddx is not None:
                score_j = last_with_ddx['score']
                prompts.append(prompt)
                chosens.append(chosen)
                rejecteds.append(last_with_ddx['output'])
                margins.append(score_i - score_j)
                full_paths.append(full_path)

            if last_without_ddx is not None:
                score_j = last_without_ddx['score']
                prompts.append(prompt)
                chosens.append(chosen)
                rejecteds.append(last_without_ddx['output'])
                margins.append(score_i - score_j)
                full_paths.append(full_path)

    print(f"Loaded {len(prompts)} prompts")
    print(f"Loaded {len(chosens)} chosen responses")
    print(f"Loaded {len(rejecteds)} rejected responses")

    print("==========example===========")
    print("prompt: ", prompts[0])
    print("chosen: ", chosens[0])
    print("rejected: ", rejecteds[0])
    print("margin: ", margins[0])


    dataset = Dataset.from_dict({"prompt": prompts, "chosen": chosens, "rejected": rejecteds, "margin": margins})
    dataset.to_json(f"{config_dir}/trainset.json", orient="records", lines=True)
    # shuffled_dataset = dataset.shuffle(seed=42)
    # subset = shuffled_dataset.select(range(min(max_samples, len(shuffled_dataset))))
    # subset.to_json(f"{config_dir}/trainset.json", orient="records", lines=True)

    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument('--model_name',type=str, required=True, help="model_name")
    # # parser.add_argument('--len_penalty',type=float, required=True, help="len_penalty")
    # # parser.add_argument('--ideal_len',type=float, required=True, help="ideal_len")
    # parser.add_argument('--trainset_name',type=str, required=True, help="trainset_name")
    # # parser.add_argument('--doctor_system_prompt',type=str, required=True, help="trainset_name")
    # # parser.add_argument('--chosen_min_score',type=float, required=True, help="chosen_min_score")
    # # parser.add_argument('--rejected_min_score',type=float, required=True, help="rejected_min_score")
    # parser.add_argument('--margin_min',type=float, required=True, help="margin_min")
    # # parser.add_argument('--repeated_penalty',type=float, required=True, help="repeated_penalty")
    # # parser.add_argument('--ddx_penalty',type=float, required=True, help="ddx_penalty")
    # parser.add_argument('--max_samples',type=int, required=True, help="max_samples")
    # parser.add_argument('--output_dir',type=str, required=True, help="output_dir")

    parser.add_argument(
        "--model_name",
        type=str,
        default="/home/zhouyang/history_taking/models/single_turn_summary_plan",
        help="Path to the base model"
    )

    parser.add_argument(
        "--trainset_name",
        type=str,
        default="note_single_turn",
        help="Name of the training set"
    )

    parser.add_argument(
        "--margin_min",
        type=float,
        default=-1.0,
        help="Minimum margin for DPO loss"
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=45000,
        help="Maximum number of training samples to use"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/zhouyang/history_taking/dpo/outputs_single_turn/test",  # leave None so you can build it dynamically just before training
        help="Directory where checkpoints and logs will be written"
    )


    args = parser.parse_args()

    model_name = args.model_name
    # len_penalty = args.len_penalty
    # ideal_len = args.ideal_len
    trainset_name = args.trainset_name
    # doctor_system_prompt = args.doctor_system_prompt
    # chosen_min_score = args.chosen_min_score
    # rejected_min_score = args.rejected_min_score
    margin_min = args.margin_min
    # repeated_penalty = args.repeated_penalty
    # ddx_penalty = args.ddx_penalty
    max_samples = args.max_samples
    output_dir = args.output_dir


    # # len_penalty = 0.025
    # len_penalty = 0.0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = "dpo_" + model_name.split("/")[-1].lower() + '_' + trainset_name + timestamp

    config_dir = f'{output_dir}/config'
    # os.makedirs(detail_dir, exist_ok=True)

    config_path = os.path.join(config_dir, 'run_config.json')
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    output_dir = f'{output_dir}/output'

    train_dataset = prepare_data_for_single_turn(max_samples, config_dir)
    # train_dataset = load_dataset("json", data_files="/home/zhouyang/history_taking/dpo/outputs_single_turn/filtered_dataset.json", split="train")
    # train_dataset = train_dataset.shuffle(seed=42)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        # torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.bos_token = tokenizer.eos_token
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Number of training samples: {len(train_dataset)}")

    print(tokenizer.all_special_tokens)

    training_args = DPOConfig(
        # per_device_train_batch_size=8,
        per_device_train_batch_size=16,
        # num_train_epochs=3,
        num_train_epochs=2,
        save_strategy="steps",
        logging_steps=10,
        save_steps=500,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        # gradient_checkpointing=False,
        # learning_rate=2e-5,
        learning_rate=1e-5,
        output_dir=output_dir,
        # lr_scheduler_type="constant_with_warmup",
        lr_scheduler_type="cosine",
        # warmup_steps=15,
        # warmup_steps=100,
        warmup_ratio=0.1,
        # optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        loss_type = "sigmoid",
        run_name=run_name,
        beta=0.1,
        # max_target_length=2048,
        # max_completion_length=2048,
        # max_prompt_length=2048,
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
        # train_dataset=tokenized_dataset,
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
