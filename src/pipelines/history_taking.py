from typing import List

from pipelines.experiment_tools import Experiment, MODELS, run_experiments

# Pre‑defined experiments ----------------------------------------------------
HISTORY_TAKING_EXPERIMENTS: List[Experiment] = [
    Experiment(
        name="multi_turn_sft_7b",
        # model_path=f"{MODELS}/Qwen2.5-7B-Instruct",
        model_path='/mnt/data/zy/zhenting/models/Qwen2.5-7B-Instruct',
        lora_path="sft/outputs/multi_turn_qwen2.5_7b_instruct_gpt",
        turn_mode="multi",
    ),
    Experiment(
        name="multi_turn_sampling_sft",
        model_path=f"{MODELS}/Qwen2.5-7B-Instruct",
        lora_path="sft/outputs/multi_turn_qwen2.5_7b_instruct_gpt_sampling_best",
        turn_mode="multi",
    ),
    Experiment(
        name="multi_turn_dpo_margin_0.0",
        model_path="models/multi_turn_qwen2.5_7b_instruct_gpt",
        lora_path="dpo/outputs/dpo_multi_turn_qwen2.5_7b_instruct_gpt_1_20250712_220148/output",
        turn_mode="multi",
    ),
    Experiment(
        name="multi_turn_dpo_margin_0.0_2",
        model_path="models/multi_turn_qwen2.5_7b_instruct_gpt",
        lora_path="dpo/outputs/dpo_multi_turn_qwen2.5_7b_instruct_gpt_1_20250719_232957/output",
        turn_mode="multi",
    ),
    # Experiment(
    #     name="single_turn_summary_sft",
    #     model_path=f"{MODELS}/Qwen2.5-7B-Instruct",
    #     lora_path="sft/outputs/single_turn_summary",
    #     turn_mode="single",
    # ),
    Experiment(
        name="single_turn_summary_plan_sft",
        model_path=f"{MODELS}/Qwen2.5-7B-Instruct",
        lora_path="sft/outputs/single_turn_summary_plan",
        turn_mode="single",
    ),
    # Experiment(
    #     name="single_turn_plan_sft",
    #     model_path=f"{MODELS}/Qwen2.5-7B-Instruct",
    #     lora_path="sft/outputs/single_turn_plan",
    #     turn_mode="single",
    # ),
    # Experiment(
    #     name="single_turn_summary_simple_plan_sft",
    #     model_path=f"{MODELS}/Qwen2.5-7B-Instruct",
    #     lora_path="sft/outputs/single_turn_summary_simple_plan",
    #     turn_mode="single",
    # ),
    Experiment(
        name="single_turn_sampling_summary_plan",
        model_path=f"{MODELS}/Qwen2.5-7B-Instruct",
        lora_path="sft/outputs/single_turn_sampling_summary_plan",
        turn_mode="single",
    ),
    Experiment(
        name="single_turn_dpo",
        model_path="models/single_turn_summary_plan",
        lora_path="dpo/outputs_single_turn/dpo_single_turn_summary_plan_note_single_turn_20250717_190315/output",
        turn_mode="single",
    ),
]

if __name__=='__main__':
    run_experiments(
        experiments=HISTORY_TAKING_EXPERIMENTS
    )