from typing import List

from pipelines.experiment_tools import Experiment, run_experiments

# Pre‑defined experiments ----------------------------------------------------
HISTORY_TAKING_EXPERIMENTS: List[Experiment] = [
    Experiment(
        name="my_multi_turn_qwen2.5_7b_sft",
        model_path="models/Qwen2.5-7B-Instruct",
        lora_path="adapters/my_multi_turn_qwen2.5_7b_sft",
        turn_mode="multi",
    ),
    Experiment(
        name="my_multi_turn_qwen2.5_7b_sft_sampling",
        model_path="models/Qwen2.5-7B-Instruct",
        lora_path="adapters/my_multi_turn_qwen2.5_7b_sft_sampling",
        turn_mode="multi",
    ),
    Experiment(
        name="my_multi_turn_qwen2.5_7b_sft_sampling_dpo",
        model_path="models/my_multi_turn_qwen2.5_7b_sft_sampling",
        lora_path="adapters/my_multi_turn_qwen2.5_7b_sft_sampling_dpo",
        turn_mode="multi",
    ),
    Experiment(
        name="my_single_turn_qwen2.5_7b_sft",
        model_path="models/Qwen2.5-7B-Instruct",
        lora_path="adapters/my_single_turn_qwen2.5_7b_sft",
        turn_mode="single",
    ),
    Experiment(
        name="my_single_turn_qwen2.5_7b_sft_sampling",
        model_path="models/Qwen2.5-7B-Instruct",
        lora_path="adapters/my_single_turn_qwen2.5_7b_sft_sampling",
        turn_mode="single",
    ),
    Experiment(
        name="my_single_turn_qwen2.5_7b_sft_sampling_dpo",
        model_path="models/my_single_turn_qwen2.5_7b_sft_sampling",
        lora_path="adapters/my_single_turn_qwen2.5_7b_sft_sampling_dpo",
        turn_mode="single",
    ),
]

if __name__=='__main__':
    run_experiments(
        experiments=HISTORY_TAKING_EXPERIMENTS
    )