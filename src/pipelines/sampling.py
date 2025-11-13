from pipelines.experiment_tools import Experiment, run_experiment


exp = Experiment(
    name = "multi_turn_sft_sampling",
    model_path = 'models/Qwen2.5-7B-Instruct',
    lora_path = "models/sft/outputs/my_multi_turn_qwen2.5_7b_sft",
    turn_mode = "multi",
    dataset_dir = "data/gpt_dialogues",
    stage = 'sampling'
)

for i in range(15):
    output_dir = f'data/sampling/{i}'
    exp.output_dir = output_dir

    run_experiment(exp)
