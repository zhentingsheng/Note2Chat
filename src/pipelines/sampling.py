import os
from pipelines.experiment_tools import Experiment, run_experiment, sentence_evaluator, diagnosis_evaluator



exp = Experiment(
    name = "multi_turn_sft_sampling",
    model_path = 'home/Note2Chat/models/Qwen2.5-7B-Instruct',
    lora_path = "adapters/multi_turn_qwen2.5_7b_instruct_gpt",
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
