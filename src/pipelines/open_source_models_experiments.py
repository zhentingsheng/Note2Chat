from typing import List
# from simulator.simulator import build_simulator_gpt, build_simulator_openllm

from pipelines.experiment_tools import Experiment, MODELS, run_experiments

# Pre‑defined experiments ----------------------------------------------------
OPEN_SOURCE_MODELS_EXPERIMENTS: List[Experiment] = [
    Experiment(
        name="zeroshot_Qwen2.5-7B-Instruct",
        model_path=f"{MODELS}/Qwen2.5-7B-Instruct",
        turn_mode="multi",
    ),
    Experiment(
        name="zeroshot_Qwen3-8B",
        model_path=f"{MODELS}/Qwen3-8B",
        turn_mode="multi",
    ),
    Experiment(
        name="zeroshot_DeepSeek-R1-0528-Qwen3-8B",
        model_path=f"{MODELS}/DeepSeek-R1-0528-Qwen3-8B",
        turn_mode="multi",
    ),
]

if __name__=='__main__':
    run_experiments(
        experiments=OPEN_SOURCE_MODELS_EXPERIMENTS
    )