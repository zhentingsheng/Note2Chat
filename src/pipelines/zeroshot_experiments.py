from typing import List
# from simulator.simulator import build_simulator_gpt, build_simulator_openllm

from pipelines.experiment_tools import Experiment, MODELS, run_experiments

# Pre‑defined experiments ----------------------------------------------------
ZERO_SHOT_EXPERIMENTS: List[Experiment] = [
    Experiment(
        name="zeroshot_Qwen2.5-7B-Instruct",
        model_path=f"{MODELS}/Qwen2.5-7B-Instruct",
        turn_mode="multi",
    ),
    Experiment(
        name="zeroshot_Medgemma-4b-it",
        model_path=f"{MODELS}/medgemma-4b-it",
        turn_mode="multi",
    ),
    Experiment(
        name="zeroshot_Medgemma-27b-text-it",
        model_path=f"{MODELS}/medgemma-27b-text-it",
        turn_mode="multi",
    ),
    Experiment(
        name="zeroshot_o4-mini",
        model_name="o4-mini",
        turn_mode="multi",
    ),
    Experiment(
        name="zeroshot_gpt-4o",
        model_name="gpt-4o",
        turn_mode="multi",
    ),
]

if __name__=='__main__':
    run_experiments(
        experiments=ZERO_SHOT_EXPERIMENTS
    )