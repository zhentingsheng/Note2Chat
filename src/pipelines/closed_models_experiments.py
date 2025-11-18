from typing import List
# from simulator.simulator import build_simulator_gpt, build_simulator_openllm

from pipelines.experiment_tools import Experiment, run_experiments

# Pre‑defined experiments ----------------------------------------------------
CLOSED_MODELS_EXPERIMENTS: List[Experiment] = [
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
        experiments=CLOSED_MODELS_EXPERIMENTS
    )