from typing import List
# from simulator.simulator import build_simulator_gpt, build_simulator_openllm

from pipelines.experiment_tools import Experiment, MODELS, run_experiments

# Pre‑defined experiments ----------------------------------------------------
DOMAIN_SPECIFIC_MODELS_EXPERIMENTS: List[Experiment] = [
    Experiment(
        name="zeroshot_HuatuoGPT-o1-8B",
        model_path=f"{MODELS}/HuatuoGPT-o1-8B",
        turn_mode="multi",
    ),
    Experiment(
        name="zeroshot_MedGemma-4B-it",
        model_path=f"{MODELS}/MedGemma-4B-it",
        turn_mode="multi",
    ),
    Experiment(
        name="zeroshot_MedGemma-27B-text-it",
        model_path=f"{MODELS}/MedGemma-27B-text-it",
        turn_mode="multi",
    ),
]

if __name__=='__main__':
    run_experiments(
        experiments=DOMAIN_SPECIFIC_MODELS_EXPERIMENTS
    )