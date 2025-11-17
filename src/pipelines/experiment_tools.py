import argparse
import fnmatch
import logging
import sys
import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from simulator.simulator import build_simulator_openllm, build_simulator_gpt
from evaluation.sentence_evaluator import SentenceEvaluator
from evaluation.diagnosis_evaluator import DiagnosisEvaluator
from evaluation.evaluation_summary import EvaluationSummary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


MODELS = "models"

PATIENT_MODEL = "models/Qwen2.5-32B-Instruct-GPTQ-Int8"
DATASET_MULTI_TURN = "data/dataset/multi_turn/test_raw"

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

COMMON_SIMULATOR_KWARGS: Dict[str, object] = {
    "max_turn": 30,
    "conv_key": "conv_inference",
    "system_instruct_version": "v1",
    "patient_model": PATIENT_MODEL,
}

EXPERIMENT_REPORT_DIR = 'logs'


@dataclass(slots=True)
class Experiment:
    """A single simulator run."""
    name: str
    turn_mode: str
    model_path: Optional[str] = None
    model_name: Optional[str] = None
    lora_path: Optional[str] = None
    dataset_dir: str = DATASET_MULTI_TURN
    output_dir: Optional[str] = None

sentence_evaluator = SentenceEvaluator(
    conv_key='conv_inference',
    eval_key='eval_inference',
    model_id=PATIENT_MODEL
)

diagnosis_evaluator = DiagnosisEvaluator(
    conv_key='conv_inference',
    eval_key='eval_inference',
    model_id=PATIENT_MODEL
)

evaluation_summary = EvaluationSummary(
    conv_key='conv_inference',
    eval_key='eval_inference',
)


def run_experiment(exp: Experiment) -> None:
    """Build a simulator and launch a dataset batch run."""
    kwargs = COMMON_SIMULATOR_KWARGS | {"turn_mode": exp.turn_mode}

    if exp.name.startswith('zeroshot'):
        kwargs['system_instruct_version'] = 'v2'

    logger.info("[%-30s] Initialising simulator…", exp.name)
    if not exp.model_name:
        simulator = build_simulator_openllm(exp.model_path, exp.lora_path, exp.name, **kwargs)
    else:
        simulator = build_simulator_gpt(exp.model_name, exp.name, **kwargs)

    logger.info(
        "[%-30s] Running on dataset '%s' → '%s'",
        exp.name,
        exp.dataset_dir,
        exp.output_dir,
    )

    os.makedirs(exp.output_dir, exist_ok=True)
    simulator.run_on_dataset_batch(exp.dataset_dir, exp.output_dir)
    logger.info("[%-30s] ✓ Completed", exp.name)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run history‑taking evaluation pipeline.")
    parser.add_argument(
        "--filter",
        metavar="PATTERN",
        help="Only run experiments whose *name* matches this fnmatch pattern (case‑insensitive).",
    )
    return parser.parse_args(argv)

def run_experiments(argv: List[str] | None = None, experiments=[]) -> None:
    args = parse_args(argv)

    if args.filter:
        pattern = args.filter.lower()
        experiments = [e for e in experiments if fnmatch.fnmatch(e.name.lower(), pattern)]
        if not experiments:
            logger.error("No experiments match pattern '%s'", args.filter)
            sys.exit(1)
    
    for exp in experiments:
        try:
            print(exp.name)
            run_experiment(exp)
            sentence_evaluator.run_on_data_batch_from_dir(exp.output_dir, max_workers=10)
            diagnosis_evaluator.run_on_data_batch_from_dir(exp.output_dir, max_workers=10)
            evaluation_summary = EvaluationSummary(
                conv_key='conv_inference',
                eval_key='eval_inference',
            )
            report = evaluation_summary.report(exp.output_dir)
            summary_output_path = os.path.join(EXPERIMENT_REPORT_DIR, f"{exp.name}_summary.json")
            with open(summary_output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

        except Exception as exc:
            logger.exception("[%-30s] ✗ Failed: %s", exp.name, exc)



