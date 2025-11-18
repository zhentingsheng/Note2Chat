import argparse
import fnmatch
import logging
import sys
import os
import json
import csv
from dataclasses import dataclass, field
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
TEST_SET_PATHS = []

testset_note_ids = []
testset_note_ids_path = "data/note_ids/testset_note_ids.csv"
with open(testset_note_ids_path, 'r', encoding='utf-8') as f:
    csv_reader = csv.reader(f)
    for line in csv_reader:
        testset_note_ids.append(line[0])

gpt_dialogue_dir = 'data/gpt_dialogues'
for root, _, files in os.walk(gpt_dialogue_dir):
    for file in files:
        note_id = file[:-len('.json')]
        if note_id in testset_note_ids:
            full_path = os.path.join(root, file)
            TEST_SET_PATHS.append(full_path)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

COMMON_SIMULATOR_KWARGS: Dict[str, object] = {
    "max_turn": 26,
    "conv_key": "conv_inference",
    "system_instruct_version": "v1",
    "patient_model": PATIENT_MODEL,
}

EXPERIMENT_REPORT_DIR = 'reports'

@dataclass(slots=True)
class Experiment:
    """A single simulator run."""
    name: str
    turn_mode: str
    model_path: Optional[str] = None
    model_name: Optional[str] = None
    lora_path: Optional[str] = None
    test_set_paths: List[str] = field(default_factory=lambda: TEST_SET_PATHS.copy())  
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

    exp.output_dir = f'{RESULTS_DIR}/{exp.name}'

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
        'testset',
        exp.output_dir,
    )

    os.makedirs(exp.output_dir, exist_ok=True)
    simulator.run_on_dataset_batch(exp.test_set_paths, exp.output_dir)
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



