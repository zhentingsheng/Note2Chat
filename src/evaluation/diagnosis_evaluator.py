import json
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.extractor import TextExtractor
from prompts.evaluation import get_match_diagnosis_prompt
from utils.models import batch_call_open_llm_api
from utils.file_readers import read_json
import glob

logger = logging.getLogger(__name__)

SAMPLE_DIR = 'data/gpt_dialogues'
DEFAULT_MODEL_ID: str = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
class DiagnosisEvaluator:
    def __init__(self, overwrite=False, eval_key="eval_inference", conv_key="conv_inference",model_id: str = DEFAULT_MODEL_ID,):
        self._overwrite = overwrite
        self._extractor = TextExtractor()
        self._eval_key = eval_key
        self._conv_key = conv_key
        self._model_id = model_id

    def _build_prompt(self, diagnosis, preliminary_diagnoses):
        try:
            ddx_list = [p['disease'] for p in preliminary_diagnoses]
        except Exception:
            ddx_list = preliminary_diagnoses

        prompt = get_match_diagnosis_prompt(diagnosis, ddx_list)
        return prompt

    def _parse_response(self, output):
        try:
            result = self._extractor.extract_first_json_object(output)
        except Exception:
            return None
        if result and "match_index" in result and isinstance(result['match_index'], int):
            return result['match_index']
        return None

    def run_single_sample(self, file_path):
        sample = read_json(file_path)

        diagnosis_rank = sample.get(self._eval_key, {}).get('diagnosis_rank')
        if diagnosis_rank is not None and not self._overwrite:
            return 

        diagnosis = sample.get('diagnosis')
        preliminary_diagnoses = sample[self._conv_key].get('preliminary_diagnoses')
        if preliminary_diagnoses is None:
            preliminary_diagnoses = sample[self._conv_key]['conversation'][-1]['content']

        prompt = self._build_prompt(diagnosis, preliminary_diagnoses)

        convo = [{'role': 'user', 'content': prompt}]
        raw_output = batch_call_open_llm_api([convo], max_tokens=2048, model=self._model_id)[0]
        match_index = self._parse_response(raw_output)
        if match_index is None:
            match_index = -1

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if self._eval_key not in data:
            data[self._eval_key] = {}

        data[self._eval_key]['diagnosis_rank'] = match_index

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def run(self, diagnosis, preliminary_diagnoses):
        prompt = self._build_prompt(diagnosis, preliminary_diagnoses)

        convo = [{'role': 'user', 'content': prompt}]
        raw_output = batch_call_open_llm_api([convo], max_tokens=2048, model=self._model_id)[0]
        match_index = self._parse_response(raw_output)
        if match_index is None:
            match_index = -1

        return match_index


    def run_on_data_batch_from_dir(self, evaluate_dir, max_workers=10):
        pattern = os.path.join(evaluate_dir, "*.json")
        file_paths = glob.glob(pattern)
        
        file_paths_remained = []

        for path in file_paths:
            sample = read_json(path)
            diagnosis_rank = sample.get(self._eval_key, {}).get('diagnosis_rank')
            if not self._overwrite and diagnosis_rank is not None:
                continue
            file_paths_remained.append(path)

        total = len(file_paths_remained)

        logger.info(f"Total samples to evaluate: {total}")

        success_total = 0
        failed_total = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.run_single_sample, path)
                for path in file_paths_remained
            ]

            for future in as_completed(futures):
                try:
                    future.result()
                    success_total += 1
                    logger.info(f"[{success_total}/{total}] ✅ Finished one sample")
                except Exception as e:
                    failed_total += 1
                    logger.exception(f"[{success_total}/{total}] ❌ Failed one sample: {e}")

        logger.info(f"Done! Total: {total}, Success: {success_total}, Failed: {failed_total}")

