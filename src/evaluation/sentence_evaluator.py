from __future__ import annotations

import json
import logging
import glob
import os
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.extractor import TextExtractor
from utils.models import call_open_llm_api
from utils.file_readers import read_json, read_jsonl
from prompts.evaluation import get_compare_sentences_with_conversation_prompt

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID: str = "models/Qwen2.5-32B-Instruct-GPTQ-Int8"
CHIEF_COMPLAINT_DIR = "data/processed/chief_complaint_hit_sentences"
HPI_SENTENCES_PATH = "data/processed/hpi_sentences.jsonl"
HPI_SENTENCE_CATEGORIES_PATH = "data/processed/hpi_sentence_categories.json"

class SentenceEvaluator:
    """Evaluate whether sentences extracted from a clinical note are supported
    by a simulated doctor–patient conversation.
    """

    def __init__(
        self,
        overwrite: bool = False,
        conv_key: str = "conv_inference",
        eval_key: str = "eval_inference",
        model_id: str = DEFAULT_MODEL_ID,
    ) -> None:
        self._overwrite = overwrite
        self._conv_key = conv_key
        self._eval_key = eval_key
        self._model_id = model_id

        self._extractor = TextExtractor()

        self._note_sentence_categories: Dict[str, Dict[str, bool]] = self._load_sentence_categories()
        self._note_sentences: Dict[str, List[str]] = self._load_note_sentences()
        self._chief_complaint_hits: Dict[str, List[str]] = self._load_chief_complaint_hits()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, note_id: str, conversation: List[Dict[str, str]], evaluate_chief_complaint: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        prompt = self._build_prompt(note_id, conversation, evaluate_chief_complaint)
        convo = [{"role": "user", "content": prompt}]
        raw_text = call_open_llm_api(convo, self._model_id, max_tokens=8192)
        supported = self._parse_supported_sentences(note_id, raw_text)
        supported_sentences, metrics = self._evaluate_single(note_id, supported, conversation, evaluate_chief_complaint)

        return supported_sentences, metrics

    def run_single_sample(self, note_id, conversation, file_path):
        try:
            supported, metrics = self.evaluate(note_id, conversation)
            sample = read_json(file_path)
            sample[self._eval_key] = {
                "supported_sentences": supported,
                "symptom": metrics
            }
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(sample, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.exception(f"{note_id} Failed")


    def run_on_data_batch_from_dir(self, evaluate_dir: str, max_workers: int=10) -> None:
        note_ids, conversations, file_paths = self._collect_samples(evaluate_dir)

        total = len(note_ids)
        done = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.run_single_sample, note_id, conversation, fp)
                for note_id, conversation, fp in zip(note_ids, conversations, file_paths)
            ]

            for future in as_completed(futures):
                try:
                    future.result()
                    done += 1
                    logger.info(f"[{done}/{total}] ✅ Finished one sample")
                except Exception as e:
                    failed += 1
                    logger.exception(f"[{done}/{total}] ❌ Failed one sample: {e}")

        logger.info(f"All done! Total: {total}, Success: {done}, Failed: {failed}")

    # ------------------------------------------------------------------
    # Private helpers – data loading
    # ------------------------------------------------------------------

    def _load_chief_complaint_hits(self) -> Dict[str, List[str]]:
        hits: Dict[str, List[str]] = {}
        pattern = os.path.join(CHIEF_COMPLAINT_DIR, "*.json")
        for path in glob.glob(pattern):
            data = read_json(path)
            note_id = data["note_id"]
            hits[note_id] = [item["sentence"] for item in data["supported_sentences"] if item.get("turn") == 0 and item.get("sentence")]
        logger.info("Loaded chief‑complaint hits for %d notes", len(hits))
        return hits

    def _load_note_sentences(self) -> Dict[str, List[str]]:
        sentences: Dict[str, List[str]] = {}
        for entry in read_jsonl(HPI_SENTENCES_PATH):
            sentences[entry["note_id"]] = entry["sentences"]
        logger.info("Loaded note sentences for %d notes", len(sentences))
        return sentences

    def _load_sentence_categories(self) -> Dict[str, Dict[str, bool]]:
        categories = read_json(HPI_SENTENCE_CATEGORIES_PATH)
        logger.info("Loaded sentence categories for %d notes", len(categories))
        return categories

    # ------------------------------------------------------------------
    # Private helpers – prompt generation & parsing
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        note_id: str,
        conversation: List[Dict[str, str]],
        evaluate_chief_complaint: bool = False,
    ) -> List[str]:
        sentences = self._note_sentences.get(note_id, [])
        indexed = [{"index": idx, "sentence": s} for idx, s in enumerate(sentences)]
        conversation_text = self._conversation_to_text(conversation, evaluate_chief_complaint)
        prompt = get_compare_sentences_with_conversation_prompt(indexed, conversation_text)
        return prompt

    def _conversation_to_text(self, conversation: List[Dict[str, str]], evaluate_chief_complaint: bool) -> str:
        for turn in conversation:
            if turn["role"] in {"assistant", "doctor"} and "</think>" in turn["content"]:
                turn["content"] = turn["content"].split("</think>")[-1]

        if conversation and conversation[0]["role"] == "system":
            conversation = conversation[1:]

        if evaluate_chief_complaint:
            conversation = conversation[:1]
            return "\n".join(f"Turn {idx} ({c['role']}): {c['content']}" for idx, c in enumerate(conversation))

        convo_body = conversation[1:-1]
        return "\n".join(
            f"Turn {idx + 1} ({c['role']}): {c['content']}" for idx, c in enumerate(convo_body)
        )

    def _parse_supported_sentences(self, note_id: str, model_output: str) -> Optional[List[Dict[str, Any]]]:
        try:
            supported_raw = self._extractor.extract_first_json_array(model_output)
        except Exception as exc:
            logger.warning("Error extracting JSON from model output for %s: %s", note_id, exc)
            return None

        note_sentences = self._note_sentences.get(note_id, [])
        supported: List[Dict[str, Any]] = [{"sentence": s} for s in note_sentences]

        for item in supported_raw:
            idx, turn = item["index"], item["turn"]
            if 0 <= idx < len(supported):
                supported[idx]["turn"] = turn
        return supported

    # ------------------------------------------------------------------
    # Private helpers – evaluation
    # ------------------------------------------------------------------

    def _evaluate_single(
        self,
        note_id: str,
        supported_sentences: List[Dict[str, Any]],
        conversation: List[Dict[str, str]],
        evaluate_chief_complaint: bool = False
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

        complaint_hits = set(self._chief_complaint_hits.get(note_id, []))

        supported_total = 0
        complaint_supported = 0
        gt_total = len(supported_sentences)

        for i in range(len(supported_sentences)):
            sentence = supported_sentences[i]['sentence']
            turn = supported_sentences[i].get('turn', -1)
            if turn > 0 and turn < len(conversation):
                if turn % 2 == 0:
                    turn -= 1
            elif evaluate_chief_complaint is False:
                turn = -1

            if sentence in complaint_hits:
                complaint_supported += 1
                supported_sentences[i]['chief_complaint'] = True
                supported_sentences[i]['patient'] = conversation[0]["content"]
                turn = 0
            
            supported_sentences[i]['turn'] = turn

            if turn > 0:
                supported_total += 1

        recall = self._safe_divide(supported_total, gt_total - complaint_supported)

        metrics = {
            "support_total": supported_total,
            "ground_truth_total": gt_total - complaint_supported,
            "recall": recall,
        }

        return supported_sentences, metrics

    # ------------------------------------------------------------------
    # Private helpers – utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_divide(numerator: int, denominator: int) -> Optional[float]:
        if denominator == 0:
            return None
        return min(numerator / denominator, 1.0)

    def _collect_samples(self, evaluate_dir: str) -> Tuple[List[str], List[List[Dict[str, str]]], List[str]]:
        pattern = os.path.join(evaluate_dir, "*.json")
        files = glob.glob(pattern)

        note_ids: List[str] = []
        conversations: List[List[Dict[str, str]]] = []
        file_paths: List[str] = []

        for fp in files:
            sample = read_json(fp)

            recall = sample.get(self._eval_key, {}).get('symptom', {}).get('recall')
            if not self._overwrite and recall is not None:
                continue
            note_ids.append(sample["note_id"])
            conversations.append(sample[self._conv_key]["conversation"])
            file_paths.append(fp)
        return note_ids, conversations, file_paths
