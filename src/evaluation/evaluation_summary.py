import os
import json
import logging

logger = logging.getLogger(__name__)

def calculate_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
class EvaluationSummary:
    def __init__(self, eval_key="eval_inference", conv_key="conv_inference"):
        self.recalls = []
        self.precisions = []
        self.f1_scores = []
        self.turns = []
        self.diagnosis_ranks = []
        self.dontknow = 0
        self.repeated = 0
        self.support_densities = []
        self.eval_key = eval_key
        self.conv_key = conv_key
        self.hpi_sentence_categories = self.get_hpi_sentence_categories()
        self.category_stats = {cat: {'total': 0, 'supported': 0} for cat in [
            "pmh_treatment", "site", "onset", "character", "radiation",
            "timing", "modifying_factors", "severity", "associated_symptoms"
        ]}


    def get_hpi_sentence_categories(self):
        path = 'data/processed/hpi_sentence_categories.json'
        with open(path, 'r', encoding='utf-8') as f:
            hpi_sentence_categories = json.load(f)
        return hpi_sentence_categories

    def run_on_data(self, directory: str):
        for root, _, files in os.walk(directory):
            for file in files:
                if not file.endswith('.json'):
                    continue

                full_path = os.path.join(root, file)
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        sample = json.load(f)
                    note_id = sample['note_id']
                    eval = sample.get(self.eval_key, {})
                    conv = sample.get(self.conv_key, {})
                    # 1) Recalls
                    recall_data = eval.get('symptom')
                    if recall_data is None:
                        recall_data = eval.get('recall')

                    if recall_data:
                        self._maybe_append(recall_data, 'recall', self.recalls, full_path)
                    else:
                        logger.warning(f"[!] {full_path} has no recall data")                    


                    # 2) Diagnosis rank
                    diag_rank = eval.get('diagnosis_rank')
                    if isinstance(diag_rank, int):
                        self.diagnosis_ranks.append(diag_rank)
                    else:
                        logger.warning(f"[!] {full_path} has no valid diagnosis_rank")

                    # 3) Turns & conversation
                    conversation = conv.get('conversation')
                    if conversation:
                        if conversation[0].get('role') == 'system':
                            conversation = conversation[1:]
                        num_turns = len(conversation)
                        self.turns.append(num_turns)

                        # support density
                        support_total = recall_data.get('support_total')
                        if isinstance(support_total, (int, float)) and num_turns > 0:
                            self.support_densities.append(support_total / num_turns)

                        # 'I don't know.' and repeated
                        for c in conversation:
                            if c['role'] == 'user' and c['content'] == "I don't know.":
                                self.dontknow += 1
                            if c['role'] == 'user' and c['content'] == "Sorry, you've already asked this question.":
                                self.repeated += 1
                    else:
                        logger.warning(f"[!] {full_path} has no valid conversation")

                    # 4) f1
                    recall = recall_data.get('recall')
                    ground_truth_total = recall_data.get('ground_truth_total')
                    if isinstance(recall, (int, float)) and isinstance(support_total, (int, float)) and isinstance(ground_truth_total, (int, float)) and num_turns > 0:
                        precision = min(support_total / (num_turns - 1) * 2, 1)
                        self.precisions.append(precision)
                        f1 = calculate_f1(precision, recall)
                        self.f1_scores.append(f1)
                    
                    # 5) categories
                    supported_sentences = eval.get('supported_sentences')
                    note_categories = self.hpi_sentence_categories.get(note_id, {})
                    if supported_sentences:
                        for support_sentence in supported_sentences:
                            sentence = support_sentence.get('sentence')
                            if sentence is None:
                                continue
                            turn = support_sentence.get('turn')
                            if turn == 0:
                                continue
                            categories = note_categories.get(sentence, {})
                            for cat, is_marked in categories.items():
                                if is_marked:
                                    self.category_stats[cat]['total'] += 1
                                    if turn > 0:
                                        self.category_stats[cat]['supported'] += 1
                            

                    
                except Exception as e:
                    logger.error(f"[!] Failed to read {full_path}: {e}")

    def _maybe_append(self, data: dict, key: str, target_list: list, context: str):
        val = data.get(key)
        if isinstance(val, (int, float)):
            target_list.append(val)
        else:
            logger.warning(f"[!] {context} has no valid {key}")

    def _compute_average(self, values: list, label: str) -> tuple:
        if not values:
            logger.warning(f"No valid {label} found.")
            return 0.0, 0
        avg = sum(values) / len(values)
        return avg, len(values)

    def compute_average_recall(self):
        return self._compute_average(self.recalls, "recall")


    def compute_average_turns(self):
        return self._compute_average(self.turns, "turns")

    def compute_average_support_density(self):
        return self._compute_average(self.support_densities, "support_density")

    def compute_average_diagnosis_rank(self):
        total = len(self.diagnosis_ranks)
        valid_ranks = [r + 1 for r in self.diagnosis_ranks if r != -1]
        invalid = total - len(valid_ranks)

        if not valid_ranks:
            logger.warning("No valid diagnosis rank found.")
            return {
                "average_rank": 0.0,
                "valid_count": 0,
                "invalid_count": invalid,
                "invalid_rate": 0.0,
                "top_k_hits": {}
            }

        avg_rank = sum(valid_ranks) / len(valid_ranks)
        top_k_hits = {
            f"top_{k}": {
                "count": sum(1 for r in valid_ranks if r <= k),
                "rate": round(sum(1 for r in valid_ranks if r <= k) / total, 4)
            }
            for k in range(1, 6)
        }

        return {
            "average_rank": round(avg_rank, 4),
            "valid_count": len(valid_ranks),
            "invalid_count": invalid,
            "invalid_rate": round(invalid / total, 4) if total else 0.0,
            "top_k_hits": top_k_hits
        }

    def compute_average_f1(self):
        return self._compute_average(self.f1_scores, "f1")
    
    def compute_average_precisions(self):
        return self._compute_average(self.precisions, "precision")
    
    def print_category_stats(self):
        print("===== Category Support Stats =====")
        for cat, stats in self.category_stats.items():
            total = stats['total']
            supported = stats['supported']
            rate = supported / total if total > 0 else 0.0
            print(f"{cat:<20} total: {total:<4}  supported: {supported:<4}  rate: {rate:.4f}")

    def report(self, directory: str):
        self.run_on_data(directory)

        results = {
            "average_recall": self.compute_average_recall(),
            "average_precision": self.compute_average_precisions(),
            "average_f1": self.compute_average_f1(),
            "average_turns": self.compute_average_turns(),
            "average_support_density": self.compute_average_support_density(),
            "diagnosis_rank": self.compute_average_diagnosis_rank(),
            "dontknow_rate": round(self.dontknow / max(len(self.recalls), 1), 4),
            "repeated_rate": round(self.repeated / max(len(self.recalls), 1), 4),
            "category_stats": self.category_stats
        }

        print("===== Evaluation Summary =====")
        print(f"Samples evaluated (recall): {results['average_recall'][1]}")
        print(f"Average F1 Score           : {results['average_f1'][0]:.4f}")
        print(f"Average Precision          : {results['average_precision'][0]:.4f}")
        print(f"Average Symptom recall     : {results['average_recall'][0]:.4f}")
        print(f"Average Turns              : {results['average_turns'][0]:.4f}")
        print(f"Average Support Density    : {results['average_support_density'][0]:.4f}")
        print(f"Average DontKnow Rate      : {results['dontknow_rate']:.4f}")
        print(f"Average Repeated Rate      : {results['repeated_rate']:.4f}")

        print("Diagnosis Rank Stats:")
        diag = results['diagnosis_rank']
        print(f"  Average Rank : {diag['average_rank']:.4f}")
        print(f"  Valid Count  : {diag['valid_count']}")
        print(f"  Invalid Count: {diag['invalid_count']}")
        print(f"  Invalid Rate : {diag['invalid_rate']:.4f}")
        print(f"  Top-K Hits   :")
        for k, v in diag['top_k_hits'].items():
            print(f"    {k:<8} -> count: {v['count']}, rate: {v['rate']:.4f}")

        self.print_category_stats()

        return results
