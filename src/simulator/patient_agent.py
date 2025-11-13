from typing import List
from utils.models import batch_call_open_llm_api


class PatientAgent:
    def __init__(self, model):
        self.model = model

    def generate_responses_in_batches(self, histories, max_token) -> List[str]:
        try:
            results_raw = batch_call_open_llm_api(histories, max_token, model=self.model)
            return results_raw
        except Exception as e:
            print(f"Error while calling the LLM: {e}")
            return [None] * len(histories)

