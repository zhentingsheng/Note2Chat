from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

class DoctorAgent:
    def __init__(self, base_model_path, lora_path=None, max_turn=26, sampling=1, exp_name=None):

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.llm = LLM(
            model=base_model_path,
            enable_lora=True,
            dtype="bfloat16",
            max_model_len=8192,
            max_lora_rank=64,
            # tensor_parallel_size=2
        )
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            repetition_penalty = 1.05,
            max_tokens=2048,
            skip_special_tokens=False,
            n=sampling
        )
        self.exp_name = exp_name
        self.lora_path = lora_path
        self.max_turn = max_turn
        # self.ddx_tag = '"{"preliminary_diagnoses":'
        self.multi_turn_ddx_tag = '{'
        self.single_turn_ddx_tag = '<think>make preliminary diagnoses</think>disease:'

    def build_single_turn_input(self, conversation):
        conversation = conversation[1:]
        turn_number = int(len(conversation) / 2)
        if len(conversation) >= 2:
            conversation = conversation[-2:]
        
        input_text = f'Current turn ({turn_number}):\n'

        for turn in conversation:
            role = turn['role']
            content = turn['content']
            sentence = f'{role}: {content}\n'
            input_text += sentence
        input_text = input_text.strip()
        return input_text
    
    def generate_responses_in_batches(self, histories, batch_size: int = 64, turn_mode='multi'):

        all_responses = []
        
        for batch_start_index in range(0, len(histories), batch_size):
            batch_end_index = batch_start_index + batch_size
            current_batch = histories[batch_start_index:batch_end_index]

            if turn_mode == 'multi':
                prompt_list = []
                for conversation in current_batch:
                    prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                    if self.exp_name and not self.exp_name.startswith('zeroshot') and len(conversation) >= self.max_turn:
                        prompt += self.multi_turn_ddx_tag
                    prompt_list.append(prompt)
                
            else:
                prompt_list = []
                for conversation in current_batch:
                    input = self.build_single_turn_input(conversation)
                    # print('input: \n', input)
                    new_conversation = [
                        {'role': 'system', 'content': conversation[0]['content']},
                        {'role': 'user', 'content': input}
                    ]
                    prompt = self.tokenizer.apply_chat_template(new_conversation, tokenize=False, add_generation_prompt=True)
                    if len(conversation) >= self.max_turn:
                        prompt += self.single_turn_ddx_tag
                    prompt_list.append(prompt)

            if self.lora_path:
                # print("Using LoRA model: ", self.lora_path)
                response_outputs = self.llm.generate(
                    prompt_list,
                    self.sampling_params,
                    lora_request=LoRARequest("lora", 1, self.lora_path)
                )
            else:
                response_outputs = self.llm.generate(
                    prompt_list,
                    self.sampling_params
                )
            for response, prompt in zip(response_outputs, prompt_list):
                for i, output in enumerate(response.outputs):
                    text = output.text
                    if prompt.endswith(self.multi_turn_ddx_tag) and self.multi_turn_ddx_tag not in text:
                        text = self.multi_turn_ddx_tag + text
                    elif prompt.endswith(self.single_turn_ddx_tag):
                        if '<think>' in text:
                            text = text.split('<think>')[-1]
                        elif '</think>' in text:
                            text = text.split('</think>')[-1]
                        text = self.single_turn_ddx_tag + text
                        
                    all_responses.append(text)

        return all_responses
    
    def close(self):
        """Gracefully release vLLM resources (any version)."""
        engine = getattr(self.llm, "llm_engine", None)

        if engine and hasattr(engine, "shutdown"):
            engine.shutdown()

        elif engine and hasattr(engine, "engine_core"):
            core = engine.engine_core
            if hasattr(core, "shutdown"):
                core.shutdown()
            elif hasattr(core, "close"):
                core.close()

        import torch, gc
        torch.cuda.empty_cache()
        gc.collect()

        self.llm = None
