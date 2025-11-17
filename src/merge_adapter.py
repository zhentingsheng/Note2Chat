from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import argparse

def merge_lora_to_base(
    base_model_name: str,
    adapter_path: str,
    output_path: str,
    save_tokenizer: bool = True,
    torch_dtype: torch.dtype = torch.bfloat16
):
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True
    )
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Merging LoRA to base model...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_path}...")
    merged_model.save_pretrained(output_path, max_shard_size="2GB")
    
    if save_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.save_pretrained(output_path)

    print("Merge completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="merge adapter")
    parser.add_argument("--base_model_path", type=str, help="base_model_path")
    parser.add_argument("--adapter_path", type=str, help="adapter_path")
    parser.add_argument("--merged_model_path", type=str, help="merged_model_path")

    args = parser.parse_args()
    base_model_path = args.base_model_path
    adapter_path = args.adapter_path
    merged_model_path = args.merged_model_path

    merge_lora_to_base(base_model_path,adapter_path,merged_model_path,save_tokenizer=True,torch_dtype=torch.bfloat16)