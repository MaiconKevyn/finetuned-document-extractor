"""
Merge the LoRA adapter into the base model weights and save a standalone model.

The merged model runs without PEFT at inference time, removing the adapter
loading overhead (~1s/call) at the cost of a larger model file on disk.

Usage:
    python scripts/merge_adapter.py \
        --base  models/Qwen2.5-1.5B-Instruct \
        --adapter models/doctune-qwen-1.5b-lora \
        --output models/doctune-qwen-1.5b-merged
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge(base_path: str, adapter_path: str, output_path: str) -> None:
    print(f"Loading base model from {base_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        device_map="cpu",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_path)

    print(f"Loading LoRA adapter from {adapter_path} ...")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging adapter weights into base model ...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_path} ...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("Done. The merged model runs without PEFT — update MODEL_ID and clear ADAPTER_PATH.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default="models/doctune-qwen-1.5b-lora")
    parser.add_argument("--output", default="models/doctune-qwen-1.5b-merged")
    args = parser.parse_args()

    merge(args.base, args.adapter, args.output)
