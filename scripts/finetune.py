import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import mlflow
import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prompts import PROMPT_VERSION, build_alpaca_prompt

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - numpy is present in the full ML env
    np = None


@dataclass
class FinetuneConfig:
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    train_file: str = "data/train.jsonl"
    val_file: str = "data/val.jsonl"
    output_dir: str = "models/doctune-qwen-1.5b-lora"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 1e-4
    epochs: float = 3
    seed: int = 42
    max_length: int = 512
    warmup_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 2
    early_stopping_patience: int = 2
    run_name: str = "qlora-qwen2.5-1.5b"


os.makedirs("models", exist_ok=True)
os.makedirs("data/logs", exist_ok=True)


def formatting_prompts_func(example):
    return build_alpaca_prompt(example["instruction"], example["input"], example["output"])


class MLflowStepCallback(TrainerCallback):
    """Log train/eval metrics to MLflow at every Trainer logging step."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and mlflow.active_run():
            step = state.global_step
            for key in ("loss", "eval_loss", "learning_rate"):
                if key in logs:
                    mlflow.log_metric(key, logs[key], step=step)


def count_jsonl(path: str) -> int:
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def configure_seed(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def write_training_run(config: FinetuneConfig, trainer) -> None:
    os.makedirs("results", exist_ok=True)
    record = {
        "project": "DocTune",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "base_model": config.model_id,
        "adapter_output": config.output_dir,
        "dataset": {
            "train": config.train_file,
            "validation": config.val_file,
            "train_samples": count_jsonl(config.train_file),
            "val_samples": count_jsonl(config.val_file),
            "generation_script": "scripts/generate_dataset.py",
        },
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_use_double_quant": True,
            "model_load_dtype": "float32",
        },
        "lora": {
            "r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "training": {
            "seed": config.seed,
            "num_train_epochs": config.epochs,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "effective_batch_size": 8,
            "learning_rate": config.learning_rate,
            "warmup_steps": config.warmup_steps,
            "optimizer": "paged_adamw_32bit",
            "fp16": False,
            "bf16": False,
            "max_length": config.max_length,
            "eval_strategy": "steps",
            "eval_steps": config.eval_steps,
            "save_steps": config.save_steps,
            "save_total_limit": config.save_total_limit,
            "early_stopping_patience": config.early_stopping_patience,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
        },
        "raw_config": asdict(config),
    }
    with open("results/training_run.json", "w") as f:
        json.dump(record, f, indent=2)


def train(config: FinetuneConfig) -> None:
    configure_seed(config.seed)

    dataset = load_dataset(
        "json",
        data_files={"train": config.train_file, "validation": config.val_file},
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Float32 model load avoids bf16/fp16 GradScaler instability on RTX 2070.
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=config.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        optim="paged_adamw_32bit",
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        logging_steps=10,
        learning_rate=config.learning_rate,
        fp16=False,
        bf16=False,
        num_train_epochs=config.epochs,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        warmup_steps=config.warmup_steps,
        report_to="tensorboard",
        max_length=config.max_length,
        dataloader_num_workers=0,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        args=sft_config,
    )

    mlflow.set_experiment("doctune-finetune")
    with mlflow.start_run(run_name=config.run_name):
        mlflow.log_params(
            {
                "model_id": config.model_id,
                "prompt_version": PROMPT_VERSION,
                "seed": config.seed,
                "lora_r": peft_config.r,
                "lora_alpha": peft_config.lora_alpha,
                "lora_dropout": peft_config.lora_dropout,
                "learning_rate": sft_config.learning_rate,
                "num_epochs": sft_config.num_train_epochs,
                "batch_size": sft_config.per_device_train_batch_size,
                "grad_accum": sft_config.gradient_accumulation_steps,
                "max_length": sft_config.max_length,
                "optimizer": sft_config.optim,
                "quant_type": bnb_config.bnb_4bit_quant_type,
            }
        )

        print("Starting DocTune QLoRA fine-tuning.")
        trainer.add_callback(MLflowStepCallback())
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience))
        trainer.train()

        trainer.save_model(config.output_dir)
        write_training_run(config, trainer)
        print(f"Adapter saved to {config.output_dir}")

        adapter_config = os.path.join(config.output_dir, "adapter_config.json")
        if os.path.exists(adapter_config):
            mlflow.log_artifact(adapter_config)
        mlflow.log_artifact("results/training_run.json")


def parse_args() -> FinetuneConfig:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for DocTune.")
    parser.add_argument("--model-id", default=FinetuneConfig.model_id)
    parser.add_argument("--train-file", default=FinetuneConfig.train_file)
    parser.add_argument("--val-file", default=FinetuneConfig.val_file)
    parser.add_argument("--output-dir", default=FinetuneConfig.output_dir)
    parser.add_argument("--lora-r", type=int, default=FinetuneConfig.lora_r)
    parser.add_argument("--lora-alpha", type=int, default=FinetuneConfig.lora_alpha)
    parser.add_argument("--lora-dropout", type=float, default=FinetuneConfig.lora_dropout)
    parser.add_argument("--learning-rate", type=float, default=FinetuneConfig.learning_rate)
    parser.add_argument("--epochs", type=float, default=FinetuneConfig.epochs)
    parser.add_argument("--seed", type=int, default=FinetuneConfig.seed)
    parser.add_argument("--max-length", type=int, default=FinetuneConfig.max_length)
    parser.add_argument("--run-name", default=FinetuneConfig.run_name)
    return FinetuneConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(parse_args())
