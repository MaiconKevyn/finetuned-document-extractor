import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
from datasets import load_dataset
from src.prompts import PROMPT_VERSION, build_alpaca_prompt
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Configurações
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_TRAIN = "data/train.jsonl"
DATASET_VAL = "data/val.jsonl"
OUTPUT_DIR = "models/doctune-qwen-1.5b-lora"

os.makedirs("models", exist_ok=True)
os.makedirs("data/logs", exist_ok=True)

def formatting_prompts_func(example):
    return build_alpaca_prompt(example["instruction"], example["input"], example["output"])


class MLflowStepCallback(TrainerCallback):
    """Logs train_loss and eval_loss to MLflow at every logging/eval step."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and mlflow.active_run():
            step = state.global_step
            for key in ("loss", "eval_loss", "learning_rate"):
                if key in logs:
                    mlflow.log_metric(key, logs[key], step=step)


def train():
    # 1. Carregar Dataset
    dataset = load_dataset("json", data_files={"train": DATASET_TRAIN, "validation": DATASET_VAL})

    # 2. Configuração BitsAndBytes (4-bit para 8GB VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, 
        bnb_4bit_use_double_quant=True,
    )

    # 3. Carregar Modelo e Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Carregamos o modelo forçando a conversão para Float32 para evitar BFloat16 na RTX 2070
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float32, 
    )

    # Preparar para k-bit training
    model = prepare_model_for_kbit_training(model)

    # 4. Configuração LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. Configuração de Treino (SFTConfig)
    # DESATIVAMOS fp16/bf16 para evitar o GradScaler problemático
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1, # Reduzido para compensar falta de fp16
        gradient_accumulation_steps=8, # Aumentado para manter batch size efetivo
        optim="paged_adamw_32bit",
        save_steps=100,
        logging_steps=10,
        learning_rate=1e-4, # Ligeiramente menor para estabilidade
        fp16=False,
        bf16=False,
        num_train_epochs=3,
        eval_strategy="steps",
        eval_steps=100,
        warmup_steps=10,
        report_to="tensorboard",
        max_length=512,
        dataloader_num_workers=0,
    )

    # 6. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        args=sft_config,
    )

    # 7. Iniciar Treino com MLflow tracking
    mlflow.set_experiment("doctune-finetune")
    with mlflow.start_run(run_name="qlora-qwen2.5-1.5b"):
        mlflow.log_params({
            "model_id":       MODEL_ID,
            "prompt_version": PROMPT_VERSION,
            "lora_r":         peft_config.r,
            "lora_alpha":     peft_config.lora_alpha,
            "lora_dropout":   peft_config.lora_dropout,
            "learning_rate":  sft_config.learning_rate,
            "num_epochs":     sft_config.num_train_epochs,
            "batch_size":     sft_config.per_device_train_batch_size,
            "grad_accum":     sft_config.gradient_accumulation_steps,
            "max_length":     sft_config.max_length,
            "optimizer":      sft_config.optim,
            "quant_type":     bnb_config.bnb_4bit_quant_type,
        })

        print("Iniciando o Fine-tuning na RTX 2070 (Modo Stable Float32 Adaptor)...")
        trainer.add_callback(MLflowStepCallback())
        trainer.train()

        # 8. Salvar o Adaptador
        trainer.save_model(OUTPUT_DIR)
        print(f"Modelo salvo em {OUTPUT_DIR}")

        mlflow.log_artifact(os.path.join(OUTPUT_DIR, "adapter_config.json"))
        if os.path.exists("results/training_run.json"):
            mlflow.log_artifact("results/training_run.json")

if __name__ == "__main__":
    train()
