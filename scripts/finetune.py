import torch
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
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
    return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"

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

    # 7. Iniciar Treino
    print("Iniciando o Fine-tuning na RTX 2070 (Modo Stable Float32 Adaptor)...")
    trainer.train()

    # 8. Salvar o Adaptador
    trainer.save_model(OUTPUT_DIR)
    print(f"Modelo salvo em {OUTPUT_DIR}")

if __name__ == "__main__":
    train()
