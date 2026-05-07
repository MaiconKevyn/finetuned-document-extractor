import os
import sys

MODEL_ID = os.getenv("MODEL_ID", "/app/models/Qwen2.5-1.5B-Instruct")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "/app/models/doctune-qwen-1.5b-lora")

print(f"Python: {sys.version}")
print(f"MODEL_ID: {MODEL_ID}")
print(f"ADAPTER_PATH: {ADAPTER_PATH}")
print()

print("[1/5] Importando torch...")
import torch
print(f"      torch={torch.__version__}, CUDA disponível: {torch.cuda.is_available()}")

print("[2/5] Importando transformers...")
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
print("      OK")

print("[3/5] Importando peft...")
from peft import PeftModel
print("      OK")

print("[4/5] Carregando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print("      OK")

print("[5/5] Carregando modelo base com 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()
print("      OK")

print()
print("Modelo carregado com sucesso.")

inputs = tokenizer("Teste de inferência:", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
print("Output:", tokenizer.decode(out[0], skip_special_tokens=True))
