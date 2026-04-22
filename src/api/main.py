import os
import torch
import uvicorn
import json
import re
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from typing import Optional

app = FastAPI(title="DocTune Extraction API")

# Configurações via env vars (produção) com fallback para dev local
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "/app/models/doctune-qwen-1.5b-lora")

class ExtractionRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty_or_too_long(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text cannot be empty")
        if len(v) > 50_000:
            raise ValueError("text exceeds maximum length of 50,000 characters")
        return v

class ExtractionResponse(BaseModel):
    data: Optional[dict]
    raw_response: str

# Variáveis globais para o modelo Singleton
model = None
tokenizer = None

# Otimização Profissional de MLOps: Lock Assíncrono da GPU
# Garante que as requisições ocorram em fila indiana, impedindo a
# GPU de estourar o limite de VRAM (CUDA Out Of Memory) durante picos.
gpu_lock = asyncio.Lock()

def load_model():
    global model, tokenizer
    if model is None:
        print(f"Carregando modelo {MODEL_ID} e adaptador {ADAPTER_PATH}...")

        if not os.path.exists(ADAPTER_PATH):
            raise RuntimeError(f"Adapter path not found: {ADAPTER_PATH}. Check ADAPTER_PATH env var or volume mount.")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )

        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        model.eval()
        print("Modelo DocTune pronto para inferência.")

@app.on_event("startup")
async def startup_event():
    load_model()

def extract_json_from_text(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        return None
    return None

def run_inference(prompt):
    """
    Isolamos a inferência para rodar num thread separado se necessário,
    mas primariamente para organizar a execução atrelada à GPU.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.post("/extract", response_model=ExtractionResponse)
async def extract_fields(request: ExtractionRequest):
    prompt = f"### Instruction:\nExtract the following fields from the document text into a JSON format: employee_name, gross_pay, tax, deductions, net_pay, pay_period, invoice_number.\n\n### Input:\n{request.text}\n\n### Response:\n"
    
    # Aguarda até que a GPU esteja livre
    async with gpu_lock:
        # Movemos a carga intensiva para um thread background.
        # Sem o 'to_thread', o 'generate' bloquearia todo o FastAPI,
        # impedindo até mesmo o Health Check do K8s de responder.
        response_text = await asyncio.to_thread(run_inference, prompt)
    
    prediction_text = response_text.split("### Response:\n")[-1]
    structured_data = extract_json_from_text(prediction_text)
    
    return ExtractionResponse(
        data=structured_data,
        raw_response=prediction_text
    )

@app.get("/health")
async def health():
    return {"status": "ok", "gpu": torch.cuda.is_available()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=5, access_log=True)
