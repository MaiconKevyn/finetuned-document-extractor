import os
import torch
import uvicorn
import json
import re
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from typing import Optional

# When true, uses lm-format-enforcer to constrain generation to valid JSON
# matching the extraction schema — eliminates data:null responses by construction.
USE_CONSTRAINED_GENERATION = os.getenv("USE_CONSTRAINED_GENERATION", "false").lower() == "true"

_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "employee_name": {"type": "string"},
        "gross_pay":     {"type": "number"},
        "tax":           {"type": "number"},
        "deductions":    {"type": "number"},
        "net_pay":       {"type": "number"},
        "pay_period":    {"type": "string"},
        "invoice_number":{"type": "string"},
    },
    "required": ["employee_name", "gross_pay", "tax", "deductions", "net_pay", "pay_period", "invoice_number"],
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(title="DocTune Extraction API", lifespan=lifespan)

MODEL_ID    = os.getenv("MODEL_ID",    "Qwen/Qwen2.5-1.5B-Instruct")
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
    constrained: bool = False


model = None
tokenizer = None

# Serializes GPU access — prevents CUDA OOM under concurrent requests
gpu_lock = asyncio.Lock()


def load_model():
    global model, tokenizer
    if model is None:
        print(f"Loading model {MODEL_ID} and adapter {ADAPTER_PATH}...")

        if not os.path.exists(ADAPTER_PATH):
            raise RuntimeError(
                f"Adapter path not found: {ADAPTER_PATH}. "
                "Check ADAPTER_PATH env var or volume mount."
            )

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
            torch_dtype=torch.float16,
        )

        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        model.eval()
        print("DocTune model ready.")


def extract_json_from_text(text: str) -> Optional[dict]:
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        return None
    return None


def run_inference(prompt: str) -> tuple[str, bool]:
    """Returns (raw_text, constrained). Tries constrained generation first if enabled."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    if USE_CONSTRAINED_GENERATION:
        try:
            from lmformatenforcer import JsonSchemaParser
            from lmformatenforcer.integrations.transformers import (
                build_transformers_prefix_allowed_tokens_fn,
            )
            parser = JsonSchemaParser(json.dumps(_EXTRACTION_SCHEMA))
            prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    prefix_allowed_tokens_fn=prefix_fn,
                )
            return tokenizer.decode(outputs[0], skip_special_tokens=True), True
        except Exception as e:
            print(f"[constrained generation failed, falling back] {e}")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True), False


@app.post("/extract", response_model=ExtractionResponse)
async def extract_fields(request: ExtractionRequest):
    prompt = (
        "### Instruction:\nExtract the following fields from the document text into a JSON format: "
        "employee_name, gross_pay, tax, deductions, net_pay, pay_period, invoice_number.\n\n"
        f"### Input:\n{request.text}\n\n### Response:\n"
    )

    async with gpu_lock:
        response_text, was_constrained = await asyncio.to_thread(run_inference, prompt)

    prediction_text = response_text.split("### Response:\n")[-1]
    structured_data = extract_json_from_text(prediction_text)

    return ExtractionResponse(
        data=structured_data,
        raw_response=prediction_text,
        constrained=was_constrained,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "gpu": torch.cuda.is_available()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=5, access_log=True)
