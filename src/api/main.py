import os
import torch
import uvicorn
import json
import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from typing import Any, Dict, Literal, Optional
from src.utils import extract_json_from_text
from src.monitoring import log_request, run_drift_report
from src.prompts import EXTRACTION_INSTRUCTION, PROMPT_VERSION, build_alpaca_prompt

# When true, uses lm-format-enforcer to constrain generation to valid JSON
# matching the extraction schema — eliminates data:null responses by construction.
USE_CONSTRAINED_GENERATION = os.getenv("USE_CONSTRAINED_GENERATION", "false").lower() == "true"

_EXTRACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "employee_name": {"type": ["string", "null"]},
        "gross_pay":     {"type": ["number", "null"]},
        "tax":           {"type": ["number", "null"]},
        "deductions":    {"type": ["number", "null"]},
        "net_pay":       {"type": ["number", "null"]},
        "pay_period":    {"type": ["string", "null"]},
        "invoice_number":{"type": ["string", "null"]},
    },
    "required": ["employee_name", "gross_pay", "tax", "deductions", "net_pay", "pay_period", "invoice_number"],
}

_METRICS = {
    "doctune_extract_requests_total": 0,
    "doctune_extract_success_total": 0,
    "doctune_extract_failure_total": 0,
    "doctune_extract_business_rule_failure_total": 0,
    "doctune_extract_latency_ms_sum": 0.0,
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


class BatchExtractionRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=32)

    @field_validator("texts")
    @classmethod
    def texts_must_be_valid(cls, values: list[str]) -> list[str]:
        for text in values:
            if not text.strip():
                raise ValueError("batch text items cannot be empty")
            if len(text) > 50_000:
                raise ValueError("batch text item exceeds maximum length of 50,000 characters")
        return values


class ExtractedFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    employee_name: Optional[str]
    gross_pay: Optional[float]
    tax: Optional[float]
    deductions: Optional[float]
    net_pay: Optional[float]
    pay_period: Optional[str]
    invoice_number: Optional[str]


class ExtractionFlags(BaseModel):
    extraction_success: bool
    valid_schema: bool
    business_rule_valid: Optional[bool]
    confidence: Literal["high", "medium", "low"]
    failure_reason: Optional[str] = None


class ExtractionResponse(BaseModel):
    request_id: str
    data: Optional[ExtractedFields] = None
    raw_response: str
    constrained: bool = False
    flags: ExtractionFlags


class BatchExtractionResponse(BaseModel):
    request_id: str
    results: list[ExtractionResponse]


model = None
tokenizer = None

# Serializes GPU access — prevents CUDA OOM under concurrent requests
gpu_lock = asyncio.Lock()


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


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


def business_rule_valid(data: ExtractedFields) -> Optional[bool]:
    values = (data.gross_pay, data.tax, data.deductions, data.net_pay)
    if any(value is None for value in values):
        return None
    return abs(data.gross_pay - data.tax - data.deductions - data.net_pay) < 1.0


def build_flags(structured_data: dict | None, parsed_data: ExtractedFields | None, schema_error: str | None) -> ExtractionFlags:
    if structured_data is None:
        return ExtractionFlags(
            extraction_success=False,
            valid_schema=False,
            business_rule_valid=None,
            confidence="low",
            failure_reason="invalid_json",
        )

    if parsed_data is None:
        return ExtractionFlags(
            extraction_success=False,
            valid_schema=False,
            business_rule_valid=None,
            confidence="low",
            failure_reason=f"schema_validation_failed: {schema_error}",
        )

    rule_valid = business_rule_valid(parsed_data)
    if rule_valid is False:
        return ExtractionFlags(
            extraction_success=True,
            valid_schema=True,
            business_rule_valid=False,
            confidence="medium",
            failure_reason="business_rule_failed",
        )

    return ExtractionFlags(
        extraction_success=True,
        valid_schema=True,
        business_rule_valid=rule_valid,
        confidence="high" if rule_valid is True else "medium",
    )


def record_extract_metrics(flags: ExtractionFlags, latency_ms: float) -> None:
    _METRICS["doctune_extract_requests_total"] += 1
    _METRICS["doctune_extract_latency_ms_sum"] += latency_ms
    if flags.extraction_success:
        _METRICS["doctune_extract_success_total"] += 1
    else:
        _METRICS["doctune_extract_failure_total"] += 1
    if flags.business_rule_valid is False:
        _METRICS["doctune_extract_business_rule_failure_total"] += 1


async def run_extraction(text: str, request_id: str) -> ExtractionResponse:
    prompt = build_alpaca_prompt(EXTRACTION_INSTRUCTION, text)

    started_at = time.perf_counter()
    async with gpu_lock:
        response_text, was_constrained = await asyncio.to_thread(run_inference, prompt)
    latency_ms = (time.perf_counter() - started_at) * 1000

    prediction_text = response_text.split("### Response:\n")[-1]
    structured_data = extract_json_from_text(prediction_text)

    parsed_data = None
    schema_error = None
    if structured_data is not None:
        try:
            parsed_data = ExtractedFields.model_validate(structured_data)
        except ValidationError as exc:
            schema_error = "; ".join(error["msg"] for error in exc.errors())

    flags = build_flags(structured_data, parsed_data, schema_error)
    record_extract_metrics(flags, latency_ms)
    log_request(text, parsed_data.model_dump() if parsed_data else None)

    return ExtractionResponse(
        request_id=request_id,
        data=parsed_data,
        raw_response=prediction_text,
        constrained=was_constrained,
        flags=flags,
    )


@app.post("/extract", response_model=ExtractionResponse)
async def extract_fields(payload: ExtractionRequest, request: Request):
    return await run_extraction(payload.text, request.state.request_id)


@app.post("/extract/batch", response_model=BatchExtractionResponse)
async def extract_fields_batch(payload: BatchExtractionRequest, request: Request):
    results = []
    for idx, text in enumerate(payload.texts):
        results.append(await run_extraction(text, f"{request.state.request_id}-{idx}"))
    return BatchExtractionResponse(request_id=request.state.request_id, results=results)


@app.get("/health")
async def health():
    return {"status": "ok", "gpu": torch.cuda.is_available()}


@app.get("/version")
async def version():
    return {
        "model_id": MODEL_ID,
        "adapter_path": ADAPTER_PATH,
        "prompt_version": PROMPT_VERSION,
    }


@app.get("/monitoring/drift")
async def drift_report():
    """Compare current request distribution against training data. Requires ≥ 30 logged requests."""
    return await asyncio.to_thread(run_drift_report)


@app.get("/metrics")
async def metrics():
    lines = [
        "# HELP doctune_extract_requests_total Total extraction requests.",
        "# TYPE doctune_extract_requests_total counter",
        f"doctune_extract_requests_total {_METRICS['doctune_extract_requests_total']}",
        "# HELP doctune_extract_success_total Total successful extractions.",
        "# TYPE doctune_extract_success_total counter",
        f"doctune_extract_success_total {_METRICS['doctune_extract_success_total']}",
        "# HELP doctune_extract_failure_total Total failed extractions.",
        "# TYPE doctune_extract_failure_total counter",
        f"doctune_extract_failure_total {_METRICS['doctune_extract_failure_total']}",
        "# HELP doctune_extract_business_rule_failure_total Total outputs failing payroll arithmetic validation.",
        "# TYPE doctune_extract_business_rule_failure_total counter",
        f"doctune_extract_business_rule_failure_total {_METRICS['doctune_extract_business_rule_failure_total']}",
        "# HELP doctune_extract_latency_ms_sum Sum of extraction latency in milliseconds.",
        "# TYPE doctune_extract_latency_ms_sum counter",
        f"doctune_extract_latency_ms_sum {_METRICS['doctune_extract_latency_ms_sum']:.6f}",
    ]
    return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=5, access_log=True)
