# DocTune: Fine-tuned Small LLM for Structured Document Extraction

A strong project idea is to build a **fine-tuning pipeline for a small LLM focused on structured document extraction**, using **PyTorch + LoRA/QLoRA**, and then serve the model on **Kubernetes**.

This project closes exactly the 3 gaps:

- fine-tuning
- PyTorch
- Kubernetes

It also fits very well with your current narrative around **GenAI for documents, extraction pipelines, and AI products**.

---

## Project Idea

## DocTune: Fine-tuned Small LLM for Structured Document Extraction

A system that takes short documents or text extracted from PDFs and returns **structured JSON** with predefined fields, such as:

- invoice number
- employee name
- gross pay
- deductions
- tax amount
- net pay
- pay period

The idea is to:

1. create or use a small synthetic dataset
2. fine-tune a compact model locally
3. compare it against a prompt-only baseline
4. package the model behind an API
5. deploy everything locally with **Minikube** or **K3s**

---

## Why This Project Is Very Strong for You

This project is strong because it:

- connects directly with your background in document extraction
- shows that you can **train models**, not only consume APIs
- demonstrates real **PyTorch** usage
- demonstrates **deployment**
- demonstrates **Kubernetes**
- gives you clear metrics for portfolio and resume

It also has a value proposition that recruiters understand quickly.

You could describe it like this:

> Built a local fine-tuning pipeline for a compact LLM using PyTorch and QLoRA to perform schema-constrained document information extraction, then deployed the model as a containerized inference service on Kubernetes.

That sounds very strong.

---

## What to Train Locally with an RTX 2070 8GB

For your GPU, the most viable path is:

### Small base models
- `Qwen2.5-1.5B-Instruct`
- `Llama 3.2 1B Instruct`
- `SmolLM2`
- `TinyLlama`

### Training approach
- fine-tuning with **LoRA** or **QLoRA**
- **4-bit quantization**
- moderate sequence length
- small batch size + gradient accumulation

With **8 GB VRAM**, full fine-tuning of large models is not realistic.

The ideal local stack is:

- Transformers
- PyTorch
- PEFT
- bitsandbytes
- TRL or standard trainer

This is totally feasible locally for a demonstrable portfolio project.

---

## Ideal Project Scope

## Phase 1 — Dataset

Build a supervised dataset of input/target pairs.

### Input
```json
{
  "document_text": "Employee: John Doe\nGross Pay: $4,500\nTax: $700\nDeductions: $300\nNet Pay: $3,500\nPeriod: Jan 2026"
}



### Target 
{
  "employee_name": "John Doe",
  "gross_pay": 4500,
  "tax": 700,
  "deductions": 300,
  "net_pay": 3500,
  "pay_period": "Jan 2026"
}