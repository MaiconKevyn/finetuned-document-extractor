import json
import torch
import re
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

def extract_json(text):
    """Extrai o objeto JSON da resposta do modelo."""
    try:
        # Tenta encontrar algo entre chaves
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        return None
    return None

def calculate_metrics(predictions, ground_truths):
    """Calcula Valid JSON Rate e Field-level Accuracy."""
    total = len(predictions)
    valid_json_count = 0
    field_scores = {}
    
    for pred, gt in zip(predictions, ground_truths):
        if pred is not None:
            valid_json_count += 1
            for field in gt.keys():
                if field not in field_scores:
                    field_scores[field] = 0
                
                p_val = str(pred.get(field, "")).strip().lower()
                g_val = str(gt.get(field, "")).strip().lower()
                
                if p_val == g_val:
                    field_scores[field] += 1
        
    metrics = {
        "valid_json_rate": round(valid_json_count / total, 4) if total > 0 else 0,
        "field_accuracy": {k: round(v / total, 4) for k, v in field_scores.items()},
        "avg_field_accuracy": round(sum(field_scores.values()) / (total * len(ground_truths[0])), 4) if total > 0 else 0
    }
    return metrics

def run_evaluation(model_id, adapter_path=None, test_file="data/val.jsonl"):
    print(f"\n--- Evaluando: {'Fine-tuned' if adapter_path else 'Baseline'} ---")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    if adapter_path:
        print(f"Carregando adaptador LoRA de: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    with open(test_file, 'r') as f:
        samples = [json.loads(line) for line in f]
    
    predictions = []
    ground_truths = []
    
    start_time = time.time()
    for sample in tqdm(samples):
        prompt = f"### Instruction:\n{sample['instruction']}\n\n### Input:\n{sample['input']}\n\n### Response:\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256, 
                temperature=0.1,
                do_sample=False # Deterministico para avaliação
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction_text = response.split("### Response:\n")[-1]
        
        predictions.append(extract_json(prediction_text))
        ground_truths.append(json.loads(sample['output']))
    
    duration = time.time() - start_time
    metrics = calculate_metrics(predictions, ground_truths)
    metrics["avg_latency_sec"] = round(duration / len(samples), 4)
    
    # Limpeza de memória GPU
    del model
    torch.cuda.empty_cache()
    
    return metrics

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    
    model_base = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter = "models/doctune-qwen-1.5b-lora"
    
    # 1. Rodar Baseline
    baseline_res = run_evaluation(model_base)
    
    # 2. Rodar Fine-tuned
    ft_res = run_evaluation(model_base, adapter_path=adapter)
    
    artifact = {
        "project": "DocTune",
        "date": "2026-04-20",
        "comparison": {
            "baseline": baseline_res,
            "fine_tuned": ft_res
        }
    }
    
    with open("results/artifact_results.json", "w") as f:
        json.dump(artifact, f, indent=2)
        
    print("\n=== ARTIFACT DE RESULTADOS GERADO em results/artifact_results.json ===")
    print(f"Baseline Valid JSON: {baseline_res['valid_json_rate']*100}%")
    print(f"Fine-tuned Valid JSON: {ft_res['valid_json_rate']*100}%")
    print(f"Fine-tuned Avg Accuracy: {ft_res['avg_field_accuracy']*100}%")
