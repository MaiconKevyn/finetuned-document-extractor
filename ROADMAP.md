# DocTune Implementation Roadmap

Este documento detalha os checkpoints, decisões técnicas e ideias de brainstorming para o projeto DocTune.

---

## 🚀 Fase 1: Dados e Preparação [CONCLUÍDA]
- [x] Configuração de Ambiente (Python 3.13 + .venv)
- [x] Estruturação de diretórios
- [x] Script de Geração de Dados Sintéticos (`scripts/generate_dataset.py`)
- [x] Split de Dados 90/10 → `data/train_dataset.jsonl` (900) + `data/val_dataset.jsonl` (100)
- [x] **Noise Injection:** Ruído aplicado apenas em chars não-numéricos (preserva IDs e valores monetários).

## 🧠 Fase 2: Fine-Tuning (SFT) [CONCLUÍDA]
- [x] Escolha do Modelo Base: `Qwen2.5-1.5B-Instruct`.
- [x] Configuração do QLoRA (4-bit quantization) para 8GB VRAM.
- [x] Implementação do Script de Treino (`scripts/finetune.py`) com estabilização FP16 para RTX 2070.
- [x] Treinamento Finalizado: Loss reduzida de 1.79 para 0.46.

## 📊 Fase 3: Avaliação e Benchmarking [CONCLUÍDA]
- [x] Criar Baseline "Prompt-Only" (Qwen 1.5B puro).
- [x] Script de Avaliação: Implementado com **field-level accuracy**.
- [x] Medir Inferência: Latência média de ~4s/doc na RTX 2070.
- [x] **Checkpoint — Artifact de Resultados:** Salvo em `results/artifact_results.json`.
    - **Resultado:** Acurácia média saltou de **63.8% (Baseline)** para **93.7% (Fine-tuned)**.

## 🚢 Fase 4: MLOps & API [CONCLUÍDA]
- [x] Desenvolver API com FastAPI em `src/api/main.py`.
- [x] Implementar Singleton para o carregamento do modelo + adaptador.
- [x] Dockerização da aplicação: `Dockerfile` criado e imagem `doctune-api:latest` gerada com sucesso.
- [x] Teste de inferência via API: **Sucesso** (extração de JSON validada localmente).

## ⚓ Fase 5: Kubernetes (K8s) [CONCLUÍDA]
- [x] Cluster K3s local instalado e node `Ready`.
- [x] NVIDIA Container Toolkit e Device Plugin configurados.
- [x] Escrita de Manifestos: `k8s/deployment.yaml` e `k8s/service.yaml`.
- [x] Importação da imagem para o K3s (`ctr images import`).
- [x] Deployment realizado: API rodando com limites de GPU orquestrados.

---

## 🛠 Guia de Comandos Operacionais (Cheat Sheet)

### 1. Ambiente e Pipeline
```bash
# Instalação PyTorch CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Geração, Treino e Avaliação
python scripts/generate_dataset.py
python scripts/finetune.py
python scripts/evaluate.py
```

### 2. Docker e Kubernetes
```bash
# Build e Importação para o Cluster
docker build -t doctune-api:latest .
docker save doctune-api:latest | sudo k3s ctr images import -

# Deploy e Monitoramento
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl get pods -w
kubectl logs -f deployment/doctune-api
kubectl port-forward service/doctune-service 8080:80
```

### 3. Debug de GPU no K8s
```bash
# Verificar se a GPU está alocável no cluster
kubectl describe node | grep nvidia.com/gpu
# Verificar logs do plugin
kubectl get pods -n kube-system | grep nvidia
```

---

## 🛠 Decisões Técnicas (Log)
- **Hardware:** NVIDIA RTX 2070 8GB.
- **Estratégia:** QLoRA 4-bit + FP16 Training (RTX 2070 incompatível com BFloat16).
- **Stack:** Python 3.13, PyTorch 2.6, K3s, FastAPI.
