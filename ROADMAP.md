# DocTune Implementation Roadmap

Este documento detalha os checkpoints, decisões técnicas e ideias de brainstorming para o projeto DocTune.

---

## 🚀 Fase 1: Dados e Preparação [CONCLUÍDA]
- [x] Configuração de Ambiente (Python 3.13 + .venv)
- [x] Estruturação de diretórios
- [x] Script de Geração de Dados Sintéticos (`scripts/generate_dataset.py`)
- [x] Split de Dados 90/10 → `data/train_dataset.jsonl` (900) + `data/val_dataset.jsonl` (100)
- [x] **Noise Injection:** Ruído aplicado apenas em chars não-numéricos (preserva IDs e valores monetários). 5 templates de documento.

## 🧠 Fase 2: Fine-Tuning (SFT)
- [ ] Escolha do Modelo Base: `Llama-3.2-1B-Instruct` ou `Qwen2.5-1.5B-Instruct`.
- [ ] Configuração do QLoRA (4-bit quantization) para 8GB VRAM.
- [ ] Implementação do Script de Treino (`scripts/finetune.py`) usando `TRL.SFTTrainer`.
- [ ] Registro de Experimentos no Tensorboard.
- [ ] **Checkpoint:** Validar se o modelo consegue fechar o JSON corretamente (completude sintática).

## 📊 Fase 3: Avaliação e Benchmarking [CONCLUÍDA]
- [x] Criar Baseline "Prompt-Only" (Qwen 1.5B puro).
- [x] Script de Avaliação: Implementado com **field-level accuracy**.
- [x] Medir Inferência: Latência média de ~4s/doc na RTX 2070.
- [x] **Checkpoint — Artifact de Resultados:** Salvo em `results/artifact_results.json`.
    - **Resultado:** Acurácia média saltou de **63.8% (Baseline)** para **93.7% (Fine-tuned)**.
    - **Insight:** Ganho massivo (+64%) na extração de valores monetários e IDs sob ruído.

## 🚢 Fase 4: MLOps & API [CONCLUÍDA]
- [x] Desenvolver API com FastAPI em `src/api/main.py`.
- [x] Implementar Singleton para o carregamento do modelo + adaptador (evitar leak de VRAM).
- [x] Dockerização da aplicação: `Dockerfile` criado com base NVIDIA CUDA 12.4.
- [x] Teste de inferência via API: **Sucesso** (extração de JSON validada localmente).
- [ ] **Next:** Build e Teste da imagem Docker local (`docker build`).

## ⚓ Fase 5: Kubernetes (K8s) [EM PROGRESSO]

### 🛠 Setup Realizado (Pré-requisitos)

#### 1. Instalar K3s
```bash
curl -sfL https://get.k3s.io | sh -
sudo k3s kubectl get nodes  # verificar se o node está Ready
```

#### 2. Configurar kubeconfig sem sudo
```bash
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $USER:$USER ~/.kube/config
sudo chmod 644 /etc/rancher/k3s/k3s.yaml
export KUBECONFIG=~/.kube/config
```

#### 3. Instalar nvidia-container-toolkit
> O repositório genérico da NVIDIA não funciona no Ubuntu 24.04. O pacote está disponível no repo CUDA já configurado.
```bash
sudo apt-get install -y nvidia-container-toolkit
```

#### 4. Configurar containerd (runtime do K3s) para usar NVIDIA
```bash
sudo nvidia-ctk runtime configure --runtime=containerd
sudo systemctl restart k3s
```

#### 5. Instalar NVIDIA Device Plugin no cluster
```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml
```

#### 6. Verificar exposição da GPU
```bash
kubectl describe node | grep nvidia.com/gpu
kubectl get pods -n kube-system | grep nvidia
```
Esperado: `nvidia.com/gpu: 1`

---

### Checkpoints
- [x] Cluster K3s local instalado e node `Ready`
- [x] NVIDIA Container Toolkit instalado (v1.19.0)
- [x] containerd configurado para GPU
- [x] NVIDIA Device Plugin aplicado no cluster
- [ ] Escrita de Manifestos:
    - `k8s/deployment.yaml`: Configuração de limites de GPU (`nvidia.com/gpu: 1`).
    - `k8s/service.yaml`: Exposição da API.
- [ ] Teste de escalonamento: Verificar se o Pod reinicia corretamente e gerencia a VRAM.

---

## 💡 Brainstorm de Ideias (Área de rascunho)

1.  **Noise Injection:** No script de geração de dados, adicionar caracteres especiais aleatórios para simular o "sujeira" de PDFs escaneados.
2.  **Schema Constrained Decoding:** Usar bibliotecas como `Guidance` ou `Outlines` na fase de API para garantir que o output seja *sempre* um JSON válido, independentemente do modelo.
3.  **HuggingFace Hub:** Fazer upload do modelo (ou apenas do adaptador) para o HF Hub privado/público para demonstrar o fluxo completo de um ML Engineer.
4.  **Comparação de Modelos:** Treinar o SmolLM2-135M e comparar com o Llama-1B para ver o "sweet spot" de performance vs hardware.

---

## 🛠 Decisões Técnicas (Log)
- **Data:** 2026-04-20
- **Hardware:** NVIDIA RTX 2070 8GB.
- **Estratégia:** QLoRA é obrigatório. Batch size pequeno (2-4) com Gradient Accumulation (4-8) para simular batch sizes maiores.
- **Stack:** Python 3.13, PyTorch, Transformers, K3s.
