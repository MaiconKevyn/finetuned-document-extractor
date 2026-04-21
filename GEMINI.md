# DocTune: Fine-tuned Small LLM for Structured Document Extraction

Este projeto visa construir um pipeline completo de fine-tuning para modelos de linguagem pequenos (SLMs) focados na extração de informações estruturadas de documentos, utilizando PyTorch e QLoRA. O objetivo final é implantar o modelo em um cluster Kubernetes (K3s/Minikube).

## Visão Geral do Projeto

O **DocTune** resolve o problema de extração de campos específicos (como nome do funcionário, valores monetários e períodos) de textos ruidosos (simulando falhas de OCR) e os converte em JSON estruturado.

### Tecnologias Principais
- **Linguagem:** Python 3.13
- **Deep Learning:** PyTorch, Hugging Face Transformers, PEFT (LoRA/QLoRA), TRL (SFTTrainer).
- **Geração de Dados:** Faker, Pydantic.
- **Hardware Target:** NVIDIA RTX 2070 8GB (Otimizado para baixa VRAM).
- **Infraestrutura:** Docker, Kubernetes (K3s).

### Arquitetura de Extração
1.  **Input:** Texto bruto de documentos com injeção de ruído (caracteres especiais e quebras de linha).
2.  **Processamento:** Modelo compactado (`Qwen2.5-1.5B` ou `Llama-3.2-1B`) carregado em 4-bit.
3.  **Output:** JSON validado via Pydantic.

---

## Comandos Principais

### Configuração do Ambiente
```bash
python3.13 -m venv .venv
source .venv/bin/activate
# Instalação otimizada para CUDA 12.4 (RTX 2070)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 1. Geração de Dataset Sintético
Gera 1000 exemplos de documentos (treino/validação) com injeção de ruído inteligente (não corrompe dígitos).
```bash
python scripts/generate_dataset.py
```

### 2. Fine-Tuning (SFT)
Executa o treinamento usando QLoRA. O script está configurado para evitar erros de `BFloat16` em GPUs de arquitetura Turing (RTX 2000).
```bash
python scripts/finetune.py
```

### 3. Avaliação e Benchmarking
Compara o modelo base (zero-shot) com o modelo fine-tuned.
```bash
python scripts/evaluate.py
```

---

## Convenções e Decisões Técnicas

- **Compatibilidade de Hardware:** A RTX 2070 não suporta `bf16`. O projeto força o uso de `fp16` ou `float32` para os adaptadores LoRA para garantir estabilidade.
- **Otimização de Memória:** 
    - Uso obrigatório de `4-bit quantization` (NF4).
    - `Gradient Accumulation` configurado para simular batch sizes maiores.
    - `Gradient Checkpointing` habilitado para economizar VRAM.
- **Qualidade de Dados:** O ruído inserido nos documentos preserva dígitos e pontuação monetária, focando em corromper apenas labels e nomes, simulando o comportamento real de falhas de OCR.
- **Estrutura de Pastas:**
    - `data/`: Datasets (`.jsonl`) e logs.
    - `scripts/`: Lógica de treino, geração e avaliação.
    - `models/`: Checkpoints e adaptadores LoRA salvos.
    - `src/api/`: (Planejado) Código para o servidor FastAPI.

## Roadmap de Implementação
Consulte o arquivo `ROADMAP.md` para o status detalhado das fases (Dados, Fine-tuning, Avaliação, MLOps e Kubernetes).
