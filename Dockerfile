# Usa imagem base da NVIDIA com CUDA e Python
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Instala Python e dependências de sistema
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia e instala requisitos (usando o índice do torch para CUDA 12.4)
COPY requirements.txt .
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install --no-cache-dir -r requirements.txt

# Copia o código da API e o modelo treinado
COPY src/ /app/src/
COPY models/ /app/models/

# Expõe a porta do FastAPI
EXPOSE 8000

# Comando para rodar a aplicação
CMD ["python3", "-m", "src.api.main"]
