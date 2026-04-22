# Usa imagem base da NVIDIA com CUDA
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Define variáveis de ambiente para não interatividade e output direto
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Instala Python 3.11 e dependências do sistema
# Python 3.11 é a versão mais estável para o stack bitsandbytes + triton + torch
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    gcc \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Cria um usuário não-root (appuser)
RUN groupadd -r -g 1001 appuser && useradd -r -u 1001 -g appuser -d /app -s /sbin/nologin appuser

# Configura o diretório de trabalho
WORKDIR /app

# Transfere a propriedade do diretório para o usuário não-root
RUN mkdir -p /tmp/hf && chown -R appuser:appuser /app /tmp/hf

# Muda para o usuário não-root
USER appuser

# Cria um ambiente virtual para instalar pacotes sem conflitos globais
RUN python3.11 -m venv /app/.venv
# Coloca o .venv no PATH para que o 'pip' e o 'python' corretos sejam usados
ENV PATH="/app/.venv/bin:$PATH"

# Copia os requisitos PRIMEIRO para aproveitar o cache de camadas do Docker
COPY --chown=appuser:appuser requirements.txt .

# Instala dependências (usando o índice do torch para CUDA 12.4)
RUN pip install --no-cache-dir --upgrade pip setuptools \
    && pip install --no-cache-dir \
       torch==2.5.1 \
       torchvision==0.20.1 \
       torchaudio==2.5.1 \
       --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir -r requirements.txt

# Copia o código da API
COPY --chown=appuser:appuser src/ /app/src/

# models/ não está na imagem — montado via bind mount no docker compose

# Expõe a porta do FastAPI
EXPOSE 8000

# Comando para rodar a aplicação
CMD ["python", "-m", "src.api.main"]