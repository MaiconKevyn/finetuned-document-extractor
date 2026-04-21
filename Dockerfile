# Usa imagem base da NVIDIA com CUDA
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Define variáveis de ambiente para não interatividade e output direto
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Instala Python 3.13 e dependências do sistema
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.13 \
    python3.13-venv \
    python3.13-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Cria um usuário não-root (appuser)
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Configura o diretório de trabalho
WORKDIR /app

# Transfere a propriedade do diretório para o usuário não-root
RUN chown -R appuser:appuser /app

# Muda para o usuário não-root
USER appuser

# Cria um ambiente virtual para instalar pacotes sem conflitos globais
RUN python3.13 -m venv /app/.venv
# Coloca o .venv no PATH para que o 'pip' e o 'python' corretos sejam usados
ENV PATH="/app/.venv/bin:$PATH"

# Copia os requisitos PRIMEIRO para aproveitar o cache de camadas do Docker
COPY --chown=appuser:appuser requirements.txt .

# Instala dependências (usando o índice do torch para CUDA 12.4)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir -r requirements.txt

# Copia o código da API
COPY --chown=appuser:appuser src/ /app/src/

# ATENÇÃO: Na indústria, os pesos do modelo são geralmente montados via Volume.
# Estamos mantendo o COPY aqui temporariamente para não quebrar seu fluxo atual.
COPY --chown=appuser:appuser models/ /app/models/

# Expõe a porta do FastAPI
EXPOSE 8000

# Comando para rodar a aplicação
CMD ["python", "-m", "src.api.main"]