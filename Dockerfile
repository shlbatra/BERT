# GPT-2 Training Docker Container - Multi-stage build

# ============================================
# Stage 1: Builder - Install dependencies
# ============================================
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04 AS builder

# Install system dependencies and Python
RUN apt-get update --allow-insecure-repositories && \
    apt-get install -y --allow-unauthenticated \
    ca-certificates \
    gnupg \
    python3 \
    python3-pip \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip3 install --no-cache-dir --break-system-packages uv

# Set uv environment variables
ENV UV_CACHE_DIR=/tmp/uv-cache
ENV UV_PYTHON=python3

# Set working directory
WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml /app/

# Create virtual environment and install dependencies
RUN uv venv .venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install dependencies from pyproject.toml
RUN uv sync --no-dev

# ============================================
# Stage 2: Runtime - Lean production image
# ============================================
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04 AS runtime

# Create non-root user for security
RUN groupadd -r gpt2 && useradd -r -g gpt2 -u 1001 -m gpt2

# Install minimal runtime dependencies only
RUN apt-get update --allow-insecure-repositories && \
    apt-get install -y --allow-unauthenticated \
    python3 \
    tmux \
    rsync \
    && rm -rf /var/lib/apt/lists/*

# Set Hugging Face cache directory
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers
ENV HF_DATASETS_CACHE=/app/.cache/huggingface/datasets
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/gcp-key.json

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY --chown=gpt2:gpt2 src/gpt_module/ /app/gpt_module/
COPY --chown=gpt2:gpt2 src/training/ /app/training/
COPY --chown=gpt2:gpt2 src/data_scripts /app/data_scripts
COPY --chown=gpt2:gpt2 src/train_gpt.py /app/

# Create cache directory and fix permissions for entire app folder
RUN mkdir -p /app/.cache/huggingface && \
    chown -R gpt2:gpt2 /app

# Switch to non-root user
USER gpt2