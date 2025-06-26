# --------------------
# STAGE 1: Builder
# --------------------
FROM python:3.10.3-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download InsightFace models
RUN mkdir -p /app/.insightface/models \
    && wget -q https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip -P /app/.insightface/models/ \
    && unzip /app/.insightface/models/antelopev2.zip -d /app/.insightface/models/

# --------------------
# STAGE 2: Runtime
# --------------------
FROM python:3.10.3-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment and models
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/.insightface /app/.insightface

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"
ENV INSIGHTFACE_CACHE_DIR=/app/.insightface

# Copy application code
COPY . .

# Fix permissions
RUN useradd -m appuser \
    && chown -R appuser:appuser /app \
    && chmod -R 755 /app/.insightface

USER appuser

# Expose and run application
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
