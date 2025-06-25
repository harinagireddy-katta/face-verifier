# --------------------
# STAGE 1: Builder
# --------------------
FROM python:3.10.3 AS builder

WORKDIR /app

# Install build tools and dependencies for dlib
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    bzip2 \
    libopenblas-dev \
    liblapack-dev \
    libssl-dev \
    pkg-config \
    libx11-dev \
    libgtk-3-dev \
    libboost-all-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first
COPY requirements.txt .

# Create and activate venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies (dlib will now build correctly)
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Download dlib model weights
RUN wget -q http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && \
    bunzip2 shape_predictor_68_face_landmarks.dat.bz2 && \
    wget -q http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2 && \
    bunzip2 dlib_face_recognition_resnet_model_v1.dat.bz2

# --------------------
# STAGE 2: Runtime
# --------------------
FROM python:3.10.3

WORKDIR /app

# Copy virtual environment and weights
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/*.dat ./

# Install runtime-only dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libopenblas-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

#RUN pip install --no-cache-dir uvicorn==0.34.3

# Copy app files
COPY . .

# Copy your custom landmark model if needed
COPY shape_predictor_5_face_landmarks.dat .

# Set venv in path
ENV PATH="/opt/venv/bin:$PATH"
ENV INSIGHTFACE_CACHE_DIR=/app/.insightface

# Run as non-root
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Run the FastAPI app
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]