# Face Verification Microservice

A production-ready microservice that verifies if two facial images belong to the same person using advanced facial recognition models. This service intelligently handles various image types including ID card photos, indoor selfies, and outdoor selfies with adaptive preprocessing.

## Features

- **Multi-model ensemble**: Uses InsightFace, FaceNet, and Dlib for robust face verification[1]
- **Adaptive preprocessing**: Automatically adjusts image enhancement based on image characteristics[2]
- **Flexible thresholds**: Configurable thresholds for different verification scenarios[1]
- **Production-ready**: Built with FastAPI, comprehensive error handling, and Docker support[3]
- **Offline operation**: Works completely offline without external API dependencies[1]

## Model Sources

The service uses the following open-source models:

- **InsightFace**: Buffalo_L model for face detection and recognition[4]
- **FaceNet**: VGGFace2 pretrained model via facenet-pytorch[4]
- **Dlib**: Face recognition ResNet model v1[4]

All models are automatically downloaded during the first run or can be pre-downloaded for faster deployment.

## API Endpoints

### POST /verify-face
Verifies if two facial images belong to the same person.

**Request:**
```json
{
  "user_id": "stu_4421",
  "image_type": "id_to_selfie",
  "image_1_base64": "<base64_image_of_id_photo>",
  "image_2_base64": "<base64_image_of_selfie>"
}
```

**Response:**
```json
{
  "user_id": "stu_4421",
  "match_score": 0.68,
  "match": true,
  "confidence_level": "medium",
  "image_type": "id_to_selfie",
  "threshold": 0.65,
  "status": "verified"
}
```

### GET /health
Returns service health status.

**Response:**
```json
{
  "status": "ok",
  "adaptive_processing": true
}
```

### GET /version
Returns model version information.

**Response:**
```json
{
  "model_version": "2.0.0",
  "adaptive_features": true
}
```

## Configuration

The service is configured via `config.json` which includes:[5]

- **Thresholds**: Different similarity thresholds for `id_to_selfie` (0.70) and `selfie_to_selfie` (0.82)
- **Preprocessing**: Image enhancement parameters including CLAHE, noise reduction, and adaptive brightness
- **Model weights**: Ensemble weights for combining multiple model predictions
- **Confidence levels**: Score ranges for low/medium/high confidence classifications

## Installation & Setup

### Prerequisites

- Python 3.8+
- Docker (optional but recommended)
- At least 4GB RAM
- CPU with AVX support (recommended)

### Local Installation

1. **Clone the repository and install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Download required Dlib models (optional - will auto-download if needed):**
```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2

wget http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
bunzip2 dlib_face_recognition_resnet_model_v1.dat.bz2
```

3. **Run the service:**
```bash
python main.py
```

The service will be available at `http://localhost:8000` with Swagger documentation at `http://localhost:8000/docs`.

## Docker Deployment

### Building the Docker Image

```bash
docker build -t face-verifier .
```

### Running the Container

```bash
docker run -p 8000:8000 face-verifier
```

### Running with Custom Configuration

```bash
docker run -p 8000:8000 -v $(pwd)/config.json:/app/config.json face-verifier
```

### Running in Detached Mode

```bash
docker run -d -p 8000:8000 --name face-verification-service face-verifier
```

### Viewing Logs

```bash
docker logs face-verification-service
```

### Stopping the Container

```bash
docker stop face-verification-service
docker rm face-verification-service
```

## Dockerfile

Create a `Dockerfile` in your project root:

```dockerfile
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
RUN pip install --no-cache-dir -r req.txt

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
```

## Testing

### Basic Health Check

```bash
curl http://localhost:8000/health
```

### Face Verification Test

```bash
curl -X POST "http://localhost:8000/verify-face" \
     -H "Content-Type: application/json" \
     -d '{
       "user_id": "test_user",
       "image_type": "selfie_to_selfie",
       "image_1_base64": "base64_encoded_image_1",
       "image_2_base64": "base64_encoded_image_2"
     }'
```

### Image Validation Test

```bash
curl -X POST "http://localhost:8000/test-validation" \
     -H "Content-Type: application/json" \
     -d '{
       "image_base64": "base64_encoded_image"
     }'
```

## Project Structure

```
face-verification-service/
├── main.py                 # FastAPI application and endpoints
├── face_matcher.py         # Face recognition and ensemble logic
├── preprocess.py          # Image preprocessing and validation
├── config.json            # Configuration file
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── README.md             # This file
└── models/               # Directory for model files (auto-created)
```

## Performance Considerations

- **Memory Usage**: Approximately 2-4GB RAM depending on loaded models
- **Processing Time**: 1-3 seconds per verification request
- **Throughput**: ~10-20 requests per second on modern hardware
- **Model Loading**: Initial startup takes 30-60 seconds to load all models

## Error Handling

The service provides detailed error responses for common issues:

- **400**: Invalid base64 image format or image validation failure
- **422**: No face detected in one or both images
- **500**: Internal server error during processing
- **503**: Service not properly initialized

## Troubleshooting

### Common Issues

1. **Models not loading**: Ensure sufficient RAM and check logs for specific model errors
2. **Face detection failures**: Verify image quality and face visibility
3. **Low accuracy**: Adjust thresholds in `config.json` based on your use case
4. **Slow performance**: Consider using only InsightFace model for faster processing

### Logs

The service provides comprehensive logging. Set log level in the startup configuration:

```python
logging.basicConfig(level=logging.DEBUG)  # For detailed logs
```

## License

This project uses open-source models and libraries. Please ensure compliance with their respective licenses:

- InsightFace: Apache 2.0
- FaceNet: MIT
- Dlib: Boost Software License