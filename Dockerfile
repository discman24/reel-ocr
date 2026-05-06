# REEL — PaddleOCR Microservice Dockerfile
# Deploy to Railway: railway up
# Or Render: set Start Command = uvicorn paddle_ocr_service:app --host 0.0.0.0 --port $PORT

FROM python:3.11-slim

# System deps for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download PaddleOCR models at build time (avoids cold start download)

COPY paddle_ocr_service.py .

EXPOSE 8000

CMD ["uvicorn", "paddle_ocr_service:app", "--host", "0.0.0.0", "--port", "8000"]
