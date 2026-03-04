FROM runpod/pytorch:1.0.3-cu1281-torch280-ubuntu2204

WORKDIR /app

# System deps for audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends libsndfile1 sox && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model weights at build time (baked into image)
RUN huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir /model

COPY handler.py .

CMD ["python", "handler.py"]
