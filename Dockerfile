FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# System dependencies for TRELLIS
RUN apt-get update && apt-get install -y \
    git \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install base Python dependencies first
RUN pip install --no-cache-dir \
    runpod>=1.6.0 \
    trimesh>=4.0.0 \
    pygltflib>=1.16.0 \
    numpy>=1.24.0 \
    Pillow>=10.0.0 \
    huggingface_hub \
    safetensors \
    einops \
    scipy \
    tqdm \
    imageio \
    imageio-ffmpeg \
    rembg \
    onnxruntime

# Clone TRELLIS
RUN git clone https://github.com/microsoft/TRELLIS.git /app/trellis

# Install TRELLIS requirements
WORKDIR /app/trellis
RUN pip install --no-cache-dir -r requirements.txt || true

# Install spconv for sparse convolutions (CUDA 11.8)
RUN pip install --no-cache-dir spconv-cu118

# Install flash-attn (may take a while to compile)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || echo "flash-attn install failed, continuing..."

# Install xformers for memory-efficient attention
RUN pip install --no-cache-dir xformers || echo "xformers install failed, continuing..."

# Add TRELLIS to Python path
ENV PYTHONPATH="/app/trellis:${PYTHONPATH}"

# Copy worker files
WORKDIR /app
COPY handler.py .
COPY src/ ./src/

# Pre-download model weights (baked into image for fast cold starts)
# This downloads ~8GB of model weights during build
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/TRELLIS-text-xlarge', local_dir='/app/models/TRELLIS-text-xlarge')" || echo "Model download will happen at runtime"

ENV TRELLIS_MODEL_PATH="/app/models/TRELLIS-text-xlarge"

CMD ["python", "-u", "handler.py"]
