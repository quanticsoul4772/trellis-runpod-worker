FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Limit ninja parallelism to avoid OOM during compilation
ENV MAX_JOBS=4
ENV NINJA_MAX_JOBS=4

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
    wget \
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

# Install prebuilt flash-attn wheel (much faster than compiling)
# Using the prebuilt wheel for PyTorch 2.1, CUDA 11.8, Python 3.10
RUN pip install --no-cache-dir flash-attn==2.5.9.post1 --no-build-isolation \
    || pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
    || echo "flash-attn install failed, will use fallback attention"

# Install xformers for memory-efficient attention (prebuilt)
RUN pip install --no-cache-dir xformers==0.0.23 || echo "xformers install failed, continuing..."

# Install spconv for sparse convolutions (CUDA 11.8)
RUN pip install --no-cache-dir spconv-cu118

# Install kaolin for 3D deep learning (prebuilt wheel for PyTorch 2.1, CUDA 11.8)
RUN pip install --no-cache-dir kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu118.html \
    || pip install --no-cache-dir kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu118.html \
    || echo "kaolin install failed, continuing..."

# Install nvdiffrast for differentiable rendering (compile with limited parallelism)
RUN pip install --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git \
    || echo "nvdiffrast install failed, continuing..."

# Clone TRELLIS
RUN git clone --depth 1 https://github.com/microsoft/TRELLIS.git /app/trellis

# Install TRELLIS requirements (with error tolerance)
WORKDIR /app/trellis
RUN pip install --no-cache-dir -r requirements.txt 2>/dev/null || \
    pip install --no-cache-dir torch torchvision torchaudio || \
    echo "Some TRELLIS requirements may be missing"

# Add TRELLIS to Python path
ENV PYTHONPATH="/app/trellis:${PYTHONPATH}"

# Copy worker files
WORKDIR /app
COPY handler.py .
COPY src/ ./src/

# Model will be downloaded at runtime to avoid build timeout
# First run will take ~5 minutes to download ~8GB of weights
ENV TRELLIS_MODEL_PATH="/app/models/TRELLIS-text-xlarge"
ENV HF_HOME="/app/hf_cache"

# Create startup script that handles model download
RUN echo '#!/bin/bash\n\
if [ ! -d "/app/models/TRELLIS-text-xlarge" ]; then\n\
    echo "Downloading TRELLIS model weights (first run only)..."\n\
    python -c "from huggingface_hub import snapshot_download; snapshot_download(\"microsoft/TRELLIS-text-xlarge\", local_dir=\"/app/models/TRELLIS-text-xlarge\")"\n\
fi\n\
exec python -u handler.py\n\
' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
