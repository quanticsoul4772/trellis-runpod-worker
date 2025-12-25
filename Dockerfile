# TRELLIS requires PyTorch 2.4.0 - use matching RunPod base image
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

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
    onnxruntime \
    easydict

# Skip flash-attn - compilation takes >30min and exceeds RunPod build timeout
# TRELLIS will use xformers as fallback for attention
RUN echo "Skipping flash-attn (build timeout issue) - using xformers fallback"

# Install xformers for memory-efficient attention (compatible with PyTorch 2.4)
RUN pip install --no-cache-dir xformers || echo "xformers install failed, continuing..."

# Install spconv for sparse convolutions (CUDA 12.4)
RUN pip install --no-cache-dir spconv-cu124 \
    || pip install --no-cache-dir spconv-cu120 \
    || echo "spconv install failed, continuing..."

# Install kaolin for 3D deep learning (prebuilt wheel for PyTorch 2.4, CUDA 12.4)
RUN pip install --no-cache-dir kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu124.html \
    || pip install --no-cache-dir kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html \
    || echo "kaolin install failed, continuing..."

# Install nvdiffrast for differentiable rendering
RUN pip install --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git \
    || echo "nvdiffrast install failed, continuing..."

# Clone TRELLIS
RUN git clone --depth 1 https://github.com/microsoft/TRELLIS.git /app/trellis

# Install TRELLIS requirements (skip torch/torchvision since base image has compatible versions)
WORKDIR /app/trellis
RUN pip install --no-cache-dir $(grep -v -E "^(torch|torchvision|torchaudio)" requirements.txt | tr '\n' ' ') 2>/dev/null \
    || echo "Some TRELLIS requirements may need manual install"

# Ensure torchvision matches base image PyTorch 2.4.0
RUN pip install --no-cache-dir torchvision==0.19.0 --no-deps || echo "torchvision already installed"

# Add TRELLIS to Python path
ENV PYTHONPATH="/app/trellis"

# Copy worker files
WORKDIR /app
COPY handler.py .

# Model will be downloaded at runtime to avoid build timeout
ENV TRELLIS_MODEL_PATH="/app/models/TRELLIS-text-xlarge"
ENV HF_HOME="/app/hf_cache"

# Verify setup during build
RUN echo "=== BUILD VERIFICATION ===" && \
    python --version && \
    python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')" && \
    ls -la /app/handler.py && \
    python -m py_compile /app/handler.py && echo "Syntax OK"

# Run the handler
CMD ["python", "-u", "/app/handler.py"]
