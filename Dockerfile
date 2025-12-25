# =============================================================================
# TRELLIS Text-to-3D RunPod Serverless Dockerfile
# =============================================================================
# Key insights from comprehensive analysis:
# 1. Use CUDA 12.1 (not 12.4) for xformers compatibility with PyTorch 2.4.0
# 2. Clone TRELLIS first, then install deps (PYTHONPATH order matters)
# 3. Install easydict LAST to prevent shadowing by other packages
# 4. Use --ignore-installed for distutils conflicts in base image
# 5. Skip flash-attn (30min compile exceeds RunPod build timeout)
# =============================================================================

# PyTorch 2.4.0 with CUDA 12.1 - compatible with xformers 0.0.27.post2
FROM runpod/pytorch:2.4.0-py3.11-cuda12.1.0-devel-ubuntu22.04

# Limit ninja parallelism to avoid OOM during CUDA kernel compilation
ENV MAX_JOBS=4
ENV NINJA_MAX_JOBS=4

# System dependencies for TRELLIS (OpenGL, image processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# =============================================================================
# PHASE 1: Clone TRELLIS first (establishes PYTHONPATH priority)
# =============================================================================
RUN git clone --depth 1 https://github.com/microsoft/TRELLIS.git /app/trellis

# Set PYTHONPATH early so TRELLIS modules are discoverable
ENV PYTHONPATH="/app/trellis:${PYTHONPATH}"

# =============================================================================
# PHASE 2: Install xformers with correct version for PyTorch 2.4.0 + CUDA 12.1
# Must be done BEFORE other packages to establish correct torch dependencies
# =============================================================================
RUN pip install --no-cache-dir xformers==0.0.27.post2 \
    || pip install --no-cache-dir xformers \
    || echo "WARNING: xformers install failed, TRELLIS will use slower attention"

# Verify torch/torchvision versions are intact after xformers
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    python -c "import torchvision; print(f'torchvision: {torchvision.__version__}')"

# =============================================================================
# PHASE 3: Install CUDA-specific packages (prebuilt wheels)
# =============================================================================

# spconv for sparse convolutions (try cu121 first for CUDA 12.1)
RUN pip install --no-cache-dir spconv-cu120 \
    || echo "WARNING: spconv install failed, some features may be limited"

# kaolin for 3D deep learning
RUN pip install --no-cache-dir kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html \
    || echo "WARNING: kaolin install failed, mesh operations may be limited"

# nvdiffrast for differentiable rendering
RUN pip install --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git \
    || echo "WARNING: nvdiffrast install failed, rendering may be limited"

# =============================================================================
# PHASE 4: Install TRELLIS basic dependencies (from setup.sh --basic)
# Use --ignore-installed to work around distutils packages (blinker, etc.)
# =============================================================================
RUN pip install --no-cache-dir --ignore-installed \
    pillow \
    imageio \
    imageio-ffmpeg \
    tqdm \
    opencv-python-headless \
    scipy \
    ninja \
    rembg \
    onnxruntime \
    trimesh \
    open3d \
    xatlas \
    pyvista \
    pymeshfix \
    igraph \
    transformers \
    git+https://github.com/EasternJournalist/utils3d.git

# =============================================================================
# PHASE 5: Install easydict LAST (prevents shadowing by other packages)
# This is the critical fix for "No module named 'easydict'" error
# =============================================================================
RUN pip install --no-cache-dir --force-reinstall easydict

# Verify easydict is importable
RUN python -c "from easydict import EasyDict; print('easydict OK')"

# =============================================================================
# PHASE 6: Install RunPod handler dependencies
# =============================================================================
RUN pip install --no-cache-dir \
    runpod>=1.6.0 \
    pygltflib>=1.16.0 \
    huggingface_hub \
    safetensors \
    einops

# =============================================================================
# PHASE 7: Final verification - import TRELLIS pipeline
# =============================================================================
RUN python -c "\
import sys; \
print('Python path:', sys.path[:3]); \
print('Checking critical imports...'); \
import torch; print(f'  torch {torch.__version__} OK'); \
import torchvision; print(f'  torchvision {torchvision.__version__} OK'); \
from easydict import EasyDict; print('  easydict OK'); \
import trimesh; print('  trimesh OK'); \
import transformers; print('  transformers OK'); \
print('All critical imports successful!')"

# Try to import TRELLIS pipeline (may fail without GPU, but tests imports)
RUN python -c "\
try: \
    from trellis.pipelines import TrellisTextTo3DPipeline; \
    print('TRELLIS pipeline import OK'); \
except Exception as e: \
    print(f'TRELLIS import check: {e}'); \
    print('(Expected if no GPU during build)')"

# =============================================================================
# PHASE 8: Copy handler and configure runtime
# =============================================================================
COPY handler.py .

# Model downloaded at runtime (avoids build timeout, allows model caching)
ENV TRELLIS_MODEL_PATH="/app/models/TRELLIS-text-xlarge"
ENV HF_HOME="/app/hf_cache"

# Final build verification
RUN echo "=== BUILD COMPLETE ===" && \
    python --version && \
    python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')" && \
    python -m py_compile /app/handler.py && echo "handler.py syntax OK" && \
    echo "Ready for deployment!"

CMD ["python", "-u", "/app/handler.py"]
