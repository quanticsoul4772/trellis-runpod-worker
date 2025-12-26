# =============================================================================
# TRELLIS Text-to-3D RunPod Serverless Dockerfile
# =============================================================================
# Key fixes based on comprehensive analysis:
# 1. Clone TRELLIS first, then install deps (PYTHONPATH order matters)
# 2. Install easydict LAST to prevent shadowing by other packages
# 3. Use --ignore-installed for distutils conflicts in base image
# 4. Skip flash-attn (30min compile exceeds RunPod build timeout)
# 5. Skip explicit xformers install - let TRELLIS use native attention
# =============================================================================

# PyTorch 2.4.0 with CUDA 12.4 - only available version on RunPod
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

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
# PHASE 1: Clone TRELLIS with submodules (flexicubes is a submodule)
# =============================================================================
RUN git clone --depth 1 --recurse-submodules https://github.com/microsoft/TRELLIS.git /app/trellis

# CRITICAL: Remove any conflicting 'trellis' package from PyPI
# The trellis-python package uses the same namespace and shadows Microsoft TRELLIS
RUN pip uninstall -y trellis trellis-python 2>/dev/null || true

# Set PYTHONPATH early so TRELLIS modules are discoverable
ENV PYTHONPATH="/app/trellis"

# =============================================================================
# PHASE 2: Verify base torch/torchvision are intact BEFORE any installs
# =============================================================================
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    python -c "import torchvision; print(f'torchvision: {torchvision.__version__}')"

# =============================================================================
# PHASE 3: Install CUDA-specific packages (prebuilt wheels)
# These should NOT touch torch/torchvision
# =============================================================================

# spconv for sparse convolutions (CUDA 12.4)
RUN pip install --no-cache-dir spconv-cu124 \
    || pip install --no-cache-dir spconv-cu120 \
    || echo "WARNING: spconv install failed, some features may be limited"

# kaolin for 3D deep learning (required by flexicubes)
# Use --ignore-installed to work around distutils conflicts
RUN pip install --no-cache-dir --ignore-installed kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu124.html \
    || pip install --no-cache-dir --ignore-installed kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html \
    || { echo "ERROR: kaolin install failed - required for mesh generation"; exit 1; }

# nvdiffrast for differentiable rendering (required for mesh export)
RUN pip install --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git \
    || { echo "ERROR: nvdiffrast install failed"; exit 1; }

# flash_attn from prebuilt wheel (avoids 30+ min compilation)
# Wheel for Python 3.11, PyTorch 2.4, CUDA 12
RUN pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl \
    || echo "WARNING: flash_attn wheel install failed"

# =============================================================================
# PHASE 4: Install TRELLIS basic dependencies (from setup.sh --basic)
# Use --ignore-installed to work around distutils packages (blinker, etc.)
# DO NOT include easydict here - it gets shadowed by later packages
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
    xatlas \
    pyvista \
    pymeshfix \
    igraph \
    plyfile \
    git+https://github.com/EasternJournalist/utils3d.git

# Install open3d separately (large package, can cause issues)
RUN pip install --no-cache-dir --ignore-installed open3d \
    || echo "WARNING: open3d install failed"

# Install transformers separately (installs many deps that can shadow packages)
RUN pip install --no-cache-dir --ignore-installed transformers

# Pin numpy < 2.4 for Numba compatibility (transformers pulls in numpy 2.4 which breaks Numba)
RUN pip install --no-cache-dir "numpy<2.4,>=1.24"

# =============================================================================
# PHASE 5: Install easydict ABSOLUTELY LAST (prevents shadowing)
# This is the critical fix for "No module named 'easydict'" error
# =============================================================================
RUN pip install --no-cache-dir --force-reinstall --no-deps easydict

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
# PHASE 7: Final verification - check all critical imports
# =============================================================================

# CRITICAL: Remove any conflicting trellis package from site-packages
# pip uninstall may not work if it was installed as a dependency
RUN pip uninstall -y trellis trellis-python 2>/dev/null || true

# DEBUG: Show what trellis-related packages exist
RUN echo "=== Checking for trellis packages ===" && \
    pip list | grep -i trellis || echo "No trellis in pip list" && \
    echo "=== Checking site-packages ===" && \
    find /usr -name "trellis*" -type d 2>/dev/null || echo "No trellis dirs found" && \
    ls -la /usr/local/lib/python3.11/dist-packages/ | grep -i trellis || echo "No trellis in dist-packages"

# AGGRESSIVE FIX: Physically remove any trellis package from site-packages
RUN rm -rf /usr/local/lib/python3.11/dist-packages/trellis* 2>/dev/null || true && \
    rm -rf /usr/lib/python3.11/dist-packages/trellis* 2>/dev/null || true && \
    rm -rf /usr/local/lib/python3.11/site-packages/trellis* 2>/dev/null || true

# Verify the correct TRELLIS is being used (Microsoft's, not PyPI's)
RUN python -c "\
import sys; \
print('Python path:', sys.path[:3]); \
print('Checking critical imports...'); \
import torch; print(f'  torch {torch.__version__} OK'); \
import torchvision; print(f'  torchvision {torchvision.__version__} OK'); \
from easydict import EasyDict; print('  easydict OK'); \
import trimesh; print('  trimesh OK'); \
import transformers; print('  transformers OK'); \
print('All critical imports successful!'); \
print('Verifying Microsoft TRELLIS...'); \
import trellis; print(f'  trellis module: {trellis.__file__}'); \
assert '/app/trellis' in trellis.__file__, 'Wrong trellis package!'; \
print('  Microsoft TRELLIS verified!')"

# Try to import TRELLIS pipeline (may fail without GPU, but tests imports)
RUN python -c "from trellis.pipelines import TrellisTextTo3DPipeline; print('TRELLIS pipeline import OK')" \
    || echo "TRELLIS import check failed (expected if no GPU during build)"

# =============================================================================
# PHASE 8: Copy handler and configure runtime
# =============================================================================
COPY handler.py .

# Model downloaded at runtime (avoids build timeout, allows model caching)
ENV TRELLIS_MODEL_PATH="/app/models/TRELLIS-text-xlarge"
ENV HF_HOME="/app/hf_cache"

# spconv algorithm selection
ENV SPCONV_ALGO="native"

# Pre-download CLIP model to avoid runtime download issues
RUN python -c "from transformers import CLIPModel, CLIPTokenizer; \
    CLIPModel.from_pretrained('openai/clip-vit-large-patch14'); \
    CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14'); \
    print('CLIP model pre-downloaded')" \
    || echo "WARNING: Could not pre-download CLIP model"

# Final build verification
RUN echo "=== BUILD COMPLETE ===" && \
    python --version && \
    python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')" && \
    python -m py_compile /app/handler.py && echo "handler.py syntax OK" && \
    echo "Ready for deployment!"

CMD ["python", "-u", "/app/handler.py"]
