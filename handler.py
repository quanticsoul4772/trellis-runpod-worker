"""
TRELLIS Text-to-3D RunPod Serverless Handler

Input:
{
    "input": {
        "prompt": "A wooden treasure chest",
        "seed": 42,                    # Optional, default: random
        "simplify": 0.95,              # Optional, mesh simplification ratio (0.0-1.0)
        "texture_size": 1024           # Optional, texture resolution (128-4096)
    }
}

Output:
{
    "glb_base64": "<base64-encoded GLB file>",
    "vertex_count": 1234,
    "face_count": 5678,
    "generation_time_seconds": 30.5
}
"""

from __future__ import annotations

# CRITICAL: Set attention backend BEFORE any other imports
# This must happen before torch or any TRELLIS code is imported
import os
import sys

# Add TRELLIS to path FIRST
sys.path.insert(0, '/app/trellis')

# Set attention backend to use PyTorch's built-in SDPA (no extra deps)
os.environ['ATTN_BACKEND'] = 'sdpa'
os.environ['SPCONV_ALGO'] = 'native'

# Debug: verify env vars are set
print(f"[EARLY] ATTN_BACKEND = {os.environ.get('ATTN_BACKEND')}")
print(f"[EARLY] SPCONV_ALGO = {os.environ.get('SPCONV_ALGO')}")

# Now import everything else
import runpod
import base64
import logging
import time
import torch
import random
from io import BytesIO
from typing import Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Validation constants
MAX_PROMPT_LENGTH = 1000
MIN_TEXTURE_SIZE = 128
MAX_TEXTURE_SIZE = 4096
MIN_SIMPLIFY = 0.0
MAX_SIMPLIFY = 1.0

# Set HuggingFace token for gated models
hf_token = os.environ.get('HF_TOKEN')
if hf_token:
    os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token
    logger.info("HuggingFace token configured")
else:
    logger.warning("HF_TOKEN not set - may fail to download gated models")

logger.info("=" * 50)
logger.info("TRELLIS Text-to-3D RunPod Worker Starting...")
logger.info(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
logger.info("=" * 50)


def fix_pipeline_paths(model_path: str) -> None:
    """Fix relative ckpts/ paths in pipeline.json to absolute paths.

    The pipeline.json has a mix of:
    - Relative paths: "ckpts/ss_flow_txt_dit_XL_16l8_fp16"
    - HuggingFace paths: "JeffreyXiang/TRELLIS-image-large/ckpts/..."

    This converts relative paths to absolute paths so they load correctly.
    """
    import json

    pipeline_json_path = os.path.join(model_path, 'pipeline.json')
    if not os.path.exists(pipeline_json_path):
        logger.warning(f"pipeline.json not found at {pipeline_json_path}")
        return

    with open(pipeline_json_path, 'r') as f:
        config = json.load(f)

    modified = False
    models = config.get('args', {}).get('models', {})

    for key, value in models.items():
        # Only fix paths that start with "ckpts/" (relative paths)
        if isinstance(value, str) and value.startswith('ckpts/'):
            absolute_path = os.path.join(model_path, value)
            models[key] = absolute_path
            logger.info(f"Fixed path {key}: {value} -> {absolute_path}")
            modified = True

    if modified:
        with open(pipeline_json_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info("Updated pipeline.json with absolute paths")


def download_model_if_needed():
    """Download TRELLIS model weights if not already present."""
    model_path = os.environ.get('TRELLIS_MODEL_PATH', '/app/models/TRELLIS-text-xlarge')

    if os.path.exists(model_path) and os.path.isdir(model_path):
        # Check if it has model files
        files = os.listdir(model_path)
        if len(files) > 0:
            logger.info(f"Model already exists at {model_path} ({len(files)} files)")
            return model_path

    logger.info("Downloading TRELLIS model weights (this may take several minutes)...")
    try:
        from huggingface_hub import snapshot_download, login

        # Login with token if available
        hf_token = os.environ.get('HF_TOKEN')
        if hf_token:
            login(token=hf_token)
            logger.info("Logged in to HuggingFace")

        # Use JeffreyXiang's repo (original author mirror)
        model_repo = "JeffreyXiang/TRELLIS-text-xlarge"
        downloaded_path = snapshot_download(
            model_repo,
            local_dir=model_path,
            local_dir_use_symlinks=False,
            token=hf_token
        )
        logger.info(f"Model downloaded to: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        # Fall back to letting the pipeline download it
        return "JeffreyXiang/TRELLIS-text-xlarge"

# Global pipeline - loaded once at startup
PIPELINE = None


def load_pipeline() -> Any:
    """Load TRELLIS pipeline (called once at worker startup).

    Returns:
        TrellisTextTo3DPipeline: The loaded and GPU-ready pipeline.

    Raises:
        RuntimeError: If pipeline fails to load.
    """
    global PIPELINE

    if PIPELINE is not None:
        return PIPELINE

    logger.info("Loading TRELLIS pipeline...")
    start = time.time()

    try:
        # Verify TRELLIS is accessible
        import os
        trellis_path = '/app/trellis'
        if os.path.exists(trellis_path):
            logger.info(f"TRELLIS path exists: {os.listdir(trellis_path)[:10]}")
        else:
            logger.error(f"TRELLIS path does not exist: {trellis_path}")

        logger.info(f"sys.path: {sys.path[:5]}")

        # Try importing TRELLIS
        logger.info("Importing trellis.pipelines...")
        from trellis.pipelines import TrellisTextTo3DPipeline
        logger.info("Import successful!")

        # Ensure model is downloaded first
        model_path = download_model_if_needed()

        # Fix relative paths in pipeline.json to be absolute
        fix_pipeline_paths(model_path)

        logger.info(f"Loading pipeline from: {model_path}")
        PIPELINE = TrellisTextTo3DPipeline.from_pretrained(model_path)
        PIPELINE.cuda()

        elapsed = time.time() - start
        logger.info(f"TRELLIS pipeline loaded in {elapsed:.1f}s")

        return PIPELINE

    except ImportError as e:
        logger.error(f"Import error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise RuntimeError(f"TRELLIS import failed: {e}") from e
    except Exception as e:
        logger.error(f"Failed to load TRELLIS pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Pipeline initialization failed: {e}") from e


def validate_input(job_input: dict[str, Any]) -> tuple[bool, Optional[str], dict[str, Any]]:
    """Validate and normalize input parameters.

    Args:
        job_input: Raw input dictionary from the request.

    Returns:
        Tuple of (is_valid, error_message, validated_params).
    """
    # Validate prompt
    prompt = job_input.get("prompt")
    if not prompt:
        return False, "Missing required 'prompt' parameter", {}
    if not isinstance(prompt, str):
        return False, "'prompt' must be a string", {}
    prompt = prompt.strip()
    if not prompt:
        return False, "'prompt' cannot be empty or whitespace", {}
    if len(prompt) > MAX_PROMPT_LENGTH:
        return False, f"'prompt' exceeds maximum length of {MAX_PROMPT_LENGTH} characters", {}

    # Validate and normalize seed
    seed = job_input.get("seed")
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    elif not isinstance(seed, int):
        return False, "'seed' must be an integer", {}
    elif seed < 0 or seed > 2**32 - 1:
        return False, "'seed' must be between 0 and 2^32-1", {}

    # Validate and normalize simplify
    simplify = job_input.get("simplify", 0.95)
    if not isinstance(simplify, (int, float)):
        return False, "'simplify' must be a number", {}
    if simplify < MIN_SIMPLIFY or simplify > MAX_SIMPLIFY:
        return False, f"'simplify' must be between {MIN_SIMPLIFY} and {MAX_SIMPLIFY}", {}

    # Validate and normalize texture_size
    texture_size = job_input.get("texture_size", 1024)
    if not isinstance(texture_size, int):
        return False, "'texture_size' must be an integer", {}
    if texture_size < MIN_TEXTURE_SIZE or texture_size > MAX_TEXTURE_SIZE:
        return False, f"'texture_size' must be between {MIN_TEXTURE_SIZE} and {MAX_TEXTURE_SIZE}", {}

    return True, None, {
        "prompt": prompt,
        "seed": seed,
        "simplify": float(simplify),
        "texture_size": texture_size,
    }


def handler(event: dict[str, Any]) -> dict[str, Any]:
    """Process a text-to-3D generation request.

    Args:
        event: RunPod event containing input parameters.

    Returns:
        Dictionary with GLB data and metadata, or error information.
    """
    start_time = time.time()

    try:
        # Ensure pipeline is loaded
        pipeline = load_pipeline()

        # Extract and validate input parameters
        job_input = event.get("input", {})
        is_valid, error_msg, params = validate_input(job_input)

        if not is_valid:
            logger.warning(f"Validation failed: {error_msg}")
            return {"error": error_msg}

        prompt = params["prompt"]
        seed = params["seed"]
        simplify = params["simplify"]
        texture_size = params["texture_size"]

        logger.info(f"Generating 3D mesh for: '{prompt[:50]}...' (len={len(prompt)})")
        logger.info(f"Settings: seed={seed}, simplify={simplify}, texture_size={texture_size}")

        # Set random seed for reproducibility
        torch.manual_seed(seed)
        random.seed(seed)

        # Generate 3D assets
        with torch.no_grad():
            outputs = pipeline.run(
                prompt,
                seed=seed,
            )

        logger.info("Generation complete, exporting to GLB...")

        # Export to GLB format
        from trellis.utils import postprocessing_utils

        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=simplify,
            texture_size=texture_size,
        )

        # Serialize GLB to bytes
        glb_buffer = BytesIO()
        glb.export(glb_buffer, file_type='glb')
        glb_bytes = glb_buffer.getvalue()

        # Get mesh statistics
        mesh = outputs['mesh'][0]
        vertex_count = len(mesh.vertices) if hasattr(mesh, 'vertices') else 0
        face_count = len(mesh.faces) if hasattr(mesh, 'faces') else 0

        generation_time = time.time() - start_time
        glb_size_mb = len(glb_bytes) / (1024 * 1024)

        logger.info(f"Export complete: vertices={vertex_count}, faces={face_count}, "
                    f"size={glb_size_mb:.2f}MB, time={generation_time:.1f}s")

        # Clear CUDA cache to free memory for next job
        torch.cuda.empty_cache()

        return {
            "glb_base64": base64.b64encode(glb_bytes).decode('utf-8'),
            "vertex_count": vertex_count,
            "face_count": face_count,
            "glb_size_bytes": len(glb_bytes),
            "generation_time_seconds": round(generation_time, 2),
            "seed_used": seed
        }

    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        logger.error(f"GPU OOM Error: {e}")
        return {"error": f"GPU out of memory. Try a simpler prompt or lower texture_size."}

    except Exception as e:
        logger.exception(f"Error during generation: {e}")
        return {"error": str(e)}


# Pre-load the model at worker startup (not per-request)
logger.info("Pre-loading TRELLIS model...")
try:
    load_pipeline()
    logger.info("Model pre-loaded successfully!")
except Exception as e:
    logger.warning(f"Could not pre-load model: {e}")
    logger.warning("Model will be loaded on first request")

# Start the RunPod serverless handler
runpod.serverless.start({"handler": handler})
