"""
TRELLIS Text-to-3D RunPod Serverless Handler

Input:
{
    "input": {
        "prompt": "A wooden treasure chest",
        "seed": 42,                    # Optional, default: random
        "simplify": 0.95,              # Optional, mesh simplification ratio
        "texture_size": 1024           # Optional, texture resolution
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

import runpod
import base64
import time
import torch
import random
import os
import sys
from io import BytesIO

# Add TRELLIS to path
sys.path.insert(0, '/app/trellis')

print("=" * 60)
print("TRELLIS Text-to-3D RunPod Worker Starting...")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("=" * 60)

# Global pipeline - loaded once at startup
PIPELINE = None


def load_pipeline():
    """Load TRELLIS pipeline (called once at worker startup)."""
    global PIPELINE

    if PIPELINE is not None:
        return PIPELINE

    print("Loading TRELLIS pipeline...")
    start = time.time()

    try:
        from trellis.pipelines import TrellisTextTo3DPipeline

        model_path = os.environ.get('TRELLIS_MODEL_PATH', 'microsoft/TRELLIS-text-xlarge')

        if os.path.exists(model_path):
            print(f"Loading from local path: {model_path}")
            PIPELINE = TrellisTextTo3DPipeline.from_pretrained(model_path)
        else:
            print(f"Loading from HuggingFace: {model_path}")
            PIPELINE = TrellisTextTo3DPipeline.from_pretrained('microsoft/TRELLIS-text-xlarge')

        PIPELINE.cuda()

        elapsed = time.time() - start
        print(f"TRELLIS pipeline loaded in {elapsed:.1f}s")

        return PIPELINE

    except Exception as e:
        print(f"ERROR loading TRELLIS: {e}")
        raise


def handler(event):
    """Process a text-to-3D generation request."""
    start_time = time.time()

    try:
        # Ensure pipeline is loaded
        pipeline = load_pipeline()

        # Extract input parameters
        job_input = event.get("input", {})
        prompt = job_input.get("prompt")

        if not prompt:
            return {"error": "Missing required 'prompt' parameter"}

        seed = job_input.get("seed", random.randint(0, 2**32 - 1))
        simplify = job_input.get("simplify", 0.95)
        texture_size = job_input.get("texture_size", 1024)

        print(f"Generating 3D mesh for: '{prompt}'")
        print(f"Settings: seed={seed}, simplify={simplify}, texture_size={texture_size}")

        # Set random seed for reproducibility
        torch.manual_seed(seed)
        random.seed(seed)

        # Generate 3D assets
        with torch.no_grad():
            outputs = pipeline.run(
                prompt,
                seed=seed,
            )

        print("Generation complete, exporting to GLB...")

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

        print(f"Export complete:")
        print(f"  - Vertices: {vertex_count}")
        print(f"  - Faces: {face_count}")
        print(f"  - GLB size: {glb_size_mb:.2f} MB")
        print(f"  - Total time: {generation_time:.1f}s")

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
        print(f"GPU OOM Error: {e}")
        return {"error": f"GPU out of memory. Try a simpler prompt or lower texture_size. Details: {str(e)}"}

    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# Pre-load the model at worker startup (not per-request)
print("Pre-loading TRELLIS model...")
try:
    load_pipeline()
    print("Model pre-loaded successfully!")
except Exception as e:
    print(f"Warning: Could not pre-load model: {e}")
    print("Model will be loaded on first request")

# Start the RunPod serverless handler
runpod.serverless.start({"handler": handler})
