# TRELLIS RunPod Worker

RunPod serverless worker for text-to-3D mesh generation using Microsoft's TRELLIS model.

## Overview

This worker accepts text prompts and generates 3D meshes in GLB format using the TRELLIS text-to-3D pipeline.

## API

### Input

```json
{
    "input": {
        "prompt": "A wooden treasure chest",
        "seed": 42,
        "simplify": 0.95,
        "texture_size": 1024
    }
}
```

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `prompt` | string | **required** | max 1000 chars | Text description of the 3D object |
| `seed` | int | random | 0 to 2^32-1 | Random seed for reproducibility |
| `simplify` | float | 0.95 | 0.0 - 1.0 | Mesh simplification ratio |
| `texture_size` | int | 1024 | 128 - 4096 | Texture resolution in pixels |

### Output

**Success:**
```json
{
    "glb_base64": "<base64-encoded GLB file>",
    "vertex_count": 1234,
    "face_count": 5678,
    "glb_size_bytes": 1234567,
    "generation_time_seconds": 30.5,
    "seed_used": 42
}
```

**Error:**
```json
{
    "error": "Missing required 'prompt' parameter"
}
```

### Error Messages

| Error | Cause |
|-------|-------|
| `Missing required 'prompt' parameter` | No prompt provided |
| `'prompt' must be a string` | Invalid prompt type |
| `'prompt' exceeds maximum length of 1000 characters` | Prompt too long |
| `'seed' must be an integer` | Invalid seed type |
| `'simplify' must be between 0.0 and 1.0` | Out of range |
| `'texture_size' must be between 128 and 4096` | Out of range |
| `GPU out of memory...` | Reduce texture_size or simplify prompt |

## Deployment

### 1. Fork/Clone this repo

```bash
git clone https://github.com/quanticsoul4772/trellis-runpod-worker
```

### 2. Deploy on RunPod

1. Go to [RunPod Console](https://console.runpod.io) → Serverless
2. Click "New Endpoint"
3. Import this GitHub repository
4. Select GPU: **A5000 (24GB)** recommended, A4000 (16GB) minimum
5. Configure:
   - Max Workers: 1-3
   - Idle Timeout: 30s
   - Flash Boot: Enabled

### 3. Test the endpoint

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "A wooden treasure chest"}}'
```

### 4. Check job status

```bash
curl "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status/JOB_ID" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## GPU Requirements

| GPU | VRAM | Status | Cost/hr |
|-----|------|--------|---------|
| A4000 | 16GB | Minimum | ~$0.20 |
| A5000 | 24GB | Recommended | ~$0.30 |
| A6000 | 48GB | Optimal | ~$0.50 |

**Note:** First run downloads ~8GB of model weights. Subsequent cold starts are faster with Flash Boot enabled.

## Cost Estimation

Based on ~30s average generation time:
- A4000: ~$0.003/mesh
- A5000: ~$0.004/mesh
- A6000: ~$0.005/mesh

## Local Testing

```bash
# Build
docker build -t trellis-worker .

# Run (requires NVIDIA GPU with 16GB+ VRAM)
docker run --gpus all -p 8000:8000 trellis-worker

# Test
curl -X POST "http://localhost:8000/run" \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "A small wooden box"}}'
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRELLIS_MODEL_PATH` | `/app/models/TRELLIS-text-xlarge` | Local model path |
| `HF_HOME` | `/app/hf_cache` | HuggingFace cache directory |
| `MAX_JOBS` | `4` | Ninja build parallelism |

## Architecture

```
Request → handler() → validate_input() → load_pipeline() → pipeline.run()
                                                              ↓
Response ← base64(GLB) ← to_glb() ← postprocessing_utils ←───┘
```

- **Singleton Pipeline**: Model loaded once at worker startup, reused across requests
- **Input Validation**: All parameters validated before processing
- **Memory Management**: CUDA cache cleared after each generation
- **Structured Logging**: Timestamps and log levels for debugging

## Credits

- [TRELLIS](https://github.com/microsoft/TRELLIS) by Microsoft Research
- [RunPod](https://runpod.io) for serverless GPU infrastructure

## License

MIT
