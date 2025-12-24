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

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | **required** | Text description of the 3D object |
| `seed` | int | random | Random seed for reproducibility |
| `simplify` | float | 0.95 | Mesh simplification ratio (0.0-1.0) |
| `texture_size` | int | 1024 | Texture resolution in pixels |

### Output

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

## GPU Requirements

| GPU | VRAM | Status |
|-----|------|--------|
| A4000 | 16GB | Minimum |
| A5000 | 24GB | Recommended |
| A6000 | 48GB | Optimal |

## Cost Estimation

- A4000: ~$0.20/hr → ~$0.003/mesh
- A5000: ~$0.30/hr → ~$0.004/mesh
- A6000: ~$0.50/hr → ~$0.005/mesh

## Local Testing

```bash
# Build
docker build -t trellis-worker .

# Run (requires NVIDIA GPU)
docker run --gpus all -p 8000:8000 trellis-worker

# Test
python -c "import requests; print(requests.post('http://localhost:8000/run', json={'input': {'prompt': 'test'}}).json())"
```

## Credits

- [TRELLIS](https://github.com/microsoft/TRELLIS) by Microsoft Research
- [RunPod](https://runpod.io) for serverless GPU infrastructure
