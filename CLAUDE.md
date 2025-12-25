# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RunPod serverless worker for text-to-3D mesh generation using Microsoft's TRELLIS model. Accepts text prompts and generates 3D meshes in GLB format.

## Build & Run Commands

```bash
# Build Docker image
docker build -t trellis-worker .

# Run locally (requires NVIDIA GPU)
docker run --gpus all -p 8000:8000 trellis-worker

# Test local endpoint
python -c "import requests; print(requests.post('http://localhost:8000/run', json={'input': {'prompt': 'test'}}).json())"

# Validate Python syntax
python -m py_compile handler.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RunPod Serverless                        │
├─────────────────────────────────────────────────────────────┤
│  POST /run                                                  │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────┐    ┌──────────────────┐                   │
│  │  handler()  │───▶│ validate_input() │                   │
│  └─────────────┘    └──────────────────┘                   │
│       │                     │                               │
│       ▼                     ▼                               │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ load_pipeline() │  │ Error Response  │                  │
│  │   (singleton)   │  └─────────────────┘                  │
│  └─────────────────┘                                        │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────┐                   │
│  │     TRELLIS Pipeline (GPU)          │                   │
│  │  TrellisTextTo3DPipeline.run()      │                   │
│  └─────────────────────────────────────┘                   │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────┐                   │
│  │  postprocessing_utils.to_glb()      │                   │
│  └─────────────────────────────────────┘                   │
│       │                                                     │
│       ▼                                                     │
│  { glb_base64, vertex_count, ... }                         │
└─────────────────────────────────────────────────────────────┘
```

**Entry Point**: `handler.py` - RunPod serverless handler with:
- `PIPELINE` - Global singleton loaded once at worker startup
- `load_pipeline()` - Lazy loads TRELLIS model from local path or HuggingFace
- `validate_input()` - Input validation with bounds checking
- `handler(event)` - Processes requests: validates → generates 3D → exports GLB → returns base64

**Docker Build Flow**:
1. Base: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
2. Installs: flash-attn, xformers, spconv-cu118, kaolin, nvdiffrast
3. Clones TRELLIS repo to `/app/trellis`
4. Model weights (~8GB) downloaded at first runtime to `/app/models/TRELLIS-text-xlarge`

## Key Files

| File | Purpose |
|------|---------|
| `handler.py` | Main RunPod handler (262 lines) |
| `Dockerfile` | Container build configuration |
| `requirements.txt` | Python dependencies |
| `test_input.json` | Example test payload |

## Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRELLIS_MODEL_PATH` | `/app/models/TRELLIS-text-xlarge` | Local model path |
| `HF_HOME` | `/app/hf_cache` | HuggingFace cache |
| `MAX_JOBS` | `4` | Ninja compilation parallelism |

## API Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `prompt` | string | **required** | max 1000 chars | Text description of 3D object |
| `seed` | int | random | 0 to 2^32-1 | Random seed for reproducibility |
| `simplify` | float | 0.95 | 0.0 - 1.0 | Mesh simplification ratio |
| `texture_size` | int | 1024 | 128 - 4096 | Texture resolution in pixels |

## Error Handling

The handler returns structured error responses:

```json
{"error": "Missing required 'prompt' parameter"}
{"error": "'simplify' must be between 0.0 and 1.0"}
{"error": "GPU out of memory. Try a simpler prompt or lower texture_size."}
```

## Validation Constants

Defined in `handler.py:44-49`:
```python
MAX_PROMPT_LENGTH = 1000
MIN_TEXTURE_SIZE = 128
MAX_TEXTURE_SIZE = 4096
MIN_SIMPLIFY = 0.0
MAX_SIMPLIFY = 1.0
```

## GPU Requirements

| GPU | VRAM | Status |
|-----|------|--------|
| A4000 | 16GB | Minimum |
| A5000 | 24GB | Recommended |
| A6000 | 48GB | Optimal |

First run downloads ~8GB model weights.

## Logging

Uses structured logging format:
```
2024-12-24 10:30:00 | INFO | Generating 3D mesh for: 'A wooden chest...' (len=25)
2024-12-24 10:30:00 | INFO | Settings: seed=42, simplify=0.95, texture_size=1024
2024-12-24 10:30:30 | INFO | Export complete: vertices=1234, faces=5678, size=2.50MB, time=30.1s
```

## Code Style

- Type hints on all functions
- Google-style docstrings
- Structured logging (no print statements)
- Input validation before processing
