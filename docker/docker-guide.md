# Atlas Docker Guide

Atlas provides per-model Dockerfiles organized by `(Hardware, Model, Quantization)` tuple.

## Directory Structure

```
docker/
  gb10/                              # NVIDIA GB10 (DGX Spark)
    qwen3-next-80b-a3b/
      nvfp4/
        Dockerfile                   # 80B model, NVFP4 quantization
    qwen3.5-35b-a3b/
      nvfp4/
        Dockerfile                   # 35B model, NVFP4 quantization
```

## Which of these images have actually been built

Being explicit, because "we ship a Dockerfile per model", "it builds", and "it
serves the model" are three different claims and only the third is what you
actually want:

| Image | Builds | Served and measured |
|---|---|---|
| `docker/gb10/Dockerfile` (multi-target) | yes | yes |
| `docker/gb10/qwen3.8-27b/nvfp4/` | yes | **yes** — 64.05 tok/s median, byte-identical output, see [`bench/qwen38-gb10/`](../bench/qwen38-gb10/) |
| the other eight per-model Dockerfiles | **yes**, all eight, verified 2026-08-24 | **no** — never served |

All nine per-model images were built on 2026-08-24 and each emitted the correct
kernel target for its model:

| Dockerfile | kernels compiled |
|---|---|
| `nemotron-3-nano-30b-a3b` | 145 `(gb10, nemotron-3-nano-30b-a3b, nvfp4)` |
| `nemotron-super-120b-a12b` | 145 `(gb10, nemotron-super-120b-a12b, nvfp4)` |
| `qwen3-next-80b-a3b` | 145 `(gb10, qwen3-next-80b-a3b, nvfp4)` |
| `qwen3-vl-30b-a3b` | 146 `(gb10, qwen3-vl-30b-a3b, nvfp4)` |
| `qwen3.5-122b-a10b` | 146 `(gb10, qwen3.5-122b-a10b, nvfp4)` |
| `qwen3.5-27b` | 145 `(gb10, qwen3.5-27b, nvfp4)` |
| `qwen3.5-35b-a3b` | 145 `(gb10, qwen3.5-35b-a3b, nvfp4)` |
| `qwen3.6-27b` | 150 `(gb10, qwen3.6-27b, nvfp4)` |
| `qwen3.8-27b` | 151 `(gb10, qwen3.8-27b, nvfp4)` |

That is a real check and not a formality: until 2026-08-23 all nine carried two
independent build-breaking bugs — a `COPY vendor/` of a directory deleted when
the grammar stack went pure-Rust, and `apt-get install "libnccl2 (>= 2.28)"`
using dpkg dependency-field syntax that apt rejects outright. Both are fixed, and
the builds above are the evidence that they are.

**What has still not been demonstrated for the eight is that they serve.** None
has had weights mounted or answered a request; most of those checkpoints are not
on the machine that built them. A correct kernel-target line proves the build
selected the right kernels, not that the model loads or decodes. If you serve one
successfully, a PR saying so — with the model revision you used — is welcome.

## Prerequisites

- NVIDIA GPU with CUDA 13.0+ drivers
- Docker with NVIDIA Container Toolkit (`nvidia-docker`)
- Model weights downloaded via `huggingface-cli`

### Download Model Weights

Use `--local-dir` to download weights as real files (no symlinks). This is the recommended approach for Docker — it avoids broken symlinks when mounting volumes.

```bash
# 80B model (~47 GB)
huggingface-cli download nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 \
  --local-dir /models/qwen3-next-80b

# 35B model (~20 GB) — both base + extra MTP weights
huggingface-cli download Kbenkhaled/Qwen3.5-35B-A3B-NVFP4 \
  --local-dir /models/qwen3.5-35b
```

## Build

All builds run from the **repository root**:

```bash
# 80B model
docker build -f docker/gb10/qwen3-next-80b-a3b/nvfp4/Dockerfile -t atlas-80b .

# 35B model
docker build -f docker/gb10/qwen3.5-35b-a3b/nvfp4/Dockerfile -t atlas-35b .
```

Build takes ~2-3 minutes (Rust compilation + CUDA kernel PTX compilation).

## Run

### Recommended: `--model-from-path` with local directory

This is the most portable approach — mount the model directory and pass the path directly.

```bash
# 80B with speculative decoding (~106 tok/s counting, ~99 tok/s diverse)
docker run --gpus all --ipc=host -p 8888:8888 \
  -v /models/qwen3-next-80b:/model \
  atlas-80b serve --model-from-path /model --speculative --num-drafts 1

# 35B with speculative decoding (~131 tok/s counting, ~127 tok/s diverse)
docker run --gpus all --ipc=host -p 8888:8888 \
  -v /models/qwen3.5-35b:/model \
  atlas-35b serve --model-from-path /model --speculative --num-drafts 1
```

### Non-speculative mode

```bash
# 80B (~82 tok/s)
docker run --gpus all --ipc=host -p 8888:8888 \
  -v /models/qwen3-next-80b:/model \
  atlas-80b serve --model-from-path /model

# 35B (~102 tok/s)
docker run --gpus all --ipc=host -p 8888:8888 \
  -v /models/qwen3.5-35b:/model \
  atlas-35b serve --model-from-path /model
```

### Alternative: HuggingFace cache mount

If you use the default HuggingFace cache (`~/.cache/huggingface/`), mount it and pass the model ID:

```bash
docker run --gpus all --ipc=host -p 8888:8888 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  atlas-80b serve nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 --speculative --num-drafts 1
```

> **Note:** The 35B model's `extra_weights.safetensors` is a symlink that may break with HF cache mounts. Use `--local-dir` download or `--model-from-path` instead.

## API

Atlas serves an OpenAI-compatible API on the configured port.

```bash
# Check server status
curl http://localhost:8888/v1/models

# Chat completion
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "atlas",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256
  }'

# Streaming
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "atlas",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256,
    "stream": true
  }'
```

## Serve Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-from-path` | — | Direct filesystem path to model weights |
| `--port` | `8888` | HTTP listening port |
| `--max-seq-len` | `4096` | Maximum sequence length (tokens) |
| `--gpu-memory-utilization` | `0.90` | GPU memory fraction (0.0-1.0) |
| `--speculative` | `false` | Enable MTP speculative decoding |
| `--num-drafts` | `1` | Draft tokens per speculative step |
| `--max-batch-size` | `8` | Max concurrent sequences per decode step |
| `--kv-cache-dtype` | `fp8` | KV cache precision (`fp8` or `bf16`) |

## Performance (NVIDIA GB10 / DGX Spark)

| Model | Mode | Counting | Diverse |
|-------|------|:--------:|:-------:|
| **35B** | Speculative (K=2) | **131 tok/s** | **127 tok/s** |
| **35B** | Non-speculative | 102 tok/s | 102 tok/s |
| **80B** | Speculative (K=2) | **106 tok/s** | **99 tok/s** |
| **80B** | Non-speculative | 82 tok/s | 82 tok/s |

## Troubleshooting

### "No MTP weights found" with speculative decoding
The 35B model stores MTP weights in `extra_weights.safetensors`. If using HF cache mounts, the file may be a broken symlink. Fix: download with `--local-dir` or use `--model-from-path`.

### Model not found
Ensure the model path is correctly mounted inside the container. With `--model-from-path`, the path must be valid **inside** the container (not the host path).

### Out of memory
- Lower `--gpu-memory-utilization` (e.g., `0.85`)
- Reduce `--max-seq-len` (e.g., `2048`)

### Slow startup
Normal — model loading takes 30-90 seconds depending on model size and storage speed.
