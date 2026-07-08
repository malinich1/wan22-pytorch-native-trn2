# WAN 2.2 T2V-A14B — Native PyTorch Beta 3 on Trainium 2

Generate 768×1280 video (81 frames, ~5s @ 16fps) from text prompts using the
**Wan 2.2 T2V-A14B** Mixture-of-Experts diffusion model on a `trn2.48xlarge`
instance with **AWS Native PyTorch Beta 3**.

> **Beta 3 (2026-06-05):** PyTorch 2.11 + `torch.compile(backend='neuron')`,
> persistent NEFF caching, async NRT, LNC2 mode, memory snapshot API.
> DLC: `421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest`

---

## What's New in Beta 3

| Feature | Details |
|---|---|
| PyTorch 2.11 | Full eager + `torch.compile` on `device='neuron'` |
| Persistent NEFF cache | No recompilation on container restart (~3 min warm vs ~16 min cold) |
| Async NRT | Enabled by default — compute/IO overlap |
| LNC2 mode | `NEURON_RT_VIRTUAL_CORE_SIZE=2` for trn2.48xlarge |
| Memory snapshot API | `--memory-snapshot` flag for OOM debugging |
| 99% ATen op coverage | No custom op wrappers needed |
| Neuron Explorer | Profiling from CLI, UI, or VS Code |

---

## Model

| Property | Value |
|---|---|
| Name | [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) |
| Architecture | 27B parameter MoE, 14B active per denoising step |
| Experts | 2 independent (high-noise / low-noise), zero shared weights |
| Weights | ~118 GB (Hugging Face) |

---

## Approach: Native PyTorch vs NXD

| Aspect | NXD (old) | Native PyTorch Beta 3 (this repo) |
|---|---|---|
| Device | XLA via `torch_neuronx` | `device='neuron'` — PyTorch 2.11 native |
| Compilation | `torch_neuronx.trace()` | `torch.compile(backend='neuron', dynamic=False)` |
| NEFF caching | Manual artifact management | Persistent cache via `NEURON_COMPILE_CACHE_URL` |
| Eager mode | Not supported | Full eager on NeuronCores — instant startup |
| Parallelism | `neuronx_distributed` | `torch.distributed` + DTensor |
| Model loading | `NxDModel.initialize()` | Standard `model.to('neuron')` |

---

## Files

| File | Description |
|---|---|
| `run_inference.py` | **Main inference script** — full Beta 3 feature set |
| `run_inference_simple.py` | Simplified entry point, defaults to eager mode |
| `compile_model.py` | Pre-compile NEFFs and populate persistent cache |
| `setup_env.sh` | Instance setup (NVMe, DLC pull, venv, env vars) |
| `download_model.py` | Download WAN 2.2 weights from HuggingFace |
| `expert_swap.py` | Expert weight swapping via `copy_()` |
| `benchmarks.py` | Performance measurement |
| `requirements.txt` | Python dependencies (install inside DLC/venv) |
| `wan22_pytorch_native_workshop.ipynb` | Workshop notebook |

---

## Quick Start

### Step 1 — Launch a trn2.48xlarge instance

Use Ubuntu 22.04 or 24.04 with Docker installed.

```bash
# Verify your account
aws sts get-caller-identity
```

### Step 2 — Pull the Beta 3 DLC

```bash
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    421672808698.dkr.ecr.us-east-1.amazonaws.com

docker pull 421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest
```

### Step 3 — Run the container

```bash
docker run -it --privileged \
    -v /mnt/nvme:/mnt/nvme \
    421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest \
    /bin/bash
```

### Step 4 — Install diffusion dependencies

```bash
pip install diffusers>=0.38.0 transformers>=4.44.0 accelerate \
            safetensors imageio imageio-ffmpeg pillow
```

### Step 5 — Download model weights (~118 GB)

```bash
python download_model.py
```

### Step 6 — Run inference

```bash
# Eager mode — instant start, no compilation wait:
python run_inference.py \
    --prompt "A cat walks gracefully through a garden" \
    --eager --height 384 --width 640 --num-frames 1 --num-inference-steps 10

# Compile mode — production quality, persistent NEFF cache:
python run_inference.py \
    --prompt "A cat walks gracefully through a garden" \
    --height 768 --width 1280 --num-inference-steps 40
```

---

## Environment Variables (Beta 3)

These are set automatically by `run_inference.py` and `setup_env.sh`:

```bash
# LNC2 mode — 2 physical NeuronCores per logical core
export NEURON_RT_VIRTUAL_CORE_SIZE=2
export NEURON_RT_NUM_CORES=64

# Compiler
export NEURON_CC_FLAGS="-O1 --auto-cast=none --enable-native-kernel=1 --remat --enable-ccop-compute-overlap"

# Async execution (Beta 3 default, explicit here)
export TORCH_NEURONX_ENABLE_ASYNC_NRT=1

# Persistent NEFF cache — survives container restarts
export NEURON_COMPILE_CACHE_URL="file:///mnt/nvme/neff_cache"
```

---

## Performance Targets (trn2.48xlarge)

| Metric | Cold cache (first run) | Warm cache |
|---|---|---|
| NEFF compilation | ~16 min (MoE) | ~3 min load |
| Per denoising step (eager) | ~2.5s | ~2.5s |
| Per denoising step (compiled) | ~0.8s | ~0.8s |
| 40-step full inference | ~32s compiled | ~32s |

---

## Known Limitations (Beta 3)

- Dynamic shapes not supported with `torch.compile` — use fixed `--height`/`--width`/`--num-frames`
- `torch.compile` modes `reduce-overhead` / `max-autotune` fall back to default (warning printed)
- `int64` tensors auto-downcast to `int32` by runtime (expected, no action needed)
- Pipeline parallelism and P2P `send`/`recv` not yet supported

---

## References

- [Beta 3 User Guide](https://quip-amazon.com/H7LEApgqbQ1K) (internal)
- [Beta 3 Release Notes](https://github.com/aws-neuron/torch-neuronx/releases/tag/private-beta-3)
- [Neuron Explorer — Getting Started](https://quip-amazon.com/vbAcA5da8hmD) (internal)
- [TorchNeuron Documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/pytorch-native-overview.html)
- [WAN 2.2 HuggingFace Model](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers)
