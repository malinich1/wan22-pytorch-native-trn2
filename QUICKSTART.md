# Quick Start — WAN 2.2 Native PyTorch Beta 3

**DLC:** `421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest`  
**Instance:** `trn2.48xlarge` (64 NeuronCores, 1.5 TB NVMe)  
**Last updated:** 2026-07-08

---

## Tested Performance (WAN 2.1 T2V-1.3B on trn2.48xlarge)

| Metric | Eager (on Neuron) | torch.compile | Speedup |
|---|---|---|---|
| **Step time** | 26.14 s/step | **0.82 s/step** | **32x** |
| Denoising (20 steps) | 522.8s | 87.9s (incl. compile) | 6x |
| Text encoding (UMT5) | 125.7s | 1.3s (cached) | 97x |
| VAE decode | 201.7s | 112.7s | 1.8x |
| **Total inference** | **850.3s (14.2 min)** | **201.9s (3.4 min)** | **4.2x** |
| Output | 384×640, 17 frames | 384×640, 17 frames | — |

> Note: First compile run includes ~71s NEFF compilation (cached for subsequent runs).
> With warm NEFF cache, total drops to ~130s (2.2 min).

---

## 30-Second Path (WAN 1.3B — recommended starting point)

```bash
# 1. Pull & run the Beta 3 DLC
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    421672808698.dkr.ecr.us-east-1.amazonaws.com

docker run -it --privileged -v /mnt/nvme:/mnt/nvme \
    421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest \
    /bin/bash

# 2. Install diffusion deps (torch/neuronx already in DLC)
pip install diffusers>=0.38.0 transformers>=4.44.0 accelerate \
            safetensors imageio imageio-ffmpeg pillow huggingface_hub

# 3. Download WAN 1.3B model (~5 GB)
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Wan-AI/Wan2.1-T2V-1.3B-Diffusers', \
  local_dir='/mnt/nvme/models/Wan2.1-T2V-1.3B-Diffusers')"

# 4. Run with torch.compile (production — 0.82s/step after NEFF compile):
torchrun --nproc-per-node 1 /mnt/nvme/run_wan_small.py \
    --model-id /mnt/nvme/models/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A graceful cat walks through a sunlit garden" \
    --height 384 --width 640 --num-frames 17 --num-steps 20

# 5. Or eager mode (instant start, no compilation — 26s/step):
torchrun --nproc-per-node 1 /mnt/nvme/run_wan_small.py \
    --model-id /mnt/nvme/models/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A graceful cat walks through a sunlit garden" \
    --eager --height 384 --width 640 --num-frames 17 --num-steps 20
```

---

## Execution Modes

| Mode | Flag | Startup | Step time | Total (20 steps, 17 frames) | Use case |
|---|---|---|---|---|---|
| Eager on Neuron | `--eager` | Instant | 26.14s | 14.2 min | Debugging, validation |
| Compile (cold) | _(default)_ | ~71s NEFF | **0.82s** | 3.4 min | First run on new instance |
| Compile (warm) | _(default)_ | ~0s | **0.82s** | ~2.2 min | Production (cached NEFFs) |

---

## Key Files

| File | Purpose |
|---|---|
| `run_inference.py` | **Main script** — all Beta 3 features |
| `run_inference_simple.py` | One-liner wrapper, defaults to eager |
| `compile_model.py` | Pre-populate NEFF cache before first inference |
| `setup_env.sh` | Full instance bootstrap (NVMe, DLC pull, env vars) |
| `download_model.py` | Download WAN 2.2 weights from HuggingFace |

---

## All CLI Options

```
run_inference.py
  --prompt               Text prompt (required)
  --negative-prompt      Negative prompt for CFG
  --model-dir            Model weights dir  [/mnt/nvme/models/Wan2.2-T2V-A14B-Diffusers]
  --output               Output file path   [auto-generated]
  --output-dir           Output directory   [/mnt/nvme/outputs]
  --neff-cache           NEFF cache path    [/mnt/nvme/neff_cache]
  --height               Frame height       [768]
  --width                Frame width        [1280]
  --num-frames           Frame count        [81]  (1 = image)
  --num-inference-steps  Denoising steps    [40]
  --guidance-scale       CFG scale          [5.0]
  --seed                 Random seed        [42]
  --fps                  Output video fps   [16]
  --eager                Eager mode — no compilation, instant start
  --memory-snapshot      Save Beta 3 memory snapshot (OOM debugging)
```

---

## Beta 3 Environment Variables

Set automatically by `run_inference.py`. Replicate manually if needed:

```bash
export NEURON_RT_VIRTUAL_CORE_SIZE=2          # LNC2 mode (trn2)
export NEURON_RT_NUM_CORES=64
export NEURON_CC_FLAGS="-O1 --auto-cast=none --enable-native-kernel=1 --remat --enable-ccop-compute-overlap"
export TORCH_NEURONX_ENABLE_ASYNC_NRT=1       # Async NRT (default in Beta 3)
export TORCH_NEURONX_ENABLE_HOST_CC=1         # Compute/comms overlap
export NEURON_COMPILE_CACHE_URL="file:///mnt/nvme/neff_cache"  # Persistent NEFF cache
```

---

## Troubleshooting

**Model not found**
```
Error: Transformer directory not found at /mnt/nvme/models/...
```
Run `python download_model.py` first. Needs ~118 GB free on `/mnt/nvme`.

**ECR 403 on docker pull**  
Your AWS account needs to be allowlisted for the Beta 3 ECR repo
(`421672808698`). Contact the Neuron team to add account access.

**Slow first run (>16 min)**  
Expected — cold NEFF compilation for MoE models. Subsequent runs use
the persistent cache at `/mnt/nvme/neff_cache` and take ~3 min.

**int32 downcast warnings**  
Expected and harmless. Beta 3 auto-downcasts `int64` → `int32` on Neuron.

**OOM on NeuronCore**  
Add `--memory-snapshot` to capture a Beta 3 memory snapshot, then reduce
resolution: `--height 384 --width 640 --num-frames 17`.

---

## Verify NeuronCores

```bash
neuron-ls
# Expected on trn2.48xlarge: 16 devices × 4 cores = 64 NeuronCores, 96 GB HBM each
```

---

## Resources

- [Beta 3 User Guide](https://quip-amazon.com/H7LEApgqbQ1K) (internal)
- [Neuron Explorer](https://quip-amazon.com/vbAcA5da8hmD) (internal)
- [TorchNeuron Docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/pytorch-native-overview.html)
- [Beta feedback survey](https://pulse.aws/survey/KHXDKDI4?p=0)
