# Quick Start — WAN 2.2 Native PyTorch Beta 3

**DLC:** `421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest`  
**Instance:** `trn2.48xlarge` (64 NeuronCores, 1.5 TB NVMe)  
**Last updated:** 2026-07-07

---

## 30-Second Path (inside the DLC container)

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
            safetensors imageio imageio-ffmpeg pillow

# 3. Download model weights (~118 GB)
python download_model.py

# 4. Smoke test — eager mode, single frame, 3 steps (~1 min)
python run_inference.py \
    --prompt "A cat walks on grass" \
    --eager --height 256 --width 256 --num-frames 1 --num-inference-steps 3

# 5. Production — torch.compile + persistent NEFF cache
#    First run: ~16 min (cold NEFF compilation)
#    Subsequent runs: ~3 min (warm cache load)
python run_inference.py \
    --prompt "A cat walks gracefully through a garden" \
    --height 768 --width 1280 --num-inference-steps 40
```

---

## Execution Modes

| Mode | Flag | Startup | Throughput | Use case |
|---|---|---|---|---|
| Eager | `--eager` | Instant | ~2.5s/step | Debugging, validation |
| Compile (cold) | _(default)_ | ~16 min | ~0.8s/step | First run on new instance |
| Compile (warm) | _(default)_ | ~3 min | ~0.8s/step | Production (cached NEFFs) |

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
