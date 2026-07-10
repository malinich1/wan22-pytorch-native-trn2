# Beta 3 Distributed Inference Guide
## WAN 2.2 T2V-A14B with Tensor Parallelism

**Created:** 2026-07-08  
**Target:** PyTorch 2.11 Beta 3 + torch-neuronx 2.11.3.x  
**Hardware:** trn2.48xlarge (64 NeuronCores)

---

## Quick Start

```bash
# TP=2 (recommended for 14B model)
./beta3_wan22_launch.sh 2 "A cat walks through a sunlit garden"

# TP=4 (more memory headroom, slightly slower)
./beta3_wan22_launch.sh 4 "Ocean waves crashing on rocky cliffs"
```

**Expected performance (TP=2, 384x640, 17 frames, 20 steps):**
- Cold start (with NEFF compilation): ~5-8 minutes
- Warm start (cached NEFFs): ~2-3 minutes
- Per-step average: ~3-5 seconds

---

## Architecture Overview

### Tensor Parallelism (TP) Strategy

WAN 2.2 is a 14B parameter DiT transformer. TP shards the model across multiple NeuronCores:

```
┌─────────────────────────────────────────────────────────────┐
│  trn2.48xlarge: 16 NeuronDevices × 4 cores = 64 NeuronCores │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  TP=2 Configuration (Recommended):                          │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │ NeuronCore 0 │  │ NeuronCore 1 │                         │
│  │   7B params  │  │   7B params  │                         │
│  │   96 GB HBM  │  │   96 GB HBM  │                         │
│  └──────┬───────┘  └───────┬──────┘                         │
│         │                  │                                 │
│         └──────all-reduce──┘ (XRT collectives)              │
│                                                               │
│  TP=4 Configuration (More Memory):                          │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐               │
│  │  Core  │ │  Core  │ │  Core  │ │  Core  │               │
│  │  0-3   │ │  1-3   │ │  2-3   │ │  3-3   │               │
│  │ 3.5B  │ │ 3.5B  │ │ 3.5B  │ │ 3.5B  │               │
│  └────┬───┘ └───┬────┘ └───┬────┘ └───┬────┘               │
│       └─────────all-reduce────────────┘                      │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Model Sharding Details

| Component | Size | Sharding Strategy |
|-----------|------|-------------------|
| **Text Encoder (T5)** | ~1.5 GB | Replicated across all ranks |
| **Transformer (DiT)** | ~14 GB | **TP-sharded** (14GB / tp_degree per rank) |
| **VAE Decoder** | ~0.5 GB | Replicated across all ranks |

**Transformer TP Sharding:**
- Attention Q/K/V/O: Column-parallel (shard output dimension)
- MLP up projection: Column-parallel
- MLP down projection: Row-parallel (shard input dimension)
- After each sharded layer: All-reduce (sum partial results)

---

## Configuration Parameters

### Recommended Configurations for WAN 2.2 14B

#### TP=2 (Recommended for Production)

```bash
NEURON_RT_VIRTUAL_CORE_SIZE=2
NEURON_RT_NUM_CORES=2
NEURON_VISIBLE_DEVICES=0,1

torchrun --nproc-per-node=2 beta3_wan22_distributed_inference.py \
    --prompt "Your prompt" \
    --tp-degree 2 \
    --height 384 --width 640 --num-frames 17
```

**Pros:**
- ✅ Fastest (minimal communication overhead)
- ✅ Each core gets 7B parameters (fits in HBM)
- ✅ 2x all-reduce is fast

**Cons:**
- ⚠️ Less memory headroom for large resolutions

**Use when:** Standard 384x640 or 768x1280 resolutions

---

#### TP=4 (More Memory Headroom)

```bash
NEURON_RT_VIRTUAL_CORE_SIZE=2
NEURON_RT_NUM_CORES=4
NEURON_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc-per-node=4 beta3_wan22_distributed_inference.py \
    --prompt "Your prompt" \
    --tp-degree 4 \
    --height 768 --width 1280 --num-frames 81
```

**Pros:**
- ✅ Each core gets 3.5B parameters
- ✅ More HBM available for activations
- ✅ Can handle 768x1280x81 (full quality)

**Cons:**
- ⚠️ More communication (4x all-reduce slower than 2x)
- ⚠️ 15-20% slower per step

**Use when:** Full resolution (768x1280, 81 frames)

---

#### TP=8 (Maximum Memory for Experimental)

```bash
NEURON_RT_VIRTUAL_CORE_SIZE=2
NEURON_RT_NUM_CORES=8
NEURON_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc-per-node=8 beta3_wan22_distributed_inference.py \
    --prompt "Your prompt" \
    --tp-degree 8 \
    --height 1024 --width 1024 --num-frames 121
```

**Pros:**
- ✅ Maximum memory per core
- ✅ Can handle very large resolutions

**Cons:**
- ⚠️ Significant communication overhead (~50% slower)
- ⚠️ Diminishing returns

**Use when:** Experimenting with >1024x1024 resolutions

---

## Environment Variables Explained

### Core Neuron Configuration

| Variable | Value | Purpose |
|----------|-------|---------|
| `NEURON_RT_VIRTUAL_CORE_SIZE` | `2` | **LNC2 mode** - 2 physical cores per logical core (trn2 optimization) |
| `NEURON_RT_NUM_CORES` | `<tp_degree>` | Number of logical cores to use (must match TP degree) |
| `NEURON_VISIBLE_DEVICES` | `0,1,...` | Which NeuronDevices are visible (comma-separated) |

**Example for TP=2:**
```bash
export NEURON_RT_VIRTUAL_CORE_SIZE=2  # LNC2 mode
export NEURON_RT_NUM_CORES=2          # 2 logical cores
export NEURON_VISIBLE_DEVICES=0,1     # Use devices 0 and 1
```

### Beta 3-Specific Features

| Variable | Value | Purpose |
|----------|-------|---------|
| `TORCH_NEURONX_ENABLE_ASYNC_NRT` | `1` | Enable async NRT execution (default in Beta 3, explicit here) |
| `TORCH_NEURONX_ENABLE_HOST_CC` | `1` | Enable host collective communications (overlap compute/comm) |
| `NEURON_COMPILE_CACHE_URL` | `file:///path` | Persistent NEFF cache (survives restarts) |
| `NEURONX_CACHE` | `/path` | Secondary cache path |

### Compiler Flags

| Flag | Purpose |
|------|---------|
| `-O1` | Optimization level 1 (good balance) |
| `--auto-cast=none` | No automatic type casting (we control precision) |
| `--enable-native-kernel=1` | Enable NKI kernels (Flash Attention, etc.) |
| `--remat` | Rematerialization (trade compute for memory) |
| `--enable-ccop-compute-overlap` | Overlap communication with compute |

---

## Launch Methods

### Method 1: Launch Script (Recommended)

```bash
# Basic usage
./beta3_wan22_launch.sh 2 "Your prompt here"

# With custom resolution
HEIGHT=768 WIDTH=1280 FRAMES=81 STEPS=40 \
    ./beta3_wan22_launch.sh 4 "A majestic eagle"

# Eager mode (instant start, no compilation)
EAGER_MODE=1 ./beta3_wan22_launch.sh 2 "Quick test"
```

### Method 2: Direct torchrun

```bash
# Set environment
export NEURON_RT_VIRTUAL_CORE_SIZE=2
export NEURON_RT_NUM_CORES=2
export NEURON_VISIBLE_DEVICES=0,1
export TORCH_NEURONX_ENABLE_ASYNC_NRT=1
export TORCH_NEURONX_ENABLE_HOST_CC=1
export NEURON_COMPILE_CACHE_URL="file:///mnt/nvme/neff_cache"

# Launch
torchrun --nproc-per-node=2 beta3_wan22_distributed_inference.py \
    --prompt "Your prompt" \
    --tp-degree 2 \
    --height 384 --width 640 --num-frames 17 --num-steps 20
```

### Method 3: Multi-Node (Future)

For multi-node distributed (not yet implemented):

```bash
# Node 0
NEURON_VISIBLE_DEVICES=0-31 torchrun \
    --nnodes=2 --nproc-per-node=32 --node-rank=0 \
    --master-addr=<node0-ip> --master-port=29500 \
    beta3_wan22_distributed_inference.py --tp-degree 64

# Node 1
NEURON_VISIBLE_DEVICES=0-31 torchrun \
    --nnodes=2 --nproc-per-node=32 --node-rank=1 \
    --master-addr=<node0-ip> --master-port=29500 \
    beta3_wan22_distributed_inference.py --tp-degree 64
```

---

## Performance Tuning

### Memory Optimization

**Symptoms of OOM:**
- `RuntimeError: NeuronCore out of memory`
- Compilation hangs at 90%+

**Solutions:**
1. **Increase TP degree** (TP=2 → TP=4)
2. **Reduce resolution** (768x1280 → 384x640)
3. **Reduce frames** (81 → 17 or 9)
4. **Use rematerialization** (already enabled via `--remat`)

### Communication Optimization

**If steps are slow:**
1. **Check network**: All-reduce uses Neuron interconnect, should be <1ms
2. **Reduce TP degree**: TP=4 → TP=2 (less communication)
3. **Enable host CC**: Already enabled via `TORCH_NEURONX_ENABLE_HOST_CC=1`

### Compilation Optimization

**First run is slow (10-20 min):**
- Normal - NEFFs are being compiled
- Subsequent runs use cache (~2-3 min)

**To pre-compile:**
```bash
# Run a short inference to populate NEFF cache
STEPS=2 FRAMES=1 ./beta3_wan22_launch.sh 2 "warmup"
```

**To clear cache:**
```bash
rm -rf /mnt/nvme/neff_cache/*
```

---

## Troubleshooting

### World Size Mismatch

```
ValueError: World size (4) must match TP degree (2)
```

**Fix:** Make sure `torchrun --nproc-per-node` matches `--tp-degree`:
```bash
torchrun --nproc-per-node=2 ... --tp-degree 2  # ✅ Match
```

### NeuronCore Not Found

```
Error: neuron-ls returned no devices
```

**Checks:**
1. `neuron-ls` shows devices?
2. `NEURON_VISIBLE_DEVICES` set correctly?
3. Another process holding cores? (`sudo systemctl restart neuron-rtd`)

### XRT Backend Error

```
RuntimeError: XRT backend not available
```

**Fix:** Install torch-neuronx 2.11.3+:
```bash
pip install torch-neuronx==2.11.3.0.1254
```

### All-Reduce Timeout

```
RuntimeError: collective operation timed out
```

**Causes:**
- One rank crashed (check logs)
- Network congestion (check `neuron-top`)
- Mismatched world sizes

**Fix:**
1. Check all ranks are alive: `ps aux | grep python`
2. Check Neuron runtime: `neuron-top`
3. Restart Neuron runtime: `sudo systemctl restart neuron-rtd`

### NEFF Compilation Hangs

```
Compilation stuck at 95%...
```

**Fixes:**
1. Wait longer (can take 15-20 min for large models)
2. Check compiler logs: `ls -lt /tmp/neuroncc-*`
3. Reduce batch size or resolution
4. Check disk space: `df -h /mnt/nvme`

---

## Performance Expectations

### WAN 2.2 14B on trn2.48xlarge

| Configuration | Resolution | Frames | Steps | Cold Start | Warm Start | Per Step |
|---------------|------------|--------|-------|------------|------------|----------|
| TP=2 | 384x640 | 17 | 20 | ~6 min | ~2.5 min | ~4s |
| TP=2 | 768x1280 | 17 | 20 | ~8 min | ~3 min | ~5s |
| TP=2 | 768x1280 | 81 | 40 | ~15 min | ~8 min | ~6s |
| TP=4 | 384x640 | 17 | 20 | ~7 min | ~3 min | ~5s |
| TP=4 | 768x1280 | 81 | 40 | ~18 min | ~10 min | ~8s |

**Cold start:** First run, NEFF compilation included  
**Warm start:** Subsequent runs, cached NEFFs

---

## Monitoring

### Check NeuronCore Utilization

```bash
neuron-top

# Expected output for TP=2:
# NC  NID   PID     %NC  ...
#  0    0  12345   95%
#  1    0  12346   95%
```

### Check Memory Usage

```bash
neuron-ls --json-output | jq '.neuron_devices[].neuron_cores[].memory_usage'
```

### Check Compilation Status

```bash
ls -lth /tmp/neuroncc-* | head -10
```

### Profile Performance

```bash
# Enable profiling
export NEURON_PROFILE=1

# Run inference
./beta3_wan22_launch.sh 2 "test"

# View profile
neuron-profile view /tmp/neuron_profile.json
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `beta3_wan22_distributed_inference.py` | Main distributed inference script |
| `beta3_wan22_launch.sh` | Launch wrapper with env configuration |
| `BETA3_DISTRIBUTED_GUIDE.md` | This guide |

---

## Advanced Topics

### Custom TP Sharding

To modify TP sharding strategy, edit `shard_transformer_for_tp()` in the inference script:

```python
def shard_transformer_for_tp(model, tp_degree, rank):
    # Custom sharding logic here
    # Example: shard only attention, keep MLP replicated
    ...
```

### Mixed Precision

Currently uses bf16 throughout. To experiment with fp16:

```python
model = model.to(dtype=torch.float16)
```

### Pipeline Parallelism

For >14B models, combine TP with Pipeline Parallelism:

```python
# Split model across pipeline stages
# Not yet implemented - future work
```

---

## Next Steps

1. **Test basic TP=2 inference** with small resolution
2. **Profile and optimize** for your workload
3. **Scale to TP=4** for full resolution
4. **Benchmark** against single-core baseline

---

**Questions?** See [BETA3_COMPLIANCE_REVIEW.md](BETA3_COMPLIANCE_REVIEW.md) for Beta 3 feature details.
