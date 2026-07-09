# WAN 2.2 T2V-A14B Runbook Execution Results

**Instance:** trn2.48xlarge (16 NeuronDevices, 64 NeuronCores, LNC=2)  
**Date:** July 8-9, 2026  
**Model:** Wan-AI/Wan2.2-T2V-A14B-Diffusers (14B parameters, MoE dual-transformer)  
**Goal:** Generate a recognizable cat video from text prompt

---

## Execution Flow

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         RUNBOOK EXECUTION FLOW                                    │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  Step 1: Environment Setup ─────────────────────────────────── ✅ PASS            │
│    └─ Beta 3 DLC container verified                                              │
│                                                                                   │
│  Step 2: neuron-framework-autoport ─────────────────────────── ⬜ NOT AVAILABLE   │
│    └─ Requires SDK 2.30 (we have 2.25/2.29)                                     │
│                                                                                   │
│  Step 3: FLUX.1-dev Precedent ──────────────────────────────── ⬜ NO ARTIFACTS    │
│    └─ Conceptual precedent only, no code to reuse                                │
│                                                                                   │
│  Step 4: torch.compile (TP=2) ──────────────────────────────── ❌ FAILED          │
│    └─ RoPE shape mismatch after naive TP sharding                                │
│                                                                                   │
│  Step 5: torch_neuronx.trace ───────────────────────────────── ❌ NOT VIABLE      │
│    └─ Not in Beta 3 DLC; 14B exceeds single-core limit                           │
│                                                                                   │
│  Step 6: NxDModel (TP=4, CP=2) ────────────────────────────── ✅ SUCCESS          │
│    └─ With --cpu_text_encoder --cpu_vae_decoder                                  │
│                                                                                   │
│  Step 7: Video Generated ───────────────────────────────────── ✅ CAT VISIBLE     │
│    └─ step6_cat_final.mp4 (1.0 MB, 480×832, 81 frames)                          │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Step-by-Step with CPU vs Neuron Breakdown

### Step 1: Environment Verification ✅

| Item | Value | Where |
|------|-------|-------|
| Container | Beta 3 DLC (`concourse-release-0461d3b:latest`) | Docker on trn2.48xlarge |
| Python | 3.12.13 | Container |
| PyTorch | 2.11.0 | Container |
| torch-neuronx | 2.11.3 | Container |
| neuronx-cc | 2.25 | Container (compiler) |
| `backend="neuron"` | Available in `torch._dynamo.list_backends()` | Container |
| Devices visible | 16 NeuronDevices (64 cores, LNC=2 → 24 GB HBM/core) | Hardware |

**Also available on host (not in container):**

| Item | Value |
|------|-------|
| Host venv | `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference` |
| Host PyTorch | 2.9.1 |
| Host torch-neuronx | 2.9.0 |
| neuronx-distributed-inference | 0.10.17970 |
| aws-neuronx-runtime-lib | 2.32.31.0 |

---

### Step 2: neuron-framework-autoport ⬜ NOT AVAILABLE

| Check | Result |
|-------|--------|
| `pip show neuron-agentic-development` | Not installed in Beta 3 DLC |
| `which neuron-framework-autoport` | CLI not found |
| SDK requirement | Needs SDK 2.30 (released May 21, 2026) |
| Our SDK | 2.25 (Beta 3 DLC) / 2.29 (host venv) |

**Verdict:** Cannot test. Would need SDK 2.30 DLAMI.

---

### Step 3: FLUX.1-dev Precedent ⬜ NO ARTIFACTS

| Check | Result |
|-------|--------|
| `find /opt -name "*flux*"` | No Neuron-specific FLUX files |
| `pip show optimum-neuron` | Not installed |
| `FluxPipeline` available | Yes (via diffusers 0.39.0) |
| `WanPipeline` available | Yes (via diffusers 0.39.0) |
| Pre-compiled FLUX NEFFs | None |

**Architecture comparison (from feasibility doc):**

| | FLUX.1-dev | WAN 2.2 T2V-A14B |
|---|---|---|
| Type | 2D DiT (images) | 3D DiT (video) |
| Params | ~12B | ~14B |
| Attention | 2D spatial | 3D spatiotemporal |
| Patch | 2D | [1, 2, 2] |
| RoPE | 2D | 3D |

**Verdict:** Same architecture family, but no reusable Neuron code.

---

### Step 4: torch.compile (TP=2, Beta 3 DLC) ❌ FAILED

**Why TP=2:** Model is 28 GB bf16, exceeds 24 GB single-core limit. Need at least 2 cores.

**Attempt 1: Naive sharding (shard QKV/FFN weights)**

| Sub-step | Where | Result |
|----------|-------|--------|
| Load 14B model on CPU | CPU (2 ranks) | ✅ 19.3s, 14.29B params |
| Shard weights (column/row parallel) | CPU | ✅ 840 tensors sharded (14.1 GB/rank), 255 replicated (0.5 GB) |
| `model.to("neuron")` | NeuronCores 0-1 | ✅ Succeeded |
| `torch.compile(backend="neuron")` | Registration | ✅ Registered |
| Forward pass (1×16×1×8×8) | NeuronCores | ❌ **FAILED** |

**Failure:** `RuntimeError: Attempting to broadcast dimension of length 5120 at -1`

**Root cause:** `norm_q.weight` (size 5120) not sharded to match QKV output (size 2560 per rank).

**Attempt 2: Fix norm weights (also shard norm_q, norm_k)**

| Sub-step | Where | Result |
|----------|-------|--------|
| Shard norm_q/norm_k weights | CPU | ✅ |
| Forward pass | NeuronCores | ❌ **FAILED** |

**Failure:** `RuntimeError: Attempting to broadcast dimension of length 64 at -1! Expected broadcastable to [1, 16, 40, 32]`

**Root cause:** After sharding, hidden dim = 2560 (5120/2). The model does `unflatten(2, (self.heads=40, -1))` which gives head_dim=64. But RoPE is precomputed for head_dim=128 and shape `[1, seq, 1, 128]`. The model hardcodes `self.heads=40` — it doesn't know about TP.

**Why this is fundamental:** torch.compile sees the model as a monolithic graph. Proper TP requires rewriting the model's internal head count, RoPE indexing, and adding all-reduce ops — exactly what NxDModel's ModelBuilder does during AOT compilation.

---

### Step 5: torch_neuronx.trace ❌ NOT VIABLE

| Check | Beta 3 DLC | Host venv (SDK 2.29) |
|-------|-----------|---------------------|
| `torch_neuronx.trace` available | **No** | Yes |
| Can trace 14B model? | N/A | No (requires single-core runtime, 14B > 24 GB) |

**Verdict:** Even if available, trace requires the full model to execute on one core. 14B doesn't fit.

---

### Step 6: NxDModel Inference ✅ SUCCESS

**This is the WORKING approach.** Uses pre-compiled models from `aws-neuron-samples` repo.

| Phase | Where | Time | Details |
|-------|-------|------|---------|
| Pipeline load | CPU | ~3s | Load diffusers pipeline (tokenizer, scheduler, VAE config) |
| Text encoding | **CPU** | 4.4s | UMT5-XXL (`pipe.encode_prompt()`, `--cpu_text_encoder`) |
| Load NxDModel | **NeuronCores 0-7** | 77.2s | Load compiled NEFF + set weights (TP=4, CP=2, world_size=8) |
| Phase 1: High-noise denoising | **NeuronCores 0-7** | 104.4s | 13 steps × 8.0s/step (transformer_1) |
| MoE weight swap | **NeuronCores 0-7** | 68.1s | `NxDModel.replace_weights()` (transformer_2 weights) |
| Phase 2: Low-noise denoising | **NeuronCores 0-7** | 216.9s | 27 steps × 8.0s/step (transformer_2) |
| VAE decode | **CPU** | 258.2s | `pipe.vae.decode()` full 81 frames (`--cpu_vae_decoder`) |
| Video save | CPU | ~1s | `export_to_video()` → mp4 |
| **TOTAL** | | **756.0s (12.6 min)** | |

**NxDModel compilation details (pre-computed, one-time ~15 min):**

| Compiled Artifact | Location | TP | CP | world_size |
|-------------------|----------|----|----|-----------|
| text_encoder | `/opt/dlami/nvme/compiled_models_t2v_a14b/text_encoder` | 4 | 2 | 8 |
| transformer (expert 1) | `.../transformer` | 4 | 2 | 8 |
| transformer_2 (expert 2) | `.../transformer_2` | 4 | 2 | 8 |
| decoder_rolling (VAE) | `.../decoder_rolling` | 4 | 2 | 8 |
| post_quant_conv | `.../post_quant_conv` | 4 | 2 | 8 |

**Known bugs found during debugging:**

| Component | Bug | Impact | Workaround |
|-----------|-----|--------|------------|
| NxD Text Encoder | Compiled with world_size=8 but UMT5 only needs TP=4; CP=2 expansion produces wrong embeddings | No recognizable content in video | `--cpu_text_encoder` |
| NxD Rolling VAE Decoder | Produces mosaic/noise output (stateful cache issue) | Corrupted video frames | `--cpu_vae_decoder` |
| NxD Transformer | **Correct** (verified 0.996 cosine similarity vs CPU reference) | None | None needed |

---

### Step 7: Final Video ✅

| Metric | Value |
|--------|-------|
| Output | `step6_cat_final.mp4` |
| Size | 1.0 MB |
| Resolution | 480×832 |
| Frames | 81 @ 16fps (~5 seconds) |
| Steps | 40 (13 high-noise + 27 low-noise) |
| CFG | 5.0 |
| Prompt | "A beautiful fluffy orange tabby cat walking through a sunlit garden with flowers, cinematic quality, photorealistic, detailed fur" |
| Cat visible | **Yes** |

---

## CPU vs NeuronCore Execution Map

```
TIME ──────────────────────────────────────────────────────────────►

CPU:  ████ Text Enc (4.4s) ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ██████████ VAE Decode (258s) ██
       │                                                                │
       └──embeddings──►                                    ◄──latents───┘
                       │                                   │
NEURON:                ████████████████████████████████████████
                       │  Load   │Phase1│Swap│  Phase 2   │
                       │  77s    │104s  │68s │   217s     │
                       │         │      │    │            │
                       NxDModel (TP=4, CP=2, 8 NeuronCores)
                       
                       Total Neuron time: ~466s
                       Total CPU time:    ~263s
                       Pipeline overhead: ~27s
                       ─────────────────────────
                       TOTAL:             756s
```

---

## Approaches Tested — Full Matrix

| Approach | Env | Model | TP | Result | Failure Mode |
|----------|-----|-------|----|---------|----|
| torch.compile (single core) | Beta 3 DLC | T2V-A14B (14B) | 1 | ❌ OOM | 28 GB > 24 GB/core |
| torch.compile (TP=2, naive shard) | Beta 3 DLC | T2V-A14B (14B) | 2 | ❌ Shape mismatch | RoPE/heads hardcoded in model |
| torch.compile (single core) | Beta 3 DLC | T2V-1.3B (WAN 2.1) | 1 | ✅ Works | 2.6 GB fits easily |
| torch.compile (single core) | Beta 3 DLC | TI2V-5B (48ch) | 1 | ❌ Compiler crash | neuronx-cc exit code 70 |
| torch_neuronx.trace | Host venv 2.29 | T2V-A14B (14B) | 1 | ❌ Not viable | 14B > 24 GB single core |
| **NxDModel (ModelBuilder)** | **Host venv 2.29** | **T2V-A14B (14B)** | **4** | **✅ Works** | — |
| neuron-framework-autoport | N/A | T2V-A14B | — | ⬜ Untested | Requires SDK 2.30 |

---

## Recommendations

### For production use of WAN 2.2 T2V-A14B:

```bash
# Activate NxD inference venv
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
cd ~/aws-neuron-samples/torch-neuronx/inference/hf_pretrained_wan2.2_t2v_a14b

# Run with proven flags
python run_wan2.2_t2v_a14b.py \
    --compiled_models_dir /opt/dlami/nvme/compiled_models_t2v_a14b \
    --prompt "Your prompt here" \
    --height 480 --width 832 --num_frames 81 \
    --num_inference_steps 40 --guidance_scale 5.0 \
    --output /mnt/nvme/outputs/video.mp4 \
    --cpu_text_encoder --cpu_vae_decoder
```

### To improve performance (future work):
1. **Fix NxD Text Encoder:** Recompile with `world_size=4` (TP=4 only, no CP)
2. **Fix NxD Rolling VAE:** Debug stateful cache issue in rolling decoder
3. **Upgrade to SDK 2.30:** Try `neuron-framework-autoport` for automatic porting
4. **torch.compile with proper TP:** Requires rewriting model internals (head counts, RoPE) — same work that ModelBuilder automates

### Model size limits for torch.compile (Beta 3 DLC):
- **≤ 3B params, in_channels ≤ 16:** torch.compile works on single core
- **> 3B or in_channels > 16:** Requires NxDModel (ModelBuilder + TP)
