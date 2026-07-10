# WAN 2.2 on Trainium2 — 1-Week Testing Summary

**Period:** July 3–10, 2026  
**Instance:** trn2.48xlarge (16 NeuronDevices, 64 NeuronCores, LNC=2, 24 GB HBM/core)  
**Region:** us-east-2  
**Goal:** Generate video (visible cat) from text using WAN 2.2 models on Neuron hardware  
**Repo:** https://github.com/malinich1/wan22-pytorch-native-trn2

---

## Executive Summary

We tested three WAN video generation models on trn2.48xlarge using multiple deployment strategies. **One approach works end-to-end: NxDModel with TP=4, CP=2 for the 14B transformer, with CPU fallback for text encoder and VAE decoder.** Total inference time is ~12.6 minutes for a 5-second 480×832 video.

| Model | Approach | Result |
|-------|----------|--------|
| WAN 2.2 T2V-A14B (14B MoE) | NxDModel (TP=4, CP=2) | **Working** |
| WAN 2.1 T2V-1.3B | torch.compile (single core) | Working (low quality) |
| WAN 2.2 TI2V-5B | torch.compile | Failed (compiler crash) |

---

## 1. Models Tested

### WAN 2.2 T2V-A14B (Primary Target)
- **Architecture:** 14B parameter MoE dual-transformer (text-to-video)
- **Memory footprint:** ~28 GB in bf16
- **Unique features:** Two transformer experts (high-noise + low-noise), 3D RoPE, spatiotemporal attention
- **Source:** `Wan-AI/Wan2.2-T2V-A14B-Diffusers`

### WAN 2.1 T2V-1.3B (Baseline)
- **Architecture:** 1.3B single transformer (text-to-video)
- **Memory footprint:** ~2.6 GB in bf16
- **Source:** `Wan-AI/Wan2.1-T2V-1.3B`

### WAN 2.2 TI2V-5B (Image-to-Video)
- **Architecture:** 5B transformer, 48 input channels (text+image)
- **Memory footprint:** ~10 GB in bf16
- **Source:** `Wan-AI/Wan2.2-TI2V-5B-Diffusers`

---

## 2. Environments Used

| Environment | PyTorch | neuronx-cc | SDK | Use Case |
|-------------|---------|------------|-----|----------|
| **Host venv** (`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`) | 2.9.1 | 2.22 | 2.29 | NxDModel inference (pre-compiled models) |
| **Beta 3 DLC** (`421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest`) | 2.11.0 | 2.25 | 2.25 | torch.compile testing |

---

## 3. Approaches Tested — Complete Matrix

| # | Approach | Model | Env | TP/CP | Result | Failure Root Cause |
|---|----------|-------|-----|-------|--------|--------------------|
| 1 | torch.compile (1 core) | T2V-A14B (14B) | Beta 3 DLC | 1/1 | **OOM** | 28 GB > 24 GB/core |
| 2 | torch.compile (TP=2, naive shard) | T2V-A14B (14B) | Beta 3 DLC | 2/1 | **Failed** | RoPE shape mismatch; model hardcodes `self.heads=40` |
| 3 | torch.compile (1 core) | T2V-1.3B | Beta 3 DLC | 1/1 | **Works** | — (but poor video quality, model too small) |
| 4 | torch.compile (1 core) | TI2V-5B | Beta 3 DLC | 1/1 | **Compiler crash** | neuronx-cc exit 70; 48 in_channels + 30 layers exceeds graph complexity |
| 5 | torch_neuronx.trace | T2V-A14B (14B) | Host 2.29 | 1/1 | **Not viable** | Not in Beta 3; 14B exceeds single-core for trace |
| 6 | neuron-framework-autoport | T2V-A14B (14B) | — | — | **Untestable** | Requires SDK 2.30 (we have 2.25/2.29) |
| 7 | **NxDModel (ModelBuilder)** | **T2V-A14B (14B)** | **Host 2.29** | **4/2** | **Working** | — |

---

## 4. Working Solution: NxDModel Pipeline

### Component Execution Map

| Component | Execution Target | Time | Status |
|-----------|-----------------|------|--------|
| Text Encoder (UMT5-XXL) | **CPU** | 4.4s | Forced via `--cpu_text_encoder` (NxD version has bug) |
| Transformer — Phase 1 (13 high-noise steps) | **Neuron** (TP=4, CP=2, 8 cores) | 104.4s | Correct (cosine=0.996 vs CPU) |
| Transformer — Weight Swap (expert 1→2) | **Neuron** | 68.6s | `NxDModel.replace_weights()` |
| Transformer — Phase 2 (27 low-noise steps) | **Neuron** (TP=4, CP=2, 8 cores) | 216.7s | Correct |
| VAE Decoder (81 frames) | **CPU** | ~258s | Forced via `--cpu_vae_decoder` (NxD version has bug) |

### Performance Metrics (Final Fresh Run — July 10, 2026)

| Metric | Value |
|--------|-------|
| Total wall-clock time | ~756s (12.6 min) |
| Denoising time (Neuron) | 321.1s (40 steps × ~8.0s/step) |
| NeuronCore utilization | 8 out of 64 cores (12.5%) |
| Per-step latency | ~8.0s/step |
| Model load time | 76.4s |
| Weight swap time | 68.6s |
| VAE decode (CPU) | ~258s |
| Text encode (CPU) | 4.4s |

### Generation Parameters

| Parameter | Value |
|-----------|-------|
| Resolution | 480×832 |
| Frames | 81 |
| FPS | 16 |
| Video duration | 5.1 seconds |
| Inference steps | 40 (13 high-noise + 27 low-noise) |
| Guidance scale | 5.0 |
| Scheduler | FlowMatchEulerDiscreteScheduler |
| Seed | 42 |

---

## 5. Bugs Found in NxD Sample Code

| Component | Bug | Evidence | Workaround | Performance Cost |
|-----------|-----|----------|------------|-----------------|
| **NxD Text Encoder** | Compiled with `world_size=8` but UMT5 only needs TP=4; CP=2 expansion corrupts relative position bias | cosine=0.31 vs CPU reference | `--cpu_text_encoder` | +4.4s (0.6% overhead) |
| **NxD Rolling VAE Decoder** | Stateful rolling decode produces mosaic/noise artifacts | Visual corruption in output frames | `--cpu_vae_decoder` | +258s (34% overhead) |
| **NxD Transformer** | None — verified correct | cosine=0.996 vs CPU reference | None needed | — |

### Text Encoder Fix Attempted (Not Successful)

| Strategy | What | Result |
|----------|------|--------|
| A: Config patch (`world_size: 8→4`) | Change config.json only | Segfault — NEFF binary has world_size=8 baked in |
| B: Full recompile (TP=4, CP=1) | Recompile from scratch | Compiles but cosine=0.31 — relative position bias sharding is wrong in `compile_text_encoder.py` |

**Conclusion:** Fix requires modifying the upstream compilation script in `aws-neuron-samples` to handle UMT5's relative position bias correctly under TP. Not a simple all-gather fix.

---

## 6. Key Technical Learnings

### torch.compile Constraints on Neuron (Beta 3 DLC)

1. **Model size limit:** ≤ 3B parameters, in_channels ≤ 16 for single-core compile
2. **No native TP support:** `torch.compile(backend="neuron")` treats the model as monolithic — no distributed primitives
3. **Manual TP breaks model invariants:** Models hardcode head counts, RoPE dimensions; naive weight sharding causes shape mismatches
4. **Compiler complexity limit:** 48 input channels + 30 transformer layers exceeds neuronx-cc graph limits (exit code 70)

### NxDModel Architecture

1. **ModelBuilder + AOT compilation:** Handles TP/CP automatically during compile phase (rewrites attention heads, inserts collectives)
2. **Weight swapping:** MoE experts can be swapped in-place on NeuronCores without re-compilation
3. **CP (Context Parallelism):** Splits sequence dimension across ranks — critical for 81-frame video latents
4. **Compiled artifacts are NOT portable:** `world_size` is baked into NEFF binaries — cannot change TP/CP post-compilation

### trn2.48xlarge Observations

1. **LNC=2 mode:** Each NeuronDevice exposes 2 logical cores with 24 GB HBM each
2. **EFA warnings are benign:** "OFI plugin initNet() failed" appears but doesn't affect intra-node communication
3. **NVMe is essential:** Model weights (28 GB) + compiled artifacts (~15 GB) need fast local storage
4. **Only 8 of 64 cores used:** Pipeline uses TP=4 × CP=2 = 8 cores, leaving 56 idle (room for batching)

---

## 7. Artifacts Produced

### Video Outputs

| File | Model | Method | Size | Resolution | Cat Visible |
|------|-------|--------|------|------------|-------------|
| `cat_fresh_nxd.mp4` | T2V-A14B (14B) | NxDModel (latest run) | 1.8 MB | 832×480, 81f | **Yes** |
| `cat_14b_final.mp4` | T2V-A14B (14B) | NxDModel | 628 KB | 832×480, 81f | **Yes** |
| `cat_14b_cpu_te.mp4` | T2V-A14B (14B) | NxDModel | 1.1 MB | 832×480, 81f | **Yes** |
| `step6_cat_final.mp4` | T2V-A14B (14B) | NxDModel | 1.0 MB | 832×480, 81f | **Yes** |
| `cat_compile_hq.mp4` | T2V-1.3B (WAN 2.1) | torch.compile | ~500 KB | various | No (too small) |
| `cat_seed7.mp4` | T2V-1.3B (WAN 2.1) | torch.compile | ~500 KB | various | No (too small) |

### Scripts (in repo)

| Script | Purpose |
|--------|---------|
| `run_wan_small.py` | WAN 2.1 1.3B torch.compile inference |
| `step4_tp2_compile_test.py` | TP=2 torch.compile attempt for 14B |
| `wan22_5b_hybrid_compile.py` | TI2V-5B compile attempt |
| `wan22_5b_hybrid_launch.sh` | TI2V-5B torchrun launcher |
| `beta3_wan22_distributed_inference.py` | Distributed inference script (Beta 3) |
| `recompile_text_encoder_cp1.py` | Text encoder recompile attempt |
| `validate_text_encoder_cp1.py` | Text encoder validation (cosine check) |
| `compare_nxd_te.py` | NxD vs CPU text encoder comparison |
| `run_wan22_nxd_fixed_te.py` | Inference with fixed text encoder attempt |
| `compile_model.py` | Model compilation utilities |

### Documentation

| Document | Purpose |
|----------|---------|
| `RUNBOOK_EXECUTION_RESULTS.md` | Detailed step-by-step execution of the runbook |
| `feasibility-wan2.2-t2v-a14b.md` | Feasibility assessment document |
| `wan2.2-trn2-runbook.md` | Original runbook with all approaches |
| `BETA3_DISTRIBUTED_GUIDE.md` | Beta 3 DLC distributed inference guide |
| `QUICKSTART.md` | Quick start instructions |
| `wan22_pytorch_native_workshop.ipynb` | Workshop notebook |

---

## 8. Recommended Production Command

```bash
# SSH to trn2.48xlarge
ssh -i ~/.ssh/wan22-test-key.pem ubuntu@<INSTANCE_IP>

# Activate NxD inference venv (SDK 2.29)
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Navigate to sample code
cd ~/aws-neuron-samples/torch-neuronx/inference/hf_pretrained_wan2.2_t2v_a14b

# Run inference (proven working)
python run_wan2.2_t2v_a14b.py \
    --compiled_models_dir /opt/dlami/nvme/compiled_models_t2v_a14b \
    --prompt "Your prompt here" \
    --height 480 --width 832 --num_frames 81 \
    --num_inference_steps 40 --guidance_scale 5.0 \
    --output /mnt/nvme/outputs/output.mp4 \
    --cpu_text_encoder --cpu_vae_decoder
```

---

## 9. Future Optimization Opportunities

| Opportunity | Expected Improvement | Effort | Dependency |
|-------------|---------------------|--------|------------|
| Fix NxD Text Encoder (recompile with correct TP) | -4.4s (marginal) | Medium | Fix in `aws-neuron-samples` compile script |
| Fix NxD Rolling VAE (debug stateful cache) | -258s → ~30s | High | Debug NxD rolling decoder |
| SDK 2.30 + neuron-framework-autoport | Unknown (may enable full torch.compile) | Low (upgrade) | New DLAMI |
| Batch inference (use all 64 cores) | 8× throughput | Medium | Launch 8 parallel pipelines |
| Higher resolution (720p) | Better quality | Low | Recompile with --height 720 --width 1280 |
| Persistent model (avoid reload) | -77s per inference | Medium | Keep NxDModel loaded across requests |

### Theoretical Best-Case Pipeline (all bugs fixed)

| Component | Target | Time |
|-----------|--------|------|
| Text Encoder | Neuron (TP=4) | ~0.5s |
| Transformer (40 steps) | Neuron (TP=4, CP=2) | 321s |
| Weight swap | Neuron | 68s |
| VAE Decoder | Neuron (rolling) | ~30s |
| **Total** | | **~420s (7 min)** vs current 756s |

---

## 10. Conclusion

**WAN 2.2 T2V-A14B runs on trn2.48xlarge** with the NxDModel approach (pre-compiled by AWS Neuron team in `aws-neuron-samples`). The 14B MoE transformer executes correctly on 8 NeuronCores with TP=4, CP=2. Current limitations are the text encoder and VAE decoder bugs which force CPU fallback, adding ~262s overhead to the pipeline. With these fixes, inference could drop from 12.6 min to ~7 min.

The key insight: **torch.compile alone cannot handle models > 3B on Neuron** — the compiler doesn't support distributed execution. NxDModel (which uses ModelBuilder for AOT compilation with automatic TP/CP) is the required path for large models until SDK 2.30's `neuron-framework-autoport` potentially bridges this gap.
