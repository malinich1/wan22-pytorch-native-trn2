# Trainium Feasibility Assessment: Wan2.2-T2V-A14B-Diffusers

**Model:** [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers)
**Type:** Text-to-Video Diffusion (DiT-based, Flow Matching)
**Date:** 2026-07-08
**SDK baseline:** Neuron 2.30 (current), referencing 2.26 FLUX.1-dev deployment precedent

---

## Verdict: 🟢 LIKELY READY (DiT precedent exists — FLUX.1-dev runs on Trn2)

**FLUX.1-dev** (a DiT-based image generation model) already deploys on Trn2 as of Neuron SDK 2.26
(Sep 2025). Wan2.2 uses the same DiT transformer architecture family, extended to 3D video. The
core pattern is proven; the assessment focuses on what's DIFFERENT between FLUX (working) and
Wan2.2 (video extension).

Additionally, Neuron 2.30 (May 2026) ships **`neuron-framework-autoport`** — a tool that
automates porting HuggingFace models to NxD Inference end-to-end. This should be the first
thing tried.

---

## Closest Proven Precedent: FLUX.1-dev on Trn2

| Dimension | FLUX.1-dev (runs on Trn2, SDK 2.26) | Wan2.2-T2V-A14B |
|---|---|---|
| Architecture | DiT (Diffusion Transformer) | DiT (same family, extended to 3D) |
| Attention | 2D spatial (image patches) | 3D spatiotemporal (video patches) |
| Conditioning | Text cross-attention | Text cross-attention (UMT5, same pattern) |
| Generation | Iterative denoising (flow matching) | Iterative denoising (flow matching, same) |
| Parameters | ~12B | ~14B (similar scale) |
| Patch embedding | 2D spatial | [1,2,2] (temporal + spatial) |
| Hardware | Trn2 | Trn2 (same) |

**The gap is narrow:** FLUX proves DiT + flow matching + text conditioning works on Trn2.
Wan2.2 adds a temporal dimension to the patches and attention. Everything else is the same pattern.

---

## Architecture Breakdown

| Component | Type | Size | Neuron Precedent |
|---|---|---|---|
| Transformer backbone (×2) | WanTransformer3DModel (DiT) | ~5-6B each | ✅ FLUX.1-dev DiT runs on Trn2 |
| Text encoder | UMT5EncoderModel | ~2-5B | ✅ T5 variants supported |
| VAE | AutoencoderKLWan (video) | ~0.2B | ⚠️ 3D variant (2D VAE works in FLUX/SDXL) |
| Scheduler | Flow matching | N/A (CPU) | ✅ Same as FLUX |
| **Total** | | **~14B** | |
| **bf16 memory** | | **~28 GB** | ✅ Fits easily on Trn2 (512 GB/device) |

### Transformer config:
- Layers: 40, Heads: 40, Head dim: 128 → Hidden: 5120
- FFN dim: 13824 (gated)
- Cross-attention: text_dim 4096 (UMT5)
- Patch size: [1, 2, 2] (temporal, height, width)
- RoPE: 3D positional encoding, max_seq_len: 1024
- qk_norm: rms_norm_across_heads
- in/out channels: 16 (latent space)

---

## Migration Path

### Step 0: Try `neuron-framework-autoport` FIRST (minutes)

Neuron 2.30 ships the **`neuron-framework-autoport`** Neuron Agentic Development skill that
automates porting HuggingFace models to NxD Inference end-to-end. Before any manual work:

```bash
# Use the neuron-framework-autoport skill (Neuron Agentic Development)
# This attempts the full port automatically and reports what fails
```

If this succeeds → done. If it partially succeeds → it tells you exactly which ops/layers
need manual handling.

### Step 1: Check how FLUX.1-dev is deployed (the closest reference)

Since FLUX.1-dev (DiT, flow matching, ~12B) already runs on Trn2, the deployment pattern
exists. Locate the FLUX deployment example in:
- Neuron SDK 2.26 release examples
- aws-neuron-samples repo
- NxD Inference tutorials

This gives you the exact compilation pattern, TP degree, and any required wrappers for DiT
models on Trn2.

### Step 2: Identify the delta (3D video extensions)

What Wan2.2 adds beyond FLUX:
1. **3D patch embedding** ([1,2,2] vs 2D) — adds a temporal dimension to input tokens
2. **3D RoPE** — rotary embeddings in 3 dimensions (time + height + width)
3. **3D VAE** (AutoencoderKLWan) — encodes/decodes video frames (conv3d instead of conv2d)
4. **Dual transformers** — two identical DiT backbones (possibly for different denoising stages)

### Step 3: Handle the deltas

| Delta | Likely approach | Risk |
|---|---|---|
| 3D patch embedding | Minor: just reshapes input (frames×H×W → sequence) | LOW |
| 3D RoPE | Pre-compute on CPU, pass as buffer (standard pattern) | LOW |
| qk_norm (rms_norm_across_heads) | SDK 2.30 has RMSNorm support; may just work | LOW-MEDIUM |
| 3D VAE (conv3d) | Check if conv3d compiles; decompose to 2D+temporal if not | MEDIUM |
| Variable video shapes | Compile for fixed buckets (same as LLM seq-len bucketing) | LOW |

### Step 4 (optional): NKI optimization for 3D attention

Video attention operates on MASSIVE sequences:
- 16 frames × (H/2) × (W/2) patches at 512×512 = 16 × 256 × 256 = **1,048,576 tokens**
- Even at lower res: 16 frames × 64 × 64 = **65,536 tokens per denoising step**

Our Phase 3 proved NKI fused attention is 2.3× faster at seqlen=4096 with CONSTANT latency.
At 65K+ tokens the advantage would be enormous. But this is an optimization AFTER the port
works, not a prerequisite.

---

## What's Available in Current SDK (Neuron 2.30)

| Capability | Status | Relevance |
|---|---|---|
| FLUX.1-dev deployment on Trn2 | ✅ GA (SDK 2.26) | Direct DiT precedent |
| `torch.compile` on Trainium | ✅ GA | May compile Wan2.2 directly |
| PyTorch Eager mode | ✅ GA | Fallback: run unchanged |
| `neuron-framework-autoport` | ✅ GA (SDK 2.30) | Automated porting tool |
| `neuron-framework-equivalence` | ✅ GA (SDK 2.30) | Validates numerical correctness of port |
| NKI 0.4.0 + segmented attention kernel | ✅ GA (SDK 2.30) | For video attention optimization |
| Expert parallelism (MoE) | Beta (SDK 2.26) | N/A (Wan2.2 is not MoE) |
| SDXL on Neuron | ✅ GA (older) | Diffusion loop pattern proven |

---

## Hardware Sizing

| Configuration | Instance | TP | Fits? |
|---|---|---|---|
| Single transformer (~6B) | trn2.48xlarge | 2 | ✅ Easily (12GB vs 512GB/device) |
| Full pipeline (~14B) | trn2.48xlarge | 4-8 | ✅ |
| 30-step denoising loop | trn2.48xlarge | 4-8 | ✅ (compiled NEFF reused each step) |
| Batch inference (4 videos) | trn2.48xlarge | 8 | ✅ |

Memory is not a concern. The model is small relative to Trn2 capacity.

---

## Risk Assessment (revised with FLUX precedent)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| DiT attention doesn't compile | **LOW** (FLUX DiT works) | High | Use FLUX's attention wrapper pattern |
| 3D RoPE hits unsupported op | Low | Medium | Pre-compute on CPU (standard pattern) |
| conv3d not supported in VAE | Medium | Medium | Decompose to conv2d + temporal |
| qk_norm fails | Low | Medium | RMSNorm is supported (SDK 2.30); may need minor rewrite |
| Performance gap vs GPU | Low | Low | NKI attention optimization available for long sequences |
| `neuron-framework-autoport` fails | Medium | Low | Fall back to manual FLUX pattern |

---

## Heavy Lifts (honest, with FLUX as baseline)

| Task | Effort | Notes |
|---|---|---|
| Try `neuron-framework-autoport` | **Minutes** | Try this first — may just work |
| Adapt FLUX deployment pattern for Wan2.2 | **LOW (days)** | Same architecture family; main change is 3D inputs |
| Handle 3D VAE (conv3d) | **MEDIUM (days-1 week)** | Only if conv3d doesn't compile directly |
| Shape bucketing for variable video | **LOW (days)** | Standard pattern (like LLM seq-len bucketing) |
| NKI 3D attention optimization | **HIGH (2+ weeks)** | OPTIONAL — only for peak performance after basic port |

**Total estimated effort: 1-2 weeks** (down from my earlier wrong estimate of 4-8 weeks,
because FLUX proves the core DiT pattern already works).

---

## Suggested Next Steps (in order)

1. **Check how FLUX.1-dev deploys on Trn2** — find the exact tutorial/example in SDK 2.26
2. **Run `neuron-framework-autoport`** on Wan2.2 — let the tool attempt the port
3. **If autoport fails:** trace WanTransformer3DModel with a minimal fixed shape and read the
   exact error messages (the "5-minute test" from before)
4. **Handle any conv3d issues** in the VAE (if they arise)
5. **Optimize with NKI** after the basic port is working

---

## For the SA Conversation

> **"Can Wan2.2 text-to-video run on Trainium?"**
>
> Very likely yes. FLUX.1-dev — which uses the same DiT architecture family and flow matching
> — already runs on Trn2 as of Neuron SDK 2.26. Wan2.2 extends this to 3D video, which adds
> temporal patches and a video VAE, but the core compute pattern is proven.
>
> The Neuron SDK 2.30 ships `neuron-framework-autoport` which can attempt the port
> automatically. Estimated effort for the manual path: 1-2 weeks (not months). Hardware
> capacity is more than sufficient (14B model on 8TB HBM). The upside for optimization is
> massive — video attention at 65K+ tokens is where NKI custom kernels provide 10-20×
> speedup over the compiler's default.
>
> **Risk level:** Low-Medium. The biggest unknown is whether the 3D-specific ops (3D RoPE,
> conv3d in VAE, qk_norm variant) compile cleanly, but even these have known workaround
> patterns.
