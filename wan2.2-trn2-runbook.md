# Wan2.2-T2V-A14B on Trn2 — Step-by-Step Runbook

**Target instance:** trn2.48xlarge (us-east-2 or any region with Trn2)
**DLAMI:** Neuron DLAMI PyTorch 2.9 / Ubuntu 24.04
**Venvs available:**
- `/opt/aws_neuronx_venv_pytorch_2_9` — base (NKI, custom tracing)
- `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference` — LLM/model inference (NxD Inference)
- `/opt/aws_neuronx_venv_pytorch_2_9_nxd_training` — training

---

## Step 1: Environment setup (2 min)

```bash
# SSH into the trn2 instance
aws ssm start-session --target <instance-id> --region <region>

# Activate the NxD INFERENCE venv (ALWAYS use this for model inference)
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Verify
python -c "import torch_neuronx; print('torch_neuronx:', torch_neuronx.__version__)"
python -c "import neuronx_distributed_inference; print('nxdi: OK')"
neuron-ls  # should show 16 Neuron devices

# Install diffusers (the pipeline library for Wan2.2)
pip install diffusers transformers accelerate sentencepiece
```

---

## Step 2: Try `neuron-framework-autoport` (5-10 min)

This is the SDK 2.30 tool that automatically ports HuggingFace models. Try it first.

```bash
# Check if the neuron-framework-autoport skill is available
python -c "from neuronx_distributed_inference.models import *; print('Available models:', dir())"

# If the autoport CLI is available:
# neuron-framework-autoport --model Wan-AI/Wan2.2-T2V-A14B-Diffusers --output /tmp/wan2_ported

# If not a CLI, check if it's a Claude Code / Kiro skill:
# The neuron-agentic-development package should have it
pip show neuron-agentic-development
```

**If autoport supports this model → you're done (just follow its output).**
**If it doesn't support diffusion models yet → continue to Step 3.**

---

## Step 3: Check how FLUX.1-dev deploys (find the DiT precedent) (10 min)

FLUX.1-dev (DiT, flow matching, ~12B) runs on Trn2 since SDK 2.26. Find its deployment pattern.

```bash
# Search for FLUX examples in the installed packages
find /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference -name "*flux*" -o -name "*dit*" 2>/dev/null
pip show optimum-neuron 2>/dev/null  # optimum-neuron may have FLUX support

# Check the Neuron samples repo for FLUX
curl -sS "https://api.github.com/repos/aws-neuron/aws-neuron-samples/git/trees/master?recursive=1" | \
  python -c "import sys,json; [print(t['path']) for t in json.load(sys.stdin).get('tree',[]) if 'flux' in t['path'].lower() or 'dit' in t['path'].lower()]"

# Also check optimum-neuron (HuggingFace's Neuron integration)
pip install optimum-neuron
python -c "from optimum.neuron import NeuronStableDiffusionXLPipeline; print('optimum-neuron SD support: OK')"
# Check if there's a FLUX or generic diffusion pipeline:
python -c "import optimum.neuron; print([x for x in dir(optimum.neuron) if 'diffusion' in x.lower() or 'flux' in x.lower() or 'stable' in x.lower()])"
```

---

## Step 4: Try `torch.compile` directly (the simplest path) (10-20 min)

Neuron SDK 2.30 supports `torch.compile` — try compiling the model directly without trace.

```bash
cat > /tmp/wan2_compile_test.py << 'PYEOF'
import os, time, torch
os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "0"
import torch_neuronx
from diffusers import WanPipeline
import torch

print("=== Wan2.2 torch.compile test on Neuron ===")

# Load the pipeline on CPU first
print("[1] Loading pipeline...")
pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    torch_dtype=torch.bfloat16
)
print("  Pipeline loaded:", type(pipe).__name__)
print("  Transformer:", type(pipe.transformer).__name__)
print("  VAE:", type(pipe.vae).__name__)
print("  Text encoder:", type(pipe.text_encoder).__name__)

# Try torch.compile on the transformer (the main compute)
print("\n[2] Attempting torch.compile on transformer...")
try:
    pipe.transformer = torch.compile(pipe.transformer, backend="neuronx")
    print("  torch.compile(backend='neuronx') SUCCEEDED")
except Exception as e:
    print("  torch.compile failed:", str(e)[:200])
    print("  Trying without explicit backend...")
    try:
        pipe.transformer = torch.compile(pipe.transformer)
        print("  torch.compile() SUCCEEDED (default backend)")
    except Exception as e2:
        print("  Also failed:", str(e2)[:200])

# Try moving to XLA device
print("\n[3] Moving transformer to XLA...")
try:
    pipe.transformer = pipe.transformer.to("xla")
    print("  .to('xla') SUCCEEDED")
except Exception as e:
    print("  .to('xla') failed:", str(e)[:200])

# Try a minimal forward pass
print("\n[4] Minimal forward pass...")
try:
    # Create minimal inputs matching the transformer's expected shape
    # batch=1, channels=16, frames=1, H=16, W=16 (tiny for compile test)
    hidden = torch.randn(1, 16, 1, 16, 16, dtype=torch.bfloat16).to("xla")
    timestep = torch.tensor([500.0]).to("xla")
    encoder_hidden = torch.randn(1, 77, 4096, dtype=torch.bfloat16).to("xla")
    
    with torch.no_grad():
        out = pipe.transformer(hidden, timestep=timestep, encoder_hidden_states=encoder_hidden)
    print("  FORWARD PASS OK! Output shape:", out.shape if hasattr(out, 'shape') else type(out))
except Exception as e:
    print("  FORWARD FAILED:", str(e)[:500])
    print("\n  This error tells you exactly which ops need fixing.")

print("\nDONE")
PYEOF

python /tmp/wan2_compile_test.py 2>&1 | tee /tmp/wan2_results.txt
```

**Read the output carefully:**
- If Step 2 (torch.compile) succeeds → the model compiles on Neuron
- If Step 4 (forward pass) succeeds → **the model RUNS on NeuronCore**
- If Step 4 fails → the error message tells you EXACTLY which op is unsupported

---

## Step 5: Try `torch_neuronx.trace` per component (the SDXL pattern) (20-30 min)

If torch.compile doesn't work directly, fall back to the proven SDXL pattern: trace each component separately.

```bash
cat > /tmp/wan2_trace_test.py << 'PYEOF'
import os, time, torch, torch_neuronx
os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "0"
from diffusers import WanPipeline

print("=== Wan2.2 component-by-component trace test ===")
pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    torch_dtype=torch.bfloat16
)

# --- Test 1: Text Encoder (UMT5) ---
print("\n[1/3] Tracing text encoder (UMT5)...")
try:
    text_input = torch.randint(0, 1000, (1, 77))  # token ids
    traced_te = torch_neuronx.trace(pipe.text_encoder, (text_input,))
    print("  TEXT ENCODER TRACE: OK")
except Exception as e:
    print("  TEXT ENCODER TRACE FAILED:", str(e)[:300])

# --- Test 2: VAE Decoder ---
print("\n[2/3] Tracing VAE decoder...")
try:
    # Latent space: (batch, latent_ch, frames, H/8, W/8)
    latent = torch.randn(1, 16, 1, 32, 32, dtype=torch.bfloat16)
    traced_vae = torch_neuronx.trace(pipe.vae.decode, (latent,))
    print("  VAE DECODE TRACE: OK")
except Exception as e:
    print("  VAE DECODE TRACE FAILED:", str(e)[:300])

# --- Test 3: Transformer (the main DiT backbone) ---
print("\n[3/3] Tracing transformer (WanTransformer3DModel)...")
try:
    # Match the transformer's forward signature
    hidden = torch.randn(1, 16, 1, 32, 32, dtype=torch.bfloat16)
    timestep = torch.tensor([500.0])
    encoder_hidden = torch.randn(1, 77, 4096, dtype=torch.bfloat16)
    
    # Wrapper to match trace requirements (positional args only)
    class TransformerWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, hidden_states, timestep, encoder_hidden_states):
            return self.model(
                hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states
            ).sample
    
    wrapper = TransformerWrapper(pipe.transformer)
    traced_dit = torch_neuronx.trace(wrapper, (hidden, timestep, encoder_hidden))
    print("  TRANSFORMER TRACE: OK")
    print("  THIS MEANS THE DiT COMPILES ON NEURON!")
except Exception as e:
    print("  TRANSFORMER TRACE FAILED:", str(e)[:500])
    print("\n  ^^^ This error tells you the specific unsupported ops.")
    print("  Compare against how FLUX.1-dev handles the same pattern.")

print("\n=== SUMMARY ===")
print("Check /tmp/wan2_results.txt for full output")
print("Any 'TRACE: OK' means that component compiles for NeuronCore")
print("Any 'FAILED' error tells you exactly what op needs a workaround")
PYEOF

python /tmp/wan2_trace_test.py 2>&1 | tee /tmp/wan2_trace_results.txt
```

---

## Step 6: If compilation succeeds — run full inference (30+ min first time)

If Steps 4 or 5 show the transformer compiles:

```bash
cat > /tmp/wan2_generate.py << 'PYEOF'
import os, time, torch
os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "0"
import torch_neuronx
from diffusers import WanPipeline

print("=== Wan2.2 Video Generation on Trn2 ===")
pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    torch_dtype=torch.bfloat16
)

# Move components to NeuronCore
pipe.transformer = pipe.transformer.to("xla")
pipe.text_encoder = pipe.text_encoder.to("xla")
# Note: VAE often stays on CPU for decode (memory for full video frames)

prompt = "A serene lake with mountains in the background, cinematic"
print(f"Generating video: '{prompt}'")
t0 = time.time()
output = pipe(
    prompt=prompt,
    num_frames=16,
    height=256,
    width=256,
    num_inference_steps=20,
)
gen_time = time.time() - t0
print(f"DONE in {gen_time:.1f}s")
print(f"Output frames: {len(output.frames[0])}")

# Save
from diffusers.utils import export_to_video
export_to_video(output.frames[0], "/tmp/wan2_output.mp4")
print("Saved to /tmp/wan2_output.mp4")
print("WAN2_NEURON_INFERENCE_OK")
PYEOF

python /tmp/wan2_generate.py
```

---

## Step 7: If specific ops fail — apply fixes from our learnings

Based on the errors from Steps 4/5, apply known fixes:

| Error | Fix |
|---|---|
| `sort is not supported` | Replace with iterative argmax (our `_neuron_topk` from Phase 3) |
| `int64 dot not supported` | Set `NEURON_CC_FLAGS='--disable-hlo-operand-type-check evrf_035'` |
| Custom attention fails | Write a Neuron-compatible attention wrapper (like SDXL example) |
| conv3d not supported | Decompose: `conv3d(x)` → `conv2d(x_per_frame)` + `conv1d(x_temporal)` |
| Dynamic shapes | Fix to a bucket: `(frames=16, H=256, W=256)` for the trace |
| qk_norm fails | Rewrite as: `F.normalize(q, dim=-1) * scale` (equivalent, standard ops) |

---

## Step 8: Performance optimization with NKI (AFTER basic port works)

Only do this after Steps 1-6 produce a working video. The NKI advantage is massive for video
because attention sequence lengths are 10-100× longer than text LLMs:

```
Video attention seq_len = frames × (H/patch_h) × (W/patch_w)
= 16 × (256/2) × (256/2) = 16 × 128 × 128 = 262,144 tokens per step!

Our Phase 3 proved: NKI 2.3× faster at 4K tokens (constant latency).
At 262K tokens: NKI advantage would be MASSIVE (estimated 50-100×).
```

Use the `nki-samples` `attention_fwd_v8a` kernel adapted for 3D RoPE + cross-attention.

---

## Decision tree (what to expect)

```
Step 2 (autoport) succeeds?
  YES → Done! Follow its output.
  NO  → Continue

Step 4 (torch.compile or .to('xla') + forward) succeeds?
  YES → Great! Run Step 6 for full generation.
  NO  → Read the error. Continue to Step 5.

Step 5 (trace per component)?
  Text encoder traces? → Almost certainly YES (T5 is well-supported)
  VAE traces? → Probably YES (2D VAE works; 3D may need conv3d decomposition)
  Transformer traces?
    YES → Port is straightforward. Do Step 6.
    NO  → The error tells you EXACTLY which ops failed.
           Apply fixes from Step 7.
           The FLUX.1-dev example shows how to handle DiT-specific issues.
```

---

## What success looks like

- **Best case (minutes):** `neuron-framework-autoport` or `torch.compile` handles it → video generates
- **Expected case (days):** Transformer needs an attention wrapper (like SDXL) + conv3d needs decomposition → 3-5 days of engineering
- **Worst case (1-2 weeks):** Multiple unsupported ops requiring individual workarounds + shape bucketing

In ALL cases: the model fits on hardware, the DiT pattern is proven (FLUX), and the gap is known ops, not fundamental architecture incompatibility.
