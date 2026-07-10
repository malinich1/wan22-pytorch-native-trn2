"""
Runbook Step 4: Try torch.compile on WAN 2.2 T2V-A14B transformer.

Tests whether the 14B transformer graph can be compiled by neuronx-cc.
Uses a TINY input shape (1 frame, 32x32) to minimize memory.
The goal is to test COMPILATION, not runtime fitness.

Expected outcomes:
  A) Compilation succeeds → torch.compile works for this architecture
  B) Exit code 70 → compiler can't handle the graph (like TI2V-5B)
  C) OOM at runtime → graph compiles but model too large for 1 core (need TP)
"""
import os
import time
import torch
import torch.distributed as dist

os.environ.setdefault("NEURON_RT_VIRTUAL_CORE_SIZE", "2")
os.environ.setdefault("NEURON_RT_NUM_CORES", "1")
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
os.environ.setdefault("NEURON_CC_FLAGS", "-O1 --auto-cast=none")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

print("=== Step 4: torch.compile test for WAN 2.2 T2V-A14B ===")
print(f"PyTorch: {torch.__version__}")

# Init process group (required in Beta 3 DLC)
dist.init_process_group(backend="neuron")
print("Process group initialized")

from diffusers import WanTransformer3DModel

# Load transformer on CPU
print("\n[1] Loading WanTransformer3DModel (14B)...")
t0 = time.time()
transformer = WanTransformer3DModel.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
).eval()
n_params = sum(p.numel() for p in transformer.parameters()) / 1e9
print(f"  Loaded in {time.time()-t0:.1f}s ({n_params:.2f}B params)")
print(f"  Config: heads={transformer.config.num_attention_heads}, "
      f"layers={transformer.config.num_layers}, "
      f"in_channels={transformer.config.in_channels}")

# Move to Neuron device
print("\n[2] Moving to device('neuron')...")
t0 = time.time()
try:
    transformer = transformer.to("neuron")
    print(f"  .to('neuron') succeeded in {time.time()-t0:.1f}s")
except Exception as e:
    print(f"  .to('neuron') FAILED: {str(e)[:200]}")
    print("  (Expected: 14B model may OOM on single core)")
    dist.destroy_process_group()
    exit(1)

# torch.compile
print("\n[3] torch.compile(backend='neuron', dynamic=False)...")
try:
    compiled = torch.compile(transformer, backend="neuron", dynamic=False)
    print("  torch.compile registered successfully")
except Exception as e:
    print(f"  torch.compile FAILED: {str(e)[:300]}")
    dist.destroy_process_group()
    exit(1)

# Minimal forward pass (tiny shape to test compilation, not runtime)
print("\n[4] Minimal forward pass (1x16x1x4x4 — tiny test shape)...")
# batch=1, channels=16, frames=1, H=4, W=4 (latent) → seq_len = 1*2*2 = 4 tokens
hidden = torch.randn(1, 16, 1, 4, 4, dtype=torch.bfloat16, device="neuron")
timestep = torch.tensor([500.0], device="neuron")
encoder_hidden = torch.randn(1, 512, 4096, dtype=torch.bfloat16, device="neuron")

t0 = time.time()
try:
    with torch.no_grad():
        output = compiled(
            hidden_states=hidden,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden,
            return_dict=False,
        )
    if isinstance(output, (tuple, list)):
        output = output[0]
    elapsed = time.time() - t0
    print(f"  FORWARD PASS SUCCEEDED in {elapsed:.1f}s!")
    print(f"  Output shape: {output.shape}")
    print(f"  Output device: {output.device}")
    print(f"\n  ✅ torch.compile WORKS for WAN 2.2 T2V-A14B on Neuron!")
    print(f"  (Compilation time includes NEFF generation)")
except RuntimeError as e:
    elapsed = time.time() - t0
    err_str = str(e)
    if "COMPILATION FAILED" in err_str or "exit code 70" in err_str:
        print(f"  ✗ COMPILATION FAILED (exit code 70) in {elapsed:.1f}s")
        print(f"    The graph is too complex for neuronx-cc 2.25")
        print(f"    Use NxDModel approach instead (proven working)")
    elif "NRT tensor allocation" in err_str or "OOM" in err_str.upper():
        print(f"  ✗ OOM at runtime in {elapsed:.1f}s")
        print(f"    Graph compiled but 14B doesn't fit on 1 core (24 GB)")
        print(f"    Need TP>=2 for this model")
        print(f"    ✅ But compilation SUCCEEDED — architecture is compatible!")
    else:
        print(f"  ✗ FORWARD FAILED in {elapsed:.1f}s: {err_str[:500]}")
except Exception as e:
    elapsed = time.time() - t0
    print(f"  ✗ UNEXPECTED ERROR in {elapsed:.1f}s: {str(e)[:500]}")

dist.destroy_process_group()
print("\nDone.")
