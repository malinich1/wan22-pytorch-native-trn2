"""
Validate the fixed text encoder (CP=1, world_size=4) against CPU reference.

Compares embeddings from:
1. CPU text encoder (ground truth)
2. Fixed NxD text encoder (world_size=4)

Success criteria: cosine similarity > 0.99
"""
import os
import sys
import json
import time
import gc

# Set NEURON_RT_NUM_CORES=0 initially to prevent NRT from grabbing cores
# during the transformers import (which triggers torch_xla init)
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_RT_NUM_CORES"] = "4"
os.environ["NEURON_RT_VISIBLE_CORES"] = "0-3"

SAMPLES_DIR = "/home/ubuntu/aws-neuron-samples/torch-neuronx/inference/hf_pretrained_wan2.2_t2v_a14b"
FIXED_COMPILED = "/opt/dlami/nvme/compiled_models_t2v_a14b_fixed"
CACHE_DIR = "/mnt/nvme/wan2.2_t2v_a14b_hf_cache_dir"
HF_SNAPSHOT = "/mnt/nvme/wan2.2_t2v_a14b_hf_cache_dir/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/5be7df9619b54f4e2667b2755bc6a756675b5cd7"

# DO NOT add SAMPLES_DIR to path yet — it triggers torch_neuronx import

import torch
import numpy as np

print("=" * 60)
print("VALIDATE FIXED TEXT ENCODER (CP=1, world_size=4)")
print("=" * 60)
print("VALIDATE FIXED TEXT ENCODER (CP=1, world_size=4)")
print("=" * 60)

prompt = "A beautiful fluffy orange tabby cat walking through a sunlit garden with flowers, cinematic quality, photorealistic, detailed fur"

# --- CPU Reference ---
print("\n[1] CPU text encoding (reference)...")
from transformers import AutoTokenizer, UMT5EncoderModel

tokenizer = AutoTokenizer.from_pretrained(HF_SNAPSHOT, subfolder="tokenizer")
text_inputs = tokenizer(
    prompt, max_length=512, padding="max_length",
    truncation=True, return_attention_mask=True, return_tensors="pt"
)
input_ids = text_inputs.input_ids
attention_mask = text_inputs.attention_mask
print(f"  Tokenized: {input_ids.shape}, non-pad: {attention_mask.sum().item()} tokens")

# Load UMT5 on CPU
t0 = time.time()
cpu_te = UMT5EncoderModel.from_pretrained(
    HF_SNAPSHOT, subfolder="text_encoder", torch_dtype=torch.bfloat16
).eval()
print(f"  CPU text encoder loaded in {time.time()-t0:.1f}s")

with torch.no_grad():
    cpu_out = cpu_te(input_ids=input_ids, attention_mask=attention_mask)
cpu_embeds = cpu_out.last_hidden_state.clone()
print(f"  CPU output: shape={cpu_embeds.shape}, mean={cpu_embeds.float().mean():.6f}, std={cpu_embeds.float().std():.6f}")

# Free CPU model IMMEDIATELY
del cpu_te, cpu_out
gc.collect()
import ctypes
ctypes.CDLL("libc.so.6").malloc_trim(0)
import time as time_mod
time_mod.sleep(2)
print("  CPU model freed")

# --- NxD Fixed Text Encoder ---
print(f"\n[2] NxD text encoder (world_size=4, CP=1)...")
te_path = os.path.join(FIXED_COMPILED, "text_encoder")
config_path = os.path.join(te_path, "config.json")

with open(config_path) as f:
    te_config = json.load(f)
print(f"  Config: {te_config}")

tp_degree = te_config["tp_degree"]
te_world_size = te_config["world_size"]

# Import NxD ONLY after CPU work is done (avoids NRT init conflicts)
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_RT_NUM_CORES"] = "4"
os.environ["NEURON_RT_VISIBLE_CORES"] = "0-3"
import torch_neuronx
from neuronx_distributed import NxDModel
from safetensors.torch import load_file

# Load NxDModel
t0 = time.time()
nxd_te = NxDModel.load(
    os.path.join(te_path, "nxd_model.pt"),
    start_rank=0, local_ranks_size=te_world_size
)

# Load weights (TP=4 only, no CP expansion since world_size=4)
te_weights = []
weights_path = os.path.join(te_path, "weights")
for rank in range(tp_degree):
    ckpt = load_file(os.path.join(weights_path, f"tp{rank}_sharded_checkpoint.safetensors"))
    ckpt = {k: v for k, v in ckpt.items() if 'master_weight' not in k}
    te_weights.append(ckpt)

# If world_size == tp_degree, use TP weights directly (no CP expansion)
if te_world_size == tp_degree:
    print(f"  Loading with TP={tp_degree} weights directly (no CP expansion)")
    nxd_te.set_weights(te_weights)
else:
    # CP expansion needed
    cp_degree = te_world_size // tp_degree
    print(f"  Loading with CP expansion (CP={cp_degree})")
    cp_weights = []
    for cp_rank in range(cp_degree):
        for tp_rank in range(tp_degree):
            world_rank = cp_rank * tp_degree + tp_rank
            ckpt = {k: v.clone() for k, v in te_weights[tp_rank].items()}
            cp_weights.append(ckpt)
    nxd_te.set_weights(cp_weights)

nxd_te.to_neuron()
print(f"  NxD text encoder loaded in {time.time()-t0:.1f}s")

# Run NxD text encoder
print("  Running NxD forward pass...")
t0 = time.time()
nxd_out = nxd_te(input_ids, attention_mask)
if isinstance(nxd_out, (tuple, list)):
    nxd_embeds = nxd_out[0]
else:
    nxd_embeds = nxd_out
nxd_time = time.time() - t0
print(f"  NxD output: shape={nxd_embeds.shape}, time={nxd_time:.3f}s")
print(f"  NxD stats: mean={nxd_embeds.float().mean():.6f}, std={nxd_embeds.float().std():.6f}")

# --- Comparison ---
print(f"\n[3] Comparison...")

if cpu_embeds.shape != nxd_embeds.shape:
    print(f"  ✗ SHAPE MISMATCH: CPU={cpu_embeds.shape} vs NxD={nxd_embeds.shape}")
    print(f"    This means the NEFF graph has world_size=8 baked in (produces different output shape)")
    print(f"    Strategy A (config patch) FAILED. Need full recompile (Strategy B).")
    sys.exit(1)

diff = (cpu_embeds.float() - nxd_embeds.float()).abs()
cos_sim = torch.nn.functional.cosine_similarity(
    cpu_embeds.flatten().float(), nxd_embeds.flatten().float(), dim=0
).item()

print(f"  Max diff:          {diff.max():.6f}")
print(f"  Mean diff:         {diff.mean():.6f}")
print(f"  Cosine similarity: {cos_sim:.6f}")

# Per-position analysis (first 10 non-padding positions)
non_pad = attention_mask.sum().item()
pos_cos = []
for p in range(min(non_pad, 10)):
    pc = torch.nn.functional.cosine_similarity(
        cpu_embeds[0, p].float().unsqueeze(0),
        nxd_embeds[0, p].float().unsqueeze(0), dim=1
    ).item()
    pos_cos.append(pc)
print(f"  Per-position cosine (first {len(pos_cos)}): min={min(pos_cos):.4f}, mean={np.mean(pos_cos):.4f}")

print(f"\n{'=' * 60}")
if cos_sim > 0.99:
    print(f"✅ FIXED TEXT ENCODER PRODUCES CORRECT EMBEDDINGS (cosine={cos_sim:.4f})")
    print(f"   Strategy A (config patch) WORKS!")
    print(f"   Proceed to run_wan22_nxd_fixed_te.py for full inference")
elif cos_sim > 0.90:
    print(f"⚠️  PARTIAL MATCH (cosine={cos_sim:.4f})")
    print(f"   Embeddings are somewhat correct but precision is degraded")
    print(f"   May produce usable but imperfect video")
else:
    print(f"✗ MISMATCH (cosine={cos_sim:.4f})")
    print(f"   Strategy A (config patch) FAILED")
    print(f"   The world_size is baked into the compiled NEFF graph")
    print(f"   Need Strategy B: full recompile with --world_size 4")
    print(f"   Run:")
    print(f"     cd {SAMPLES_DIR}")
    print(f"     PYTHONPATH={SAMPLES_DIR}:$PYTHONPATH python neuron_wan2_2_t2v_a14b/compile_text_encoder.py \\")
    print(f"       --max_sequence_length 512 --tp_degree 4 --world_size 4 \\")
    print(f"       --compiled_models_dir {FIXED_COMPILED} --cache_dir {CACHE_DIR}")
print("=" * 60)
