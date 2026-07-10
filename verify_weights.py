"""
Verify that compiled NxD model weights match the original HuggingFace model weights.
Compare a few key tensors between:
1. Original HF model weights
2. Sharded weights loaded by the inference script
"""
import os
os.environ["NEURON_RT_NUM_CORES"] = "1"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_RT_VISIBLE_CORES"] = "0"

import torch
import numpy as np
from safetensors.torch import load_file

# Load original HF model weights
hf_path = "/mnt/nvme/wan2.2_t2v_a14b_hf_cache_dir/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/5be7df9619b54f4e2667b2755bc6a756675b5cd7"
print("Loading original HF transformer weights...")
hf_weights = {}
transformer_path = os.path.join(hf_path, "transformer")
for f in sorted(os.listdir(transformer_path)):
    if f.endswith('.safetensors'):
        w = load_file(os.path.join(transformer_path, f))
        hf_weights.update(w)
        print(f"  {f}: {len(w)} tensors")
print(f"Total HF transformer tensors: {len(hf_weights)}")

# Load sharded weights from compiled model
compiled_path = "/opt/dlami/nvme/compiled_models_t2v_a14b/transformer/weights"
print(f"\nLoading sharded weights from {compiled_path}...")
sharded_weights = []
tp_degree = 4
for rank in range(tp_degree):
    f = os.path.join(compiled_path, f"tp{rank}_sharded_checkpoint.safetensors")
    w = load_file(f)
    sharded_weights.append(w)
    if rank == 0:
        print(f"  tp0 keys (first 10): {list(w.keys())[:10]}")
        print(f"  tp0 total tensors: {len(w)}")

# Compare specific weights
print("\n=== WEIGHT COMPARISON ===")

# Key to check: first block's self-attention Q projection weight
# In HF: blocks.0.attn1.to_q.weight -> sharded across TP
hf_key = "blocks.0.attn1.to_q.weight"
nxd_key = "transformer.blocks.0.attn1.to_q.weight"

if hf_key in hf_weights:
    hf_w = hf_weights[hf_key]
    print(f"\nHF {hf_key}: shape={hf_w.shape}, dtype={hf_w.dtype}")
    print(f"  range=[{hf_w.min():.6f}, {hf_w.max():.6f}], mean={hf_w.mean():.6f}")
    
    # Reconstruct from sharded
    if nxd_key in sharded_weights[0]:
        shards = [sharded_weights[r][nxd_key] for r in range(tp_degree)]
        print(f"  NxD shard shape: {shards[0].shape} (per rank)")
        reconstructed = torch.cat(shards, dim=0)  # Column parallel: cat on dim 0
        print(f"  Reconstructed: shape={reconstructed.shape}")
        print(f"  range=[{reconstructed.min():.6f}, {reconstructed.max():.6f}], mean={reconstructed.mean():.6f}")
        
        # Check if they match
        if hf_w.shape == reconstructed.shape:
            diff = (hf_w.float() - reconstructed.float()).abs()
            print(f"  Max diff: {diff.max():.8f}")
            print(f"  Mean diff: {diff.mean():.8f}")
            if diff.max() < 0.01:
                print(f"  ✓ MATCH (within bf16 precision)")
            else:
                print(f"  ✗ MISMATCH!")
        else:
            print(f"  Shape mismatch: HF={hf_w.shape} vs Reconstructed={reconstructed.shape}")
    else:
        print(f"  NxD key '{nxd_key}' not found!")
        # Try finding similar keys
        similar = [k for k in sharded_weights[0].keys() if 'blocks.0.attn1' in k]
        print(f"  Similar keys in NxD: {similar[:5]}")
else:
    print(f"HF key '{hf_key}' not found!")
    similar = [k for k in hf_weights.keys() if 'blocks.0.attn1' in k]
    print(f"  Similar keys in HF: {similar[:5]}")

# Also check condition_embedder (not sharded)
hf_key2 = "condition_embedder.time_embedder.linear_1.weight"
nxd_key2 = "transformer.condition_embedder.time_embedder.linear_1.weight"
print(f"\nHF {hf_key2}:")
if hf_key2 in hf_weights:
    hf_w2 = hf_weights[hf_key2]
    print(f"  shape={hf_w2.shape}, range=[{hf_w2.min():.6f}, {hf_w2.max():.6f}]")
    if nxd_key2 in sharded_weights[0]:
        nxd_w2 = sharded_weights[0][nxd_key2]
        print(f"  NxD: shape={nxd_w2.shape}, range=[{nxd_w2.min():.6f}, {nxd_w2.max():.6f}]")
        diff2 = (hf_w2.float() - nxd_w2.float()).abs()
        print(f"  Max diff: {diff2.max():.8f}")
        if diff2.max() < 0.01:
            print(f"  ✓ MATCH")
        else:
            print(f"  ✗ MISMATCH!")

# Check norm weights
hf_key3 = "blocks.0.norm1.weight"
nxd_key3 = "transformer.blocks.0.norm1.weight"
print(f"\nHF {hf_key3}:")
if hf_key3 in hf_weights:
    hf_w3 = hf_weights[hf_key3]
    print(f"  shape={hf_w3.shape}, range=[{hf_w3.min():.6f}, {hf_w3.max():.6f}]")
    if nxd_key3 in sharded_weights[0]:
        nxd_w3 = sharded_weights[0][nxd_key3]
        print(f"  NxD: shape={nxd_w3.shape}, range=[{nxd_w3.min():.6f}, {nxd_w3.max():.6f}]")
        diff3 = (hf_w3.float() - nxd_w3.float()).abs()
        print(f"  Max diff: {diff3.max():.8f}")
        if diff3.max() < 0.01:
            print(f"  ✓ MATCH")
        else:
            print(f"  ✗ MISMATCH!")

print("\nDone!")
