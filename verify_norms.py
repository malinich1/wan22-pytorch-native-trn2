"""Verify norm weights and scale_shift_table match between HF and NxD."""
import os
os.environ["NEURON_RT_NUM_CORES"] = "1"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_RT_VISIBLE_CORES"] = "0"

from safetensors.torch import load_file
import torch

hf_path = '/mnt/nvme/wan2.2_t2v_a14b_hf_cache_dir/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/5be7df9619b54f4e2667b2755bc6a756675b5cd7/transformer'
nxd_path = '/opt/dlami/nvme/compiled_models_t2v_a14b/transformer/weights'

# Load all HF weights
hf_weights = {}
for f in sorted(os.listdir(hf_path)):
    if f.endswith('.safetensors'):
        hf_weights.update(load_file(os.path.join(hf_path, f)))

# Load NxD tp0 weights
nxd_w = load_file(os.path.join(nxd_path, 'tp0_sharded_checkpoint.safetensors'))

# Check norm weights (should be full size, not sharded)
checks = [
    ('blocks.0.norm1.weight', 'transformer.blocks.0.norm1.weight'),
    ('blocks.0.norm2.weight', 'transformer.blocks.0.norm2.weight'),
    ('blocks.0.norm3.weight', 'transformer.blocks.0.norm3.weight'),
    ('blocks.0.scale_shift_table', 'transformer.blocks.0.scale_shift_table'),
    ('norm_out.weight', 'transformer.norm_out.weight'),
    ('scale_shift_table', 'transformer.scale_shift_table'),
    ('patch_embedding.proj.weight', 'transformer.patch_embedding.proj.weight'),
    ('proj_out.weight', 'transformer.proj_out.weight'),
]

print("=== WEIGHT VERIFICATION ===\n")
all_match = True
for hf_key, nxd_key in checks:
    if hf_key not in hf_weights:
        print(f"SKIP: {hf_key} not in HF weights")
        continue
    if nxd_key not in nxd_w:
        print(f"MISSING: {nxd_key} not in NxD weights!")
        all_match = False
        continue
    
    hf_t = hf_weights[hf_key]
    nxd_t = nxd_w[nxd_key]
    
    if hf_t.shape != nxd_t.shape:
        print(f"SHAPE MISMATCH: {hf_key}: HF={hf_t.shape} vs NxD={nxd_t.shape}")
        all_match = False
        continue
    
    diff = (hf_t.float() - nxd_t.float()).abs().max().item()
    status = "✓" if diff < 0.01 else "✗"
    if diff >= 0.01:
        all_match = False
    print(f"{status} {hf_key}: shape={hf_t.shape}, max_diff={diff:.6f}")

print(f"\n{'ALL WEIGHTS MATCH' if all_match else 'SOME WEIGHTS DO NOT MATCH'}")

# Also check: what's the RoPE cache shape?
rope = torch.load('/opt/dlami/nvme/compiled_models_t2v_a14b/transformer/rope_cache.pt', weights_only=True)
print(f"\nRoPE cache:")
print(f"  cos: {rope['rotary_emb_cos'].shape}, dtype={rope['rotary_emb_cos'].dtype}")
print(f"  sin: {rope['rotary_emb_sin'].shape}, dtype={rope['rotary_emb_sin'].dtype}")
print(f"  cos range: [{rope['rotary_emb_cos'].min():.4f}, {rope['rotary_emb_cos'].max():.4f}]")
