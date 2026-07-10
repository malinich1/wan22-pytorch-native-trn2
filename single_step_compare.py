"""
Single-step comparison: run the SAME input through both:
1. Official HF WanTransformer3DModel (CPU) 
2. NxDModel compiled transformer (Neuron)

If outputs differ significantly, the bug is in compilation/execution.
We use a small 256x256x1 frame to make CPU feasible.
"""
import os
os.environ["NEURON_RT_NUM_CORES"] = "8"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_RT_VISIBLE_CORES"] = "0-7"

import torch
import time
import json
import gc
import numpy as np

# Load the official transformer on CPU
print("Loading official WanTransformer3DModel on CPU...")
from diffusers import WanTransformer3DModel
cache_dir = "/mnt/nvme/wan2.2_t2v_a14b_hf_cache_dir"
hf_path = "/mnt/nvme/wan2.2_t2v_a14b_hf_cache_dir/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/5be7df9619b54f4e2667b2755bc6a756675b5cd7"

transformer_cpu = WanTransformer3DModel.from_pretrained(
    hf_path, subfolder="transformer", torch_dtype=torch.float32
).eval()
print(f"  Loaded. Config: heads={transformer_cpu.config.num_attention_heads}, layers={transformer_cpu.config.num_layers}")

# Create test inputs matching 480x832x81 (the compiled resolution)
# Latent shape: [1, 16, 21, 60, 104]
print("\nCreating test inputs (480x832x81 latent space)...")
torch.manual_seed(123)
hidden_states = torch.randn(1, 16, 21, 60, 104, dtype=torch.float32)
timestep = torch.tensor([999.0], dtype=torch.float32)  # High noise timestep

# Create a simple text embedding
encoder_hidden_states = torch.randn(1, 512, 4096, dtype=torch.float32)

# Compute RoPE using the model's own RoPE module
print("Computing RoPE...")
rope_output = transformer_cpu.rope(hidden_states)
rotary_emb_cos, rotary_emb_sin = rope_output
print(f"  RoPE cos: {rotary_emb_cos.shape}, sin: {rotary_emb_sin.shape}")

# Run official model on CPU (single step)
print("\nRunning official transformer (CPU)...")
t0 = time.time()
with torch.no_grad():
    cpu_output = transformer_cpu(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        return_dict=False,
    )[0]
cpu_time = time.time() - t0
print(f"  CPU output: shape={cpu_output.shape}, time={cpu_time:.1f}s")
print(f"  range=[{cpu_output.min():.4f}, {cpu_output.max():.4f}], mean={cpu_output.mean():.4f}, std={cpu_output.std():.4f}")

# Save CPU output for comparison
torch.save({
    "hidden_states": hidden_states,
    "timestep": timestep,
    "encoder_hidden_states": encoder_hidden_states,
    "rotary_emb_cos": rotary_emb_cos,
    "rotary_emb_sin": rotary_emb_sin,
    "cpu_output": cpu_output,
}, "/mnt/nvme/single_step_inputs.pt")
print("Saved inputs + CPU output to /mnt/nvme/single_step_inputs.pt")

# Free CPU transformer memory
del transformer_cpu
gc.collect()

# Now run NxDModel
print("\n\nRunning NxDModel compiled transformer (Neuron)...")
from neuronx_distributed import NxDModel
from safetensors.torch import load_file

compiled_path = "/opt/dlami/nvme/compiled_models_t2v_a14b/transformer"
config_file = os.path.join(compiled_path, "config.json")
with open(config_file) as f:
    config = json.load(f)

tp_degree = config["tp_degree"]
cp_degree = config["cp_degree"]
world_size = config["world_size"]

# Load RoPE cache (same as what inference script uses)
rope_cache = torch.load(os.path.join(compiled_path, "rope_cache.pt"), weights_only=True)
nxd_rope_cos = rope_cache["rotary_emb_cos"]  # bf16
nxd_rope_sin = rope_cache["rotary_emb_sin"]  # bf16

print(f"  NxD RoPE cos: {nxd_rope_cos.shape}, dtype={nxd_rope_cos.dtype}")
print(f"  Official RoPE cos: {rotary_emb_cos.shape}, dtype={rotary_emb_cos.dtype}")

# Compare RoPE values
rope_diff = (rotary_emb_cos.to(torch.bfloat16).float() - nxd_rope_cos.float()).abs().max()
print(f"  RoPE cos max diff: {rope_diff:.6f}")
if rope_diff < 0.01:
    print("  ✓ RoPE MATCH")
else:
    print("  ✗ RoPE MISMATCH - THIS IS THE BUG!")

# Load weights
weights_path = os.path.join(compiled_path, "weights")
tp_checkpoints = []
for rank in range(tp_degree):
    ckpt_path = os.path.join(weights_path, f"tp{rank}_sharded_checkpoint.safetensors")
    raw_ckpt = load_file(ckpt_path)
    ckpt = {k: v for k, v in raw_ckpt.items() if 'master_weight' not in k}
    tp_checkpoints.append(ckpt)

# Prepare CP checkpoints
cp_checkpoints = []
for cp_rank in range(cp_degree):
    for tp_rank in range(tp_degree):
        world_rank = cp_rank * tp_degree + tp_rank
        ckpt = {k: v.clone() for k, v in tp_checkpoints[tp_rank].items()}
        ckpt["transformer.global_rank.rank"] = torch.tensor([world_rank], dtype=torch.int32)
        cp_checkpoints.append(ckpt)

# Load NxDModel
nxd_model_path = os.path.join(compiled_path, "nxd_model.pt")
print(f"  Loading NxDModel (TP={tp_degree}, CP={cp_degree}, world_size={world_size})...")
t0 = time.time()
nxd_model = NxDModel.load(nxd_model_path, start_rank=0, local_ranks_size=world_size)
nxd_model.set_weights(cp_checkpoints)
nxd_model.to_neuron()
print(f"  NxDModel loaded in {time.time()-t0:.1f}s")

# Run NxDModel with SAME inputs
print("  Running NxDModel forward...")
t0 = time.time()
nxd_output = nxd_model(
    hidden_states.to(torch.bfloat16),
    timestep,
    encoder_hidden_states.to(torch.bfloat16),
    nxd_rope_cos,
    nxd_rope_sin,
)
if isinstance(nxd_output, (tuple, list)):
    nxd_output = nxd_output[0]
nxd_time = time.time() - t0
print(f"  NxD output: shape={nxd_output.shape}, time={nxd_time:.1f}s")
print(f"  range=[{nxd_output.min():.4f}, {nxd_output.max():.4f}], mean={nxd_output.mean():.4f}, std={nxd_output.std():.4f}")

# Compare outputs
print("\n=== OUTPUT COMPARISON ===")
print(f"CPU output: mean={cpu_output.mean():.4f}, std={cpu_output.std():.4f}, range=[{cpu_output.min():.4f}, {cpu_output.max():.4f}]")
print(f"NxD output: mean={nxd_output.mean():.4f}, std={nxd_output.std():.4f}, range=[{nxd_output.min():.4f}, {nxd_output.max():.4f}]")

output_diff = (cpu_output.float() - nxd_output.float()).abs()
print(f"Max diff: {output_diff.max():.4f}")
print(f"Mean diff: {output_diff.mean():.4f}")
print(f"Cosine similarity: {torch.nn.functional.cosine_similarity(cpu_output.flatten().float(), nxd_output.flatten().float(), dim=0):.4f}")

if output_diff.max() < 1.0 and output_diff.mean() < 0.1:
    print("\n✓ OUTPUTS MATCH (within bf16 tolerance)")
else:
    print("\n✗ OUTPUTS DO NOT MATCH - BUG CONFIRMED IN COMPILED MODEL")
