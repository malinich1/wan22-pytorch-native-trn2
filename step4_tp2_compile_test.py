"""
Runbook Step 4: TP=2 torch.compile test for WAN 2.2 T2V-A14B (14B model).

Shards the transformer across 2 NeuronCores using manual tensor parallelism,
then applies torch.compile(backend="neuron") on each rank's subgraph.

Sharding strategy:
  - QKV projections (to_q, to_k, to_v): Column parallel (shard output dim)
  - Output projections (to_out.0): Row parallel (shard input dim)
  - FFN gate/up (net.0.proj): Column parallel
  - FFN down (net.2): Row parallel
  - Norms, embeddings, patch_embedding: Replicated (full copy on each rank)

Launch:
  torchrun --nproc-per-node 2 step4_tp2_compile_test.py

Expected memory per core (bf16):
  - Sharded params: ~14 GB (half of 28 GB total)
  - Replicated params: ~2 GB (norms, embedders, patch_embed)
  - Activations: ~4-8 GB (depends on seq_len)
  - Total: ~20-24 GB (tight fit in 24 GB/core with LNC=2)
"""
import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn

os.environ.setdefault("NEURON_CC_FLAGS", "-O1 --auto-cast=none")
os.environ.setdefault("NEURON_RT_VIRTUAL_CORE_SIZE", "2")
os.environ.setdefault("NEURON_RT_NUM_CORES", "2")

print(f"=== Step 4: TP=2 torch.compile for WAN 2.2 T2V-A14B ===")
print(f"PyTorch: {torch.__version__}")

# Initialize distributed
dist.init_process_group(backend="neuron")
rank = dist.get_rank()
world_size = dist.get_world_size()
print(f"[Rank {rank}/{world_size}] Initialized")

from diffusers import WanTransformer3DModel

# Load full model on CPU (all ranks load independently)
if rank == 0:
    print(f"\n[1] Loading WanTransformer3DModel on CPU...")
t0 = time.time()
transformer = WanTransformer3DModel.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    cache_dir="/mnt/nvme/wan2.2_t2v_a14b_hf_cache_dir",
).eval()
if rank == 0:
    n_params = sum(p.numel() for p in transformer.parameters()) / 1e9
    print(f"  Loaded in {time.time()-t0:.1f}s ({n_params:.2f}B params)")
    print(f"  Config: heads={transformer.config.num_attention_heads}, "
          f"layers={transformer.config.num_layers}, "
          f"in_channels={transformer.config.in_channels}, "
          f"hidden={transformer.config.num_attention_heads * transformer.config.attention_head_dim}")

# ============================================================
# [2] Shard weights for TP=2
# ============================================================
if rank == 0:
    print(f"\n[2] Sharding model for TP={world_size}...")

t0 = time.time()
sharded_count = 0
replicated_count = 0

with torch.no_grad():
    for name, param in list(transformer.named_parameters()):
        # Column parallel: shard output dimension (dim=0)
        # These are QKV projections and FFN up/gate
        if any(k in name for k in ['to_q.weight', 'to_k.weight', 'to_v.weight',
                                     'to_q.bias', 'to_k.bias', 'to_v.bias',
                                     'net.0.proj.weight', 'net.0.proj.bias',
                                     'norm_q.weight', 'norm_k.weight']):
            dim = 0
            chunk_size = param.shape[dim] // world_size
            param.data = param.data.narrow(dim, rank * chunk_size, chunk_size).contiguous()
            sharded_count += 1

        # Row parallel: shard input dimension (dim=1 for weight, keep bias full on rank 0)
        # These are output projections and FFN down
        elif any(k in name for k in ['to_out.0.weight', 'net.2.weight']):
            dim = 1
            chunk_size = param.shape[dim] // world_size
            param.data = param.data.narrow(dim, rank * chunk_size, chunk_size).contiguous()
            sharded_count += 1

        # Row parallel bias: only rank 0 keeps it, others zero
        elif any(k in name for k in ['to_out.0.bias', 'net.2.bias']):
            if rank != 0:
                param.data = torch.zeros_like(param.data)
            replicated_count += 1

        # Everything else: replicated (norms, embeddings, condition_embedder, etc.)
        else:
            replicated_count += 1

if rank == 0:
    total_sharded_gb = sum(
        p.numel() * p.element_size() for n, p in transformer.named_parameters()
        if any(k in n for k in ['to_q', 'to_k', 'to_v', 'to_out.0', 'net.0.proj', 'net.2'])
    ) / 1e9
    total_replicated_gb = sum(
        p.numel() * p.element_size() for n, p in transformer.named_parameters()
        if not any(k in n for k in ['to_q', 'to_k', 'to_v', 'to_out.0', 'net.0.proj', 'net.2'])
    ) / 1e9
    total_per_rank_gb = total_sharded_gb + total_replicated_gb
    print(f"  Sharded: {sharded_count} tensors ({total_sharded_gb:.1f} GB per rank)")
    print(f"  Replicated: {replicated_count} tensors ({total_replicated_gb:.1f} GB per rank)")
    print(f"  Total per rank: {total_per_rank_gb:.1f} GB")
    print(f"  Sharding done in {time.time()-t0:.1f}s")

# ============================================================
# [3] Move to Neuron device
# ============================================================
if rank == 0:
    print(f"\n[3] Moving sharded model to device('neuron')...")
t0 = time.time()
try:
    transformer = transformer.to("neuron")
    if rank == 0:
        print(f"  .to('neuron') succeeded in {time.time()-t0:.1f}s")
except RuntimeError as e:
    if rank == 0:
        print(f"  ✗ .to('neuron') FAILED (likely OOM): {str(e)[:300]}")
    dist.destroy_process_group()
    exit(1)

# ============================================================
# [4] torch.compile
# ============================================================
if rank == 0:
    print(f"\n[4] torch.compile(backend='neuron', dynamic=False)...")
try:
    compiled = torch.compile(transformer, backend="neuron", dynamic=False)
    if rank == 0:
        print(f"  torch.compile registered")
except Exception as e:
    if rank == 0:
        print(f"  ✗ torch.compile FAILED: {str(e)[:300]}")
    dist.destroy_process_group()
    exit(1)

# ============================================================
# [5] Forward pass test (tiny shape)
# ============================================================
if rank == 0:
    print(f"\n[5] Forward pass test (tiny: 1x16x1x8x8)...")

# Tiny input: batch=1, ch=16, frames=1, H=8, W=8 latent
hidden = torch.randn(1, 16, 1, 8, 8, dtype=torch.bfloat16, device="neuron")
timestep = torch.tensor([999.0], device="neuron")
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

    # All-reduce to combine partial results from row-parallel layers
    dist.all_reduce(output, op=dist.ReduceOp.SUM)

    elapsed = time.time() - t0
    if rank == 0:
        print(f"  ✅ FORWARD PASS SUCCEEDED in {elapsed:.1f}s")
        print(f"  Output shape: {output.shape}, device: {output.device}")
        print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
        print(f"\n  torch.compile + TP=2 WORKS for WAN 2.2 T2V-A14B!")

except RuntimeError as e:
    elapsed = time.time() - t0
    err = str(e)
    if rank == 0:
        if "COMPILATION FAILED" in err or "exit code 70" in err:
            print(f"  ✗ COMPILATION FAILED (neuronx-cc crash) in {elapsed:.1f}s")
            print(f"    Graph too complex for neuronx-cc 2.25")
        elif "NRT tensor allocation" in err or "oom" in err.lower():
            print(f"  ✗ OOM at runtime in {elapsed:.1f}s")
            print(f"    Sharded model still too large for 24 GB/core")
            print(f"    Need TP=4 or reduce activation memory")
        else:
            print(f"  ✗ FAILED in {elapsed:.1f}s: {err[:500]}")

except Exception as e:
    if rank == 0:
        print(f"  ✗ UNEXPECTED: {str(e)[:500]}")

dist.barrier()
dist.destroy_process_group()
if rank == 0:
    print("\nDone.")
