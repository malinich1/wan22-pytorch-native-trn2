"""Compare NxD text encoder (world_size=4) against saved CPU reference."""
import os
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_RT_NUM_CORES"] = "4"
os.environ["NEURON_RT_VISIBLE_CORES"] = "0-3"

import torch
import json
import time
import torch_neuronx
from neuronx_distributed import NxDModel
from safetensors.torch import load_file

TE_PATH = "/opt/dlami/nvme/compiled_models_t2v_a14b_fixed/text_encoder"

# Load CPU reference
ref = torch.load("/mnt/nvme/cpu_te_reference.pt", weights_only=False)
cpu_embeds = ref["embeds"]
input_ids = ref["ids"]
attention_mask = ref["mask"]
print(f"CPU ref: {cpu_embeds.shape}, mean={cpu_embeds.float().mean():.6f}")

# Load NxD text encoder
with open(os.path.join(TE_PATH, "config.json")) as f:
    cfg = json.load(f)
print(f"NxD config: {cfg}")

t0 = time.time()
nxd_te = NxDModel.load(os.path.join(TE_PATH, "nxd_model.pt"), start_rank=0, local_ranks_size=cfg["world_size"])
weights = []
for r in range(cfg["tp_degree"]):
    w = load_file(os.path.join(TE_PATH, "weights", f"tp{r}_sharded_checkpoint.safetensors"))
    weights.append({k: v for k, v in w.items() if "master_weight" not in k})
nxd_te.set_weights(weights)
nxd_te.to_neuron()
print(f"NxD loaded in {time.time()-t0:.1f}s")

# Run
nxd_out = nxd_te(input_ids, attention_mask)
if isinstance(nxd_out, dict):
    nxd_embeds = nxd_out.get("last_hidden_state", list(nxd_out.values())[0])
elif isinstance(nxd_out, (tuple, list)):
    nxd_embeds = nxd_out[0]
else:
    nxd_embeds = nxd_out
print(f"NxD output: {nxd_embeds.shape}, mean={nxd_embeds.float().mean():.6f}")

# Compare
if cpu_embeds.shape != nxd_embeds.shape:
    print(f"\nSHAPE MISMATCH: CPU={cpu_embeds.shape} vs NxD={nxd_embeds.shape}")
    print("Strategy A FAILED — NEFF has world_size=8 baked in")
    exit(1)

cos = torch.nn.functional.cosine_similarity(
    cpu_embeds.flatten().float(), nxd_embeds.flatten().float(), dim=0
).item()
diff = (cpu_embeds.float() - nxd_embeds.float()).abs()
print(f"\nCosine similarity: {cos:.6f}")
print(f"Max diff: {diff.max():.6f}")
print(f"Mean diff: {diff.mean():.6f}")

if cos > 0.99:
    print(f"\n✅ SUCCESS — NxD text encoder (world_size=4) MATCHES CPU! cosine={cos:.4f}")
elif cos > 0.90:
    print(f"\n⚠️ Partial match: cosine={cos:.4f}")
else:
    print(f"\n✗ MISMATCH: cosine={cos:.4f} — Strategy A failed, need full recompile")
