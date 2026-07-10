"""Debug: inspect NxD text encoder output structure and compare with CPU."""
import os
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_RT_NUM_CORES"] = "4"
os.environ["NEURON_RT_VISIBLE_CORES"] = "0-3"

import torch
import json
import torch_neuronx
from neuronx_distributed import NxDModel
from safetensors.torch import load_file

TE_PATH = "/opt/dlami/nvme/compiled_models_t2v_a14b_fixed/text_encoder"

# Load CPU reference
ref = torch.load("/mnt/nvme/cpu_te_reference.pt", weights_only=False)
cpu_embeds = ref["embeds"]
input_ids = ref["ids"]
mask = ref["mask"]
print(f"CPU ref: shape={cpu_embeds.shape}, mean={cpu_embeds.float().mean():.6f}")

# Load NxD model
with open(os.path.join(TE_PATH, "config.json")) as f:
    cfg = json.load(f)
print(f"Config: {cfg}")

nxd_te = NxDModel.load(os.path.join(TE_PATH, "nxd_model.pt"), start_rank=0, local_ranks_size=4)
weights = []
for r in range(4):
    w = load_file(os.path.join(TE_PATH, "weights", f"tp{r}_sharded_checkpoint.safetensors"))
    weights.append({k: v for k, v in w.items() if "master_weight" not in k})
nxd_te.set_weights(weights)
nxd_te.to_neuron()
print("NxD loaded")

# Run
out = nxd_te(input_ids, mask)

# Inspect output structure
print(f"\nOutput type: {type(out)}")
if isinstance(out, dict):
    print(f"  Keys: {list(out.keys())}")
    for k, v in out.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, mean={v.float().mean():.6f}")
elif isinstance(out, (tuple, list)):
    print(f"  Length: {len(out)}")
    for i, v in enumerate(out):
        if hasattr(v, "shape"):
            print(f"  [{i}]: shape={v.shape}, dtype={v.dtype}, mean={v.float().mean():.6f}")
        elif v is None:
            print(f"  [{i}]: None")
else:
    print(f"  shape={out.shape}, mean={out.float().mean():.6f}")

# Get the main output tensor
if isinstance(out, dict):
    nxd_embeds = list(out.values())[0]
elif isinstance(out, (tuple, list)):
    nxd_embeds = out[0]
else:
    nxd_embeds = out

print(f"\nNxD embeds: shape={nxd_embeds.shape}, mean={nxd_embeds.float().mean():.6f}, std={nxd_embeds.float().std():.6f}")
print(f"CPU embeds: shape={cpu_embeds.shape}, mean={cpu_embeds.float().mean():.6f}, std={cpu_embeds.float().std():.6f}")

# Detailed comparison
print(f"\nCPU [0,0,:10]: {cpu_embeds[0,0,:10].float().tolist()}")
print(f"NxD [0,0,:10]: {nxd_embeds[0,0,:10].float().tolist()}")
print(f"\nCPU [0,5,:10]: {cpu_embeds[0,5,:10].float().tolist()}")
print(f"NxD [0,5,:10]: {nxd_embeds[0,5,:10].float().tolist()}")

# Check if it's a gather issue (every 4th value matches)
cos_full = torch.nn.functional.cosine_similarity(
    cpu_embeds.flatten().float(), nxd_embeds.flatten().float(), dim=0
).item()
print(f"\nFull cosine: {cos_full:.4f}")

# Check if NxD output is just rank 0's portion repeated 4x
quarter = cpu_embeds.shape[-1] // 4  # 1024
for i in range(4):
    chunk_cos = torch.nn.functional.cosine_similarity(
        cpu_embeds[0, :, i*quarter:(i+1)*quarter].flatten().float(),
        nxd_embeds[0, :, i*quarter:(i+1)*quarter].flatten().float(),
        dim=0
    ).item()
    print(f"  Quarter {i} (dims {i*quarter}-{(i+1)*quarter}): cosine={chunk_cos:.4f}")
