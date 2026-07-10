"""Simple validation: CPU TE vs NxD TE (world_size=4) in one process."""
import os
# Don't set NEURON env vars at the top — let NRT init with defaults
# We'll set them later before NxDModel loading
# os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
# os.environ["NEURON_RT_NUM_CORES"] = "4"
# os.environ["NEURON_RT_VISIBLE_CORES"] = "0-3"

import torch
import json
import time
import numpy as np

HF = "/mnt/nvme/wan2.2_t2v_a14b_hf_cache_dir/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/5be7df9619b54f4e2667b2755bc6a756675b5cd7"
TE_PATH = "/opt/dlami/nvme/compiled_models_t2v_a14b_fixed/text_encoder"

prompt = "A beautiful fluffy orange tabby cat"

# [1] CPU text encoding
print("[1] CPU text encoding...")
from transformers import AutoTokenizer, UMT5EncoderModel
tokenizer = AutoTokenizer.from_pretrained(HF, subfolder="tokenizer")
ids = tokenizer(prompt, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
print(f"  Tokens: {ids.input_ids.shape}, non-pad: {ids.attention_mask.sum().item()}")

te = UMT5EncoderModel.from_pretrained(HF, subfolder="text_encoder", torch_dtype=torch.bfloat16).eval()
with torch.no_grad():
    cpu_out = te(input_ids=ids.input_ids, attention_mask=ids.attention_mask).last_hidden_state.clone()
print(f"  CPU output: {cpu_out.shape}, mean={cpu_out.float().mean():.6f}, std={cpu_out.float().std():.6f}")
del te
import gc; gc.collect()

# [2] NxD text encoder (fixed, world_size=4)
print("\n[2] NxD text encoder (world_size=4)...")

# NOW set Neuron env vars before NxD loading
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_RT_NUM_CORES"] = "4"
os.environ["NEURON_RT_VISIBLE_CORES"] = "0-3"

# Lazy import to avoid NRT init during CPU work
import importlib
torch_neuronx = importlib.import_module("torch_neuronx")
NxDModel = importlib.import_module("neuronx_distributed").NxDModel
load_file = importlib.import_module("safetensors.torch").load_file

with open(os.path.join(TE_PATH, "config.json")) as f:
    cfg = json.load(f)
print(f"  Config: {cfg}")

t0 = time.time()
nxd_te = NxDModel.load(os.path.join(TE_PATH, "nxd_model.pt"), start_rank=0, local_ranks_size=cfg["world_size"])
weights = []
for r in range(cfg["tp_degree"]):
    w = load_file(os.path.join(TE_PATH, "weights", f"tp{r}_sharded_checkpoint.safetensors"))
    weights.append({k: v for k, v in w.items() if "master_weight" not in k})
nxd_te.set_weights(weights)
nxd_te.to_neuron()
print(f"  Loaded in {time.time()-t0:.1f}s")

nxd_out = nxd_te(ids.input_ids, ids.attention_mask)
nxd_embeds = nxd_out[0] if isinstance(nxd_out, (tuple, list)) else nxd_out
print(f"  NxD output: {nxd_embeds.shape}, mean={nxd_embeds.float().mean():.6f}, std={nxd_embeds.float().std():.6f}")

# [3] Compare
print("\n[3] Comparison...")
if cpu_out.shape != nxd_embeds.shape:
    print(f"  SHAPE MISMATCH: CPU={cpu_out.shape} vs NxD={nxd_embeds.shape}")
    print(f"  Strategy A FAILED — world_size baked into NEFF graph")
    exit(1)

cos = torch.nn.functional.cosine_similarity(cpu_out.flatten().float(), nxd_embeds.flatten().float(), dim=0).item()
diff = (cpu_out.float() - nxd_embeds.float()).abs()
print(f"  Cosine similarity: {cos:.6f}")
print(f"  Max diff: {diff.max():.6f}")
print(f"  Mean diff: {diff.mean():.6f}")

if cos > 0.99:
    print(f"\n✅ SUCCESS — Fixed text encoder matches CPU! (cosine={cos:.4f})")
elif cos > 0.90:
    print(f"\n⚠️  Partial match (cosine={cos:.4f}) — may work for video but degraded")
else:
    print(f"\n✗ MISMATCH (cosine={cos:.4f}) — Strategy A failed, need full recompile")
    print(f"  Run: cd {os.path.dirname(TE_PATH)}")
    print(f"  python neuron_wan2_2_t2v_a14b/compile_text_encoder.py --tp_degree 4 --world_size 4 ...")
