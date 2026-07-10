"""
Compare CPU text encoder vs NxD compiled text encoder.
Loads ONLY the text encoder (not full pipeline) to avoid OOM.
"""
import os, sys, json, time, gc
os.environ["NEURON_RT_NUM_CORES"] = "8"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_RT_VISIBLE_CORES"] = "0-7"

import torch
import numpy as np
sys.path.insert(0, "/home/ubuntu/aws-neuron-samples/torch-neuronx/inference/hf_pretrained_wan2.2_t2v_a14b")

prompt = "A fluffy orange tabby cat walking gracefully through a sunlit garden, detailed fur, green grass, realistic"
cache_dir = "/mnt/nvme/wan2.2_t2v_a14b_hf_cache_dir"
hf_path = "/mnt/nvme/wan2.2_t2v_a14b_hf_cache_dir/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/5be7df9619b54f4e2667b2755bc6a756675b5cd7"

# === TOKENIZE ===
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(hf_path, subfolder="tokenizer")
text_inputs = tokenizer(
    prompt, padding="max_length", max_length=512,
    truncation=True, return_attention_mask=True, return_tensors="pt",
)
input_ids = text_inputs.input_ids
attention_mask = text_inputs.attention_mask
print(f"Tokenized: input_ids={input_ids.shape}, non-pad tokens={attention_mask.sum().item()}")

# === CPU TEXT ENCODER ===
print("\n=== CPU Text Encoder (UMT5) ===")
from transformers import UMT5EncoderModel
t0 = time.time()
cpu_te = UMT5EncoderModel.from_pretrained(
    hf_path, subfolder="text_encoder", torch_dtype=torch.float32
).eval()
print(f"Loaded in {time.time()-t0:.1f}s")

with torch.no_grad():
    cpu_out = cpu_te(input_ids=input_ids, attention_mask=attention_mask)
cpu_embeds = cpu_out.last_hidden_state
print(f"  Output: shape={cpu_embeds.shape}, dtype={cpu_embeds.dtype}")
print(f"  mean={cpu_embeds.mean():.6f}, std={cpu_embeds.std():.6f}")
print(f"  range=[{cpu_embeds.min():.4f}, {cpu_embeds.max():.4f}]")
print(f"  first 5 values [0,0,:5]: {cpu_embeds[0,0,:5].tolist()}")
print(f"  first 5 values [0,10,:5]: {cpu_embeds[0,10,:5].tolist()}")

# Free CPU model
del cpu_te
gc.collect()

# Force garbage collection and free memory
import ctypes
gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
print(f"Memory freed. Proceeding to NxD.")

# === NxD TEXT ENCODER ===
print("\n=== NxD Compiled Text Encoder ===")
from neuronx_distributed import NxDModel
from safetensors.torch import load_file
from neuron_wan2_2_t2v_a14b.neuron_commons import InferenceTextEncoderWrapperV2

te_path = "/opt/dlami/nvme/compiled_models_t2v_a14b/text_encoder"
with open(os.path.join(te_path, "config.json")) as f:
    te_cfg = json.load(f)
print(f"Config: {te_cfg}")

t0 = time.time()
nxd_te = NxDModel.load(os.path.join(te_path, "nxd_model.pt"), start_rank=0, local_ranks_size=te_cfg["world_size"])

# Load weights
te_checkpoints = []
for rank in range(te_cfg["tp_degree"]):
    ckpt = load_file(os.path.join(te_path, "weights", f"tp{rank}_sharded_checkpoint.safetensors"))
    ckpt = {k: v for k, v in ckpt.items() if 'master_weight' not in k}
    te_checkpoints.append(ckpt)

nxd_te.set_weights(te_checkpoints)
nxd_te.to_neuron()
print(f"NxD TE loaded in {time.time()-t0:.1f}s")

# Run NxD text encoder directly (not through wrapper first)
print("\nRunning NxD directly...")
nxd_raw_out = nxd_te(input_ids, attention_mask)
if isinstance(nxd_raw_out, (tuple, list)):
    nxd_raw = nxd_raw_out[0]
else:
    nxd_raw = nxd_raw_out
print(f"  Raw NxD output: shape={nxd_raw.shape}, dtype={nxd_raw.dtype}")
print(f"  mean={nxd_raw.mean():.6f}, std={nxd_raw.std():.6f}")
print(f"  range=[{nxd_raw.min():.4f}, {nxd_raw.max():.4f}]")
print(f"  first 5 values [0,0,:5]: {nxd_raw[0,0,:5].tolist()}")
print(f"  first 5 values [0,10,:5]: {nxd_raw[0,10,:5].tolist()}")

# === COMPARISON ===
print("\n" + "=" * 60)
print("COMPARISON: CPU vs NxD Text Encoder")
print("=" * 60)

if cpu_embeds.shape == nxd_raw.shape:
    diff = (cpu_embeds.float() - nxd_raw.float()).abs()
    cos_sim = torch.nn.functional.cosine_similarity(
        cpu_embeds.flatten().float(), nxd_raw.flatten().float(), dim=0
    )
    print(f"  Max diff: {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    
    # Per-position analysis
    pos_cos = []
    for p in range(min(20, cpu_embeds.shape[1])):
        pc = torch.nn.functional.cosine_similarity(
            cpu_embeds[0, p].float().unsqueeze(0),
            nxd_raw[0, p].float().unsqueeze(0), dim=1
        ).item()
        pos_cos.append(pc)
    print(f"  Per-position cosine (first 20): min={min(pos_cos):.4f}, max={max(pos_cos):.4f}, mean={np.mean(pos_cos):.4f}")
    
    if cos_sim > 0.99:
        print("\n✓ TEXT ENCODERS MATCH — bug is elsewhere")
    elif cos_sim > 0.90:
        print(f"\n⚠ PARTIAL MATCH (cosine={cos_sim:.4f}) — numerical precision issue")
    else:
        print(f"\n✗ MISMATCH (cosine={cos_sim:.4f}) — NxD text encoder is broken!")
        if nxd_raw.abs().max() < 0.001:
            print("  DIAGNOSIS: NxD output is near-zero")
        elif nxd_raw.std() < 0.01:
            print("  DIAGNOSIS: NxD output collapsed")
        else:
            print(f"  DIAGNOSIS: NxD produces different values")
            print(f"  CPU std={cpu_embeds.std():.4f}, NxD std={nxd_raw.std():.4f}")
            print(f"  Ratio: {nxd_raw.std()/cpu_embeds.std():.4f}")
else:
    print(f"  SHAPE MISMATCH: CPU={cpu_embeds.shape} vs NxD={nxd_raw.shape}")
