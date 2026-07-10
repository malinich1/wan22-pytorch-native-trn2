"""
Compare CPU text encoder output vs NxD compiled text encoder output.
Runs both on the same prompt and compares embeddings.
"""
import os, sys, json, time, gc
os.environ["NEURON_RT_NUM_CORES"] = "8"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_RT_VISIBLE_CORES"] = "0-7"

import torch
import numpy as np

# Monkey-patch xm.mark_step
import torch_xla.core.xla_model as xm
xm.mark_step = lambda *a, **kw: None

sys.path.insert(0, "/home/ubuntu/aws-neuron-samples/torch-neuronx/inference/hf_pretrained_wan2.2_t2v_a14b")

prompt = "A fluffy orange tabby cat walking gracefully through a sunlit garden, detailed fur, green grass, realistic"
negative = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

# === CPU TEXT ENCODER ===
print("=" * 60)
print("STEP 1: CPU Text Encoding (reference)")
print("=" * 60)

from diffusers import WanPipeline
cache_dir = "/mnt/nvme/wan2.2_t2v_a14b_hf_cache_dir"
pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    cache_dir=cache_dir,
    torch_dtype=torch.float32,
)
print("Pipeline loaded.")

t0 = time.time()
cpu_prompt_embeds, cpu_neg_embeds = pipe.encode_prompt(
    prompt=prompt,
    negative_prompt=negative,
    do_classifier_free_guidance=True,
    num_videos_per_prompt=1,
    max_sequence_length=512,
    device=torch.device("cpu"),
)
print(f"CPU encoding time: {time.time()-t0:.1f}s")
print(f"  prompt_embeds: shape={cpu_prompt_embeds.shape}, dtype={cpu_prompt_embeds.dtype}")
print(f"    mean={cpu_prompt_embeds.mean():.6f}, std={cpu_prompt_embeds.std():.6f}")
print(f"    range=[{cpu_prompt_embeds.min():.4f}, {cpu_prompt_embeds.max():.4f}]")
print(f"    first 5 values: {cpu_prompt_embeds[0, 0, :5].tolist()}")
print(f"  negative_embeds: shape={cpu_neg_embeds.shape}")
print(f"    mean={cpu_neg_embeds.mean():.6f}, std={cpu_neg_embeds.std():.6f}")

# Save for reference
torch.save({
    "prompt_embeds": cpu_prompt_embeds,
    "negative_embeds": cpu_neg_embeds,
}, "/mnt/nvme/cpu_text_embeds.pt")

# Free pipeline memory (keep only text encoder config)
te_config = pipe.text_encoder.config
del pipe
gc.collect()

# === NxD TEXT ENCODER ===
print("\n" + "=" * 60)
print("STEP 2: NxD Compiled Text Encoding")
print("=" * 60)

from neuronx_distributed import NxDModel
from safetensors.torch import load_file
from neuron_wan2_2_t2v_a14b.neuron_commons import InferenceTextEncoderWrapperV2

te_path = "/opt/dlami/nvme/compiled_models_t2v_a14b/text_encoder"
te_config_file = os.path.join(te_path, "config.json")
with open(te_config_file) as f:
    te_cfg = json.load(f)
te_tp = te_cfg["tp_degree"]
te_ws = te_cfg["world_size"]
print(f"NxD text encoder config: TP={te_tp}, world_size={te_ws}")

# Load NxDModel
nxd_te_path = os.path.join(te_path, "nxd_model.pt")
print("Loading NxDModel...")
t0 = time.time()
nxd_te = NxDModel.load(nxd_te_path, start_rank=0, local_ranks_size=te_ws)

# Load weights
te_weights_path = os.path.join(te_path, "weights")
te_checkpoints = []
for rank in range(te_tp):
    ckpt_path = os.path.join(te_weights_path, f"tp{rank}_sharded_checkpoint.safetensors")
    raw_ckpt = load_file(ckpt_path)
    ckpt = {k: v for k, v in raw_ckpt.items() if 'master_weight' not in k}
    te_checkpoints.append(ckpt)

nxd_te.set_weights(te_checkpoints)
nxd_te.to_neuron()
print(f"NxDModel loaded in {time.time()-t0:.1f}s")

# Create wrapper
wrapper = InferenceTextEncoderWrapperV2(nxd_te, te_config)

# Now we need to tokenize and run through the wrapper the same way the pipeline does
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    subfolder="tokenizer",
    cache_dir=cache_dir,
)

# Tokenize prompt
text_inputs = tokenizer(
    prompt,
    padding="max_length",
    max_length=512,
    truncation=True,
    return_attention_mask=True,
    return_tensors="pt",
)
input_ids = text_inputs.input_ids
attention_mask = text_inputs.attention_mask
print(f"\nTokenized prompt: input_ids shape={input_ids.shape}")
print(f"  Non-padding tokens: {attention_mask.sum().item()}")

# Run through wrapper
print("Running NxD text encoder...")
t0 = time.time()
with torch.no_grad():
    nxd_output = wrapper(input_ids=input_ids, attention_mask=attention_mask)
nxd_time = time.time() - t0
print(f"NxD encoding time: {nxd_time:.1f}s")

# Extract last_hidden_state
if hasattr(nxd_output, 'last_hidden_state'):
    nxd_embeds = nxd_output.last_hidden_state
elif isinstance(nxd_output, (tuple, list)):
    nxd_embeds = nxd_output[0]
else:
    nxd_embeds = nxd_output
print(f"  NxD output: shape={nxd_embeds.shape}, dtype={nxd_embeds.dtype}")
print(f"    mean={nxd_embeds.mean():.6f}, std={nxd_embeds.std():.6f}")
print(f"    range=[{nxd_embeds.min():.4f}, {nxd_embeds.max():.4f}]")
print(f"    first 5 values: {nxd_embeds[0, 0, :5].tolist()}")

# === COMPARISON ===
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

# The cpu_prompt_embeds came from pipe.encode_prompt which also applies attention_mask
# Let's compare raw last_hidden_state values
# cpu_prompt_embeds already has the prompt_attention_mask applied and is the final embedding
print(f"\nCPU prompt_embeds: mean={cpu_prompt_embeds.mean():.6f}, std={cpu_prompt_embeds.std():.6f}")
print(f"NxD last_hidden:   mean={nxd_embeds.mean():.6f}, std={nxd_embeds.std():.6f}")

# Check if shapes match
if cpu_prompt_embeds.shape == nxd_embeds.shape:
    diff = (cpu_prompt_embeds.float() - nxd_embeds.float()).abs()
    cos_sim = torch.nn.functional.cosine_similarity(
        cpu_prompt_embeds.flatten().float(),
        nxd_embeds.flatten().float(),
        dim=0
    )
    print(f"\nElement-wise comparison:")
    print(f"  Max diff: {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    
    # Check non-padding positions only
    mask_expanded = attention_mask.unsqueeze(-1).expand_as(cpu_prompt_embeds)
    masked_diff = diff * mask_expanded
    non_pad_count = mask_expanded.sum()
    print(f"  Mean diff (non-padding only): {masked_diff.sum() / non_pad_count:.6f}")
    
    if cos_sim > 0.99:
        print("\n✓ TEXT ENCODERS MATCH")
    elif cos_sim > 0.90:
        print(f"\n⚠ TEXT ENCODERS PARTIALLY MATCH (cosine={cos_sim:.4f})")
    else:
        print(f"\n✗ TEXT ENCODERS DO NOT MATCH (cosine={cos_sim:.4f})")
        print("  This explains why the NxD pipeline produces no cat!")
        
        # Diagnose: check if it's all zeros, random, or shifted
        if nxd_embeds.abs().max() < 0.001:
            print("  DIAGNOSIS: NxD output is near-zero (model not running properly)")
        elif nxd_embeds.std() < cpu_prompt_embeds.std() * 0.1:
            print("  DIAGNOSIS: NxD output has collapsed variance")
        else:
            print(f"  DIAGNOSIS: NxD output has structure but wrong values")
            print(f"    CPU std={cpu_prompt_embeds.std():.4f} vs NxD std={nxd_embeds.std():.4f}")
            # Check correlation per position
            print(f"    Position 0 correlation: {torch.corrcoef(torch.stack([cpu_prompt_embeds[0,0].float(), nxd_embeds[0,0].float()]))[0,1]:.4f}")
            print(f"    Position 10 correlation: {torch.corrcoef(torch.stack([cpu_prompt_embeds[0,10].float(), nxd_embeds[0,10].float()]))[0,1]:.4f}")
else:
    print(f"\n✗ SHAPE MISMATCH: CPU={cpu_prompt_embeds.shape} vs NxD={nxd_embeds.shape}")
