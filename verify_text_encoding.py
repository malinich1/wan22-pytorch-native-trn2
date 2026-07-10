"""
Verify that NxD text encoding matches the official pipeline's text encoding.
"""
import os
os.environ["NEURON_RT_NUM_CORES"] = "8"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_RT_VISIBLE_CORES"] = "0-7"

import torch
import time

# Monkey-patch xm.mark_step
import torch_xla.core.xla_model as xm
xm.mark_step = lambda *a, **kw: None

from diffusers import WanPipeline

# Load pipeline
print("Loading pipeline...")
cache_dir = "/mnt/nvme/wan2.2_t2v_a14b_hf_cache_dir"
pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    cache_dir=cache_dir,
    torch_dtype=torch.float32,
)
print("Pipeline loaded.")

# Encode prompt using official pipeline method
prompt = "A fluffy orange tabby cat walking gracefully through a sunlit garden, detailed fur, green grass, realistic"
negative = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

print(f"Prompt: {prompt[:80]}...")
print("Encoding with pipeline.encode_prompt (CPU)...")
t0 = time.time()
prompt_embeds, negative_embeds = pipe.encode_prompt(
    prompt=prompt,
    negative_prompt=negative,
    do_classifier_free_guidance=True,
    num_videos_per_prompt=1,
    max_sequence_length=512,
    device=torch.device("cpu"),
)
print(f"  Encoded in {time.time()-t0:.1f}s")
print(f"  prompt_embeds: shape={prompt_embeds.shape}, mean={prompt_embeds.mean():.4f}, std={prompt_embeds.std():.4f}")
print(f"  negative_embeds: shape={negative_embeds.shape}, mean={negative_embeds.mean():.4f}, std={negative_embeds.std():.4f}")
print(f"  prompt_embeds range: [{prompt_embeds.min():.4f}, {prompt_embeds.max():.4f}]")

# Now check what the NxD script's subprocess text encoder produces
# The subprocess saves prompt_embeds to a file - let's check the log
# Actually, let's load the text encoder submodule from NxD and compare
print("\n\nNow loading NxD text encoder to compare...")
import json
from neuronx_distributed import NxDModel
from safetensors.torch import load_file
import sys
sys.path.insert(0, "/home/ubuntu/aws-neuron-samples/torch-neuronx/inference/hf_pretrained_wan2.2_t2v_a14b")
from neuron_wan2_2_t2v_a14b.neuron_commons import InferenceTextEncoderWrapperV2

te_path = "/opt/dlami/nvme/compiled_models_t2v_a14b/text_encoder"
te_config_path = os.path.join(te_path, "config.json")
with open(te_config_path) as f:
    te_config = json.load(f)
te_tp = te_config["tp_degree"]
te_ws = te_config["world_size"]
print(f"  NxD text encoder: TP={te_tp}, world_size={te_ws}")

# Load NxDModel text encoder
nxd_te_path = os.path.join(te_path, "nxd_model.pt")
nxd_te = NxDModel.load(nxd_te_path, start_rank=0, local_ranks_size=te_ws)

# Load weights
te_weights_path = os.path.join(te_path, "weights")
te_checkpoints = []
for rank in range(te_tp):
    ckpt_path = os.path.join(te_weights_path, f"tp{rank}_sharded_checkpoint.safetensors")
    raw_ckpt = load_file(ckpt_path)
    ckpt = {k: v for k, v in raw_ckpt.items() if 'master_weight' not in k}
    te_checkpoints.append(ckpt)

# For text encoder, no CP (world_size = tp_degree)
nxd_te.set_weights(te_checkpoints)
nxd_te.to_neuron()
print("  NxD text encoder loaded to Neuron")

# Create wrapper and replace in pipeline
wrapper = InferenceTextEncoderWrapperV2(nxd_te, pipe.text_encoder.config)
original_te = pipe.text_encoder
pipe.text_encoder = wrapper

# Encode with NxD text encoder
print("Encoding with NxD text encoder...")
t0 = time.time()
nxd_prompt_embeds, nxd_negative_embeds = pipe.encode_prompt(
    prompt=prompt,
    negative_prompt=negative,
    do_classifier_free_guidance=True,
    num_videos_per_prompt=1,
    max_sequence_length=512,
    device=torch.device("cpu"),
)
print(f"  Encoded in {time.time()-t0:.1f}s")
print(f"  nxd_prompt_embeds: shape={nxd_prompt_embeds.shape}, mean={nxd_prompt_embeds.mean():.4f}, std={nxd_prompt_embeds.std():.4f}")
print(f"  nxd_negative_embeds: shape={nxd_negative_embeds.shape}, mean={nxd_negative_embeds.mean():.4f}, std={nxd_negative_embeds.std():.4f}")

# Compare
print("\n=== TEXT ENCODING COMPARISON ===")
diff_prompt = (prompt_embeds.float() - nxd_prompt_embeds.float()).abs()
diff_neg = (negative_embeds.float() - nxd_negative_embeds.float()).abs()
cos_sim = torch.nn.functional.cosine_similarity(
    prompt_embeds.flatten().float(), nxd_prompt_embeds.flatten().float(), dim=0
)
print(f"  Prompt embeds max diff: {diff_prompt.max():.4f}")
print(f"  Prompt embeds mean diff: {diff_prompt.mean():.4f}")
print(f"  Prompt embeds cosine sim: {cos_sim:.4f}")
print(f"  Negative embeds max diff: {diff_neg.max():.4f}")

if cos_sim > 0.99:
    print("\n✓ TEXT ENCODING MATCHES")
else:
    print(f"\n✗ TEXT ENCODING MISMATCH (cosine sim={cos_sim:.4f}) - THIS MAY BE THE BUG!")
