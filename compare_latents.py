"""
Compare NxD-produced latents vs reference pipeline latents.
Run just 5 steps of the official diffusers pipeline at 256x256 to get reference latents,
then compare stats and structure against the NxD debug latents.
"""
import os
os.environ["NEURON_RT_NUM_CORES"] = "1"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_RT_VISIBLE_CORES"] = "0"

import torch
import numpy as np

# Monkey-patch xm.mark_step
import torch_xla.core.xla_model as xm
xm.mark_step = lambda *a, **kw: None

from diffusers import WanPipeline

# Load NxD latents
nxd_latents = torch.load(
    "/home/ubuntu/aws-neuron-samples/torch-neuronx/inference/hf_pretrained_wan2.2_t2v_a14b/debug_latents.pt",
    map_location="cpu", weights_only=True
)
print(f"NxD latents: shape={nxd_latents.shape}")
print(f"  range=[{nxd_latents.min():.4f}, {nxd_latents.max():.4f}]")
print(f"  mean={nxd_latents.mean():.4f}, std={nxd_latents.std():.4f}")
print(f"  Per-channel means: {nxd_latents[0].mean(dim=(1,2,3))[:4].tolist()}")

# Run reference pipeline with matching seed to get reference latents
print("\nLoading pipeline for reference latent generation...")
cache_dir = "/mnt/nvme/wan2.2_t2v_a14b_hf_cache_dir"
pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    cache_dir=cache_dir,
    torch_dtype=torch.float32,
)

# Generate with output_type="latent" to get raw latents
print("Generating reference latents (256x256, 1 frame, 5 steps)...")
output = pipe(
    prompt="A fluffy orange tabby cat sitting on grass, looking at camera, realistic",
    negative_prompt="blurry, distorted",
    height=256,
    width=256,
    num_frames=1,
    num_inference_steps=5,
    guidance_scale=4.0,
    generator=torch.Generator().manual_seed(42),
    output_type="latent",
)
ref_latents = output.frames
print(f"\nReference latents: shape={ref_latents.shape}")
print(f"  range=[{ref_latents.min():.4f}, {ref_latents.max():.4f}]")
print(f"  mean={ref_latents.mean():.4f}, std={ref_latents.std():.4f}")
print(f"  Per-channel means: {ref_latents[0].mean(dim=(1,2,3))[:4].tolist()}")

# Now generate more steps at same resolution to see what good latents look like
print("\nGenerating reference latents (256x256, 1 frame, 40 steps - same as NxD)...")
output40 = pipe(
    prompt="A fluffy orange tabby cat sitting on grass, looking at camera, realistic",
    negative_prompt="blurry, distorted",
    height=256,
    width=256,
    num_frames=1,
    num_inference_steps=40,
    guidance_scale=4.0,
    generator=torch.Generator().manual_seed(42),
    output_type="latent",
)
ref40_latents = output40.frames
print(f"Reference 40-step latents: shape={ref40_latents.shape}")
print(f"  range=[{ref40_latents.min():.4f}, {ref40_latents.max():.4f}]")
print(f"  mean={ref40_latents.mean():.4f}, std={ref40_latents.std():.4f}")
print(f"  Per-channel means: {ref40_latents[0].mean(dim=(1,2,3))[:4].tolist()}")

print("\n=== COMPARISON ===")
print(f"NxD (480x832, 81 frames, 40 steps): std={nxd_latents.std():.4f}, mean={nxd_latents.mean():.4f}")
print(f"Ref (256x256, 1 frame, 40 steps):   std={ref40_latents.std():.4f}, mean={ref40_latents.mean():.4f}")
print(f"Pure noise would have:               std≈1.0, mean≈0.0")
print(f"Properly denoised should have:        std≈0.3-0.6, abs(mean)<0.1")
