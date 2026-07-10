"""
Reference test: Run WAN 2.2 T2V-A14B using the official diffusers pipeline on CPU.
Only generates 1 frame with 5 steps to verify the pipeline produces coherent output.
This bypasses all NxDModel/Neuron code to isolate whether the issue is in:
- The model weights
- The prompt encoding
- The diffusers pipeline itself
"""
import torch
import os
import time

os.environ["NEURON_RT_NUM_CORES"] = "1"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_RT_VISIBLE_CORES"] = "0"

# Monkey-patch to disable xm.mark_step()
import torch_xla.core.xla_model as xm
xm.mark_step = lambda *a, **kw: None

from diffusers import WanPipeline
from PIL import Image
import numpy as np

print("Loading WAN 2.2 T2V-A14B pipeline on CPU (this will be slow)...")
t0 = time.time()

# Load pipeline
cache_dir = "/mnt/nvme/wan2.2_t2v_a14b_hf_cache_dir"
pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    cache_dir=cache_dir,
    torch_dtype=torch.float32,
)
print(f"Pipeline loaded in {time.time()-t0:.1f}s")
print(f"  boundary_ratio: {pipe.config.boundary_ratio}")
print(f"  expand_timesteps: {pipe.config.expand_timesteps}")

# Generate just 1 frame with minimal steps for quick validation
prompt = "A fluffy orange tabby cat sitting on grass, looking at camera, realistic"
print(f"\nPrompt: {prompt}")
print("Generating 1 frame, 5 steps, 256x256 (minimal test)...")

t0 = time.time()
output = pipe(
    prompt=prompt,
    negative_prompt="blurry, distorted",
    height=256,
    width=256,
    num_frames=1,
    num_inference_steps=5,
    guidance_scale=4.0,
    generator=torch.Generator().manual_seed(42),
    output_type="np",
)
print(f"Generation done in {time.time()-t0:.1f}s")

# Save output
video = output.frames  # shape depends on output format
print(f"Output type: {type(video)}, shape info: {video[0].shape if hasattr(video[0], 'shape') else 'list'}")

# Handle different output formats
if isinstance(video, np.ndarray):
    frame = video[0][0]  # [batch, frames, H, W, C]
elif isinstance(video, list):
    frame = video[0][0] if isinstance(video[0], (list, np.ndarray)) else np.array(video[0])
else:
    frame = video

if frame.max() <= 1.0:
    frame = (frame * 255).astype(np.uint8)

img = Image.fromarray(frame)
img.save("/mnt/nvme/outputs/reference_test.png")
print(f"Saved: /mnt/nvme/outputs/reference_test.png, size={frame.shape}, range=[{frame.min()}, {frame.max()}]")
