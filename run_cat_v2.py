"""WAN 2.1 1.3B - Cat video generation v2
Uses simpler prompt and higher resolution for better cat recognition.
Uses the WanPipeline API with torch.compile on Neuron.
"""
import os, sys, time, logging, torch
import torch_neuronx
import torch.distributed as dist

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_RT_NUM_CORES"] = "1"
os.environ["NEURON_CC_FLAGS"] = "-O1 --auto-cast=none"
os.environ["NEURON_COMPILE_CACHE_URL"] = "file:///mnt/nvme/neff_cache_v2"

dist.init_process_group(backend="neuron")
rank = dist.get_rank()

from diffusers import WanPipeline
from diffusers.utils import export_to_video

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
logger.info(f"Loading pipeline from {model_id}")

pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# Move transformer to neuron
pipe.transformer = pipe.transformer.to(torch.device("neuron"))

# Compile transformer for speed
pipe.transformer = torch.compile(pipe.transformer, backend="neuron", fullgraph=True, dynamic=False)

# Keep VAE on CPU (float32 for quality)
pipe.vae = pipe.vae.to(torch.float32)

# Simple prompt - small models do better with short, clear descriptions
prompt = "a cat, orange fur, sitting, looking at camera, simple background"
negative_prompt = "blurry, distorted, ugly, deformed"

logger.info(f"Prompt: {prompt}")
logger.info(f"Negative: {negative_prompt}")
logger.info(f"Config: 480x832, 33 frames, 50 steps, cfg=6.0, seed=42")

t0 = time.time()
output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=33,
    height=480,
    width=832,
    num_inference_steps=50,
    guidance_scale=6.0,
    generator=torch.Generator().manual_seed(42),
)
elapsed = time.time() - t0
logger.info(f"Generation done in {elapsed:.1f}s")

# Save video
os.makedirs("/mnt/nvme/outputs", exist_ok=True)
out_path = "/mnt/nvme/outputs/cat_v2.mp4"
export_to_video(output.frames[0], out_path, fps=16)
logger.info(f"Saved to {out_path}, size={os.path.getsize(out_path)/1024:.0f}KB")
logger.info(f"TOTAL: {elapsed:.1f}s")
