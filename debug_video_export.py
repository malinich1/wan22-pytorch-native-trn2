"""Debug the video export: decode latents on CPU and save as individual PNGs + video."""
import os
import sys
import numpy as np
import torch

# Check ffmpeg
import imageio
try:
    exe = imageio.plugins.ffmpeg.get_exe()
    print(f"ffmpeg found: {exe}")
except:
    import shutil
    exe = shutil.which("ffmpeg")
    print(f"System ffmpeg: {exe}")

# Load debug latents from previous NxD run
latents_path = "/home/ubuntu/aws-neuron-samples/torch-neuronx/inference/hf_pretrained_wan2.2_t2v_a14b/debug_latents.pt"
latents = torch.load(latents_path, map_location="cpu", weights_only=True)
print(f"Latents: {latents.shape}, range=[{latents.min():.3f}, {latents.max():.3f}], std={latents.std():.3f}")

# Load VAE
from diffusers import AutoencoderKLWan
vae_path = "/mnt/nvme/wan2.2_t2v_a14b_hf_cache_dir/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/5be7df9619b54f4e2667b2755bc6a756675b5cd7/vae"
vae = AutoencoderKLWan.from_pretrained(vae_path, torch_dtype=torch.float32).eval()
print("VAE loaded")

# Denormalize (same formula as pipeline and script)
latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
latents_denorm = latents.float() / latents_std + latents_mean
print(f"Denormalized: range=[{latents_denorm.min():.3f}, {latents_denorm.max():.3f}]")

# Decode first 3 temporal frames (produces ~9 video frames due to 4x temporal upsample)
small = latents_denorm[:, :, :3, :, :]
print(f"Decoding {small.shape} on CPU...")
with torch.no_grad():
    video = vae.decode(small, return_dict=False)[0]
print(f"Decoded video: {video.shape}, range=[{video.min():.3f}, {video.max():.3f}]")

# Post-process exactly like the script
video_np = video[0].permute(1, 2, 3, 0).float().cpu().numpy()  # [F, H, W, C]
video_np = ((video_np + 1.0) / 2.0).clip(0, 1)
print(f"Post-processed: shape={video_np.shape}, range=[{video_np.min():.4f}, {video_np.max():.4f}], mean={video_np.mean():.3f}")

# Save individual frames as PNG
os.makedirs("/mnt/nvme/outputs", exist_ok=True)
from PIL import Image
for i in range(min(5, video_np.shape[0])):
    frame_uint8 = (video_np[i] * 255).astype(np.uint8)
    Image.fromarray(frame_uint8).save(f"/mnt/nvme/outputs/debug_frame{i}.png")
    print(f"  Frame {i}: shape={frame_uint8.shape}, min={frame_uint8.min()}, max={frame_uint8.max()}, mean={frame_uint8.mean():.1f}")

# Save as video using export_to_video
from diffusers.utils import export_to_video
frames_list = [video_np[i] for i in range(video_np.shape[0])]
export_to_video(frames_list, "/mnt/nvme/outputs/debug_cpu_vae.mp4", fps=16)
size = os.path.getsize("/mnt/nvme/outputs/debug_cpu_vae.mp4")
print(f"\nVideo saved: /mnt/nvme/outputs/debug_cpu_vae.mp4 ({size/1024:.0f} KB, {len(frames_list)} frames)")
print("Done! Download debug_frame0.png to verify visual quality.")
