"""Check video quality by loading debug latents and analyzing pixel output."""
import torch
import numpy as np

# Load debug latents saved during inference
latents = torch.load('/home/ubuntu/aws-neuron-samples/torch-neuronx/inference/hf_pretrained_wan2.2_t2v_a14b/debug_latents.pt', map_location='cpu', weights_only=True)
print(f"Debug latents: shape={latents.shape}, dtype={latents.dtype}")
print(f"  range=[{latents.min():.4f}, {latents.max():.4f}], mean={latents.mean():.4f}, std={latents.std():.4f}")

# Check if latents look like noise (mean~0, std~1) or denoised (structured)
if latents.std() > 0.8 and abs(latents.mean()) < 0.1:
    print("  WARNING: Latents look like pure noise (not denoised properly)")
elif latents.std() < 0.01:
    print("  WARNING: Latents are near-zero (collapsed)")
else:
    print(f"  Latents appear denoised (std={latents.std():.3f}, not 1.0)")

# Now try CPU VAE decode on first 2 frames to verify
print("\nLoading VAE for CPU decode verification...")
from diffusers import AutoencoderKLWan
vae = AutoencoderKLWan.from_pretrained(
    "/mnt/nvme/wan2.2_t2v_a14b_hf_cache_dir/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/5be7df9619b54f4e2667b2755bc6a756675b5cd7/vae",
    torch_dtype=torch.float32
).eval()

# Denormalize latents
latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
latents_denorm = latents / (1.0 / latents_std) + latents_mean
print(f"Denormalized: range=[{latents_denorm.min():.3f}, {latents_denorm.max():.3f}]")

# Decode just first 2 temporal frames (to save memory)
small_latents = latents_denorm[:, :, :2, :, :]  # [1, 16, 2, 60, 104]
print(f"Decoding {small_latents.shape} on CPU...")
with torch.no_grad():
    video = vae.decode(small_latents).sample
print(f"Decoded: shape={video.shape}, range=[{video.min():.3f}, {video.max():.3f}]")

# Convert to uint8
video_np = ((video[0].permute(1,2,3,0).float().cpu().numpy() + 1.0) / 2.0).clip(0, 1)
video_uint8 = (video_np * 255).astype(np.uint8)
print(f"Pixels: shape={video_uint8.shape}, min={video_uint8.min()}, max={video_uint8.max()}, mean={video_uint8.mean():.1f}")

from PIL import Image
img = Image.fromarray(video_uint8[0])
img.save('/mnt/nvme/outputs/cpu_vae_frame0.png')
print("Saved /mnt/nvme/outputs/cpu_vae_frame0.png")
