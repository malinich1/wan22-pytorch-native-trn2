"""Compare pipeline VAE decode logic with our script."""
from diffusers import WanPipeline
import inspect

src = inspect.getsource(WanPipeline.__call__)
lines = src.split('\n')
# Print lines around latent denormalization and video output
for i, line in enumerate(lines):
    if any(k in line for k in ['latents_mean', 'latents_std', 'video_processor', 'decode']):
        # Print context around this line
        start = max(0, i-1)
        end = min(len(lines), i+2)
        for j in range(start, end):
            print(f"{j}: {lines[j]}")
        print("---")
