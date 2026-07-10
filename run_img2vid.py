"""
WAN 2.1 T2V-1.3B — Image-to-Video on Neuron (Beta 3)

Uses an existing image as the first frame conditioning to generate a video.
The approach:
  1. Load the reference image and encode it through the VAE
  2. Use the encoded latent as conditioning for the first temporal frame
  3. Run the diffusion process with the T2V transformer
  4. Decode and save the output video

This provides a pseudo image-to-video capability using the T2V 1.3B model.

Usage:
    torchrun --nproc-per-node 1 run_img2vid.py \
        --image /path/to/input_image.png \
        --prompt "A cat walking gracefully" \
        --height 384 --width 640 --num-frames 17 --num-steps 30

    # Quick test:
    torchrun --nproc-per-node 1 run_img2vid.py \
        --image /path/to/input_image.png \
        --prompt "A cat walking" \
        --height 256 --width 256 --num-frames 9 --num-steps 20
"""

import argparse
import logging
import os
import time

import torch
import torch.distributed as dist
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_HEIGHT = 384
DEFAULT_WIDTH = 640
DEFAULT_NUM_FRAMES = 17
DEFAULT_NUM_STEPS = 30
DEFAULT_GUIDANCE = 5.0
DEFAULT_SEED = 42
DEFAULT_OUTPUT = "/mnt/nvme/outputs/img2vid_output.mp4"
DEFAULT_NEFF_CACHE = "/mnt/nvme/neff_cache"
DEFAULT_IMG_STRENGTH = 0.7  # How strongly the image conditions generation (0=ignore, 1=full)

# torch.compile config
torch._dynamo.config.cache_size_limit = 64
torch.set_default_dtype(torch.float32)


# ============================================================================
# Neuron environment
# ============================================================================

def setup_neuron_env(neff_cache: str = DEFAULT_NEFF_CACHE):
    """Configure Beta 3 environment for single-device inference."""
    os.environ["NEURON_CC_FLAGS"] = "-O1 --auto-cast=none"
    os.environ["TORCH_NEURONX_ENABLE_ASYNC_NRT"] = "1"
    os.makedirs(neff_cache, exist_ok=True)
    os.environ["NEURON_COMPILE_CACHE_URL"] = f"file://{neff_cache}"
    os.environ["NEURONX_CACHE"] = neff_cache
    logger.info(f"Neuron env configured. NEFF cache: {neff_cache}")


# ============================================================================
# Image loading and preprocessing
# ============================================================================

def load_and_preprocess_image(image_path: str, height: int, width: int) -> torch.Tensor:
    """Load image and preprocess to tensor in [-1, 1] range."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((width, height), Image.LANCZOS)
    
    import numpy as np
    img_np = np.array(img).astype(np.float32) / 255.0
    # Normalize to [-1, 1]
    img_np = img_np * 2.0 - 1.0
    # Convert to tensor: (H, W, C) -> (C, H, W) -> (1, C, H, W)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    
    logger.info(f"Loaded image: {image_path}, resized to {width}x{height}")
    logger.info(f"Image tensor shape: {img_tensor.shape}, range: [{img_tensor.min():.2f}, {img_tensor.max():.2f}]")
    return img_tensor


# ============================================================================
# Pipeline
# ============================================================================

def run_img2vid_inference(**kwargs):
    """
    Run image-to-video inference using WAN 2.1 1.3B on Neuron.
    
    The approach uses the reference image to condition the video generation:
    1. Encode the reference image through VAE to get its latent
    2. Use this latent to initialize/condition the first frame of the video latent
    3. Run the standard T2V diffusion process
    4. Decode and save
    """
    image_path = kwargs.get("image")
    prompt = kwargs.get("prompt", "A cat walking gracefully")
    height = kwargs.get("height", DEFAULT_HEIGHT)
    width = kwargs.get("width", DEFAULT_WIDTH)
    num_frames = kwargs.get("num_frames", DEFAULT_NUM_FRAMES)
    num_steps = kwargs.get("num_steps", DEFAULT_NUM_STEPS)
    guidance = kwargs.get("guidance", DEFAULT_GUIDANCE)
    seed = kwargs.get("seed", DEFAULT_SEED)
    output_path = kwargs.get("output", DEFAULT_OUTPUT)
    eager = kwargs.get("eager", False)
    neff_cache = kwargs.get("neff_cache", DEFAULT_NEFF_CACHE)
    model_id = kwargs.get("model_id", MODEL_ID)
    img_strength = kwargs.get("img_strength", DEFAULT_IMG_STRENGTH)

    # --- Step 1: Initialize distributed (Neuron backend) ---
    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    logger.info(f"Rank {rank}/{world_size}: WAN 2.1 1.3B Image-to-Video Inference")
    logger.info(f"  Model:      {model_id}")
    logger.info(f"  Image:      {image_path}")
    logger.info(f"  Prompt:     {prompt}")
    logger.info(f"  Resolution: {width}x{height}, {num_frames} frames")
    logger.info(f"  Steps:      {num_steps}, CFG: {guidance}")
    logger.info(f"  Img strength: {img_strength}")
    logger.info(f"  Mode:       {'eager' if eager else 'torch.compile'}")

    device = torch.device("neuron")
    neuron_device_idx = torch.neuron.current_device()
    logger.info(f"Rank {rank}: Using device neuron (index {neuron_device_idx})")

    # --- Step 2: Load pipeline components ---
    from diffusers import AutoencoderKLWan, WanTransformer3DModel, UniPCMultistepScheduler
    from transformers import AutoTokenizer, UMT5EncoderModel

    logger.info(f"Rank {rank}: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")

    logger.info(f"Rank {rank}: Loading text encoder (UMT5)...")
    t0 = time.time()
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16,
    ).eval()
    logger.info(f"Rank {rank}: Text encoder loaded in {time.time()-t0:.1f}s")

    logger.info(f"Rank {rank}: Loading transformer (1.3B)...")
    t0 = time.time()
    transformer = WanTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
    ).eval()
    logger.info(f"Rank {rank}: Transformer loaded in {time.time()-t0:.1f}s")

    logger.info(f"Rank {rank}: Loading VAE...")
    t0 = time.time()
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.bfloat16,
    ).eval()
    logger.info(f"Rank {rank}: VAE loaded in {time.time()-t0:.1f}s")

    logger.info(f"Rank {rank}: Loading scheduler...")
    scheduler = UniPCMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")

    # --- Step 3: Move to Neuron device ---
    logger.info(f"Rank {rank}: Moving transformer to {device}...")
    transformer = transformer.to(device)
    logger.info(f"Rank {rank}: Moving VAE to {device}...")
    vae = vae.to(device)

    # --- Step 4: torch.compile ---
    if not eager:
        logger.info(f"Rank {rank}: Compiling transformer with torch.compile(backend='neuron')...")
        transformer.forward = torch.compile(
            transformer.forward, backend="neuron", fullgraph=True, dynamic=False
        )
        logger.info(f"Rank {rank}: Compilation registered (NEFFs built on first pass)")

    dist.barrier()

    # --- Step 5: Load and encode reference image ---
    logger.info(f"Rank {rank}: Loading reference image...")
    img_tensor = load_and_preprocess_image(image_path, height, width)
    
    logger.info(f"Rank {rank}: Encoding reference image through VAE...")
    t0 = time.time()
    with torch.no_grad():
        img_for_vae = img_tensor.to(device, dtype=torch.bfloat16)
        # Encode: (1, C, H, W) -> (1, latent_ch, H/8, W/8)
        img_latent = vae.encode(img_for_vae).latent_dist.sample()
        # Apply scaling
        latents_mean = torch.tensor(vae.config.latents_mean, dtype=img_latent.dtype, device=device).view(1, -1, 1, 1)
        latents_std = torch.tensor(vae.config.latents_std, dtype=img_latent.dtype, device=device).view(1, -1, 1, 1)
        img_latent = (img_latent - latents_mean) / latents_std
    logger.info(f"Rank {rank}: Image encoded in {time.time()-t0:.1f}s, latent shape: {img_latent.shape}")

    # --- Step 6: Run inference ---
    logger.info(f"Rank {rank}: Starting image-to-video inference...")
    total_t0 = time.time()
    torch.manual_seed(seed)

    # Text encoding (CPU)
    logger.info(f"Rank {rank}: Encoding text...")
    t0 = time.time()
    text_inputs = tokenizer(
        prompt, max_length=512, padding="max_length", truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        prompt_embeds = text_encoder(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
        ).last_hidden_state.to(torch.bfloat16)

    neg_inputs = tokenizer(
        "", max_length=512, padding="max_length", truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        neg_embeds = text_encoder(
            input_ids=neg_inputs["input_ids"],
            attention_mask=neg_inputs["attention_mask"],
        ).last_hidden_state.to(torch.bfloat16)
    logger.info(f"Rank {rank}: Text encoded in {time.time()-t0:.1f}s")

    # Prepare latents - Initialize with noise but blend in the image latent for the first frame
    latent_ch = transformer.config.in_channels
    latent_h = height // 8
    latent_w = width // 8
    latent_t = (num_frames - 1) // 4 + 1 if num_frames > 1 else 1
    
    # Generate random noise for full video latent
    latents = torch.randn(1, latent_ch, latent_t, latent_h, latent_w, dtype=torch.float32)
    
    # Condition the first temporal frame with the image latent
    # img_latent shape: (1, latent_ch, latent_h, latent_w) -> expand to (1, latent_ch, 1, latent_h, latent_w)
    img_latent_cpu = img_latent.cpu().float().unsqueeze(2)  # Add temporal dimension
    
    # Blend: use image latent for first frame, noise for the rest
    # img_strength controls how much of the image latent is used
    latents[:, :, :1, :, :] = (1.0 - img_strength) * latents[:, :, :1, :, :] + img_strength * img_latent_cpu
    
    logger.info(f"Rank {rank}: Latent shape: {list(latents.shape)}, first frame conditioned with strength={img_strength}")

    # Denoising loop
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps

    logger.info(f"Rank {rank}: Denoising ({num_steps} steps)...")
    denoise_t0 = time.time()
    for i, t in enumerate(timesteps):
        step_t0 = time.time()

        x = latents.to(device, dtype=torch.bfloat16)
        t_in = t.expand(1).to(device)
        pe = prompt_embeds.to(device)
        ne = neg_embeds.to(device)

        with torch.no_grad():
            # Conditional pass
            noise_pred = transformer(
                hidden_states=x, timestep=t_in,
                encoder_hidden_states=pe, return_dict=False,
            )[0]

            # Unconditional pass (CFG)
            if guidance > 1.0:
                noise_uncond = transformer(
                    hidden_states=x, timestep=t_in,
                    encoder_hidden_states=ne, return_dict=False,
                )[0]
                noise_pred = noise_uncond + guidance * (noise_pred - noise_uncond)

        # Scheduler step (CPU)
        latents = scheduler.step(noise_pred.cpu(), t, latents, return_dict=False)[0]

        step_time = time.time() - step_t0
        if (i + 1) % 5 == 0 or (i + 1) == num_steps:
            logger.info(f"  Step {i+1}/{num_steps}: {step_time:.2f}s/step")

    denoise_time = time.time() - denoise_t0
    logger.info(f"Rank {rank}: Denoising done in {denoise_time:.1f}s ({denoise_time/num_steps:.2f}s/step)")

    # VAE decode
    logger.info(f"Rank {rank}: VAE decoding...")
    t0 = time.time()
    with torch.no_grad():
        latents_mean = torch.tensor(vae.config.latents_mean, dtype=latents.dtype).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(vae.config.latents_std, dtype=latents.dtype).view(1, -1, 1, 1, 1)
        latents_scaled = latents * latents_std + latents_mean
        video = vae.decode(latents_scaled.to(device, dtype=torch.bfloat16)).sample.cpu()
    logger.info(f"Rank {rank}: VAE decode in {time.time()-t0:.1f}s, shape={video.shape}")

    total_time = time.time() - total_t0
    logger.info(f"Rank {rank}: TOTAL INFERENCE: {total_time:.1f}s ({total_time/60:.1f} min)")

    # --- Step 7: Save output ---
    if rank == 0:
        save_output(video, output_path, num_frames)

    dist.barrier()
    dist.destroy_process_group()
    logger.info(f"Rank {rank}: Done.")


# ============================================================================
# Output saving
# ============================================================================

def save_output(video: torch.Tensor, output_path: str, num_frames: int, fps: int = 16):
    """Save video output."""
    import imageio

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    video = video.squeeze(0).permute(1, 2, 3, 0)  # (T, H, W, C)
    video = ((video.float() / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8).numpy()

    if video.shape[0] == 1 or num_frames == 1:
        output_path = output_path.rsplit(".", 1)[0] + ".png"
        from PIL import Image
        img = Image.fromarray(video[0])
        img.save(output_path)
        size_kb = os.path.getsize(output_path) / 1024
        logger.info(f"Saved image: {output_path} ({size_kb:.0f} KB)")
    else:
        output_path = output_path.rsplit(".", 1)[0] + ".mp4"
        writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
        for frame in video:
            writer.append_data(frame)
        writer.close()
        size_mb = os.path.getsize(output_path) / 1e6
        logger.info(f"Saved video: {output_path} ({size_mb:.1f} MB, {len(video)} frames @ {fps}fps)")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="WAN 2.1 T2V-1.3B Image-to-Video on Neuron (Beta 3)"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, default="A cat walking gracefully through a garden")
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--neff-cache", type=str, default=DEFAULT_NEFF_CACHE)
    parser.add_argument("--img-strength", type=float, default=DEFAULT_IMG_STRENGTH,
                        help="Strength of image conditioning (0.0-1.0)")
    parser.add_argument("--eager", action="store_true",
                        help="Skip torch.compile, run in eager mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Input image not found: {args.image}")

    if not args.eager:
        setup_neuron_env(neff_cache=args.neff_cache)

    run_img2vid_inference(
        image=args.image,
        prompt=args.prompt,
        model_id=args.model_id,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_steps=args.num_steps,
        guidance=args.guidance,
        seed=args.seed,
        output=args.output,
        neff_cache=args.neff_cache,
        img_strength=args.img_strength,
        eager=args.eager,
    )
