"""
WAN 2.1 T2V-1.3B — Inference on Neuron with torch.compile (Beta 3)

Follows the same pattern as the Qwen2 torch_compile example:
  1. dist.init_process_group(backend="neuron")
  2. Load model on CPU → move to torch.neuron.current_device()
  3. torch.compile(model.forward, backend="neuron", fullgraph=True, dynamic=False)
  4. Run inference

The 1.3B model (~2.6 GB in bfloat16) easily fits on a single NeuronCore pair (LNC2 = 48 GB HBM).
No tensor parallelism needed — single-process, single-device.

DLC: 421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest

Usage:
    # Single NeuronCore, torch.compile:
    torchrun --nproc-per-node 1 run_wan_small.py \
        --prompt "A cat walks through a garden" \
        --height 480 --width 832 --num-frames 81 --num-steps 30

    # Quick test (smaller resolution):
    torchrun --nproc-per-node 1 run_wan_small.py \
        --prompt "A cat walks on grass" \
        --height 384 --width 640 --num-frames 17 --num-steps 20

    # Eager mode (no compilation, instant start):
    torchrun --nproc-per-node 1 run_wan_small.py \
        --prompt "A cat" --eager \
        --height 256 --width 256 --num-frames 1 --num-steps 5
"""

import argparse
import logging
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 832
DEFAULT_NUM_FRAMES = 81
DEFAULT_NUM_STEPS = 30
DEFAULT_GUIDANCE = 5.0
DEFAULT_SEED = 42
DEFAULT_OUTPUT = "/mnt/nvme/outputs/wan_small_beta3.mp4"
DEFAULT_NEFF_CACHE = "/mnt/nvme/neff_cache"

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

    # Persistent NEFF cache
    os.makedirs(neff_cache, exist_ok=True)
    os.environ["NEURON_COMPILE_CACHE_URL"] = f"file://{neff_cache}"
    os.environ["NEURONX_CACHE"] = neff_cache

    logger.info(f"Neuron env configured. NEFF cache: {neff_cache}")


# ============================================================================
# Pipeline
# ============================================================================

def run_wan_inference(**kwargs):
    """
    Run WAN 2.1 1.3B T2V inference on Neuron following Qwen2 example pattern.

    Steps:
      1. init_process_group(backend="neuron")
      2. Load pipeline components on CPU
      3. Move transformer to neuron device
      4. torch.compile the transformer forward
      5. Run diffusion pipeline
      6. Save output
    """
    prompt = kwargs.get("prompt", "A cat walks on grass")
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

    # --- Step 1: Initialize distributed (Neuron backend) ---
    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    logger.info(f"Rank {rank}/{world_size}: WAN 2.1 1.3B T2V Inference")
    logger.info(f"  Model:      {model_id}")
    logger.info(f"  Prompt:     {prompt}")
    logger.info(f"  Resolution: {width}x{height}, {num_frames} frames")
    logger.info(f"  Steps:      {num_steps}, CFG: {guidance}")
    logger.info(f"  Mode:       {'eager' if eager else 'torch.compile'}")

    # Get Neuron device for this rank
    neuron_device_idx = torch.neuron.current_device()
    device = torch.device("neuron")
    logger.info(f"Rank {rank}: Using device neuron (index {neuron_device_idx})")

    # --- Step 2: Load pipeline components on CPU ---
    from diffusers import AutoencoderKLWan, WanTransformer3DModel, UniPCMultistepScheduler
    from transformers import AutoTokenizer, UMT5EncoderModel

    logger.info(f"Rank {rank}: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")

    logger.info(f"Rank {rank}: Loading text encoder (UMT5)...")
    t0 = time.time()
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16,
    ).eval()
    logger.info(f"Rank {rank}: Text encoder loaded in {time.time()-t0:.1f}s (stays on CPU)")

    logger.info(f"Rank {rank}: Loading transformer (1.3B)...")
    t0 = time.time()
    transformer = WanTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
    ).eval()
    n_params = sum(p.numel() for p in transformer.parameters()) / 1e9
    logger.info(f"Rank {rank}: Transformer loaded in {time.time()-t0:.1f}s ({n_params:.2f}B params)")

    logger.info(f"Rank {rank}: Loading VAE...")
    t0 = time.time()
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.bfloat16,
    ).eval()
    logger.info(f"Rank {rank}: VAE loaded in {time.time()-t0:.1f}s")

    logger.info(f"Rank {rank}: Loading scheduler...")
    scheduler = UniPCMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")

    # --- Step 3: Move transformer + VAE to Neuron device ---
    logger.info(f"Rank {rank}: Moving transformer to {device}...")
    t0 = time.time()
    transformer = transformer.to(device)
    logger.info(f"Rank {rank}: Transformer on device in {time.time()-t0:.1f}s")

    logger.info(f"Rank {rank}: Moving VAE to {device}...")
    t0 = time.time()
    vae = vae.to(device)
    logger.info(f"Rank {rank}: VAE on device in {time.time()-t0:.1f}s")

    # --- Step 4: torch.compile (unless --eager) ---
    if not eager:
        logger.info(f"Rank {rank}: Compiling transformer with torch.compile(backend='neuron')...")
        transformer.forward = torch.compile(
            transformer.forward, backend="neuron", fullgraph=True, dynamic=False
        )
        # Note: VAE decode uses tiled/loop-based decoding which isn't fullgraph-compatible.
        # Keep VAE in eager mode on Neuron device — it's a single pass anyway.
        logger.info(f"Rank {rank}: Compilation registered (NEFFs built on first pass)")
        logger.info(f"Rank {rank}: VAE stays in eager mode (tiled decode not fullgraph-compatible)")
    else:
        logger.info(f"Rank {rank}: Eager mode — skipping torch.compile")

    dist.barrier()

    # --- Step 5: Run inference ---
    logger.info(f"Rank {rank}: Starting inference...")
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

    # Prepare latents
    latent_ch = transformer.config.in_channels
    latent_h = height // 8
    latent_w = width // 8
    latent_t = (num_frames - 1) // 4 + 1 if num_frames > 1 else 1
    latents = torch.randn(1, latent_ch, latent_t, latent_h, latent_w, dtype=torch.float32)
    logger.info(f"Rank {rank}: Latent shape: {list(latents.shape)}")

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

    # --- Step 6: Save output (rank 0 only) ---
    if rank == 0:
        save_output(video, output_path, num_frames)

    dist.barrier()
    dist.destroy_process_group()
    logger.info(f"Rank {rank}: Done.")


# ============================================================================
# Output saving
# ============================================================================

def save_output(video: torch.Tensor, output_path: str, num_frames: int, fps: int = 16):
    """Save video/image output."""
    from PIL import Image

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    video = video.squeeze(0).permute(1, 2, 3, 0)  # (T, H, W, C)
    video = ((video.float() / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8).numpy()

    if video.shape[0] == 1 or num_frames == 1:
        output_path = output_path.rsplit(".", 1)[0] + ".png"
        img = Image.fromarray(video[0])
        img.save(output_path)
        size_kb = os.path.getsize(output_path) / 1024
        logger.info(f"Saved image: {output_path} ({size_kb:.0f} KB)")
    else:
        import imageio
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
        description="WAN 2.1 T2V-1.3B inference on Neuron (Beta 3, torch.compile pattern)"
    )
    parser.add_argument("--prompt", type=str, default="A cat walks through a sunlit garden")
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--neff-cache", type=str, default=DEFAULT_NEFF_CACHE)
    parser.add_argument("--eager", action="store_true",
                        help="Skip torch.compile, run in eager mode on Neuron device")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.eager:
        setup_neuron_env(neff_cache=args.neff_cache)

    run_wan_inference(
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
        eager=args.eager,
    )
