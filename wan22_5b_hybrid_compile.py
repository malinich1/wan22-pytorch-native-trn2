"""
WAN 2.2 TI2V-5B — Native PyTorch Inference on Neuron with torch.compile (Beta 3 DLC)

Generates text-to-video (T2V) using the WAN 2.2 TI2V-5B (5B dense) model on
Trainium 2 using the PyTorch Native approach:
  1. dist.init_process_group(backend="neuron")
  2. Load model on CPU → move to torch.device("neuron")
  3. torch.compile(model.forward, backend="neuron", fullgraph=True, dynamic=False)
  4. Run inference

The 5B model (~10 GB in bfloat16) fits on a single NeuronCore pair (LNC=2 = 48 GB HBM).
No tensor parallelism needed — single-process, single-device.

Model: Wan-AI/Wan2.2-TI2V-5B-Diffusers
  - 5B dense transformer (NOT MoE — single transformer, no expert switching)
  - Supports both T2V and I2V (via expand_timesteps=True)
  - UMT5-XXL text encoder
  - WAN 2.2 VAE (16x16x4 compression)

DLC: 421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest
  - PyTorch 2.11, torch-neuronx 2.11.3, neuronx-cc 2.25
  - Has backend="neuron" registered for torch.compile

Usage (inside Beta 3 DLC container):
    # Text-to-Video (default):
    torchrun --nproc-per-node 1 wan22_5b_hybrid_compile.py \\
        --prompt "A fluffy orange cat walking through a garden" \\
        --height 480 --width 832 --num-frames 33 --num-steps 30

    # Higher quality (more steps, higher guidance):
    torchrun --nproc-per-node 1 wan22_5b_hybrid_compile.py \\
        --prompt "A cat sits on a windowsill watching rain" \\
        --height 480 --width 832 --num-frames 81 --num-steps 50 \\
        --guidance 5.0

    # Quick test (single frame, eager mode):
    torchrun --nproc-per-node 1 wan22_5b_hybrid_compile.py \\
        --prompt "A cat" --eager \\
        --height 256 --width 256 --num-frames 1 --num-steps 5

    # 720P (requires sufficient HBM):
    torchrun --nproc-per-node 1 wan22_5b_hybrid_compile.py \\
        --prompt "Ocean waves at sunset" \\
        --height 704 --width 1280 --num-frames 81 --num-steps 50
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

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 832
DEFAULT_NUM_FRAMES = 33
DEFAULT_NUM_STEPS = 30
DEFAULT_GUIDANCE = 5.0
DEFAULT_SEED = 42
DEFAULT_OUTPUT = "/mnt/nvme/outputs/wan22_5b_t2v.mp4"
DEFAULT_NEFF_CACHE = "/mnt/nvme/neff_cache_5b"

# torch.compile config
torch._dynamo.config.cache_size_limit = 64
torch.set_default_dtype(torch.float32)


# ============================================================================
# Neuron environment
# ============================================================================

def setup_neuron_env(neff_cache: str):
    """Configure Beta 3 environment for single-device inference."""
    os.environ["NEURON_CC_FLAGS"] = "-O1 --auto-cast=none"
    os.environ["TORCH_NEURONX_ENABLE_ASYNC_NRT"] = "1"
    os.makedirs(neff_cache, exist_ok=True)
    os.environ["NEURON_COMPILE_CACHE_URL"] = f"file://{neff_cache}"
    os.environ["NEURONX_CACHE"] = neff_cache
    logger.info(f"Neuron env configured. NEFF cache: {neff_cache}")


# ============================================================================
# Pipeline
# ============================================================================

def run_inference(**kwargs):
    """
    Run WAN 2.2 TI2V-5B T2V inference on Neuron using torch.compile.

    Steps:
      1. init_process_group(backend="neuron")
      2. Load pipeline components on CPU
      3. Move transformer to neuron device
      4. torch.compile the transformer forward
      5. Run diffusion with manual denoising loop
      6. Decode with VAE on CPU and save output
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

    # --- Step 1: Initialize distributed (Neuron backend) ---
    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()

    logger.info(f"Rank {rank}: WAN 2.2 TI2V-5B T2V Inference")
    logger.info(f"  Model:      {MODEL_ID}")
    logger.info(f"  Prompt:     {prompt}")
    logger.info(f"  Resolution: {width}x{height}, {num_frames} frames")
    logger.info(f"  Steps:      {num_steps}, CFG: {guidance}")
    logger.info(f"  Mode:       {'eager' if eager else 'torch.compile'}")

    device = torch.device("neuron")

    # --- Step 2: Load pipeline components on CPU ---
    from diffusers import AutoencoderKLWan, WanTransformer3DModel, UniPCMultistepScheduler
    from transformers import AutoTokenizer, UMT5EncoderModel

    logger.info(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")

    logger.info(f"Loading text encoder (UMT5)...")
    t0 = time.time()
    text_encoder = UMT5EncoderModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16,
    ).eval()
    logger.info(f"Text encoder loaded in {time.time()-t0:.1f}s (stays on CPU)")

    logger.info(f"Loading transformer (5B)...")
    t0 = time.time()
    transformer = WanTransformer3DModel.from_pretrained(
        MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16,
    ).eval()
    n_params = sum(p.numel() for p in transformer.parameters()) / 1e9
    logger.info(f"Transformer loaded in {time.time()-t0:.1f}s ({n_params:.2f}B params)")

    logger.info(f"Loading VAE...")
    t0 = time.time()
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.float32,
    ).eval()
    logger.info(f"VAE loaded in {time.time()-t0:.1f}s (stays on CPU for decode)")

    logger.info(f"Loading scheduler...")
    scheduler = UniPCMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    # --- Step 3: Move transformer to Neuron device ---
    logger.info(f"Moving transformer to {device}...")
    t0 = time.time()
    transformer = transformer.to(device)
    logger.info(f"Transformer on device in {time.time()-t0:.1f}s")

    # --- Step 4: torch.compile (unless --eager) ---
    if not eager:
        logger.info(f"Compiling transformer with torch.compile(backend='neuron')...")
        transformer.forward = torch.compile(
            transformer.forward, backend="neuron", fullgraph=True, dynamic=False
        )
        logger.info(f"Compilation registered (NEFFs built on first forward pass)")
    else:
        logger.info(f"Eager mode — skipping torch.compile")

    dist.barrier()

    # --- Step 5: Run inference ---
    logger.info(f"Starting inference...")
    total_t0 = time.time()
    torch.manual_seed(seed)

    # Text encoding (CPU)
    logger.info(f"Encoding text on CPU...")
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
    logger.info(f"Text encoded in {time.time()-t0:.1f}s")

    # Prepare latents
    latent_ch = transformer.config.in_channels
    latent_h = height // 8
    latent_w = width // 8
    latent_t = (num_frames - 1) // 4 + 1 if num_frames > 1 else 1
    latents = torch.randn(1, latent_ch, latent_t, latent_h, latent_w, dtype=torch.float32)
    logger.info(f"Latent shape: {list(latents.shape)}")

    # Denoising loop
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps

    logger.info(f"Denoising ({num_steps} steps)...")
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
    logger.info(f"Denoising done in {denoise_time:.1f}s ({denoise_time/num_steps:.2f}s/step)")

    # VAE decode (CPU)
    logger.info(f"VAE decoding on CPU...")
    t0 = time.time()
    with torch.no_grad():
        latents_mean = torch.tensor(vae.config.latents_mean, dtype=latents.dtype).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(vae.config.latents_std, dtype=latents.dtype).view(1, -1, 1, 1, 1)
        latents_scaled = latents * latents_std + latents_mean
        video = vae.decode(latents_scaled).sample
    logger.info(f"VAE decode in {time.time()-t0:.1f}s, shape={video.shape}")

    total_time = time.time() - total_t0
    logger.info(f"TOTAL INFERENCE: {total_time:.1f}s ({total_time/60:.1f} min)")

    # --- Step 6: Save output ---
    if rank == 0:
        save_output(video, output_path, num_frames)

    dist.barrier()
    dist.destroy_process_group()
    logger.info(f"Done.")


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
        description="WAN 2.2 TI2V-5B T2V inference on Neuron (Beta 3, torch.compile)"
    )
    parser.add_argument("--prompt", type=str, default="A fluffy orange tabby cat walking through a sunlit garden, realistic, cinematic")
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

    run_inference(
        prompt=args.prompt,
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
