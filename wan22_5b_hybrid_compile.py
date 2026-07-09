"""
WAN 2.2 TI2V-5B — Native PyTorch Inference on Neuron (torch.compile)

⚠️  KNOWN LIMITATION: torch.compile FAILS for WAN 2.2 TI2V-5B on current Neuron SDK.
    neuronx-cc crashes with exit code 70 at ALL resolutions (even 256x256x1).
    Root cause: The 5B transformer with 48 input channels + 30 layers creates a
    computation graph that exceeds the compiler's internal complexity limits.

    WORKING ALTERNATIVE: Use the NxDModel approach (AOT compilation via ModelBuilder):
      - github.com/malinich1/NeuronStuff/Wan2.2-TI2V-5B/ (trn2.3xlarge notebook)
      - github.com/whn09/aws-neuron-samples/torch-neuronx/inference/hf_pretrained_wan2.2_ti2v
      - Venv: /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference (SDK 2.29)
      - Parallelism: TP=4, world_size=4

    This script is kept as a reference and will work once neuronx-cc supports
    larger transformer graphs (expected in future SDK releases).

Uses the PyTorch Native pattern from the Trainium2 workshop:
  1. model.to("neuron")
  2. torch.compile(model, backend="neuron")
  3. Run inference

No distributed setup needed — single NeuronCore, single process.

Model: Wan-AI/Wan2.2-TI2V-5B-Diffusers
  - 5B dense transformer (supports both T2V and I2V)
  - in_channels=48 (combined video+image conditioning)
  - UMT5-XXL text encoder (CPU)
  - WAN 2.2 VAE (CPU decode)

Environment:
  - Instance: trn2.48xlarge (or any trn2 with >= 1 NeuronCore)
  - Venv: /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference (SDK 2.29+)
    OR Beta 3 DLC container (PyTorch 2.11, torch-neuronx 2.11.3)
  - Required: backend="neuron" support in torch.compile

Usage:
    # Single process, no torchrun needed:
    python wan22_5b_hybrid_compile.py \\
        --prompt "A fluffy orange cat walking through a garden" \\
        --height 480 --width 832 --num-frames 33 --num-steps 30

    # Quick test (small resolution):
    python wan22_5b_hybrid_compile.py \\
        --prompt "A cat" --height 256 --width 256 --num-frames 1 --num-steps 5

    # Eager mode (no compilation, for debugging):
    python wan22_5b_hybrid_compile.py \\
        --prompt "A cat" --eager --height 256 --width 256 --num-frames 1 --num-steps 5

Note: If torch.compile fails with exit code 70, the model graph is too complex
for the current neuronx-cc version. Use the NxDModel approach instead:
  - github.com/malinich1/NeuronStuff/Wan2.2-TI2V-5B/
  - github.com/whn09/aws-neuron-samples/torch-neuronx/inference/hf_pretrained_wan2.2_ti2v
"""

import argparse
import logging
import os
import time

import torch
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


# ============================================================================
# Neuron environment
# ============================================================================

def setup_neuron_env(neff_cache: str):
    """Configure Neuron environment for inference."""
    os.environ.setdefault("NEURON_CC_FLAGS", "-O1 --auto-cast=none")
    os.environ.setdefault("NEURON_RT_VIRTUAL_CORE_SIZE", "2")
    os.environ.setdefault("NEURON_RT_NUM_CORES", "1")
    os.makedirs(neff_cache, exist_ok=True)
    os.environ["NEURON_COMPILE_CACHE_URL"] = f"file://{neff_cache}"
    os.environ["NEURONX_CACHE"] = neff_cache
    logger.info(f"Neuron env configured. NEFF cache: {neff_cache}")


# ============================================================================
# Pipeline
# ============================================================================

def run_inference(args):
    """
    Run WAN 2.2 TI2V-5B T2V inference using PyTorch Native on Neuron.

    Pattern (from Trainium2 workshop):
      1. Load model on CPU
      2. model.to("neuron")
      3. torch.compile(model, backend="neuron")
      4. Run forward passes — first call triggers NEFF compilation
    """
    logger.info(f"WAN 2.2 TI2V-5B T2V Inference (PyTorch Native)")
    logger.info(f"  Model:      {MODEL_ID}")
    logger.info(f"  Prompt:     {args.prompt}")
    logger.info(f"  Resolution: {args.width}x{args.height}, {args.num_frames} frames")
    logger.info(f"  Steps:      {args.num_steps}, CFG: {args.guidance}")
    logger.info(f"  Mode:       {'eager' if args.eager else 'torch.compile(backend=neuron)'}")

    # --- Load pipeline components on CPU ---
    from diffusers import AutoencoderKLWan, WanTransformer3DModel, UniPCMultistepScheduler
    from transformers import AutoTokenizer, UMT5EncoderModel

    logger.info(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")

    logger.info(f"Loading text encoder (UMT5-XXL) on CPU...")
    t0 = time.time()
    text_encoder = UMT5EncoderModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16,
    ).eval()
    logger.info(f"  Text encoder loaded in {time.time()-t0:.1f}s")

    logger.info(f"Loading transformer (5B) on CPU...")
    t0 = time.time()
    transformer = WanTransformer3DModel.from_pretrained(
        MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16,
    ).eval()
    n_params = sum(p.numel() for p in transformer.parameters()) / 1e9
    logger.info(f"  Transformer loaded in {time.time()-t0:.1f}s ({n_params:.2f}B params)")

    logger.info(f"Loading VAE on CPU...")
    t0 = time.time()
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.float32,
    ).eval()
    logger.info(f"  VAE loaded in {time.time()-t0:.1f}s (stays on CPU)")

    logger.info(f"Loading scheduler...")
    scheduler = UniPCMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    # --- Move transformer to Neuron device ---
    logger.info(f"Moving transformer to device('neuron')...")
    t0 = time.time()
    transformer = transformer.to("neuron")
    logger.info(f"  On device in {time.time()-t0:.1f}s")

    # --- torch.compile ---
    if not args.eager:
        logger.info(f"Compiling with torch.compile(backend='neuron', dynamic=False)...")
        transformer = torch.compile(transformer, backend="neuron", dynamic=False)
        logger.info(f"  Compilation registered (NEFFs built on first forward pass)")
    else:
        logger.info(f"  Eager mode — no compilation")

    # --- Text encoding (CPU) ---
    logger.info(f"Encoding text on CPU...")
    total_t0 = time.time()
    t0 = time.time()

    text_inputs = tokenizer(
        args.prompt, max_length=512, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        prompt_embeds = text_encoder(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
        ).last_hidden_state.to(torch.bfloat16)

    neg_inputs = tokenizer(
        "", max_length=512, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        neg_embeds = text_encoder(
            input_ids=neg_inputs["input_ids"],
            attention_mask=neg_inputs["attention_mask"],
        ).last_hidden_state.to(torch.bfloat16)

    text_time = time.time() - t0
    logger.info(f"  Text encoded in {text_time:.1f}s, shape={prompt_embeds.shape}")

    # --- Prepare latents ---
    torch.manual_seed(args.seed)
    latent_ch = transformer.config.in_channels if hasattr(transformer, 'config') else 48
    latent_h = args.height // 8
    latent_w = args.width // 8
    latent_t = (args.num_frames - 1) // 4 + 1 if args.num_frames > 1 else 1

    latents = torch.randn(1, latent_ch, latent_t, latent_h, latent_w, dtype=torch.float32)
    logger.info(f"Latent shape: {list(latents.shape)}")

    # --- Denoising loop ---
    scheduler.set_timesteps(args.num_steps)
    timesteps = scheduler.timesteps
    device = torch.device("neuron")

    logger.info(f"Denoising ({args.num_steps} steps)...")
    denoise_t0 = time.time()

    for i, t in enumerate(timesteps):
        step_t0 = time.time()

        # Move inputs to Neuron
        x = latents.to(device, dtype=torch.bfloat16)
        t_in = t.expand(1).to(device)
        pe = prompt_embeds.to(device)
        ne = neg_embeds.to(device)

        with torch.no_grad():
            # Conditional prediction
            noise_pred = transformer(
                hidden_states=x, timestep=t_in,
                encoder_hidden_states=pe, return_dict=False,
            )[0]

            # Unconditional prediction (CFG)
            if args.guidance > 1.0:
                noise_uncond = transformer(
                    hidden_states=x, timestep=t_in,
                    encoder_hidden_states=ne, return_dict=False,
                )[0]
                noise_pred = noise_uncond + args.guidance * (noise_pred - noise_uncond)

        # Scheduler step on CPU
        latents = scheduler.step(noise_pred.cpu(), t, latents, return_dict=False)[0]

        step_time = time.time() - step_t0
        if (i + 1) % 5 == 0 or (i + 1) == args.num_steps or (i + 1) == 1:
            logger.info(f"  Step {i+1}/{args.num_steps}: {step_time:.2f}s/step")

    denoise_time = time.time() - denoise_t0
    logger.info(f"Denoising done in {denoise_time:.1f}s ({denoise_time/args.num_steps:.2f}s/step avg)")

    # --- VAE decode (CPU) ---
    logger.info(f"VAE decoding on CPU...")
    t0 = time.time()
    with torch.no_grad():
        latents_mean = torch.tensor(vae.config.latents_mean, dtype=latents.dtype).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(vae.config.latents_std, dtype=latents.dtype).view(1, -1, 1, 1, 1)
        latents_scaled = latents * latents_std + latents_mean
        video = vae.decode(latents_scaled).sample
    vae_time = time.time() - t0
    logger.info(f"  VAE decode in {vae_time:.1f}s, shape={video.shape}")

    total_time = time.time() - total_t0
    logger.info(f"TOTAL: {total_time:.1f}s ({total_time/60:.1f} min)")

    # --- Save output ---
    save_output(video, args.output, args.num_frames)

    # --- Summary ---
    logger.info(f"\n{'='*60}")
    logger.info(f"INFERENCE COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Text encoding:  {text_time:.1f}s (CPU)")
    logger.info(f"  Denoising:      {denoise_time:.1f}s ({args.num_steps} steps, {denoise_time/args.num_steps:.2f}s/step)")
    logger.info(f"  VAE decode:     {vae_time:.1f}s (CPU)")
    logger.info(f"  Total:          {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"  Output:         {args.output}")
    logger.info(f"{'='*60}")


# ============================================================================
# Output saving
# ============================================================================

def save_output(video: torch.Tensor, output_path: str, num_frames: int, fps: int = 16):
    """Save video/image output."""
    from PIL import Image

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    video = video.squeeze(0).permute(1, 2, 3, 0)  # [C, T, H, W] -> [T, H, W, C]
    video = ((video.float() / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8).numpy()

    if video.shape[0] == 1 or num_frames == 1:
        output_path = output_path.rsplit(".", 1)[0] + ".png"
        img = Image.fromarray(video[0])
        img.save(output_path)
        logger.info(f"Saved image: {output_path} ({os.path.getsize(output_path)/1024:.0f} KB)")
    else:
        import imageio
        output_path = output_path.rsplit(".", 1)[0] + ".mp4"
        writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
        for frame in video:
            writer.append_data(frame)
        writer.close()
        logger.info(f"Saved video: {output_path} ({os.path.getsize(output_path)/1e6:.1f} MB, {len(video)} frames @ {fps}fps)")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WAN 2.2 TI2V-5B inference on Neuron (PyTorch Native, torch.compile)"
    )
    parser.add_argument("--prompt", type=str,
                        default="A fluffy orange tabby cat walking through a sunlit garden, realistic, cinematic")
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
    args = parser.parse_args()

    setup_neuron_env(neff_cache=args.neff_cache)
    run_inference(args)
