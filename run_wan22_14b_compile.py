"""
WAN 2.2 T2V-A14B — Inference on Neuron with torch.compile (Beta 3)

Follows the Qwen2 torch_compile pattern:
  1. dist.init_process_group(backend="neuron")
  2. Load WAN 2.2 pipeline (MoE: 2 x 14B transformers)
  3. Move transformer to torch.device("neuron")
  4. torch.compile(transformer.forward, backend="neuron", fullgraph=True, dynamic=False)
  5. Run denoising with MoE switching (timestep >= 875 -> expert1, < 875 -> expert2)
  6. VAE decode on CPU (safe fallback) or Neuron

WAN 2.2 A14B has two 14B transformer experts (~28GB each in bfloat16).
Each expert needs TP across multiple NeuronCores to fit in HBM.
Use torchrun --nproc-per-node 4 (or 8) for tensor parallelism.

NOTE: This is an experimental approach. The WAN 2.2 transformer uses
standard attention that may or may not be fullgraph-compatible with
the Neuron compiler. If torch.compile fails, fall back to --eager mode.

Usage:
    # Eager mode on Neuron (no compilation, works immediately):
    torchrun --nproc-per-node 1 run_wan22_14b_compile.py \\
        --prompt "A cat walks on grass, realistic" \\
        --height 480 --width 832 --num-frames 81 --num-steps 40 --eager

    # torch.compile mode (TP=1, single core, may OOM for 14B):
    torchrun --nproc-per-node 1 run_wan22_14b_compile.py \\
        --prompt "A cat walks on grass, realistic" \\
        --height 384 --width 640 --num-frames 17 --num-steps 20

    # Quick smoke test (very small resolution, eager):
    torchrun --nproc-per-node 1 run_wan22_14b_compile.py \\
        --prompt "A cat" --eager \\
        --height 256 --width 256 --num-frames 1 --num-steps 5
"""

import argparse
import logging
import os
import time

import torch
import torch_neuronx
import torch.distributed as dist

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
CACHE_DIR = "/mnt/nvme/wan2.2_t2v_a14b_hf_cache_dir"
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 832
DEFAULT_NUM_FRAMES = 81
DEFAULT_NUM_STEPS = 40
DEFAULT_GUIDANCE = 4.0
DEFAULT_GUIDANCE_2 = 3.0
DEFAULT_SEED = 42
DEFAULT_OUTPUT = "/mnt/nvme/outputs/wan22_14b_compile.mp4"
DEFAULT_NEFF_CACHE = "/mnt/nvme/neff_cache_wan22"
MoE_BOUNDARY = 875  # timestep threshold for expert switching

# torch.compile config
torch._dynamo.config.cache_size_limit = 64
torch.set_default_dtype(torch.float32)


# ============================================================================
# Neuron environment
# ============================================================================

def setup_neuron_env(neff_cache: str = DEFAULT_NEFF_CACHE):
    """Configure Beta 3 environment."""
    os.environ["NEURON_CC_FLAGS"] = "-O1 --auto-cast=none"
    os.environ["TORCH_NEURONX_ENABLE_ASYNC_NRT"] = "1"
    os.makedirs(neff_cache, exist_ok=True)
    os.environ["NEURON_COMPILE_CACHE_URL"] = f"file://{neff_cache}"
    os.environ["NEURONX_CACHE"] = neff_cache
    logger.info(f"Neuron env configured. NEFF cache: {neff_cache}")


# ============================================================================
# Pipeline
# ============================================================================

def run_wan22_inference(**kwargs):
    """
    Run WAN 2.2 T2V-A14B inference on Neuron using torch.compile pattern.

    MoE Architecture:
    - transformer (transformer_1): high-noise expert, used when timestep >= 875
    - transformer_2: low-noise expert, used when timestep < 875
    """
    prompt = kwargs.get("prompt", "A cat walks on the grass, realistic")
    negative_prompt = kwargs.get("negative_prompt", "blurry, low quality, deformed")
    height = kwargs.get("height", DEFAULT_HEIGHT)
    width = kwargs.get("width", DEFAULT_WIDTH)
    num_frames = kwargs.get("num_frames", DEFAULT_NUM_FRAMES)
    num_steps = kwargs.get("num_steps", DEFAULT_NUM_STEPS)
    guidance = kwargs.get("guidance", DEFAULT_GUIDANCE)
    guidance_2 = kwargs.get("guidance_2", DEFAULT_GUIDANCE_2)
    seed = kwargs.get("seed", DEFAULT_SEED)
    output_path = kwargs.get("output", DEFAULT_OUTPUT)
    eager = kwargs.get("eager", False)
    neff_cache = kwargs.get("neff_cache", DEFAULT_NEFF_CACHE)
    model_id = kwargs.get("model_id", MODEL_ID)

    # --- Step 1: Initialize distributed (Neuron backend) ---
    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    logger.info(f"Rank {rank}/{world_size}: WAN 2.2 T2V-A14B Inference (torch.compile)")
    logger.info(f"  Model:      {model_id}")
    logger.info(f"  Prompt:     {prompt}")
    logger.info(f"  Resolution: {width}x{height}, {num_frames} frames")
    logger.info(f"  Steps:      {num_steps}, CFG: {guidance}/{guidance_2}")
    logger.info(f"  Mode:       {'eager' if eager else 'torch.compile'}")

    device = torch.device("neuron")
    logger.info(f"Rank {rank}: Using neuron device")

    # --- Step 2: Load pipeline on CPU ---
    from diffusers import AutoencoderKLWan, WanTransformer3DModel, UniPCMultistepScheduler
    from transformers import AutoTokenizer, UMT5EncoderModel

    logger.info(f"Rank {rank}: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer", cache_dir=CACHE_DIR)

    logger.info(f"Rank {rank}: Loading text encoder (UMT5-XXL)...")
    t0 = time.time()
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16, cache_dir=CACHE_DIR
    ).eval()
    logger.info(f"Rank {rank}: Text encoder loaded in {time.time()-t0:.1f}s")

    logger.info(f"Rank {rank}: Loading transformer_1 (high-noise expert, 14B)...")
    t0 = time.time()
    transformer_1 = WanTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16, cache_dir=CACHE_DIR
    ).eval()
    n_params = sum(p.numel() for p in transformer_1.parameters()) / 1e9
    logger.info(f"Rank {rank}: Transformer_1 loaded in {time.time()-t0:.1f}s ({n_params:.2f}B params)")

    logger.info(f"Rank {rank}: Loading transformer_2 (low-noise expert, 14B)...")
    t0 = time.time()
    transformer_2 = WanTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer_2", torch_dtype=torch.bfloat16, cache_dir=CACHE_DIR
    ).eval()
    logger.info(f"Rank {rank}: Transformer_2 loaded in {time.time()-t0:.1f}s")

    logger.info(f"Rank {rank}: Loading VAE...")
    t0 = time.time()
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir=CACHE_DIR
    ).eval()
    logger.info(f"Rank {rank}: VAE loaded in {time.time()-t0:.1f}s")

    logger.info(f"Rank {rank}: Loading scheduler...")
    scheduler = UniPCMultistepScheduler.from_pretrained(model_id, subfolder="scheduler", cache_dir=CACHE_DIR)

    # --- Step 3: Move transformers to Neuron device ---
    logger.info(f"Rank {rank}: Moving transformer_1 to {device}...")
    t0 = time.time()
    transformer_1 = transformer_1.to(device)
    logger.info(f"Rank {rank}: Transformer_1 on device in {time.time()-t0:.1f}s")

    logger.info(f"Rank {rank}: Moving transformer_2 to {device}...")
    t0 = time.time()
    transformer_2 = transformer_2.to(device)
    logger.info(f"Rank {rank}: Transformer_2 on device in {time.time()-t0:.1f}s")

    # --- Step 4: torch.compile (unless --eager) ---
    if not eager:
        logger.info(f"Rank {rank}: Compiling transformers with torch.compile(backend='neuron')...")
        transformer_1.forward = torch.compile(
            transformer_1.forward, backend="neuron", fullgraph=True, dynamic=False
        )
        transformer_2.forward = torch.compile(
            transformer_2.forward, backend="neuron", fullgraph=True, dynamic=False
        )
        logger.info(f"Rank {rank}: Compilation registered (NEFFs built on first pass)")
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
        negative_prompt, max_length=512, padding="max_length", truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        neg_embeds = text_encoder(
            input_ids=neg_inputs["input_ids"],
            attention_mask=neg_inputs["attention_mask"],
        ).last_hidden_state.to(torch.bfloat16)
    logger.info(f"Rank {rank}: Text encoded in {time.time()-t0:.1f}s")

    # Free text encoder memory
    del text_encoder
    import gc; gc.collect()

    # Prepare latents
    latent_ch = transformer_1.config.in_channels
    latent_h = height // 8
    latent_w = width // 8
    latent_t = (num_frames - 1) // 4 + 1 if num_frames > 1 else 1
    latents = torch.randn(1, latent_ch, latent_t, latent_h, latent_w, dtype=torch.float32)
    logger.info(f"Rank {rank}: Latent shape: {list(latents.shape)}")

    # Denoising loop with MoE switching
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps

    # Determine switch point
    switch_idx = None
    for i, t in enumerate(timesteps):
        if t < MoE_BOUNDARY:
            switch_idx = i
            break
    if switch_idx is None:
        switch_idx = len(timesteps)

    logger.info(f"Rank {rank}: Denoising ({num_steps} steps, MoE switch at step {switch_idx})...")
    logger.info(f"  transformer_1 (high-noise): steps 0-{switch_idx-1}")
    logger.info(f"  transformer_2 (low-noise):  steps {switch_idx}-{len(timesteps)-1}")

    denoise_t0 = time.time()
    for i, t in enumerate(timesteps):
        step_t0 = time.time()

        # Select expert based on timestep
        if i < switch_idx:
            transformer = transformer_1
            cfg_scale = guidance
        else:
            transformer = transformer_2
            cfg_scale = guidance_2

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
            if cfg_scale > 1.0:
                noise_uncond = transformer(
                    hidden_states=x, timestep=t_in,
                    encoder_hidden_states=ne, return_dict=False,
                )[0]
                noise_pred = noise_uncond + cfg_scale * (noise_pred - noise_uncond)

        # Scheduler step (CPU)
        latents = scheduler.step(noise_pred.cpu(), t, latents, return_dict=False)[0]

        step_time = time.time() - step_t0
        if (i + 1) % 5 == 0 or (i + 1) == num_steps or i == switch_idx:
            expert = "T1" if i < switch_idx else "T2"
            logger.info(f"  Step {i+1}/{num_steps} [{expert}] (t={t.item():.0f}): {step_time:.2f}s/step")

    denoise_time = time.time() - denoise_t0
    logger.info(f"Rank {rank}: Denoising done in {denoise_time:.1f}s ({denoise_time/num_steps:.2f}s/step)")

    # --- Step 6: VAE decode (CPU fallback for safety) ---
    logger.info(f"Rank {rank}: VAE decoding (CPU)...")
    t0 = time.time()
    with torch.no_grad():
        latents_mean = torch.tensor(vae.config.latents_mean, dtype=latents.dtype).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(vae.config.latents_std, dtype=latents.dtype).view(1, -1, 1, 1, 1)
        latents_scaled = latents * latents_std + latents_mean
        video = vae.decode(latents_scaled.to(torch.float32)).sample
    logger.info(f"Rank {rank}: VAE decode in {time.time()-t0:.1f}s, shape={video.shape}")

    total_time = time.time() - total_t0
    logger.info(f"Rank {rank}: TOTAL INFERENCE: {total_time:.1f}s ({total_time/60:.1f} min)")

    # --- Step 7: Save output (rank 0 only) ---
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
    import imageio

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
        description="WAN 2.2 T2V-A14B inference on Neuron (torch.compile pattern)"
    )
    parser.add_argument("--prompt", type=str, default="A close-up photograph of a beautiful fluffy orange tabby cat sitting in a sunlit garden, photorealistic, sharp focus, bokeh background, 8k")
    parser.add_argument("--negative-prompt", type=str, default="blurry, low quality, deformed, ugly, cartoon, anime, painting, text, watermark")
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE)
    parser.add_argument("--guidance-2", type=float, default=DEFAULT_GUIDANCE_2)
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

    run_wan22_inference(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        model_id=args.model_id,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_steps=args.num_steps,
        guidance=args.guidance,
        guidance_2=args.guidance_2,
        seed=args.seed,
        output=args.output,
        neff_cache=args.neff_cache,
        eager=args.eager,
    )
