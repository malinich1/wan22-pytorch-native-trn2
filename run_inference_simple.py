"""
WAN 2.2 T2V-A14B — Simplified PyTorch Native Inference (Single-Core Starter)

This is a simplified version that runs on a single NeuronCore to validate
the pipeline logic before scaling to distributed TP/CP.

Key simplifications:
1. Single-process, single-core execution
2. Eager mode by default (no compilation initially)
3. Uses HuggingFace DiffusionPipeline directly
4. Smaller test resolution (384x640, 17 frames)

Usage:
    # CPU eager mode (for testing without Neuron hardware)
    python run_inference_simple.py --prompt "A cat walks on grass" --device cpu --eager

    # Single NeuronCore (on trn2 instance)
    python run_inference_simple.py --prompt "A cat walks on grass" --device neuron

    # Generate image instead of video (faster)
    python run_inference_simple.py --prompt "A cat" --image --device cpu
"""

import os
import sys
import time
import argparse
import torch
from pathlib import Path

DEFAULT_MODEL_DIR = "/mnt/nvme/models/Wan2.2-T2V-A14B-Diffusers"
DEFAULT_OUTPUT_DIR = "/mnt/nvme/outputs"


def setup_environment(device: str):
    """Setup environment variables for Neuron."""
    if device == "neuron":
        # Basic Neuron configuration
        os.environ["NEURON_CC_FLAGS"] = "-O1"
        os.environ["NEURON_RT_NUM_CORES"] = "1"  # Single core for now
        print("Neuron environment configured (single core)")


def generate_image(
    prompt: str,
    model_dir: str,
    output_path: str,
    device: str = "cpu",
    height: int = 384,
    width: int = 640,
    num_steps: int = 20,
    guidance_scale: float = 5.0,
    seed: int = 42,
):
    """Generate a single image using WAN 2.2 pipeline."""
    from diffusers import WanPipeline
    from PIL import Image

    print("=" * 60)
    print("WAN 2.2 Image Generation (Simplified)")
    print("=" * 60)
    print(f"Prompt:     {prompt}")
    print(f"Resolution: {width}x{height}")
    print(f"Steps:      {num_steps}")
    print(f"Device:     {device}")
    print("=" * 60)

    t0 = time.time()

    # Load pipeline
    print("\nLoading pipeline...")
    load_t0 = time.time()

    try:
        pipeline = WanPipeline.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
    except Exception as e:
        print(f"Error loading WanPipeline: {e}")
        print("\nNote: WanPipeline may not be available in diffusers yet.")
        print("Falling back to manual component loading...")
        return generate_image_manual(
            prompt, model_dir, output_path, device, height, width,
            num_steps, guidance_scale, seed
        )

    if device == "neuron":
        try:
            import torch_neuronx
            print("Moving pipeline to Neuron device...")
            pipeline = pipeline.to("neuron")
        except Exception as e:
            print(f"Warning: Could not move to Neuron device: {e}")
            print("Falling back to CPU")
            device = "cpu"
            pipeline = pipeline.to("cpu")
    else:
        pipeline = pipeline.to(device)

    print(f"Pipeline loaded in {time.time() - load_t0:.1f}s")

    # Generate image
    print("\nGenerating image...")
    torch.manual_seed(seed)

    gen_t0 = time.time()
    output = pipeline(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=1,  # Single frame = image
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
    )
    gen_time = time.time() - gen_t0

    # Save
    from PIL import Image as PILImage
    import numpy as np
    frame = output.frames[0][0]  # First frame of first video
    if isinstance(frame, np.ndarray):
        # Convert float [0,1] array to uint8 [0,255]
        if frame.dtype != np.uint8:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        # Handle single-pixel height dimension (1, H, W, 3) -> (H, W, 3)
        if frame.ndim == 4:
            frame = frame.squeeze(0)
        image = PILImage.fromarray(frame)
    else:
        image = frame
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)

    total_time = time.time() - t0

    print("\n" + "=" * 60)
    print("✅ Image Generated")
    print("=" * 60)
    print(f"Output:     {output_path}")
    print(f"Load time:  {load_t0:.1f}s")
    print(f"Gen time:   {gen_time:.1f}s")
    print(f"Total time: {total_time:.1f}s")
    print("=" * 60)


def generate_image_manual(
    prompt: str,
    model_dir: str,
    output_path: str,
    device: str = "cpu",
    height: int = 384,
    width: int = 640,
    num_steps: int = 20,
    guidance_scale: float = 5.0,
    seed: int = 42,
):
    """Manual pipeline implementation when WanPipeline is not available."""
    from transformers import T5EncoderModel, AutoTokenizer
    from diffusers import WanTransformer3DModel, AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
    from PIL import Image
    import imageio

    print("\n--- Manual Pipeline Mode ---")
    print("Loading components individually...")

    t0 = time.time()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, subfolder="tokenizer")

    # Load text encoder
    print("Loading text encoder...")
    text_encoder = T5EncoderModel.from_pretrained(
        model_dir,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
    )
    text_encoder.eval()
    text_encoder = text_encoder.to(device)

    # Load transformer (DiT)
    print("Loading transformer...")
    transformer = WanTransformer3DModel.from_pretrained(
        model_dir,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    transformer.eval()
    transformer = transformer.to(device)

    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        model_dir,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    )
    vae.eval()
    vae = vae.to(device)

    # Load scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_dir, subfolder="scheduler"
    )

    print(f"All components loaded in {time.time() - t0:.1f}s")

    # Encode prompt
    print("\nEncoding prompt...")
    inputs = tokenizer(
        prompt,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        text_embeddings = text_encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        ).last_hidden_state

    # Prepare latents
    torch.manual_seed(seed)
    latent_h = height // 8
    latent_w = width // 8
    latent_t = 1  # Single frame

    latents = torch.randn(
        1, 16, latent_t, latent_h, latent_w,
        dtype=torch.bfloat16,
        device=device,
    )

    # Denoising loop
    print(f"Denoising ({num_steps} steps)...")
    scheduler.set_timesteps(num_steps)

    denoise_t0 = time.time()
    for i, t in enumerate(scheduler.timesteps):
        # Classifier-free guidance
        latent_input = torch.cat([latents, latents], dim=0)
        text_input = torch.cat([
            torch.zeros_like(text_embeddings),
            text_embeddings
        ], dim=0)
        t_input = t.expand(2).to(device)

        with torch.no_grad():
            noise_pred = transformer(
                hidden_states=latent_input,
                timestep=t_input,
                encoder_hidden_states=text_input,
            ).sample

        noise_uncond, noise_cond = noise_pred.chunk(2, dim=0)
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

        if (i + 1) % 5 == 0:
            print(f"  Step {i+1}/{num_steps}")

    denoise_time = time.time() - denoise_t0
    print(f"Denoising complete: {denoise_time:.1f}s")

    # Decode
    print("Decoding...")
    decode_t0 = time.time()
    with torch.no_grad():
        latents_scaled = latents / vae.config.scaling_factor
        image_tensor = vae.decode(latents_scaled).sample

    decode_time = time.time() - decode_t0

    # Convert to PIL
    image_tensor = image_tensor.squeeze(0).squeeze(1)  # Remove batch and time
    image_tensor = (image_tensor.permute(1, 2, 0) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    image = Image.fromarray(image_tensor)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)

    total_time = time.time() - t0

    print("\n" + "=" * 60)
    print("✅ Image Generated (Manual Pipeline)")
    print("=" * 60)
    print(f"Output:        {output_path}")
    print(f"Denoise time:  {denoise_time:.1f}s ({denoise_time/num_steps*1000:.0f}ms/step)")
    print(f"Decode time:   {decode_time:.1f}s")
    print(f"Total time:    {total_time:.1f}s")
    print("=" * 60)


def generate_video(
    prompt: str,
    model_dir: str,
    output_path: str,
    device: str = "cpu",
    height: int = 384,
    width: int = 640,
    num_frames: int = 17,
    num_steps: int = 20,
    guidance_scale: float = 5.0,
    seed: int = 42,
    fps: int = 16,
):
    """Generate video using manual pipeline (WanPipeline may not support video yet)."""
    from transformers import T5EncoderModel, AutoTokenizer
    from diffusers import WanTransformer3DModel, AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
    import imageio

    print("=" * 60)
    print("WAN 2.2 Video Generation (Simplified)")
    print("=" * 60)
    print(f"Prompt:     {prompt}")
    print(f"Resolution: {width}x{height}, {num_frames} frames")
    print(f"Steps:      {num_steps}")
    print(f"Device:     {device}")
    print("=" * 60)

    t0 = time.time()

    # Load components
    print("\nLoading pipeline components...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, subfolder="tokenizer")

    text_encoder = T5EncoderModel.from_pretrained(
        model_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    text_encoder.eval().to(device)

    transformer = WanTransformer3DModel.from_pretrained(
        model_dir, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    transformer.eval().to(device)

    vae = AutoencoderKLWan.from_pretrained(
        model_dir, subfolder="vae", torch_dtype=torch.bfloat16
    )
    vae.eval().to(device)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_dir, subfolder="scheduler"
    )

    print(f"Components loaded in {time.time() - t0:.1f}s")

    # Encode text
    print("\nEncoding text...")
    inputs = tokenizer(prompt, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        text_embeddings = text_encoder(**inputs).last_hidden_state

    # Prepare latents
    torch.manual_seed(seed)
    latent_h = height // 8
    latent_w = width // 8
    latent_t = (num_frames - 1) // 4 + 1

    latents = torch.randn(
        1, 16, latent_t, latent_h, latent_w,
        dtype=torch.bfloat16,
        device=device,
    )

    # Denoise
    print(f"\nDenoising ({num_steps} steps)...")
    scheduler.set_timesteps(num_steps)

    denoise_t0 = time.time()
    for i, t in enumerate(scheduler.timesteps):
        latent_input = torch.cat([latents, latents], dim=0)
        text_input = torch.cat([torch.zeros_like(text_embeddings), text_embeddings], dim=0)
        t_input = t.expand(2).to(device)

        with torch.no_grad():
            noise_pred = transformer(
                hidden_states=latent_input,
                timestep=t_input,
                encoder_hidden_states=text_input,
            ).sample

        noise_uncond, noise_cond = noise_pred.chunk(2, dim=0)
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        if (i + 1) % 5 == 0:
            print(f"  Step {i+1}/{num_steps}")

    denoise_time = time.time() - denoise_t0

    # Decode
    print("Decoding video...")
    decode_t0 = time.time()
    with torch.no_grad():
        video = vae.decode(latents / vae.config.scaling_factor).sample
    decode_time = time.time() - decode_t0

    # Save
    video_np = video.squeeze(0).permute(1, 2, 3, 0)
    video_np = (video_np * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
    for frame in video_np:
        writer.append_data(frame)
    writer.close()

    total_time = time.time() - t0

    print("\n" + "=" * 60)
    print("✅ Video Generated")
    print("=" * 60)
    print(f"Output:        {output_path}")
    print(f"File size:     {os.path.getsize(output_path)/1e6:.1f} MB")
    print(f"Denoise time:  {denoise_time:.1f}s ({denoise_time/num_steps*1000:.0f}ms/step)")
    print(f"Decode time:   {decode_time:.1f}s")
    print(f"Total time:    {total_time:.1f}s")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="WAN 2.2 Simplified Inference")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "neuron"], default="cpu")
    parser.add_argument("--image", action="store_true", help="Generate image instead of video")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--eager", action="store_true", help="Unused (always eager in simple version)")
    args = parser.parse_args()

    # Setup
    setup_environment(args.device)

    # Determine output path
    if args.output is None:
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ext = "png" if args.image else "mp4"
        args.output = os.path.join(DEFAULT_OUTPUT_DIR, f"wan22_simple_{timestamp}.{ext}")

    # Generate
    if args.image:
        generate_image(
            prompt=args.prompt,
            model_dir=args.model_dir,
            output_path=args.output,
            device=args.device,
            height=args.height,
            width=args.width,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
    else:
        generate_video(
            prompt=args.prompt,
            model_dir=args.model_dir,
            output_path=args.output,
            device=args.device,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            fps=args.fps,
        )

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
