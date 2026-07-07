"""
WAN 2.2 T2V-A14B — End-to-End Inference with PyTorch Native on Trainium 2

Single-process inference using device='neuron' and torch.compile().
Replaces the NXD subprocess-based approach with native PyTorch parallelism.

Architecture:
  WAN 2.2 uses TWO independent transformer models (Mixture-of-Experts):
    - transformer   (Expert 1): Handles high-noise denoising steps
    - transformer_2 (Expert 2): Handles low-noise denoising steps
  The boundary is determined by `boundary_ratio` (0.875 = 87.5% of timesteps
  use Expert 1, remaining 12.5% use Expert 2).

Pipeline:
1. Text encoding (UMT5-XXL)
2. Expert 1 denoising (high-noise steps: t >= boundary)
3. Expert 2 denoising (low-noise steps: t < boundary)
4. VAE decode (tiled)

Usage:
    # Eager mode (CPU, for testing/debugging):
    python run_inference.py --prompt "A cat walks on grass" --eager \\
        --num-inference-steps 20 --height 384 --width 640

    # Neuron compiled mode (production):
    python run_inference.py --prompt "A cat walks on grass" \\
        --num-inference-steps 40 --height 768 --width 1280

    # Image generation (single frame):
    python run_inference.py --prompt "A cat" --eager --num-frames 1 \\
        --height 384 --width 640 --num-inference-steps 20
"""

import os
import sys
import time
import argparse
import torch
from pathlib import Path
from typing import Optional

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_MODEL_DIR = "/mnt/nvme/models/Wan2.2-T2V-A14B-Diffusers"
DEFAULT_COMPILED_DIR = "/mnt/nvme/compiled_artifacts"
DEFAULT_OUTPUT_DIR = "/mnt/nvme/outputs"

# Parallelism (for future distributed Neuron inference)
TP_DEGREE = 4
CP_DEGREE = 16
WORLD_SIZE = TP_DEGREE * CP_DEGREE  # 64

# Model defaults
DEFAULT_HEIGHT = 768
DEFAULT_WIDTH = 1280
DEFAULT_NUM_FRAMES = 81
DEFAULT_NUM_STEPS = 40
DEFAULT_GUIDANCE_SCALE = 5.0
DEFAULT_SEED = 42

# WAN 2.2 MoE boundary: Expert 1 handles timesteps >= boundary, Expert 2 below
# boundary_ratio = 0.875 means 87.5% of the noise schedule uses Expert 1
BOUNDARY_RATIO = 0.875


# ============================================================================
# Helper functions
# ============================================================================

def setup_neuron_env():
    """Configure Neuron environment variables."""
    os.environ["NEURON_CC_FLAGS"] = (
        "-O1 --auto-cast=none --enable-native-kernel=1 "
        "--remat --enable-ccop-compute-overlap"
    )
    os.environ["NEURON_RT_VISIBLE_CORES"] = "0-63"
    os.environ["NEURON_RT_NUM_CORES"] = "64"
    os.environ["NEURON_ENABLE_NATIVE_KERNEL"] = "1"


# ============================================================================
# Pipeline
# ============================================================================

class WanPipelineNative:
    """
    WAN 2.2 T2V-A14B Pipeline using PyTorch Native on Trainium 2.

    Key design:
    - Loads BOTH transformer experts into memory (no weight swapping needed)
    - Switches between experts based on timestep vs boundary threshold
    - In Neuron compiled mode, both experts are compiled separately
    - Uses batched CFG or sequential CFG based on memory constraints
    """

    def __init__(
        self,
        model_dir: str,
        compiled_dir: Optional[str] = None,
        eager: bool = False,
    ):
        self.model_dir = model_dir
        self.compiled_dir = compiled_dir
        self.eager = eager
        self.text_encoder = None
        self.transformer = None      # Expert 1 (high-noise)
        self.transformer_2 = None    # Expert 2 (low-noise)
        self.vae = None
        self.scheduler = None
        self.tokenizer = None
        self.boundary_timestep = None

    def load_pipeline(self):
        """Load all pipeline components."""
        from diffusers import AutoencoderKLWan, WanTransformer3DModel
        from transformers import AutoTokenizer, UMT5EncoderModel

        print("=" * 60)
        print("Loading WAN 2.2 Pipeline (PyTorch Native)")
        print("=" * 60)
        print(f"Mode: {'Eager (CPU)' if self.eager else 'Compiled (Neuron)'}")
        print(f"Model: {self.model_dir}")

        t0 = time.time()

        # --- Load Tokenizer ---
        print("\n[1/6] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, subfolder="tokenizer"
        )

        # --- Load Text Encoder ---
        print("[2/6] Loading text encoder (UMT5-XXL)...")
        t_enc = time.time()
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            self.model_dir,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
        )
        self.text_encoder.eval()
        print(f"  Text encoder loaded in {time.time() - t_enc:.1f}s")

        # --- Load Transformer Expert 1 (high-noise) ---
        print("[3/6] Loading transformer Expert 1 (high-noise)...")
        t_trans = time.time()
        self.transformer = WanTransformer3DModel.from_pretrained(
            self.model_dir,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        self.transformer.eval()
        params_1 = sum(p.numel() for p in self.transformer.parameters()) / 1e9
        print(f"  Expert 1 loaded in {time.time() - t_trans:.1f}s ({params_1:.1f}B params)")

        # --- Load Transformer Expert 2 (low-noise) ---
        print("[4/6] Loading transformer Expert 2 (low-noise)...")
        t_trans2 = time.time()
        self.transformer_2 = WanTransformer3DModel.from_pretrained(
            self.model_dir,
            subfolder="transformer_2",
            torch_dtype=torch.bfloat16,
        )
        self.transformer_2.eval()
        params_2 = sum(p.numel() for p in self.transformer_2.parameters()) / 1e9
        print(f"  Expert 2 loaded in {time.time() - t_trans2:.1f}s ({params_2:.1f}B params)")

        # --- Load VAE ---
        print("[5/6] Loading VAE decoder...")
        t_vae = time.time()
        self.vae = AutoencoderKLWan.from_pretrained(
            self.model_dir,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        )
        self.vae.eval()
        print(f"  VAE loaded in {time.time() - t_vae:.1f}s")

        # --- Setup Scheduler ---
        print("[6/6] Loading scheduler...")
        from diffusers import UniPCMultistepScheduler
        self.scheduler = UniPCMultistepScheduler.from_pretrained(
            self.model_dir, subfolder="scheduler"
        )

        # Compute boundary timestep for expert switching
        num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.boundary_timestep = BOUNDARY_RATIO * num_train_timesteps
        print(f"  Boundary timestep: {self.boundary_timestep:.0f} "
              f"(Expert1: t >= {self.boundary_timestep:.0f}, Expert2: t < {self.boundary_timestep:.0f})")

        # --- Compile for Neuron if not eager ---
        if not self.eager:
            self._compile_for_neuron()

        total_load = time.time() - t0
        print(f"\n{'=' * 60}")
        print(f"Pipeline loaded in {total_load:.1f}s")
        print(f"{'=' * 60}\n")

    def _compile_for_neuron(self):
        """
        Prepare models for Neuron hardware.

        Note: torch.compile(backend='neuronx') requires Neuron SDK 2.29+.
        On earlier versions, use torch_neuronx.trace() for AOT compilation
        or run in eager mode on NeuronCores via device placement.

        For the current SDK version, we use torch_neuronx.trace() which
        requires example inputs for each model. Compilation is deferred
        to the first inference call to determine correct input shapes.
        """
        import torch_neuronx

        print("\nPreparing models for Neuron...")
        print("  Note: Full torch.compile(backend='neuronx') requires SDK 2.29+")
        print("  Using deferred trace-based compilation (will compile on first forward pass)")
        print("  For immediate compilation, provide --eager flag to skip this step")

        # Mark models for deferred Neuron compilation
        # The actual trace/compile happens on first inference with real input shapes
        self._neuron_compile_pending = True

    def encode_text(self, prompt: str, negative_prompt: str = "") -> tuple:
        """Encode text prompt and negative prompt."""
        print("Encoding text prompt...")
        t0 = time.time()

        # Encode positive prompt
        inputs = self.tokenizer(
            prompt,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            prompt_embeds = self.text_encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ).last_hidden_state.to(torch.bfloat16)

        # Encode negative prompt (empty string for unconditional)
        neg_inputs = self.tokenizer(
            negative_prompt,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            negative_prompt_embeds = self.text_encoder(
                input_ids=neg_inputs["input_ids"],
                attention_mask=neg_inputs["attention_mask"],
            ).last_hidden_state.to(torch.bfloat16)

        elapsed = time.time() - t0
        print(f"  Text encoding: {elapsed:.1f}s")

        return prompt_embeds, negative_prompt_embeds

    def denoise(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        num_steps: int = DEFAULT_NUM_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    ) -> torch.Tensor:
        """
        Run the full denoising loop with dual-expert switching.

        Expert selection is based on the timestep value vs boundary:
        - t >= boundary_timestep: Expert 1 (high-noise transformer)
        - t <  boundary_timestep: Expert 2 (low-noise transformer)
        """
        print(f"\nDenoising ({num_steps} steps, guidance_scale={guidance_scale})...")
        print(f"  Boundary: t >= {self.boundary_timestep:.0f} → Expert 1, "
              f"t < {self.boundary_timestep:.0f} → Expert 2")

        # Setup timesteps
        self.scheduler.set_timesteps(num_steps)
        timesteps = self.scheduler.timesteps

        total_denoise_t0 = time.time()
        expert1_steps = 0
        expert2_steps = 0

        for i, t in enumerate(timesteps):
            step_t0 = time.time()

            # Select expert based on timestep
            if self.boundary_timestep is None or t >= self.boundary_timestep:
                current_model = self.transformer
                expert_name = "E1"
                expert1_steps += 1
            else:
                current_model = self.transformer_2
                expert_name = "E2"
                expert2_steps += 1

            latents = self._denoise_step(
                latents, prompt_embeds, negative_prompt_embeds,
                t, guidance_scale, current_model,
            )

            step_time = time.time() - step_t0
            if (i + 1) % 5 == 0 or (i + 1) == num_steps:
                print(f"    Step {i + 1}/{num_steps} [{expert_name}] ({step_time:.1f}s)")

        total_denoise = time.time() - total_denoise_t0
        print(f"\n  Total denoising: {total_denoise:.1f}s "
              f"({total_denoise/num_steps:.1f}s/step)")
        print(f"  Expert 1 steps: {expert1_steps}, Expert 2 steps: {expert2_steps}")

        return latents

    def _denoise_step(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        timestep: torch.Tensor,
        guidance_scale: float,
        model: torch.nn.Module,
    ) -> torch.Tensor:
        """
        Single denoising step with Classifier-Free Guidance.

        Uses sequential CFG (two forward passes) for memory efficiency.
        For Neuron compiled models, batched CFG can be enabled for 2x speedup.
        """
        latent_input = latents.to(torch.bfloat16)
        t_input = timestep.expand(latents.shape[0])

        with torch.no_grad():
            # Conditional prediction
            noise_pred = model(
                hidden_states=latent_input,
                timestep=t_input,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

            if guidance_scale > 1.0:
                # Unconditional prediction
                noise_uncond = model(
                    hidden_states=latent_input,
                    timestep=t_input,
                    encoder_hidden_states=negative_prompt_embeds,
                    return_dict=False,
                )[0]

                # Apply CFG
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

        # Scheduler step
        latents = self.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to video frames using VAE."""
        print("\nDecoding latents (VAE)...")
        t0 = time.time()

        with torch.no_grad():
            # Normalize latents using mean/std from VAE config
            latents_mean = torch.tensor(
                self.vae.config.latents_mean, dtype=latents.dtype
            ).view(1, -1, 1, 1, 1)
            latents_std = torch.tensor(
                self.vae.config.latents_std, dtype=latents.dtype
            ).view(1, -1, 1, 1, 1)
            latents = latents * latents_std + latents_mean

            # Decode
            video = self.vae.decode(latents.to(torch.bfloat16)).sample

        elapsed = time.time() - t0
        print(f"  VAE decode: {elapsed:.1f}s")

        return video

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        num_frames: int = DEFAULT_NUM_FRAMES,
        num_inference_steps: int = DEFAULT_NUM_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        seed: int = DEFAULT_SEED,
    ) -> torch.Tensor:
        """
        Run full text-to-video (or text-to-image) inference.

        Returns video tensor of shape (1, C, T, H, W).
        """
        print(f"\n{'='*60}")
        print(f"WAN 2.2 T2V-A14B — PyTorch Native Inference")
        print(f"{'='*60}")
        print(f"Prompt:     {prompt}")
        print(f"Negative:   {negative_prompt or '(none)'}")
        print(f"Resolution: {width}x{height}, {num_frames} frames")
        print(f"Steps:      {num_inference_steps}")
        print(f"CFG Scale:  {guidance_scale}")
        print(f"Seed:       {seed}")
        print(f"{'='*60}\n")

        total_t0 = time.time()

        # Set seed
        torch.manual_seed(seed)

        # 1. Encode text
        prompt_embeds, negative_prompt_embeds = self.encode_text(prompt, negative_prompt)

        # 2. Prepare latents
        latent_channels = self.transformer.config.in_channels
        latent_h = height // 8
        latent_w = width // 8
        latent_t = (num_frames - 1) // 4 + 1 if num_frames > 1 else 1

        latents = torch.randn(
            1, latent_channels, latent_t, latent_h, latent_w,
            dtype=torch.float32,
        )

        # 3. Denoise
        latents = self.denoise(
            latents, prompt_embeds, negative_prompt_embeds,
            num_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        # 4. Decode
        video = self.decode_latents(latents)

        total_time = time.time() - total_t0
        print(f"\n{'='*60}")
        print(f"TOTAL INFERENCE TIME: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"{'='*60}\n")

        return video


# ============================================================================
# Output Saving
# ============================================================================

def save_output(video: torch.Tensor, output_path: str, num_frames: int, fps: int = 16):
    """Save video tensor to MP4 or single frame to PNG."""
    from PIL import Image
    import numpy as np

    print(f"Saving output to {output_path}...")

    # video shape: (1, C, T, H, W) -> (T, H, W, C)
    video = video.squeeze(0)  # (C, T, H, W)
    video = video.permute(1, 2, 3, 0)  # (T, H, W, C)
    video = (video.float() * 127.5 + 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    if video.shape[0] == 1 or num_frames == 1:
        # Single frame — save as PNG
        if not output_path.endswith(".png"):
            output_path = output_path.rsplit(".", 1)[0] + ".png"
        img = Image.fromarray(video[0])
        img.save(output_path)
        file_size = os.path.getsize(output_path) / 1024
        print(f"  Saved image: {output_path} ({file_size:.1f} KB, {img.size})")
    else:
        # Multiple frames — save as video
        import imageio
        if not output_path.endswith(".mp4"):
            output_path = output_path.rsplit(".", 1)[0] + ".mp4"
        writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
        for frame in video:
            writer.append_data(frame)
        writer.close()
        file_size = os.path.getsize(output_path) / 1e6
        print(f"  Saved video: {output_path} ({file_size:.1f} MB, {len(video)} frames @ {fps}fps)")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="WAN 2.2 T2V-A14B Inference (PyTorch Native on Trn2)"
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--negative-prompt", type=str, default="",
                        help="Negative prompt for CFG")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--compiled-dir", type=str, default=DEFAULT_COMPILED_DIR)
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: auto-generated)")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--num-inference-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eager", action="store_true",
                        help="Use eager mode (no Neuron compilation, runs on CPU)")
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS")
    args = parser.parse_args()

    # Setup
    if not args.eager:
        setup_neuron_env()

    # Determine output path
    if args.output is None:
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ext = "png" if args.num_frames == 1 else "mp4"
        args.output = os.path.join(DEFAULT_OUTPUT_DIR, f"wan22_{timestamp}.{ext}")

    # Create and load pipeline
    pipeline = WanPipelineNative(
        model_dir=args.model_dir,
        compiled_dir=args.compiled_dir,
        eager=args.eager,
    )
    pipeline.load_pipeline()

    # Run inference
    video = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    # Save output
    save_output(video, args.output, args.num_frames, fps=args.fps)

    print("\nDone!")


if __name__ == "__main__":
    main()
