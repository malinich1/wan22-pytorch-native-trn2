"""
WAN 2.2 T2V-A14B — End-to-End Inference with PyTorch Native on Trainium 2

Single-process inference using device='neuron' and torch.compile().
Replaces the NXD subprocess-based approach with native PyTorch parallelism.

Pipeline:
1. Text encoding (T5-XXL, TP=4)
2. Expert 1 denoising (high-noise steps, TP=4 CP=16, 64 cores)
3. Expert swap via copy_()
4. Expert 2 denoising (low-noise steps, TP=4 CP=16, 64 cores)
5. VAE decode (tiled, 8 NeuronCores)

Usage:
    # Basic:
    python run_inference.py --prompt "A cat walks on the grass, realistic style"
    
    # With custom parameters:
    python run_inference.py \\
        --prompt "An astronaut floating in space, cinematic" \\
        --num-frames 81 \\
        --height 768 --width 1280 \\
        --num-inference-steps 40 \\
        --guidance-scale 5.0 \\
        --output /mnt/nvme/outputs/output.mp4
    
    # Eager mode (debugging):
    python run_inference.py --prompt "test" --eager --num-inference-steps 2
"""

import os
import sys
import time
import argparse
import torch
import torch_neuronx
from pathlib import Path
from typing import Optional, Tuple

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_MODEL_DIR = "/mnt/nvme/models/Wan2.2-T2V-A14B-Diffusers"
DEFAULT_COMPILED_DIR = "/mnt/nvme/compiled_artifacts"
DEFAULT_OUTPUT_DIR = "/mnt/nvme/outputs"

# Parallelism
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

# WAN 2.2 MoE schedule: Expert 1 handles high-noise, Expert 2 handles low-noise
# Default split: first 13 steps = Expert 1, last 27 steps = Expert 2
EXPERT_1_STEPS = 13
EXPERT_2_STEPS = 27


# ============================================================================
# Helper functions
# ============================================================================

def setup_neuron_env():
    """Configure Neuron environment variables."""
    os.environ["NEURON_CC_FLAGS"] = (
        "-O1 --auto-cast=none --enable-native-kernel=1 "
        "--remat --enable-ccop-compute-overlap"
    )
    # Use all 64 NeuronCores
    os.environ["NEURON_RT_VISIBLE_CORES"] = "0-63"
    os.environ["NEURON_RT_NUM_CORES"] = "64"
    # Enable native kernel (NKI Flash Attention)
    os.environ["NEURON_ENABLE_NATIVE_KERNEL"] = "1"


def init_distributed(world_size: int = WORLD_SIZE):
    """
    Initialize torch.distributed for Neuron multi-core inference.
    
    PyTorch Native approach: uses standard torch.distributed with
    Neuron's XRT backend for inter-core communication.
    """
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="xla",  # Neuron uses XLA backend for collectives
            world_size=world_size,
        )
    return torch.distributed.get_rank()


# ============================================================================
# Pipeline Components
# ============================================================================

class WanPipelineNative:
    """
    WAN 2.2 T2V-A14B Pipeline using PyTorch Native on Trainium 2.
    
    Differences from NXD-based pipeline:
    - No subprocess isolation (single process, shared memory)
    - Expert swap via copy_() (no model reload)
    - torch.compile() for graph optimization
    - Standard torch.distributed for communication
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
        self.transformer = None
        self.vae = None
        self.scheduler = None
        self.tokenizer = None
        self.expert_swap_manager = None
        
    def load_pipeline(self):
        """Load all pipeline components."""
        from diffusers import WanPipeline, FlowMatchEulerDiscreteScheduler
        from diffusers import WanTransformer3DModel, AutoencoderKLWan
        from transformers import T5EncoderModel, AutoTokenizer
        from expert_swap import ExpertSwapManager
        
        print("=" * 60)
        print("Loading WAN 2.2 Pipeline (PyTorch Native)")
        print("=" * 60)
        
        t0 = time.time()
        
        # --- Load Tokenizer ---
        print("\n[1/5] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, subfolder="tokenizer"
        )
        
        # --- Load Text Encoder ---
        print("[2/5] Loading text encoder (T5-XXL)...")
        t_enc = time.time()
        if self.compiled_dir and not self.eager:
            compiled_path = os.path.join(self.compiled_dir, "text_encoder_compiled.pt")
            if os.path.exists(compiled_path):
                self.text_encoder = torch.jit.load(compiled_path)
                print(f"  Loaded compiled text encoder from {compiled_path}")
            else:
                self._load_text_encoder_fresh()
        else:
            self._load_text_encoder_fresh()
        print(f"  Text encoder loaded in {time.time() - t_enc:.1f}s")
        
        # --- Load Transformer ---
        print("[3/5] Loading transformer (DiT)...")
        t_trans = time.time()
        self.transformer = WanTransformer3DModel.from_pretrained(
            self.model_dir,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        self.transformer.eval()
        
        if not self.eager:
            # Compile with Neuron backend
            self.transformer = torch.compile(
                self.transformer,
                backend="neuronx",
            )
        print(f"  Transformer loaded in {time.time() - t_trans:.1f}s")
        
        # --- Load VAE ---
        print("[4/5] Loading VAE decoder...")
        t_vae = time.time()
        self.vae = AutoencoderKLWan.from_pretrained(
            self.model_dir,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        )
        self.vae.eval()
        
        if not self.eager:
            self.vae.decoder = torch.compile(
                self.vae.decoder,
                backend="neuronx",
            )
        print(f"  VAE loaded in {time.time() - t_vae:.1f}s")
        
        # --- Setup Scheduler ---
        print("[5/5] Loading scheduler...")
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.model_dir, subfolder="scheduler"
        )
        
        # --- Setup Expert Swap Manager ---
        self.expert_swap_manager = ExpertSwapManager(
            model_dir=self.model_dir,
            model=self.transformer,
            tp_degree=TP_DEGREE,
            cp_degree=CP_DEGREE,
        )
        self.expert_swap_manager.preload_experts()
        
        total_load = time.time() - t0
        print(f"\n{'=' * 60}")
        print(f"Pipeline loaded in {total_load:.1f}s")
        print(f"{'=' * 60}\n")
    
    def _load_text_encoder_fresh(self):
        """Load and optionally compile text encoder from scratch."""
        from transformers import T5EncoderModel
        
        self.text_encoder = T5EncoderModel.from_pretrained(
            self.model_dir,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
        )
        self.text_encoder.eval()
        
        if not self.eager:
            self.text_encoder = torch.compile(
                self.text_encoder,
                backend="neuronx",
            )
    
    def encode_text(self, prompt: str) -> torch.Tensor:
        """Encode text prompt using T5 text encoder."""
        print("Encoding text prompt...")
        t0 = time.time()
        
        inputs = self.tokenizer(
            prompt,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            encoder_output = self.text_encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        
        text_embeddings = encoder_output.last_hidden_state.to(torch.bfloat16)
        elapsed = time.time() - t0
        print(f"  Text encoding: {elapsed:.1f}s")
        
        return text_embeddings
    
    def denoise(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        num_steps: int = DEFAULT_NUM_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    ) -> torch.Tensor:
        """
        Run the full denoising loop with expert swapping.
        
        Schedule:
        - Steps 0..EXPERT_1_STEPS-1: Expert 1 (high noise)
        - Steps EXPERT_1_STEPS..num_steps-1: Expert 2 (low noise)
        
        Uses batched CFG: batch_size=2 (conditional + unconditional) in
        a single forward pass, matching the NXD optimized approach.
        """
        print(f"\nDenoising ({num_steps} steps, guidance_scale={guidance_scale})...")
        
        # Setup timesteps
        self.scheduler.set_timesteps(num_steps)
        timesteps = self.scheduler.timesteps
        
        total_denoise_t0 = time.time()
        
        # --- Expert 1: High-noise steps ---
        print(f"\n  Expert 1: steps 0-{EXPERT_1_STEPS - 1} (high noise)")
        expert1_t0 = time.time()
        swap_time_1 = self.expert_swap_manager.activate_expert(0)
        
        for i, t in enumerate(timesteps[:EXPERT_1_STEPS]):
            latents = self._denoise_step(
                latents, text_embeddings, t, guidance_scale
            )
            if (i + 1) % 5 == 0:
                print(f"    Step {i + 1}/{EXPERT_1_STEPS}")
        
        expert1_time = time.time() - expert1_t0
        print(f"  Expert 1 complete: {expert1_time:.1f}s "
              f"(swap: {swap_time_1:.1f}s, denoise: {expert1_time - swap_time_1:.1f}s)")
        
        # --- Expert 2: Low-noise steps ---
        expert2_steps = num_steps - EXPERT_1_STEPS
        print(f"\n  Expert 2: steps {EXPERT_1_STEPS}-{num_steps - 1} (low noise)")
        expert2_t0 = time.time()
        swap_time_2 = self.expert_swap_manager.activate_expert(1)
        
        for i, t in enumerate(timesteps[EXPERT_1_STEPS:]):
            latents = self._denoise_step(
                latents, text_embeddings, t, guidance_scale
            )
            if (i + 1) % 5 == 0:
                print(f"    Step {i + 1}/{expert2_steps}")
        
        expert2_time = time.time() - expert2_t0
        print(f"  Expert 2 complete: {expert2_time:.1f}s "
              f"(swap: {swap_time_2:.1f}s, denoise: {expert2_time - swap_time_2:.1f}s)")
        
        total_denoise = time.time() - total_denoise_t0
        print(f"\n  Total denoising: {total_denoise:.1f}s")
        print(f"  Expert swap overhead: {swap_time_1 + swap_time_2:.1f}s")
        
        return latents
    
    def _denoise_step(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        timestep: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """
        Single denoising step with batched Classifier-Free Guidance.
        
        Batched CFG: concatenates conditional and unconditional inputs
        along batch dimension, runs ONE forward pass, then separates.
        This halves the per-step time compared to sequential CFG.
        """
        # Batched CFG: [unconditional, conditional] in one batch
        latent_input = torch.cat([latents, latents], dim=0)  # batch=2
        
        # Text embeddings: [null_embedding, text_embedding]
        null_embedding = torch.zeros_like(text_embeddings)
        text_input = torch.cat([null_embedding, text_embeddings], dim=0)
        
        # Timestep for both
        t_input = timestep.expand(2)
        
        # Single forward pass (batched)
        with torch.no_grad():
            noise_pred = self.transformer(
                hidden_states=latent_input,
                timestep=t_input,
                encoder_hidden_states=text_input,
            ).sample
        
        # Split and apply CFG
        noise_uncond, noise_cond = noise_pred.chunk(2, dim=0)
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        
        # Scheduler step
        latents = self.scheduler.step(noise_pred, timestep, latents).prev_sample
        
        return latents
    
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to video frames using tiled VAE."""
        print("\nDecoding latents to video (tiled VAE)...")
        t0 = time.time()
        
        with torch.no_grad():
            # Scale latents
            latents = latents / self.vae.config.scaling_factor
            
            # Decode (uses tiled decoding internally for memory efficiency)
            video = self.vae.decode(latents).sample
        
        elapsed = time.time() - t0
        print(f"  VAE decode: {elapsed:.1f}s")
        
        return video
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        num_frames: int = DEFAULT_NUM_FRAMES,
        num_inference_steps: int = DEFAULT_NUM_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        seed: int = DEFAULT_SEED,
    ) -> torch.Tensor:
        """
        Run full text-to-video inference.
        
        Returns video tensor of shape (1, num_frames, 3, height, width).
        """
        print(f"\n{'='*60}")
        print(f"WAN 2.2 T2V-A14B — PyTorch Native Inference")
        print(f"{'='*60}")
        print(f"Prompt:     {prompt}")
        print(f"Resolution: {width}x{height}, {num_frames} frames")
        print(f"Steps:      {num_inference_steps} (Expert1: {EXPERT_1_STEPS}, Expert2: {num_inference_steps - EXPERT_1_STEPS})")
        print(f"CFG Scale:  {guidance_scale}")
        print(f"Seed:       {seed}")
        print(f"{'='*60}\n")
        
        total_t0 = time.time()
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        
        # 1. Encode text
        text_embeddings = self.encode_text(prompt)
        
        # 2. Initialize latents
        latent_h = height // 8
        latent_w = width // 8
        latent_t = (num_frames - 1) // 4 + 1
        latent_channels = 16
        
        latents = torch.randn(
            1, latent_channels, latent_t, latent_h, latent_w,
            dtype=torch.bfloat16,
        )
        
        # 3. Denoise with expert swapping
        latents = self.denoise(
            latents, text_embeddings,
            num_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        
        # 4. Decode to video
        video = self.decode_latents(latents)
        
        total_time = time.time() - total_t0
        print(f"\n{'='*60}")
        print(f"TOTAL INFERENCE TIME: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"{'='*60}\n")
        
        return video


# ============================================================================
# Video Saving
# ============================================================================

def save_video(video: torch.Tensor, output_path: str, fps: int = 16):
    """Save video tensor to MP4 file."""
    import imageio
    
    print(f"Saving video to {output_path}...")
    
    # Convert from (1, C, T, H, W) to (T, H, W, C) uint8
    video = video.squeeze(0)  # Remove batch
    video = video.permute(1, 2, 3, 0)  # C,T,H,W -> T,H,W,C
    video = (video * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
    for frame in video:
        writer.append_data(frame)
    writer.close()
    
    file_size = os.path.getsize(output_path) / 1e6
    print(f"  Saved: {output_path} ({file_size:.1f} MB, {len(video)} frames @ {fps} fps)")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="WAN 2.2 T2V-A14B Inference (PyTorch Native on Trn2)"
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--compiled-dir", type=str, default=DEFAULT_COMPILED_DIR)
    parser.add_argument("--output", type=str, default=None,
                        help="Output video path (default: auto-generated)")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--num-inference-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eager", action="store_true",
                        help="Use eager mode (no compilation, for debugging)")
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS")
    args = parser.parse_args()
    
    # Setup
    setup_neuron_env()
    
    # Determine output path
    if args.output is None:
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(DEFAULT_OUTPUT_DIR, f"wan22_{timestamp}.mp4")
    
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
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    
    # Save output
    save_video(video, args.output, fps=args.fps)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
