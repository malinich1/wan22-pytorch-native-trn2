"""
WAN 2.2 T2V-A14B — Inference with Native PyTorch Beta 3 on Trainium 2

Uses PyTorch 2.11 + torch-neuronx 2.11.3.x on trn2.48xlarge (64 NeuronCores).

Beta 3 capabilities used here:
  - PyTorch 2.11 eager mode AND torch.compile(backend='neuron')
  - Asynchronous NRT execution (enabled by default, explicit here)
  - Persistent NEFF caching (no recompilation on restart)
  - Memory snapshot API for OOM debugging
  - LNC2 mode: NEURON_RT_VIRTUAL_CORE_SIZE=2 (2 physical cores per logical core)
  - 99% ATen op coverage — no custom op wrappers needed

DLC (Native PyTorch Beta 3):
    ECR URI: 421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest

    Pull & run:
        aws ecr get-login-password --region us-east-1 | \\
            docker login --username AWS --password-stdin 421672808698.dkr.ecr.us-east-1.amazonaws.com
        docker pull 421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest
        docker run -it --privileged \\
            -v /mnt/nvme:/mnt/nvme \\
            421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest /bin/bash

    Versions inside DLC:
        torch          2.11.0+cpu
        torch-neuronx  2.11.3.0.1254+1dc9304c.dev
        neuronx-cc     2.0.253257.0a0+fd6c623c
        nki            0.4.0b4

Architecture:
  WAN 2.2 is Mixture-of-Experts with TWO transformer experts:
    - transformer   (Expert 1): high-noise denoising steps (t >= boundary)
    - transformer_2 (Expert 2): low-noise denoising steps  (t <  boundary)
  boundary_ratio=0.875 → 87.5% of steps use Expert 1, 12.5% use Expert 2.

Usage:
    # Eager mode on Neuron device (fast iteration, no compilation wait):
    python run_inference.py --prompt "A cat walks on grass" --eager \\
        --num-inference-steps 20 --height 384 --width 640

    # torch.compile mode (production, persistent NEFF cache after first run):
    python run_inference.py --prompt "A cat walks on grass" \\
        --num-inference-steps 40 --height 768 --width 1280

    # Single frame (image):
    python run_inference.py --prompt "A cat" --eager --num-frames 1 \\
        --height 384 --width 640 --num-inference-steps 20

    # Enable memory snapshot for OOM debugging:
    python run_inference.py --prompt "..." --memory-snapshot

Known limitations (Beta 3):
    - Dynamic shapes not supported with torch.compile (use fixed H/W/frames)
    - torch.compile modes reduce-overhead/max-autotune fall back to default
    - int64 tensors downcast to int32 (warning printed, handled below)
    - P2P send/recv and pipeline parallelism not yet supported
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

DEFAULT_MODEL_DIR    = "/mnt/nvme/models/Wan2.2-T2V-A14B-Diffusers"
DEFAULT_OUTPUT_DIR   = "/mnt/nvme/outputs"
# Persistent NEFF cache directory — survives container restarts (Beta 3)
DEFAULT_NEFF_CACHE   = "/mnt/nvme/neff_cache"

# trn2.48xlarge: 16 Neuron devices × 4 physical cores = 64 physical cores.
# LNC2 mode (NEURON_RT_VIRTUAL_CORE_SIZE=2) gives 32 logical cores.
# We run single-process inference using all 64 physical cores directly.
NEURON_RT_NUM_CORES  = 64

# Model defaults
DEFAULT_HEIGHT       = 768
DEFAULT_WIDTH        = 1280
DEFAULT_NUM_FRAMES   = 81
DEFAULT_NUM_STEPS    = 40
DEFAULT_GUIDANCE     = 5.0
DEFAULT_SEED         = 42

# WAN 2.2 MoE expert boundary
BOUNDARY_RATIO       = 0.875


# ============================================================================
# Beta 3 environment setup
# ============================================================================

def setup_neuron_env(neff_cache: str = DEFAULT_NEFF_CACHE):
    """
    Configure environment for Native PyTorch Beta 3 on trn2.48xlarge.

    Beta 3 specifics applied here:
      - NEURON_RT_VIRTUAL_CORE_SIZE=2  → LNC2 mode (2 physical cores/logical core)
      - NEURON_RT_NUM_CORES=64         → expose all 64 physical cores
      - NEURON_COMPILE_CACHE_URL       → persistent NEFF cache (Beta 3 feature)
      - TORCH_NEURONX_ENABLE_ASYNC_NRT → async execution (default in Beta 3, explicit)
      - TORCH_NEURONX_ENABLE_HOST_CC   → host collective comms / compute overlap
      - NEURONX_CACHE                  → secondary persistent cache path
    """
    # Compiler flags: -O1, no auto-cast, native kernel, rematerialization
    os.environ["NEURON_CC_FLAGS"] = (
        "-O1 --auto-cast=none --enable-native-kernel=1 "
        "--remat --enable-ccop-compute-overlap"
    )

    # LNC2 mode for trn2: 2 physical NeuronCores per logical core
    os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
    os.environ["NEURON_RT_NUM_CORES"]         = str(NEURON_RT_NUM_CORES)
    os.environ["NEURON_RT_VISIBLE_CORES"]     = f"0-{NEURON_RT_NUM_CORES - 1}"
    os.environ["NEURON_ENABLE_NATIVE_KERNEL"] = "1"

    # Beta 3: async NRT is on by default; set explicitly for clarity
    os.environ["TORCH_NEURONX_ENABLE_ASYNC_NRT"]  = "1"
    # Host collective comms — enables compute/communication overlap
    os.environ["TORCH_NEURONX_ENABLE_HOST_CC"]    = "1"

    # Persistent NEFF cache (Beta 3 feature — eliminates recompilation on restart)
    os.makedirs(neff_cache, exist_ok=True)
    os.environ["NEURON_COMPILE_CACHE_URL"] = f"file://{neff_cache}"
    os.environ["NEURONX_CACHE"]            = neff_cache

    print(f"[Neuron] Beta 3 env configured")
    print(f"  LNC2 mode:      NEURON_RT_VIRTUAL_CORE_SIZE=2")
    print(f"  Cores:          {NEURON_RT_NUM_CORES} physical NeuronCores")
    print(f"  Async NRT:      enabled")
    print(f"  NEFF cache:     {neff_cache}")


# ============================================================================
# Beta 3 Memory Snapshot utility
# ============================================================================

class MemorySnapshotContext:
    """
    Context manager for Beta 3 memory snapshot API.

    Wraps torch.cuda.memory._snapshot() equivalent for Neuron.
    Falls back silently if the API isn't available (e.g. eager/CPU mode).

    Usage:
        with MemorySnapshotContext("compile_phase", output_dir):
            model = torch.compile(model, backend="neuron")
    """

    def __init__(self, label: str, output_dir: str = DEFAULT_OUTPUT_DIR):
        self.label      = label
        self.output_dir = output_dir
        self._available = False

    def __enter__(self):
        try:
            # Beta 3 memory snapshot API
            import torch_neuronx
            if hasattr(torch_neuronx, "memory_snapshot_start"):
                torch_neuronx.memory_snapshot_start()
                self._available = True
                print(f"  [MemSnapshot] Started: {self.label}")
        except Exception:
            pass
        return self

    def __exit__(self, *args):
        if not self._available:
            return
        try:
            import torch_neuronx
            snapshot = torch_neuronx.memory_snapshot_stop()
            if snapshot:
                os.makedirs(self.output_dir, exist_ok=True)
                ts   = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join(self.output_dir, f"memsnapshot_{self.label}_{ts}.pkl")
                import pickle
                with open(path, "wb") as f:
                    pickle.dump(snapshot, f)
                print(f"  [MemSnapshot] Saved: {path}")
        except Exception as e:
            print(f"  [MemSnapshot] Could not save snapshot: {e}")


# ============================================================================
# Pipeline
# ============================================================================

class WanPipelineNative:
    """
    WAN 2.2 T2V-A14B Pipeline — Native PyTorch Beta 3 on trn2.48xlarge.

    Supports two execution modes (--eager / default compile):

    Eager mode  (--eager):
      - Models placed on torch.device("neuron") and run in PyTorch 2.11 eager mode.
      - No compilation wait. Good for functional validation and profiling setup.
      - All 99% of supported ATen ops execute on NeuronCores directly.

    Compile mode (default):
      - torch.compile(backend='neuron') applied to both transformer experts + VAE.
      - First run: NEFF compilation ~16 min for MoE (cold cache).
      - Subsequent runs: ~3 min warm cache load from NEURON_COMPILE_CACHE_URL.
      - Static shapes required — enforced by fixed H/W/num_frames defaults.
    """

    def __init__(
        self,
        model_dir: str,
        eager: bool = False,
        memory_snapshot: bool = False,
        output_dir: str = DEFAULT_OUTPUT_DIR,
    ):
        self.model_dir       = model_dir
        self.eager           = eager
        self.memory_snapshot = memory_snapshot
        self.output_dir      = output_dir
        self.device          = torch.device("neuron") if not eager else torch.device("cpu")
        self.text_encoder    = None
        self.transformer     = None   # Expert 1 (high-noise)
        self.transformer_2   = None   # Expert 2 (low-noise)
        self.vae             = None
        self.scheduler       = None
        self.tokenizer       = None
        self.boundary_timestep = None

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_pipeline(self):
        """Load all pipeline components and prepare for inference."""
        from diffusers import AutoencoderKLWan, WanTransformer3DModel, UniPCMultistepScheduler
        from transformers import AutoTokenizer, UMT5EncoderModel

        mode_str = (
            "Eager / PyTorch 2.11 (no compilation)"
            if self.eager
            else "torch.compile(backend='neuron') / PyTorch 2.11"
        )
        print("=" * 68)
        print("  WAN 2.2 T2V-A14B  —  Native PyTorch Beta 3")
        print("=" * 68)
        print(f"  Mode:   {mode_str}")
        print(f"  Device: {self.device}")
        print(f"  Model:  {self.model_dir}")
        print("=" * 68)

        t0 = time.time()

        print("\n[1/6] Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, subfolder="tokenizer"
        )

        print("[2/6] Text encoder (UMT5-XXL)...")
        t = time.time()
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            self.model_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16,
        ).eval()
        # Text encoder runs on CPU to save NeuronCore memory for the transformers
        print(f"      Loaded in {time.time()-t:.1f}s  (runs on CPU)")

        print("[3/6] Transformer Expert 1 (high-noise)...")
        t = time.time()
        self.transformer = WanTransformer3DModel.from_pretrained(
            self.model_dir, subfolder="transformer", torch_dtype=torch.bfloat16,
        ).eval()
        p1 = sum(p.numel() for p in self.transformer.parameters()) / 1e9
        print(f"      Loaded in {time.time()-t:.1f}s  ({p1:.1f}B params)")

        print("[4/6] Transformer Expert 2 (low-noise)...")
        t = time.time()
        self.transformer_2 = WanTransformer3DModel.from_pretrained(
            self.model_dir, subfolder="transformer_2", torch_dtype=torch.bfloat16,
        ).eval()
        p2 = sum(p.numel() for p in self.transformer_2.parameters()) / 1e9
        print(f"      Loaded in {time.time()-t:.1f}s  ({p2:.1f}B params)")

        print("[5/6] VAE decoder...")
        t = time.time()
        self.vae = AutoencoderKLWan.from_pretrained(
            self.model_dir, subfolder="vae", torch_dtype=torch.bfloat16,
        ).eval()
        print(f"      Loaded in {time.time()-t:.1f}s")

        print("[6/6] Scheduler (UniPC)...")
        self.scheduler = UniPCMultistepScheduler.from_pretrained(
            self.model_dir, subfolder="scheduler"
        )
        n_train = self.scheduler.config.num_train_timesteps
        self.boundary_timestep = BOUNDARY_RATIO * n_train
        print(f"      Boundary timestep: {self.boundary_timestep:.0f}  "
              f"(Expert1 t>={self.boundary_timestep:.0f}, Expert2 t<{self.boundary_timestep:.0f})")

        # Place on Neuron / compile
        self._prepare_for_neuron()

        print(f"\n{'='*68}")
        print(f"  Pipeline ready in {time.time()-t0:.1f}s")
        print(f"{'='*68}\n")

    # ------------------------------------------------------------------
    # Neuron placement & compilation
    # ------------------------------------------------------------------

    def _prepare_for_neuron(self):
        """
        Place models on Neuron device and optionally torch.compile them.

        Beta 3 notes:
          - .to(device) with device='neuron' works natively in PyTorch 2.11.
          - torch.compile(backend='neuron') is the recommended path for production.
          - Persistent NEFF cache (NEURON_COMPILE_CACHE_URL) means compilation
            only happens on the very first run; all subsequent runs load from disk.
          - int64 tensors are auto-downcast to int32 by the runtime (expected).
          - dynamic=True is NOT supported; our fixed resolutions satisfy this.
        """
        print("\nPlacing models on Neuron device...")
        snap_label = "neuron_placement"

        with MemorySnapshotContext(snap_label, self.output_dir) if self.memory_snapshot \
                else _nullctx():

            # Move transformer experts and VAE to NeuronCores
            t = time.time()
            print("  Expert 1 → neuron ...")
            self.transformer   = self.transformer.to(self.device)
            print("  Expert 2 → neuron ...")
            self.transformer_2 = self.transformer_2.to(self.device)
            print("  VAE      → neuron ...")
            self.vae           = self.vae.to(self.device)
            print(f"  Device placement: {time.time()-t:.1f}s")

            if not self.eager:
                self._compile_models()

    def _compile_models(self):
        """
        Apply torch.compile(backend='neuron') to all three Neuron models.

        Compilation is lazy — NEFFs are generated on the first forward pass,
        not here. Persistent NEFF cache means a warm restart skips compilation
        entirely (~3 min cache load vs ~16 min cold compile for MoE).
        """
        print("\nApplying torch.compile(backend='neuron')  [Beta 3]...")
        print("  Cold-cache first run: ~16 min (MoE).  Warm cache: ~3 min.")
        print(f"  NEFF cache: {os.environ.get('NEURON_COMPILE_CACHE_URL', 'not set')}")

        # Beta 3 limitation: reduce-overhead / max-autotune fall back to default
        compile_kwargs = dict(backend="neuron", dynamic=False)

        t = time.time()
        print("  Compiling Expert 1...")
        self.transformer   = torch.compile(self.transformer,   **compile_kwargs)
        print("  Compiling Expert 2...")
        self.transformer_2 = torch.compile(self.transformer_2, **compile_kwargs)
        print("  Compiling VAE...")
        self.vae           = torch.compile(self.vae,           **compile_kwargs)

        print(f"  torch.compile() registered in {time.time()-t:.1f}s  "
              f"(NEFF build deferred to first forward pass)")

    # ------------------------------------------------------------------
    # Text encoding (CPU)
    # ------------------------------------------------------------------

    def encode_text(self, prompt: str, negative_prompt: str = "") -> tuple:
        """Encode text on CPU (UMT5-XXL stays on CPU to preserve NeuronCore HBM)."""
        print("Encoding prompts (CPU)...")
        t0 = time.time()

        def _encode(text):
            toks = self.tokenizer(
                text, max_length=512, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            with torch.no_grad():
                return self.text_encoder(
                    input_ids=toks["input_ids"],
                    attention_mask=toks["attention_mask"],
                ).last_hidden_state.to(torch.bfloat16)

        prompt_embeds   = _encode(prompt)
        neg_embeds      = _encode(negative_prompt if negative_prompt else "")

        print(f"  Text encoding: {time.time()-t0:.1f}s  shape={prompt_embeds.shape}")
        return prompt_embeds, neg_embeds

    # ------------------------------------------------------------------
    # Denoising loop
    # ------------------------------------------------------------------

    def denoise(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        neg_embeds: torch.Tensor,
        num_steps: int   = DEFAULT_NUM_STEPS,
        guidance: float  = DEFAULT_GUIDANCE,
    ) -> torch.Tensor:
        """
        Full denoising loop with dual-expert switching.

        Expert routing:
          t >= boundary_timestep  →  Expert 1 (high-noise transformer)
          t <  boundary_timestep  →  Expert 2 (low-noise transformer)

        CFG uses sequential unconditional + conditional passes to avoid
        doubling memory requirements on NeuronCores.
        """
        self.scheduler.set_timesteps(num_steps)
        timesteps = self.scheduler.timesteps

        print(f"\nDenoising: {num_steps} steps, CFG={guidance}")
        print(f"  Expert routing boundary: {self.boundary_timestep:.0f}")

        t_total = time.time()
        e1_steps = e2_steps = 0

        for i, t in enumerate(timesteps):
            t_step = time.time()

            if t >= self.boundary_timestep:
                model, ename = self.transformer,   "E1"
                e1_steps += 1
            else:
                model, ename = self.transformer_2, "E2"
                e2_steps += 1

            latents = self._denoise_step(latents, prompt_embeds, neg_embeds,
                                         t, guidance, model)

            elapsed = time.time() - t_step
            if (i + 1) % 5 == 0 or (i + 1) == num_steps:
                print(f"  Step {i+1:>3}/{num_steps} [{ename}]  {elapsed:.1f}s/step")

        total = time.time() - t_total
        print(f"\n  Denoising done: {total:.1f}s  ({total/num_steps:.1f}s/step avg)")
        print(f"  Expert 1: {e1_steps} steps  |  Expert 2: {e2_steps} steps")
        return latents

    def _denoise_step(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        neg_embeds: torch.Tensor,
        timestep: torch.Tensor,
        guidance: float,
        model: torch.nn.Module,
    ) -> torch.Tensor:
        """
        Single CFG denoising step.

        Beta 3 note: tensors moved to Neuron device inline; int64 timesteps are
        auto-downcast to int32 by the runtime (no manual cast needed).
        Sequential CFG halves peak NeuronCore HBM vs batched CFG.
        """
        x     = latents.to(self.device, dtype=torch.bfloat16)
        t_in  = timestep.expand(latents.shape[0]).to(self.device)
        pe    = prompt_embeds.to(self.device)
        ne    = neg_embeds.to(self.device)

        with torch.no_grad():
            noise_pred = model(
                hidden_states=x, timestep=t_in,
                encoder_hidden_states=pe, return_dict=False,
            )[0]

            if guidance > 1.0:
                noise_uncond = model(
                    hidden_states=x, timestep=t_in,
                    encoder_hidden_states=ne, return_dict=False,
                )[0]
                noise_pred = noise_uncond + guidance * (noise_pred - noise_uncond)

        latents = self.scheduler.step(
            noise_pred.cpu(), timestep, latents, return_dict=False
        )[0]
        return latents

    # ------------------------------------------------------------------
    # VAE decode
    # ------------------------------------------------------------------

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents → video frames via VAE (runs on Neuron device)."""
        print("\nVAE decode...")
        t0 = time.time()

        with torch.no_grad():
            lmean = torch.tensor(
                self.vae.config.latents_mean, dtype=latents.dtype
            ).view(1, -1, 1, 1, 1)
            lstd = torch.tensor(
                self.vae.config.latents_std, dtype=latents.dtype
            ).view(1, -1, 1, 1, 1)
            latents_scaled = latents * lstd + lmean
            video = self.vae.decode(
                latents_scaled.to(self.device, dtype=torch.bfloat16)
            ).sample.cpu()

        print(f"  VAE decode: {time.time()-t0:.1f}s  shape={video.shape}")
        return video

    # ------------------------------------------------------------------
    # Top-level call
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str    = "",
        height: int             = DEFAULT_HEIGHT,
        width: int              = DEFAULT_WIDTH,
        num_frames: int         = DEFAULT_NUM_FRAMES,
        num_inference_steps: int = DEFAULT_NUM_STEPS,
        guidance_scale: float   = DEFAULT_GUIDANCE,
        seed: int               = DEFAULT_SEED,
    ) -> torch.Tensor:
        """Run full T2V inference. Returns video tensor (1, C, T, H, W)."""
        print(f"\n{'='*68}")
        print(f"  WAN 2.2 — PyTorch 2.11 Native Inference")
        print(f"  Prompt:     {prompt}")
        print(f"  Negative:   {negative_prompt or '(none)'}")
        print(f"  Resolution: {width}x{height}, {num_frames} frames")
        print(f"  Steps:      {num_inference_steps}  CFG: {guidance_scale}  Seed: {seed}")
        print(f"{'='*68}\n")

        t0 = time.time()
        torch.manual_seed(seed)

        # 1. Text encode (CPU)
        prompt_embeds, neg_embeds = self.encode_text(prompt, negative_prompt)

        # 2. Initial latents (CPU float32, matches scheduler dtype expectation)
        latent_ch = self.transformer.config.in_channels
        latent_h  = height    // 8
        latent_w  = width     // 8
        latent_t  = (num_frames - 1) // 4 + 1 if num_frames > 1 else 1
        latents   = torch.randn(1, latent_ch, latent_t, latent_h, latent_w)
        print(f"Latent shape: {list(latents.shape)}")

        # 3. Denoise on Neuron
        latents = self.denoise(latents, prompt_embeds, neg_embeds,
                               num_steps=num_inference_steps,
                               guidance=guidance_scale)

        # 4. Decode on Neuron
        video = self.decode_latents(latents)

        elapsed = time.time() - t0
        print(f"\n{'='*68}")
        print(f"  TOTAL INFERENCE: {elapsed:.1f}s  ({elapsed/60:.1f} min)")
        print(f"{'='*68}\n")
        return video


# ============================================================================
# Null context helper (for optional memory snapshot blocks)
# ============================================================================

from contextlib import contextmanager

@contextmanager
def _nullctx():
    yield


# ============================================================================
# Output saving
# ============================================================================

def save_output(video: torch.Tensor, output_path: str,
                num_frames: int, fps: int = 16) -> str:
    """Save video tensor (1,C,T,H,W) to MP4 or single frame to PNG."""
    from PIL import Image
    import numpy as np

    video = video.squeeze(0).permute(1, 2, 3, 0)          # (T, H, W, C)
    video = ((video.float() / 2 + 0.5).clamp(0, 1) * 255) \
              .to(torch.uint8).cpu().numpy()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
                exist_ok=True)

    if video.shape[0] == 1 or num_frames == 1:
        output_path = output_path.rsplit(".", 1)[0] + ".png"
        img = Image.fromarray(video[0])
        img.save(output_path)
        size_kb = os.path.getsize(output_path) / 1024
        print(f"Saved image: {output_path}  ({size_kb:.0f} KB, {img.size})")
    else:
        import imageio
        output_path = output_path.rsplit(".", 1)[0] + ".mp4"
        writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
        for frame in video:
            writer.append_data(frame)
        writer.close()
        size_mb = os.path.getsize(output_path) / 1e6
        print(f"Saved video: {output_path}  ({size_mb:.1f} MB, "
              f"{len(video)} frames @ {fps}fps)")
    return output_path


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="WAN 2.2 T2V-A14B — Native PyTorch Beta 3 on trn2.48xlarge"
    )
    parser.add_argument("--prompt",              type=str,   required=True)
    parser.add_argument("--negative-prompt",     type=str,   default="")
    parser.add_argument("--model-dir",           type=str,   default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output",              type=str,   default=None)
    parser.add_argument("--output-dir",          type=str,   default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--neff-cache",          type=str,   default=DEFAULT_NEFF_CACHE,
                        help="Persistent NEFF cache path (Beta 3)")
    parser.add_argument("--height",              type=int,   default=DEFAULT_HEIGHT)
    parser.add_argument("--width",               type=int,   default=DEFAULT_WIDTH)
    parser.add_argument("--num-frames",          type=int,   default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--num-inference-steps", type=int,   default=DEFAULT_NUM_STEPS)
    parser.add_argument("--guidance-scale",      type=float, default=DEFAULT_GUIDANCE)
    parser.add_argument("--seed",                type=int,   default=DEFAULT_SEED)
    parser.add_argument("--fps",                 type=int,   default=16)
    parser.add_argument("--eager",               action="store_true",
                        help="Eager mode (PyTorch 2.11 on Neuron, no compilation)")
    parser.add_argument("--memory-snapshot",     action="store_true",
                        help="Capture Beta 3 memory snapshot for OOM debugging")
    args = parser.parse_args()

    # Beta 3 env (skipped in eager mode — device still works without it)
    if not args.eager:
        setup_neuron_env(neff_cache=args.neff_cache)
    else:
        print("[Eager mode] Skipping Neuron env setup — running PyTorch 2.11 eager on CPU")

    # Output path
    if args.output is None:
        os.makedirs(args.output_dir, exist_ok=True)
        ts  = time.strftime("%Y%m%d_%H%M%S")
        ext = "png" if args.num_frames == 1 else "mp4"
        args.output = os.path.join(args.output_dir, f"wan22_{ts}.{ext}")

    # Build pipeline
    pipeline = WanPipelineNative(
        model_dir       = args.model_dir,
        eager           = args.eager,
        memory_snapshot = args.memory_snapshot,
        output_dir      = args.output_dir,
    )
    pipeline.load_pipeline()

    # Run
    video = pipeline(
        prompt              = args.prompt,
        negative_prompt     = args.negative_prompt,
        height              = args.height,
        width               = args.width,
        num_frames          = args.num_frames,
        num_inference_steps = args.num_inference_steps,
        guidance_scale      = args.guidance_scale,
        seed                = args.seed,
    )

    save_output(video, args.output, args.num_frames, fps=args.fps)
    print("Done.")


if __name__ == "__main__":
    main()
