"""
WAN 2.2 T2V-A14B — Model Compilation for Native PyTorch Beta 3

Compiles transformer experts and VAE using torch.compile(backend='neuron').
Replaces the old torch_neuronx.trace() approach with the Beta 3 native path.

Beta 3 compilation features used:
  - torch.compile(backend='neuron', dynamic=False)  — PyTorch 2.11 native
  - Persistent NEFF cache via NEURON_COMPILE_CACHE_URL  — no recompile on restart
  - LNC2 mode (NEURON_RT_VIRTUAL_CORE_SIZE=2) for trn2.48xlarge
  - Asynchronous NRT execution (TORCH_NEURONX_ENABLE_ASYNC_NRT=1)

Cold-cache compile times (trn2.48xlarge, MoE model):
  - Expert 1 transformer:  ~8 min
  - Expert 2 transformer:  ~8 min
  - VAE decoder:           ~2 min
  - Total first run:       ~18 min

Subsequent runs (warm NEFF cache at /mnt/nvme/neff_cache):
  - All models:  ~3 min load

Usage:
    # Compile all components (run once, then use run_inference.py):
    python compile_model.py

    # Compile a specific component:
    python compile_model.py --component transformer
    python compile_model.py --component vae

    # Dry-run: verify shapes/device placement without compiling:
    python compile_model.py --dry-run
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

DEFAULT_MODEL_DIR  = "/mnt/nvme/models/Wan2.2-T2V-A14B-Diffusers"
DEFAULT_NEFF_CACHE = "/mnt/nvme/neff_cache"

# trn2.48xlarge: 64 physical NeuronCores in LNC2 mode
NEURON_RT_NUM_CORES = 64

# Static shapes — dynamic=True is NOT supported in Beta 3
HEIGHT      = 768
WIDTH       = 1280
NUM_FRAMES  = 81
LATENT_H    = HEIGHT    // 8        # 96
LATENT_W    = WIDTH     // 8        # 160
LATENT_T    = (NUM_FRAMES - 1) // 4 + 1  # 21
LATENT_CH   = 16
BATCH_CFG   = 1    # Sequential CFG (not batched) to save HBM


# ============================================================================
# Environment setup
# ============================================================================

def setup_compile_env(neff_cache: str):
    """Configure Beta 3 environment for compilation."""
    os.environ["NEURON_CC_FLAGS"] = (
        "-O1 --auto-cast=none --enable-native-kernel=1 "
        "--remat --enable-ccop-compute-overlap"
    )
    os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"]    = "2"
    os.environ["NEURON_RT_NUM_CORES"]            = str(NEURON_RT_NUM_CORES)
    os.environ["NEURON_RT_VISIBLE_CORES"]        = f"0-{NEURON_RT_NUM_CORES - 1}"
    os.environ["NEURON_ENABLE_NATIVE_KERNEL"]    = "1"
    os.environ["TORCH_NEURONX_ENABLE_ASYNC_NRT"] = "1"
    os.environ["TORCH_NEURONX_ENABLE_HOST_CC"]   = "1"

    # Persistent NEFF cache (Beta 3) — compiled NEFFs survive container restarts
    os.makedirs(neff_cache, exist_ok=True)
    os.environ["NEURON_COMPILE_CACHE_URL"] = f"file://{neff_cache}"
    os.environ["NEURONX_CACHE"]            = neff_cache

    print(f"[Compile env]  LNC2 mode, {NEURON_RT_NUM_CORES} cores")
    print(f"[Compile env]  NEFF cache: {neff_cache}")


# ============================================================================
# Compile helpers
# ============================================================================

def _compile_model(model: torch.nn.Module, label: str) -> torch.nn.Module:
    """
    Apply torch.compile(backend='neuron') to a model.

    Beta 3 notes:
      - dynamic=False required (dynamic shapes not supported).
      - NEFF is built on the first forward pass, not here.
      - Persistent cache means subsequent runs skip recompilation.
      - reduce-overhead / max-autotune modes fall back to default (warning only).
    """
    print(f"  Applying torch.compile(backend='neuron') to {label}...")
    compiled = torch.compile(model, backend="neuron", dynamic=False)
    print(f"  {label}: torch.compile registered (NEFF built on first forward pass)")
    return compiled


def _warmup(model, example_inputs: dict, label: str):
    """Run one forward pass to trigger NEFF compilation and cache it."""
    print(f"  Warming up {label} (triggering NEFF compilation)...")
    t0 = time.time()
    with torch.no_grad():
        _ = model(**example_inputs)
    elapsed = time.time() - t0
    print(f"  {label} NEFF compiled and cached in {elapsed:.1f}s")


# ============================================================================
# Component compilers
# ============================================================================

def compile_transformer(model_dir: str, subfolder: str, label: str,
                        dry_run: bool = False):
    """
    Compile a WAN 2.2 transformer expert (Expert 1 or Expert 2).

    Both experts have identical architecture — only weights differ.
    Compiled separately so each gets its own NEFF in the persistent cache.
    """
    from diffusers import WanTransformer3DModel

    print(f"\n{'='*60}")
    print(f"  Compiling {label}")
    print(f"{'='*60}")
    t0 = time.time()

    print(f"  Loading {label} from {model_dir}/{subfolder} ...")
    model = WanTransformer3DModel.from_pretrained(
        model_dir, subfolder=subfolder, torch_dtype=torch.bfloat16,
    ).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Loaded: {n_params:.1f}B params in {time.time()-t0:.1f}s")

    if dry_run:
        print(f"  [dry-run] Skipping device placement and compilation")
        return model

    # Move to Neuron device (PyTorch 2.11 native)
    print(f"  Moving {label} to device='neuron' ...")
    model = model.to(torch.device("neuron"))

    # Compile
    model = _compile_model(model, label)

    # Build example inputs matching inference static shapes
    latent_seq = LATENT_T * LATENT_H * LATENT_W
    example = {
        "hidden_states": torch.randn(
            BATCH_CFG, LATENT_CH, LATENT_T, LATENT_H, LATENT_W,
            dtype=torch.bfloat16, device="neuron"
        ),
        "timestep": torch.tensor([500.0], dtype=torch.bfloat16, device="neuron"),
        "encoder_hidden_states": torch.randn(
            BATCH_CFG, 512, model._orig_mod.config.cross_attention_dim
            if hasattr(model, "_orig_mod") else 4096,
            dtype=torch.bfloat16, device="neuron"
        ),
        "return_dict": False,
    }

    _warmup(model, example, label)
    print(f"  {label} total: {time.time()-t0:.1f}s")
    return model


def compile_vae(model_dir: str, dry_run: bool = False):
    """
    Compile the VAE decoder for Neuron.

    Static input shape: (1, 16, 21, 96, 160) — matches 768x1280 / 81 frames.
    """
    from diffusers import AutoencoderKLWan

    print(f"\n{'='*60}")
    print(f"  Compiling VAE Decoder")
    print(f"{'='*60}")
    t0 = time.time()

    print(f"  Loading VAE from {model_dir}/vae ...")
    vae = AutoencoderKLWan.from_pretrained(
        model_dir, subfolder="vae", torch_dtype=torch.bfloat16,
    ).eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    if dry_run:
        print(f"  [dry-run] Skipping device placement and compilation")
        return vae

    print(f"  Moving VAE to device='neuron' ...")
    vae = vae.to(torch.device("neuron"))
    vae = _compile_model(vae, "VAE")

    example = {
        "sample": torch.randn(
            1, LATENT_CH, LATENT_T, LATENT_H, LATENT_W,
            dtype=torch.bfloat16, device="neuron"
        ),
        "return_dict": False,
    }
    _warmup(vae, example, "VAE")
    print(f"  VAE total: {time.time()-t0:.1f}s")
    return vae


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compile WAN 2.2 components for Native PyTorch Beta 3"
    )
    parser.add_argument("--model-dir",   type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--neff-cache",  type=str, default=DEFAULT_NEFF_CACHE,
                        help="Persistent NEFF cache path (Beta 3)")
    parser.add_argument("--component",   type=str,
                        choices=["all", "transformer", "vae"],
                        default="all")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Verify shapes/placement without compiling NEFFs")
    args = parser.parse_args()

    print("=" * 60)
    print("  WAN 2.2 — Native PyTorch Beta 3 Compilation")
    print("=" * 60)
    print(f"  Model dir:   {args.model_dir}")
    print(f"  NEFF cache:  {args.neff_cache}")
    print(f"  Component:   {args.component}")
    print(f"  Mode:        {'dry-run' if args.dry_run else 'compile'}")
    print("=" * 60)

    if not args.dry_run:
        setup_compile_env(args.neff_cache)

    t_total = time.time()

    if args.component in ("all", "transformer"):
        compile_transformer(args.model_dir, "transformer",   "Expert 1 (high-noise)", args.dry_run)
        compile_transformer(args.model_dir, "transformer_2", "Expert 2 (low-noise)",  args.dry_run)

    if args.component in ("all", "vae"):
        compile_vae(args.model_dir, args.dry_run)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  All compilations complete in {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"  NEFFs cached at: {args.neff_cache}")
    print(f"  Run inference:  python run_inference.py --prompt '...'")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
