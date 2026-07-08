"""
WAN 2.2 T2V-A14B — Simple Inference (Beta 3 / Single Entry Point)

Lightweight wrapper around run_inference.py for quick testing.
Defaults to --eager mode so it works immediately without waiting for
NEFF compilation — useful for validating the pipeline on a fresh instance.

Beta 3 DLC:
    421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest

Usage:
    # Quick CPU smoke test (no Neuron hardware needed):
    python run_inference_simple.py --prompt "A cat walks on grass" \
        --height 256 --width 256 --num-frames 1 --num-steps 3

    # Eager mode on Neuron device (PyTorch 2.11, no compilation wait):
    python run_inference_simple.py --prompt "A cat walks on grass" \
        --height 384 --width 640 --num-frames 1 --num-steps 10

    # Full compile mode (uses persistent NEFF cache, ~16 min cold / ~3 min warm):
    python run_inference_simple.py --prompt "A cat walks on grass" --compile \
        --height 768 --width 1280 --num-steps 40

For the full feature set (memory snapshots, custom NEFF cache path, etc.)
use run_inference.py directly.
"""

import os
import sys
import time
import argparse
import torch
from pathlib import Path

DEFAULT_MODEL_DIR  = "/mnt/nvme/models/Wan2.2-T2V-A14B-Diffusers"
DEFAULT_OUTPUT_DIR = "/mnt/nvme/outputs"
DEFAULT_NEFF_CACHE = "/mnt/nvme/neff_cache"


def is_neuron_available() -> bool:
    """Check whether NeuronCores are accessible on this host."""
    try:
        import subprocess
        result = subprocess.run(["neuron-ls"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="WAN 2.2 simple inference — Native PyTorch Beta 3"
    )
    parser.add_argument("--prompt",              type=str,   required=True)
    parser.add_argument("--negative-prompt",     type=str,   default="")
    parser.add_argument("--model-dir",           type=str,   default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output",              type=str,   default=None)
    parser.add_argument("--height",              type=int,   default=384)
    parser.add_argument("--width",               type=int,   default=640)
    parser.add_argument("--num-frames",          type=int,   default=1,
                        help="1 = image, >1 = video (81 for full 5s @ 16fps)")
    parser.add_argument("--num-steps",           type=int,   default=20)
    parser.add_argument("--guidance-scale",      type=float, default=5.0)
    parser.add_argument("--seed",                type=int,   default=42)
    parser.add_argument("--fps",                 type=int,   default=16)
    parser.add_argument("--compile",             action="store_true",
                        help="Use torch.compile mode (default: eager)")
    parser.add_argument("--neff-cache",          type=str,   default=DEFAULT_NEFF_CACHE)
    args = parser.parse_args()

    # Eager by default unless --compile is explicitly requested
    eager = not args.compile

    # Auto-detect: if no Neuron device available, force CPU eager
    if not is_neuron_available():
        print("[simple] No NeuronCores detected — running in CPU/eager mode")
        eager = True

    print("=" * 60)
    print("  WAN 2.2 — Native PyTorch Beta 3 (Simple)")
    print("=" * 60)
    print(f"  Prompt:   {args.prompt}")
    print(f"  Size:     {args.width}x{args.height}, {args.num_frames} frame(s)")
    print(f"  Steps:    {args.num_steps}")
    print(f"  Mode:     {'torch.compile (Neuron)' if not eager else 'eager (no compilation)'}")
    print("=" * 60)

    # Delegate to the main inference module
    # Import here so this file works as a standalone entry point too
    sys.path.insert(0, str(Path(__file__).parent))

    from run_inference import WanPipelineNative, save_output, setup_neuron_env

    if not eager:
        setup_neuron_env(neff_cache=args.neff_cache)
    else:
        # Even in eager mode, set core visibility if Neuron is available
        if is_neuron_available():
            os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
            os.environ["NEURON_RT_NUM_CORES"]         = "64"
            os.environ["TORCH_NEURONX_ENABLE_ASYNC_NRT"] = "1"

    # Output path
    if args.output is None:
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        ts  = time.strftime("%Y%m%d_%H%M%S")
        ext = "png" if args.num_frames == 1 else "mp4"
        args.output = os.path.join(DEFAULT_OUTPUT_DIR, f"wan22_simple_{ts}.{ext}")

    pipeline = WanPipelineNative(
        model_dir  = args.model_dir,
        eager      = eager,
        output_dir = DEFAULT_OUTPUT_DIR,
    )
    pipeline.load_pipeline()

    video = pipeline(
        prompt              = args.prompt,
        negative_prompt     = args.negative_prompt,
        height              = args.height,
        width               = args.width,
        num_frames          = args.num_frames,
        num_inference_steps = args.num_steps,
        guidance_scale      = args.guidance_scale,
        seed                = args.seed,
    )

    save_output(video, args.output, args.num_frames, fps=args.fps)
    print(f"\nDone. Output: {args.output}")


if __name__ == "__main__":
    main()
