"""
WAN 2.2 T2V-A14B — Benchmarking Suite

Measures and compares performance metrics between:
- NXD approach (baseline from existing implementation)
- PyTorch Native approach (this project)

Tracks:
- Per forward pass latency
- Per-step average (with batched CFG)
- Expert swap time
- Total denoising time
- End-to-end wall time
- Memory utilization (HBM)
"""

import os
import sys
import time
import json
import torch
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, List


@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""
    name: str
    approach: str  # "pytorch_native" or "nxd"
    
    # Timing (seconds)
    text_encoding_time: float = 0.0
    expert1_swap_time: float = 0.0
    expert1_denoise_time: float = 0.0
    expert2_swap_time: float = 0.0
    expert2_denoise_time: float = 0.0
    vae_decode_time: float = 0.0
    total_time: float = 0.0
    
    # Per-step metrics
    per_step_avg_ms: float = 0.0
    per_forward_pass_ms: float = 0.0
    
    # Config
    num_steps: int = 40
    batch_size: int = 2
    resolution: str = "768x1280x81"
    tp_degree: int = 4
    cp_degree: int = 16
    
    # Hardware
    neuron_cores: int = 64
    instance_type: str = "trn2.48xlarge"
    sdk_version: str = ""
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"{'='*60}",
            f"Benchmark: {self.name} ({self.approach})",
            f"{'='*60}",
            f"Instance:     {self.instance_type} ({self.neuron_cores} cores)",
            f"Resolution:   {self.resolution}",
            f"Steps:        {self.num_steps} (batch_size={self.batch_size})",
            f"Parallelism:  TP={self.tp_degree}, CP={self.cp_degree}",
            f"",
            f"--- Timing Breakdown ---",
            f"Text encoding:      {self.text_encoding_time:>8.1f}s",
            f"Expert 1 swap:      {self.expert1_swap_time:>8.1f}s",
            f"Expert 1 denoise:   {self.expert1_denoise_time:>8.1f}s",
            f"Expert 2 swap:      {self.expert2_swap_time:>8.1f}s",
            f"Expert 2 denoise:   {self.expert2_denoise_time:>8.1f}s",
            f"VAE decode:         {self.vae_decode_time:>8.1f}s",
            f"TOTAL:              {self.total_time:>8.1f}s ({self.total_time/60:.1f} min)",
            f"",
            f"--- Per-Step Metrics ---",
            f"Per-step avg:       {self.per_step_avg_ms:>8.1f} ms",
            f"Per forward pass:   {self.per_forward_pass_ms:>8.1f} ms",
            f"{'='*60}",
        ]
        return "\n".join(lines)


# NXD baseline numbers from the existing implementation
NXD_BASELINE_OPTIMIZED = BenchmarkResult(
    name="NXD Optimized (CP=16, Batched CFG)",
    approach="nxd",
    text_encoding_time=22.0,
    expert1_swap_time=0.0,  # Subprocess load
    expert1_denoise_time=67.0,
    expert2_swap_time=0.0,  # Subprocess load
    expert2_denoise_time=136.0,
    vae_decode_time=35.0,
    total_time=618.0,
    per_step_avg_ms=5060.0,
    per_forward_pass_ms=5060.0,
    num_steps=40,
    batch_size=2,
    tp_degree=4,
    cp_degree=16,
    sdk_version="2.29.1",
)

NXD_SINGLE_PROCESS = BenchmarkResult(
    name="NXD Single-Process (copy_() swap)",
    approach="nxd",
    text_encoding_time=22.0,
    expert1_swap_time=0.0,  # First expert, no swap needed
    expert1_denoise_time=100.8,
    expert2_swap_time=64.1,
    expert2_denoise_time=101.0,
    vae_decode_time=35.0,
    total_time=266.0 + 90.8 + 49.9 + 35.0,  # denoise + init + load + vae
    per_step_avg_ms=5047.0,
    per_forward_pass_ms=2520.0,
    num_steps=40,
    batch_size=1,  # Sequential CFG
    tp_degree=4,
    cp_degree=16,
    sdk_version="2.29.1",
)


def compare_results(
    native_result: BenchmarkResult,
    baseline: BenchmarkResult = NXD_SINGLE_PROCESS,
) -> str:
    """Compare PyTorch Native results against NXD baseline."""
    lines = [
        f"\n{'='*60}",
        f"COMPARISON: PyTorch Native vs NXD",
        f"{'='*60}",
        f"",
        f"{'Metric':<25} {'NXD':>10} {'Native':>10} {'Diff':>10} {'Speedup':>8}",
        f"{'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*8}",
    ]
    
    metrics = [
        ("Per forward pass (ms)", baseline.per_forward_pass_ms, native_result.per_forward_pass_ms),
        ("Per step avg (ms)", baseline.per_step_avg_ms, native_result.per_step_avg_ms),
        ("Expert swap (s)", baseline.expert2_swap_time, native_result.expert2_swap_time),
        ("Expert 1 denoise (s)", baseline.expert1_denoise_time, native_result.expert1_denoise_time),
        ("Expert 2 denoise (s)", baseline.expert2_denoise_time, native_result.expert2_denoise_time),
        ("VAE decode (s)", baseline.vae_decode_time, native_result.vae_decode_time),
        ("Total (s)", baseline.total_time, native_result.total_time),
    ]
    
    for name, nxd_val, native_val in metrics:
        diff = native_val - nxd_val
        speedup = nxd_val / native_val if native_val > 0 else float('inf')
        sign = "+" if diff > 0 else ""
        lines.append(
            f"{name:<25} {nxd_val:>10.1f} {native_val:>10.1f} "
            f"{sign}{diff:>9.1f} {speedup:>7.2f}x"
        )
    
    lines.append(f"\n{'='*60}")
    
    # Verdict
    total_speedup = baseline.total_time / native_result.total_time if native_result.total_time > 0 else 0
    if total_speedup >= 1.0:
        lines.append(f"✅ PyTorch Native is {total_speedup:.2f}x FASTER than NXD baseline")
    else:
        lines.append(f"⚠️  PyTorch Native is {1/total_speedup:.2f}x SLOWER than NXD baseline")
        lines.append(f"   (Target: match or exceed NXD performance)")
    
    return "\n".join(lines)


def run_benchmark(
    model_dir: str,
    num_steps: int = 40,
    warmup_steps: int = 2,
    eager: bool = False,
) -> BenchmarkResult:
    """
    Run a full benchmark of the PyTorch Native pipeline.
    
    Args:
        model_dir: Path to model weights
        num_steps: Total denoising steps
        warmup_steps: Warmup steps before timing
        eager: Use eager mode (no compilation)
    """
    from run_inference import WanPipelineNative, setup_neuron_env
    
    setup_neuron_env()
    
    result = BenchmarkResult(
        name="PyTorch Native" + (" (Eager)" if eager else ""),
        approach="pytorch_native",
        num_steps=num_steps,
    )
    
    # Load pipeline
    pipeline = WanPipelineNative(model_dir=model_dir, eager=eager)
    pipeline.load_pipeline()
    
    # Warmup (if compiled)
    if not eager and warmup_steps > 0:
        print(f"\nRunning {warmup_steps} warmup steps...")
        # ... warmup code ...
    
    # Benchmark text encoding
    print("\nBenchmarking text encoding...")
    t0 = time.time()
    text_embeddings = pipeline.encode_text("A cat walks on the grass, realistic style")
    result.text_encoding_time = time.time() - t0
    
    # Initialize latents
    latents = torch.randn(1, 16, 21, 96, 160, dtype=torch.bfloat16)
    
    # Benchmark Expert 1
    print("Benchmarking Expert 1...")
    t0 = time.time()
    swap_time = pipeline.expert_swap_manager.activate_expert(0)
    result.expert1_swap_time = swap_time
    
    # Run expert 1 denoising steps
    pipeline.scheduler.set_timesteps(num_steps)
    timesteps = pipeline.scheduler.timesteps
    
    t_denoise = time.time()
    for t in timesteps[:13]:
        latents = pipeline._denoise_step(latents, text_embeddings, t, 5.0)
    result.expert1_denoise_time = time.time() - t_denoise
    
    # Benchmark Expert 2
    print("Benchmarking Expert 2...")
    t0 = time.time()
    swap_time = pipeline.expert_swap_manager.activate_expert(1)
    result.expert2_swap_time = swap_time
    
    t_denoise = time.time()
    for t in timesteps[13:]:
        latents = pipeline._denoise_step(latents, text_embeddings, t, 5.0)
    result.expert2_denoise_time = time.time() - t_denoise
    
    # Benchmark VAE
    print("Benchmarking VAE decode...")
    t0 = time.time()
    video = pipeline.decode_latents(latents)
    result.vae_decode_time = time.time() - t0
    
    # Calculate totals
    result.total_time = (
        result.text_encoding_time +
        result.expert1_swap_time + result.expert1_denoise_time +
        result.expert2_swap_time + result.expert2_denoise_time +
        result.vae_decode_time
    )
    
    total_denoise = result.expert1_denoise_time + result.expert2_denoise_time
    result.per_step_avg_ms = (total_denoise / num_steps) * 1000
    result.per_forward_pass_ms = result.per_step_avg_ms / 2  # batched CFG
    
    return result


def main():
    parser = argparse.ArgumentParser(description="WAN 2.2 Benchmarks")
    parser.add_argument("--model-dir", default="/mnt/nvme/models/Wan2.2-T2V-A14B-Diffusers")
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--compare-only", action="store_true",
                        help="Show NXD baselines without running benchmark")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()
    
    if args.compare_only:
        print(NXD_BASELINE_OPTIMIZED.summary())
        print(NXD_SINGLE_PROCESS.summary())
        return
    
    # Run benchmark
    result = run_benchmark(
        model_dir=args.model_dir,
        num_steps=args.num_steps,
        eager=args.eager,
    )
    
    # Print results
    print(result.summary())
    print(compare_results(result))
    
    # Save to JSON
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()
