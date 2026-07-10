"""
WAN 2.2 T2V-A14B — Distributed Inference with Tensor Parallelism (Beta 3 Native)

Uses the Beta 3 DLC PyTorch 2.11 native Neuron backend with torch.compile for
distributed tensor-parallel inference of the WAN 2.2 14B MoE video diffusion model.

Architecture:
  - 2 x WanTransformer3DModel (14B params each, MoE switching at timestep 875)
  - UMT5-XXL text encoder (replicated on rank 0)
  - AutoencoderKLWan VAE (replicated, decode on CPU)
  - Tensor Parallelism: shard transformer across N NeuronCores

Pattern (follows Qwen2 torch_compile reference):
  1. dist.init_process_group(backend="neuron")
  2. Load transformer on CPU
  3. Shard weights across TP ranks (column/row parallel linear)
  4. Move sharded model to torch.device("neuron")
  5. torch.compile(forward, backend="neuron", fullgraph=True, dynamic=False)
  6. All-reduce after forward for proper TP communication
  7. MoE switching: timestep >= 875 -> expert_1, < 875 -> expert_2

Usage (inside Beta 3 DLC container):
  # TP=2 (recommended starting point for 14B)
  torchrun --nproc-per-node 2 beta3_wan22_distributed_inference.py \
      --prompt "A cat walks through a sunlit garden" \
      --height 384 --width 640 --num-frames 17 --num-steps 20

  # TP=4 for higher resolution
  torchrun --nproc-per-node 4 beta3_wan22_distributed_inference.py \
      --prompt "Ocean waves crashing on rocky cliffs" \
      --height 768 --width 1280 --num-frames 81 --num-steps 40

  # Eager mode (no compilation, instant start, slower per-step):
  torchrun --nproc-per-node 2 beta3_wan22_distributed_inference.py \
      --prompt "A cat" --eager \
      --height 256 --width 256 --num-frames 1 --num-steps 5

DLC: 421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest
"""

import argparse
import gc
import logging
import os
import time

import torch
import torch.distributed as dist

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [Rank %(process)d] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
CACHE_DIR = "/mnt/nvme/wan2.2_t2v_a14b_hf_cache_dir"
DEFAULT_HEIGHT = 384
DEFAULT_WIDTH = 640
DEFAULT_NUM_FRAMES = 17
DEFAULT_NUM_STEPS = 20
DEFAULT_GUIDANCE = 4.0
DEFAULT_GUIDANCE_2 = 3.0
DEFAULT_SEED = 42
DEFAULT_OUTPUT = "/mnt/nvme/outputs/beta3_wan22_tp.mp4"
DEFAULT_NEFF_CACHE = "/mnt/nvme/neff_cache_beta3_wan22"
MOE_BOUNDARY = 875  # timestep threshold for expert switching

# torch.compile config
torch._dynamo.config.cache_size_limit = 64
torch.set_default_dtype(torch.float32)


# ============================================================================
# Environment Setup
# ============================================================================

def setup_neuron_env(neff_cache: str):
    """Configure Beta 3 Neuron environment for distributed inference."""
    os.environ["NEURON_CC_FLAGS"] = "-O1 --auto-cast=none"
    os.environ["TORCH_NEURONX_ENABLE_ASYNC_NRT"] = "1"
    os.environ["TORCH_NEURONX_ENABLE_HOST_CC"] = "1"
    os.environ.setdefault("NEURON_RT_VIRTUAL_CORE_SIZE", "2")  # LNC2
    os.makedirs(neff_cache, exist_ok=True)
    os.environ["NEURON_COMPILE_CACHE_URL"] = f"file://{neff_cache}"
    os.environ["NEURONX_CACHE"] = neff_cache
    logger.info(f"Beta 3 Neuron env configured. NEFF cache: {neff_cache}")


# ============================================================================
# Tensor Parallelism Utilities
# ============================================================================

def shard_linear_column(weight, bias, rank, world_size):
    """Shard a Linear layer's weight along output dimension (column parallel)."""
    out_features = weight.shape[0]
    shard_size = out_features // world_size
    start = rank * shard_size
    end = start + shard_size
    sharded_weight = weight[start:end].contiguous()
    sharded_bias = bias[start:end].contiguous() if bias is not None else None
    return sharded_weight, sharded_bias


def shard_linear_row(weight, bias, rank, world_size):
    """Shard a Linear layer's weight along input dimension (row parallel)."""
    in_features = weight.shape[1]
    shard_size = in_features // world_size
    start = rank * shard_size
    end = start + shard_size
    sharded_weight = weight[:, start:end].contiguous()
    # Bias is NOT sharded for row parallel (added after all-reduce)
    return sharded_weight, bias


def shard_transformer_weights(transformer, rank, world_size):
    """
    Apply tensor parallelism sharding to WanTransformer3DModel.
    
    Strategy:
    - QKV projections: column parallel (shard output heads)
    - Output projections: row parallel (shard input, all-reduce after)
    - FFN first linear: column parallel
    - FFN second linear: row parallel
    - Norms, embeddings: replicated
    
    For simplicity in this implementation, we shard the attention and FFN
    layers and rely on torch.compile to handle the communication.
    """
    logger.info(f"Sharding transformer weights for rank {rank}/{world_size}")
    
    state_dict = transformer.state_dict()
    new_state_dict = {}
    
    sharded_count = 0
    replicated_count = 0
    
    for key, tensor in state_dict.items():
        # Identify shardable layers by name pattern
        # WanTransformer3DModel has blocks with attn and ffn sublayers
        if any(pattern in key for pattern in [
            'to_q.weight', 'to_k.weight', 'to_v.weight',  # QKV: column parallel
            'ffn.0.weight', 'ffn.0.bias',  # FFN first layer: column parallel  
            'ffn_2.0.weight', 'ffn_2.0.bias',  # FFN2 first layer: column parallel
        ]):
            if 'weight' in key:
                if tensor.dim() == 2:
                    sharded, _ = shard_linear_column(tensor, None, rank, world_size)
                    new_state_dict[key] = sharded
                    sharded_count += 1
                    continue
            elif 'bias' in key:
                out_features = tensor.shape[0]
                shard_size = out_features // world_size
                start = rank * shard_size
                new_state_dict[key] = tensor[start:start + shard_size].contiguous()
                sharded_count += 1
                continue
                
        elif any(pattern in key for pattern in [
            'to_out.0.weight',  # Output projection: row parallel
            'ffn.2.weight',  # FFN second layer: row parallel
            'ffn_2.2.weight',  # FFN2 second layer: row parallel
        ]):
            if tensor.dim() == 2:
                sharded, _ = shard_linear_row(tensor, None, rank, world_size)
                new_state_dict[key] = sharded
                sharded_count += 1
                continue
        
        # Default: replicate
        new_state_dict[key] = tensor
        replicated_count += 1
    
    logger.info(f"  Sharded: {sharded_count} tensors, Replicated: {replicated_count} tensors")
    
    # Apply sharded weights in-place by modifying parameters directly
    for name, param in transformer.named_parameters():
        if name in new_state_dict:
            new_data = new_state_dict[name]
            if new_data.shape != param.shape:
                # Replace parameter with resized version
                param.data = new_data.to(param.dtype)
    
    for name, buf in transformer.named_buffers():
        if name in new_state_dict:
            new_data = new_state_dict[name]
            if new_data.shape != buf.shape:
                buf.data = new_data.to(buf.dtype)
    
    return transformer


# ============================================================================
# Distributed Forward Wrapper
# ============================================================================

class DistributedTransformerWrapper(torch.nn.Module):
    """Wraps transformer forward with all-reduce for tensor parallelism."""
    
    def __init__(self, transformer, world_size):
        super().__init__()
        self.transformer = transformer
        self.world_size = world_size
    
    def forward(self, hidden_states, timestep, encoder_hidden_states, **kwargs):
        output = self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]
        
        # All-reduce across TP ranks
        if self.world_size > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
            # Average (since each rank computes partial output)
            output = output / self.world_size
        
        return output


# ============================================================================
# Main Inference
# ============================================================================

def run_inference(args):
    """Run WAN 2.2 T2V-A14B distributed inference."""
    
    # --- Initialize distributed ---
    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    logger.info(f"WAN 2.2 T2V-A14B Distributed Inference (Beta 3)")
    logger.info(f"  TP={world_size}, Rank={rank}")
    logger.info(f"  Prompt: {args.prompt}")
    logger.info(f"  Resolution: {args.width}x{args.height}, {args.num_frames} frames")
    logger.info(f"  Steps: {args.num_steps}, CFG: {args.guidance}/{args.guidance_2}")
    logger.info(f"  Mode: {'eager' if args.eager else 'torch.compile'}")
    
    device = torch.device("neuron")
    
    # --- Load pipeline components ---
    from diffusers import AutoencoderKLWan, WanTransformer3DModel, UniPCMultistepScheduler
    from transformers import AutoTokenizer, UMT5EncoderModel
    
    # Tokenizer (all ranks)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer", cache_dir=CACHE_DIR)
    
    # Text encoder (rank 0 only, broadcast embeddings)
    if rank == 0:
        logger.info("Loading text encoder (UMT5-XXL)...")
        t0 = time.time()
        text_encoder = UMT5EncoderModel.from_pretrained(
            MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16, cache_dir=CACHE_DIR
        ).eval()
        logger.info(f"Text encoder loaded in {time.time()-t0:.1f}s")
    
    # Load both transformer experts
    logger.info("Loading transformer_1 (high-noise expert)...")
    t0 = time.time()
    transformer_1 = WanTransformer3DModel.from_pretrained(
        MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16, cache_dir=CACHE_DIR
    ).eval()
    n_params = sum(p.numel() for p in transformer_1.parameters()) / 1e9
    logger.info(f"Transformer_1 loaded in {time.time()-t0:.1f}s ({n_params:.2f}B params)")
    
    logger.info("Loading transformer_2 (low-noise expert)...")
    t0 = time.time()
    transformer_2 = WanTransformer3DModel.from_pretrained(
        MODEL_ID, subfolder="transformer_2", torch_dtype=torch.bfloat16, cache_dir=CACHE_DIR
    ).eval()
    logger.info(f"Transformer_2 loaded in {time.time()-t0:.1f}s")
    
    # Scheduler
    scheduler = UniPCMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler", cache_dir=CACHE_DIR)
    
    # VAE (rank 0 only, for decode)
    if rank == 0:
        logger.info("Loading VAE...")
        vae = AutoencoderKLWan.from_pretrained(
            MODEL_ID, subfolder="vae", torch_dtype=torch.float32, cache_dir=CACHE_DIR
        ).eval()
    
    dist.barrier()
    
    # --- Shard transformer weights across TP ranks ---
    logger.info("Sharding transformer_1...")
    transformer_1 = shard_transformer_weights(transformer_1, rank, world_size)
    logger.info("Sharding transformer_2...")
    transformer_2 = shard_transformer_weights(transformer_2, rank, world_size)
    
    # --- Move to Neuron device ---
    logger.info(f"Moving transformer_1 to {device}...")
    t0 = time.time()
    transformer_1 = transformer_1.to(device)
    logger.info(f"Transformer_1 on device in {time.time()-t0:.1f}s")
    
    logger.info(f"Moving transformer_2 to {device}...")
    t0 = time.time()
    transformer_2 = transformer_2.to(device)
    logger.info(f"Transformer_2 on device in {time.time()-t0:.1f}s")
    
    # --- Wrap with distributed forward ---
    dist_t1 = DistributedTransformerWrapper(transformer_1, world_size)
    dist_t2 = DistributedTransformerWrapper(transformer_2, world_size)
    
    # --- torch.compile ---
    if not args.eager:
        logger.info("Compiling transformers with torch.compile(backend='neuron')...")
        dist_t1.forward = torch.compile(
            dist_t1.forward, backend="neuron", fullgraph=True, dynamic=False
        )
        dist_t2.forward = torch.compile(
            dist_t2.forward, backend="neuron", fullgraph=True, dynamic=False
        )
        logger.info("Compilation registered (NEFFs built on first pass)")
    
    dist.barrier()
    
    # --- Text Encoding (rank 0, broadcast) ---
    logger.info("Encoding text...")
    t0 = time.time()
    
    if rank == 0:
        text_inputs = tokenizer(
            args.prompt, max_length=512, padding="max_length", truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            prompt_embeds = text_encoder(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            ).last_hidden_state.to(torch.bfloat16)
        
        neg_inputs = tokenizer(
            args.negative_prompt, max_length=512, padding="max_length", truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            neg_embeds = text_encoder(
                input_ids=neg_inputs["input_ids"],
                attention_mask=neg_inputs["attention_mask"],
            ).last_hidden_state.to(torch.bfloat16)
        
        # Free text encoder
        del text_encoder
        gc.collect()
    else:
        prompt_embeds = torch.zeros(1, 512, 4096, dtype=torch.bfloat16)
        neg_embeds = torch.zeros(1, 512, 4096, dtype=torch.bfloat16)
    
    # Broadcast embeddings to all ranks
    dist.broadcast(prompt_embeds, src=0)
    dist.broadcast(neg_embeds, src=0)
    logger.info(f"Text encoded and broadcast in {time.time()-t0:.1f}s")
    
    # --- Prepare latents ---
    torch.manual_seed(args.seed + rank)  # Different seed per rank for diversity
    latent_ch = transformer_1.config.in_channels
    latent_h = args.height // 8
    latent_w = args.width // 8
    latent_t = (args.num_frames - 1) // 4 + 1 if args.num_frames > 1 else 1
    
    # Same latents across all ranks (use base seed)
    torch.manual_seed(args.seed)
    latents = torch.randn(1, latent_ch, latent_t, latent_h, latent_w, dtype=torch.float32)
    logger.info(f"Latent shape: {list(latents.shape)}")
    
    # --- Denoising with MoE switching ---
    scheduler.set_timesteps(args.num_steps)
    timesteps = scheduler.timesteps
    
    switch_idx = None
    for i, t in enumerate(timesteps):
        if t < MOE_BOUNDARY:
            switch_idx = i
            break
    if switch_idx is None:
        switch_idx = len(timesteps)
    
    logger.info(f"Denoising: {args.num_steps} steps, MoE switch at step {switch_idx}")
    
    total_t0 = time.time()
    denoise_t0 = time.time()
    
    for i, t in enumerate(timesteps):
        step_t0 = time.time()
        
        # Select expert
        if i < switch_idx:
            dist_transformer = dist_t1
            cfg_scale = args.guidance
        else:
            dist_transformer = dist_t2
            cfg_scale = args.guidance_2
        
        x = latents.to(device, dtype=torch.bfloat16)
        t_in = t.expand(1).to(device)
        pe = prompt_embeds.to(device)
        ne = neg_embeds.to(device)
        
        with torch.no_grad():
            # Conditional pass
            noise_pred = dist_transformer(x, t_in, pe)
            
            # Unconditional pass (CFG)
            if cfg_scale > 1.0:
                noise_uncond = dist_transformer(x, t_in, ne)
                noise_pred = noise_uncond + cfg_scale * (noise_pred - noise_uncond)
        
        # Scheduler step (CPU)
        latents = scheduler.step(noise_pred.cpu(), t, latents, return_dict=False)[0]
        
        step_time = time.time() - step_t0
        if rank == 0 and ((i + 1) % 5 == 0 or (i + 1) == args.num_steps or i == switch_idx):
            expert = "T1" if i < switch_idx else "T2"
            logger.info(f"  Step {i+1}/{args.num_steps} [{expert}] (t={t.item():.0f}): {step_time:.2f}s")
    
    denoise_time = time.time() - denoise_t0
    if rank == 0:
        logger.info(f"Denoising done in {denoise_time:.1f}s ({denoise_time/args.num_steps:.2f}s/step)")
    
    # --- VAE Decode (rank 0 only, CPU) ---
    if rank == 0:
        logger.info("VAE decoding (CPU)...")
        t0 = time.time()
        with torch.no_grad():
            latents_mean = torch.tensor(vae.config.latents_mean, dtype=latents.dtype).view(1, -1, 1, 1, 1)
            latents_std = torch.tensor(vae.config.latents_std, dtype=latents.dtype).view(1, -1, 1, 1, 1)
            latents_scaled = latents * latents_std + latents_mean
            video = vae.decode(latents_scaled.to(torch.float32)).sample
        logger.info(f"VAE decode in {time.time()-t0:.1f}s, shape={video.shape}")
        
        total_time = time.time() - total_t0
        logger.info(f"TOTAL INFERENCE: {total_time:.1f}s ({total_time/60:.1f} min)")
        
        # Save output
        save_output(video, args.output, args.num_frames)
    
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        logger.info("Done.")


# ============================================================================
# Output Saving
# ============================================================================

def save_output(video, output_path, num_frames, fps=16):
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
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="WAN 2.2 T2V-A14B Distributed Inference (Beta 3)")
    parser.add_argument("--prompt", type=str, default="A close-up photograph of a beautiful fluffy orange tabby cat sitting in a sunlit garden, photorealistic, sharp focus, 8k")
    parser.add_argument("--negative-prompt", type=str, default="blurry, low quality, deformed, ugly, cartoon, anime, painting, text, watermark")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE)
    parser.add_argument("--guidance-2", type=float, default=DEFAULT_GUIDANCE_2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--neff-cache", type=str, default=DEFAULT_NEFF_CACHE)
    parser.add_argument("--eager", action="store_true", help="Skip torch.compile, run eager mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if not args.eager:
        setup_neuron_env(args.neff_cache)
    
    run_inference(args)
