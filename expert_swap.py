"""
WAN 2.2 T2V-A14B — Expert Weight Swapping via tensor.copy_()

PyTorch Native approach for swapping MoE expert weights in-place
WITHOUT reinitializing the model or NEFF.

Key insight: On Neuron, compiled NEFFs read weight tensors via DMA from
the CPU tensor buffer. Using tensor.copy_() updates the buffer in-place,
so the next forward pass reads the new weights automatically — no 
recompilation or re-initialization needed.

Performance from NXD approach (target to match):
- copy_() swap time: 64.1s across 64 ranks
- ~1ms per tensor × ~64K tensors

This module implements the same swap for PyTorch Native models.
"""

import os
import time
import torch
from pathlib import Path
from typing import Dict, Optional
from safetensors import safe_open


def load_expert_weights(
    model_dir: str,
    expert_id: int,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, torch.Tensor]:
    """
    Load expert weights from safetensors checkpoint.
    
    WAN 2.2 A14B has 2 experts:
      - Expert 1 (expert_id=0): High-noise denoising steps
      - Expert 2 (expert_id=1): Low-noise denoising steps
    They share architecture but have completely independent weights.
    
    Args:
        model_dir: Path to model directory
        expert_id: 0 or 1
        dtype: Target dtype (bfloat16 for Trn2)
    
    Returns:
        Dictionary mapping parameter names to tensors
    """
    expert_dir = os.path.join(model_dir, "transformer", f"expert_{expert_id}")
    
    # If experts are in a single safetensors with prefix
    # Adjust path based on actual model layout
    safetensors_files = list(Path(model_dir, "transformer").glob("*.safetensors"))
    
    weights = {}
    for sf_file in safetensors_files:
        with safe_open(str(sf_file), framework="pt") as f:
            for key in f.keys():
                # Filter to this expert's weights
                if f"expert_{expert_id}" in key or expert_id == 0:
                    tensor = f.get_tensor(key).to(dtype)
                    weights[key] = tensor
    
    print(f"  Loaded expert {expert_id}: {len(weights)} tensors, "
          f"{sum(t.numel() * t.element_size() for t in weights.values()) / 1e9:.1f} GB")
    
    return weights


def swap_expert_weights_native(
    model: torch.nn.Module,
    new_weights: Dict[str, torch.Tensor],
    world_size: int = 64,
) -> float:
    """
    Swap expert weights in-place using tensor.copy_() — PyTorch Native version.
    
    This is the PyTorch Native equivalent of the NXD copy_() workaround.
    Instead of NxDModel.weights[rank][key].copy_(), we directly update
    the model's state_dict parameters in-place.
    
    For distributed models (TP+CP), each rank holds a shard of the full
    weight. We copy the corresponding shard to each rank's parameter tensor.
    
    Args:
        model: The compiled PyTorch model (on Neuron device)
        new_weights: Dictionary of new weight tensors
        world_size: Number of distributed ranks (64 for full trn2.48xlarge)
    
    Returns:
        Time taken for the swap in seconds
    """
    t0 = time.time()
    
    swap_count = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in new_weights:
                # In-place copy — updates the DMA buffer without recompilation
                param.data.copy_(new_weights[name])
                swap_count += 1
            else:
                # Try without module prefix
                short_name = name.split(".", 1)[-1] if "." in name else name
                if short_name in new_weights:
                    param.data.copy_(new_weights[short_name])
                    swap_count += 1
    
    elapsed = time.time() - t0
    print(f"  Expert swap complete: {swap_count} tensors in {elapsed:.1f}s")
    
    return elapsed


def swap_expert_weights_distributed(
    model: torch.nn.Module,
    new_weights: Dict[str, torch.Tensor],
    tp_degree: int = 4,
    cp_degree: int = 16,
    rank: int = 0,
) -> float:
    """
    Swap expert weights for distributed model with TP+CP sharding.
    
    For a model sharded across 64 NeuronCores (TP=4, CP=16):
    - TP-sharded params: split along attention head / hidden dim
    - CP doesn't shard weights (only activations/sequence)
    
    Each TP rank holds 1/4 of attention weights and 1/4 of MLP weights.
    CP ranks share the same weight shard (only sequence is split).
    
    Args:
        model: Distributed model (this rank's shard)
        new_weights: Full expert weights (will be sharded here)
        tp_degree: Tensor parallelism degree
        cp_degree: Context parallelism degree
        rank: This process's global rank
    
    Returns:
        Time taken for the swap
    """
    tp_rank = rank % tp_degree  # 0-3
    # cp_rank = rank // tp_degree  # 0-15 (not needed for weight swap)
    
    t0 = time.time()
    swap_count = 0
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in new_weights:
                continue
            
            full_weight = new_weights[name]
            
            # Determine if this parameter is TP-sharded
            if _is_tp_sharded(name, model):
                # Shard along the TP dimension
                shard_dim = _get_shard_dim(name, model)
                shard_size = full_weight.shape[shard_dim] // tp_degree
                start = tp_rank * shard_size
                end = start + shard_size
                
                # Extract this rank's shard
                shard = full_weight.narrow(shard_dim, start, shard_size).contiguous()
                param.data.copy_(shard)
            else:
                # Non-sharded parameter — copy full tensor
                param.data.copy_(full_weight)
            
            swap_count += 1
    
    elapsed = time.time() - t0
    print(f"  [Rank {rank}] Expert swap: {swap_count} tensors in {elapsed:.3f}s")
    
    return elapsed


def _is_tp_sharded(param_name: str, model: torch.nn.Module) -> bool:
    """Check if a parameter is sharded across TP ranks."""
    # WAN 2.2 DiT shards: attention Q/K/V/O projections and MLP layers
    tp_patterns = [
        "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out",
        "ff.net.0",  # MLP up projection
        "ff.net.2",  # MLP down projection
    ]
    return any(pattern in param_name for pattern in tp_patterns)


def _get_shard_dim(param_name: str, model: torch.nn.Module) -> int:
    """Get the dimension to shard along for TP."""
    # Column-parallel (output dim sharded): Q, K, V, MLP up
    # Row-parallel (input dim sharded): O proj, MLP down
    row_parallel_patterns = ["attn.to_out", "ff.net.2"]
    if any(pattern in param_name for pattern in row_parallel_patterns):
        return 1  # Shard along input dimension
    return 0  # Shard along output dimension


class ExpertSwapManager:
    """
    Manages expert weight swapping for the WAN 2.2 MoE model.
    
    Preloads both expert weights into CPU memory, then uses copy_()
    to swap between them without recompilation.
    
    Usage:
        manager = ExpertSwapManager(model_dir, model)
        manager.preload_experts()
        
        # Switch to expert 1 (high noise)
        manager.activate_expert(0)
        # ... run denoising steps ...
        
        # Switch to expert 2 (low noise)  
        manager.activate_expert(1)
        # ... run denoising steps ...
    """
    
    def __init__(
        self,
        model_dir: str,
        model: torch.nn.Module,
        tp_degree: int = 4,
        cp_degree: int = 16,
        rank: int = 0,
    ):
        self.model_dir = model_dir
        self.model = model
        self.tp_degree = tp_degree
        self.cp_degree = cp_degree
        self.rank = rank
        self.expert_weights: Dict[int, Dict[str, torch.Tensor]] = {}
        self.active_expert: Optional[int] = None
    
    def preload_experts(self):
        """Preload both expert weight sets into CPU memory."""
        print("Preloading expert weights...")
        t0 = time.time()
        
        for expert_id in [0, 1]:
            self.expert_weights[expert_id] = load_expert_weights(
                self.model_dir, expert_id, dtype=torch.bfloat16
            )
        
        total_size = sum(
            sum(t.numel() * t.element_size() for t in w.values())
            for w in self.expert_weights.values()
        )
        elapsed = time.time() - t0
        print(f"Both experts preloaded: {total_size / 1e9:.1f} GB in {elapsed:.1f}s")
    
    def activate_expert(self, expert_id: int) -> float:
        """
        Activate an expert by swapping its weights into the compiled model.
        
        Returns swap time in seconds.
        """
        if expert_id == self.active_expert:
            print(f"  Expert {expert_id} already active, skipping swap")
            return 0.0
        
        if expert_id not in self.expert_weights:
            raise ValueError(f"Expert {expert_id} not preloaded. Call preload_experts() first.")
        
        print(f"Activating expert {expert_id}...")
        swap_time = swap_expert_weights_distributed(
            self.model,
            self.expert_weights[expert_id],
            tp_degree=self.tp_degree,
            cp_degree=self.cp_degree,
            rank=self.rank,
        )
        
        self.active_expert = expert_id
        return swap_time


if __name__ == "__main__":
    # Quick test / standalone usage
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/mnt/nvme/models/Wan2.2-T2V-A14B-Diffusers")
    args = parser.parse_args()
    
    print("Testing expert weight loading...")
    for eid in [0, 1]:
        weights = load_expert_weights(args.model_dir, eid)
        print(f"  Expert {eid}: {len(weights)} params")
