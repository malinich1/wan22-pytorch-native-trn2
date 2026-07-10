"""Quick test: load UMT5 on CPU and run forward pass."""
import torch
import time
from transformers import UMT5EncoderModel

HF = "/mnt/nvme/wan2.2_t2v_a14b_hf_cache_dir/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/5be7df9619b54f4e2667b2755bc6a756675b5cd7"

print("Loading UMT5...")
t0 = time.time()
te = UMT5EncoderModel.from_pretrained(HF, subfolder="text_encoder", torch_dtype=torch.bfloat16).eval()
print(f"Loaded in {time.time()-t0:.1f}s ({sum(p.numel() for p in te.parameters())/1e9:.1f}B)")

ids = torch.ones(1, 512, dtype=torch.long)
mask = torch.ones(1, 512, dtype=torch.long)

print("Forward pass...")
t0 = time.time()
with torch.no_grad():
    out = te(input_ids=ids, attention_mask=mask)
print(f"Done in {time.time()-t0:.1f}s")
print(f"Output: {out.last_hidden_state.shape}, mean={out.last_hidden_state.float().mean():.4f}")
print("SUCCESS")
