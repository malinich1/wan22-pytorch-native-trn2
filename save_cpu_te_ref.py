"""Save CPU text encoder reference embeddings to disk."""
import torch
from transformers import AutoTokenizer, UMT5EncoderModel

HF = "/mnt/nvme/wan2.2_t2v_a14b_hf_cache_dir/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/5be7df9619b54f4e2667b2755bc6a756675b5cd7"
prompt = "A beautiful fluffy orange tabby cat walking through a sunlit garden"

tokenizer = AutoTokenizer.from_pretrained(HF, subfolder="tokenizer")
ids = tokenizer(prompt, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
te = UMT5EncoderModel.from_pretrained(HF, subfolder="text_encoder", torch_dtype=torch.bfloat16).eval()
with torch.no_grad():
    out = te(input_ids=ids.input_ids, attention_mask=ids.attention_mask).last_hidden_state
torch.save({"embeds": out, "ids": ids.input_ids, "mask": ids.attention_mask}, "/mnt/nvme/cpu_te_reference.pt")
print(f"Saved: {out.shape}, mean={out.float().mean():.6f}, std={out.float().std():.6f}")
