"""
Recompile WAN 2.2 T2V-A14B Text Encoder with CP=1 (world_size=4).

The original text encoder was compiled with world_size=8 (TP=4, CP=2),
but UMT5 doesn't need context parallelism. The CP=2 expansion causes
incorrect embeddings.

Fix: Recompile with world_size=4 (TP=4, CP=1).

Uses the same compilation code from aws-neuron-samples but overrides world_size.
"""
import os
import sys
import json
import shutil

# Configuration
SAMPLES_DIR = "/home/ubuntu/aws-neuron-samples/torch-neuronx/inference/hf_pretrained_wan2.2_t2v_a14b"
ORIGINAL_COMPILED = "/opt/dlami/nvme/compiled_models_t2v_a14b"
FIXED_COMPILED = "/opt/dlami/nvme/compiled_models_t2v_a14b_fixed"
CACHE_DIR = "/mnt/nvme/wan2.2_t2v_a14b_hf_cache_dir"

TP_DEGREE = 4
WORLD_SIZE = 4  # CP=1 (was 8 = CP=2)
MAX_SEQ_LEN = 512

# Add the samples dir to path for imports
sys.path.insert(0, SAMPLES_DIR)
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "0"

print("=" * 60)
print("RECOMPILE TEXT ENCODER: world_size=4 (TP=4, CP=1)")
print("=" * 60)

# Strategy A: Try config-patch approach first
# Just copy the existing compiled model and change config.json
print("\n[Strategy A] Config patch — change world_size in config.json")
print("  (Works if world_size is not baked into the NEFF graph)")

te_fixed_dir = os.path.join(FIXED_COMPILED, "text_encoder")
os.makedirs(te_fixed_dir, exist_ok=True)

# Copy existing compiled text encoder
te_orig_dir = os.path.join(ORIGINAL_COMPILED, "text_encoder")
if os.path.exists(te_orig_dir):
    # Copy nxd_model.pt and weights
    for item in os.listdir(te_orig_dir):
        src = os.path.join(te_orig_dir, item)
        dst = os.path.join(te_fixed_dir, item)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    print(f"  Copied {te_orig_dir} → {te_fixed_dir}")

    # Patch config.json
    config_path = os.path.join(te_fixed_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    print(f"  Original config: {config}")
    config["world_size"] = WORLD_SIZE
    config["cp_degree"] = 1  # Explicitly mark CP=1
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"  Patched config:  {config}")
    print(f"  Saved to: {config_path}")
else:
    print(f"  ERROR: Original text encoder not found at {te_orig_dir}")
    sys.exit(1)

# Also copy transformer and other artifacts (symlink to original)
print("\n[Setup] Linking other compiled artifacts to fixed directory...")
for component in ["transformer", "transformer_2", "decoder_rolling", "post_quant_conv"]:
    src = os.path.join(ORIGINAL_COMPILED, component)
    dst = os.path.join(FIXED_COMPILED, component)
    if os.path.exists(src) and not os.path.exists(dst):
        os.symlink(src, dst)
        print(f"  Symlinked {component}")

print(f"\n[Strategy A] Done. Fixed models at: {FIXED_COMPILED}")
print(f"  Next: Run validate_text_encoder_cp1.py to check if config patch works")
print(f"  If it doesn't (cosine < 0.99), Strategy B (full recompile) is needed.")

# Strategy B: Full recompile using the samples compile script
print("\n" + "=" * 60)
print("[Strategy B] Full recompile (if Strategy A fails)")
print("=" * 60)

compile_script = os.path.join(SAMPLES_DIR, "neuron_wan2_2_t2v_a14b", "compile_text_encoder.py")
if os.path.exists(compile_script):
    print(f"  Compile script found: {compile_script}")
    print(f"  To recompile from scratch:")
    print(f"    cd {SAMPLES_DIR}")
    print(f"    PYTHONPATH={SAMPLES_DIR}:$PYTHONPATH python {compile_script} \\")
    print(f"      --max_sequence_length {MAX_SEQ_LEN} \\")
    print(f"      --tp_degree {TP_DEGREE} --world_size {WORLD_SIZE} \\")
    print(f"      --compiled_models_dir {FIXED_COMPILED} \\")
    print(f"      --cache_dir {CACHE_DIR}")
else:
    print(f"  Compile script not found at expected location")
    print(f"  Available scripts: {os.listdir(os.path.join(SAMPLES_DIR, 'neuron_wan2_2_t2v_a14b'))}")

print("\n✅ Strategy A setup complete. Run validate_text_encoder_cp1.py next.")
