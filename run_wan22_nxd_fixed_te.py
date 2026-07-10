"""
WAN 2.2 T2V-A14B inference with FIXED text encoder (CP=1, world_size=4).

Identical to run_wan2.2_t2v_a14b.py but uses the fixed text encoder
from /opt/dlami/nvme/compiled_models_t2v_a14b_fixed/.

Run ONLY after validate_text_encoder_cp1.py confirms cosine > 0.99.

Usage:
    python run_wan22_nxd_fixed_te.py \
        --compiled_models_dir /opt/dlami/nvme/compiled_models_t2v_a14b_fixed \
        --prompt "A cat" \
        --cpu_vae_decoder
"""
import sys
import os

SAMPLES_DIR = "/home/ubuntu/aws-neuron-samples/torch-neuronx/inference/hf_pretrained_wan2.2_t2v_a14b"
sys.path.insert(0, SAMPLES_DIR)
os.environ["PATH"] = "/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin:" + os.environ.get("PATH", "")

# Just delegate to the existing inference script with the fixed compiled dir
from run_wan2_2_t2v_a14b import main
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled_models_dir", default="/opt/dlami/nvme/compiled_models_t2v_a14b_fixed")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--guidance_scale_2", type=float, default=3.0)
    parser.add_argument("--prompt", default="A fluffy orange tabby cat walking through a sunlit garden")
    parser.add_argument("--negative_prompt", default="Bright tones, overexposed, static, blurred details")
    parser.add_argument("--output", default="/mnt/nvme/outputs/cat_fixed_te.mp4")
    parser.add_argument("--cpu_text_encoder", action="store_true", default=False)
    parser.add_argument("--cpu_vae_decoder", action="store_true", default=False)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    args = parser.parse_args()
    main(args)
