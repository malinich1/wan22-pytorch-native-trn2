"""
Download WAN 2.2 T2V-A14B model weights from Hugging Face.
Saves to /mnt/nvme/models/ (~118 GB).
"""

import os
import argparse
from huggingface_hub import snapshot_download


MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
DEFAULT_MODEL_DIR = "/mnt/nvme/models/Wan2.2-T2V-A14B-Diffusers"


def main():
    parser = argparse.ArgumentParser(description="Download WAN 2.2 T2V-A14B model")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=f"Directory to save model (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (if model is gated)",
    )
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    print(f"Downloading {MODEL_ID} to {args.model_dir}")
    print("This will download ~118 GB of model weights...")

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=args.model_dir,
        token=args.token,
        ignore_patterns=["*.msgpack", "*.h5"],  # Only get safetensors
    )

    print(f"\nModel downloaded successfully to: {args.model_dir}")
    print(f"Total size: {sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fn in os.walk(args.model_dir) for f in fn) / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
