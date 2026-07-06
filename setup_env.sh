#!/bin/bash
# =============================================================================
# WAN 2.2 T2V-A14B — Environment Setup for PyTorch Native on Trainium 2
# Run on: trn2.48xlarge with Deep Learning AMI Neuron (Ubuntu 24.04)
# =============================================================================

set -euo pipefail

echo "=== WAN 2.2 PyTorch Native — Environment Setup ==="

# --- NVMe Setup (trn2.48xlarge has local NVMe storage) ---
echo "[1/5] Setting up NVMe storage..."
if [ -b /dev/nvme1n1 ]; then
    sudo mkfs.ext4 -F /dev/nvme1n1
    sudo mkdir -p /mnt/nvme
    sudo mount /dev/nvme1n1 /mnt/nvme
    sudo chown $USER:$USER /mnt/nvme
    echo "NVMe mounted at /mnt/nvme"
else
    echo "No NVMe device found, using EBS storage"
    mkdir -p /mnt/nvme
fi

# --- Create working directories ---
echo "[2/5] Creating directories..."
mkdir -p /mnt/nvme/models
mkdir -p /mnt/nvme/compiled_artifacts
mkdir -p /mnt/nvme/outputs

# --- Activate Neuron PyTorch venv ---
echo "[3/5] Activating Neuron PyTorch venv..."
# SDK 2.29.1+ provides this venv on the DLAMI
VENV_PATH="/opt/aws_neuronx_venv_pytorch_2_9"
if [ -d "$VENV_PATH" ]; then
    source "${VENV_PATH}/bin/activate"
else
    # Fallback: try the nxd_inference venv
    VENV_PATH="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference"
    if [ -d "$VENV_PATH" ]; then
        source "${VENV_PATH}/bin/activate"
    else
        echo "ERROR: No suitable Neuron PyTorch venv found!"
        echo "Expected: /opt/aws_neuronx_venv_pytorch_2_9 or /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference"
        exit 1
    fi
fi
echo "Using venv: $VENV_PATH"

# --- Install additional dependencies ---
echo "[4/5] Installing Python dependencies..."
pip install --upgrade pip
pip install diffusers>=0.38.0 \
            transformers \
            accelerate \
            safetensors \
            huggingface_hub \
            pillow \
            imageio \
            imageio-ffmpeg \
            tqdm

# --- Verify Neuron SDK ---
echo "[5/5] Verifying Neuron SDK installation..."
echo "--- Neuron packages ---"
pip list | grep -i neuron || true
echo ""
echo "--- neuronx-cc version ---"
neuronx-cc --version || echo "neuronx-cc not found in PATH"
echo ""
echo "--- NeuronCore count ---"
neuron-ls 2>/dev/null | head -20 || echo "neuron-ls not available"
echo ""

# --- Verify torch device support ---
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'Available devices: {torch.device(\"neuron\") if hasattr(torch, \"neuron\") else \"checking...\"}')
try:
    import torch_neuronx
    print(f'torch_neuronx version: {torch_neuronx.__version__}')
except ImportError:
    print('torch_neuronx not found')
try:
    import torch_xla.core.xla_model as xm
    print(f'XLA devices: {xm.get_xla_supported_devices()}')
except ImportError:
    print('torch_xla not available (expected for pure native path)')
"

echo ""
echo "=== Setup Complete ==="
echo "Model weights dir: /mnt/nvme/models"
echo "Compiled artifacts: /mnt/nvme/compiled_artifacts"
echo "Output dir:         /mnt/nvme/outputs"
echo ""
echo "Next steps:"
echo "  python download_model.py"
echo "  python run_inference.py --prompt 'A cat walks on the grass'"
