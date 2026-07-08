#!/bin/bash
# =============================================================================
# WAN 2.2 T2V-A14B — Environment Setup for Native PyTorch Beta 3 on Trn2
#
# Supports two installation paths:
#   Option A: Run inside the Beta 3 DLC container (recommended)
#   Option B: Extract DLC artifacts and install into a local venv
#
# DLC: 421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest
#
# Run on: trn2.48xlarge with Ubuntu 22.04 or 24.04
# =============================================================================

set -euo pipefail

echo "========================================================================"
echo "  WAN 2.2 — Native PyTorch Beta 3 Environment Setup"
echo "========================================================================"

# --- NVMe Setup --------------------------------------------------------------
echo ""
echo "[1/6] Setting up NVMe storage..."
if [ -b /dev/nvme1n1 ] && ! mountpoint -q /mnt/nvme; then
    sudo mkfs.ext4 -F /dev/nvme1n1
    sudo mkdir -p /mnt/nvme
    sudo mount /dev/nvme1n1 /mnt/nvme
    sudo chown "$USER":"$USER" /mnt/nvme
    echo "  NVMe mounted at /mnt/nvme"
else
    echo "  /mnt/nvme already mounted or no NVMe device — skipping format"
    sudo mkdir -p /mnt/nvme
    sudo chown "$USER":"$USER" /mnt/nvme 2>/dev/null || true
fi

mkdir -p /mnt/nvme/models
mkdir -p /mnt/nvme/outputs
mkdir -p /mnt/nvme/neff_cache   # Persistent NEFF cache (Beta 3)

# --- Docker ECR login --------------------------------------------------------
echo ""
echo "[2/6] Logging in to Beta 3 ECR..."
ECR_ACCOUNT="421672808698"
ECR_REGION="us-east-1"
ECR_REGISTRY="${ECR_ACCOUNT}.dkr.ecr.${ECR_REGION}.amazonaws.com"
DLC_IMAGE="${ECR_REGISTRY}/concourse-release-0461d3b:latest"

aws ecr get-login-password --region "${ECR_REGION}" | \
    docker login --username AWS --password-stdin "${ECR_REGISTRY}"
echo "  ECR login OK"
echo "  DLC: ${DLC_IMAGE}"

# --- Choose setup path -------------------------------------------------------
echo ""
echo "[3/6] Select installation method:"
echo "  A) Run inside DLC container  (recommended — fully self-contained)"
echo "  B) Extract DLC + local venv  (for bare-metal / custom images)"
echo ""
read -rp "Choice [A/b]: " CHOICE
CHOICE="${CHOICE:-A}"

if [[ "${CHOICE,,}" == "a" ]]; then
    # -------------------------------------------------------------------------
    # Option A: DLC container
    # -------------------------------------------------------------------------
    echo ""
    echo "[4/6] Pulling Beta 3 DLC..."
    docker pull "${DLC_IMAGE}"
    IMAGE_ID=$(docker images -q --filter "reference=${DLC_IMAGE}")

    echo ""
    echo "[5/6] DLC pulled. Run your workload with:"
    echo ""
    echo "  docker run -it --privileged \\"
    echo "    -v /mnt/nvme:/mnt/nvme \\"
    echo "    ${DLC_IMAGE} /bin/bash"
    echo ""
    echo "  # Inside the container:"
    echo "  pip install diffusers>=0.38.0 transformers>=4.44.0 accelerate \\"
    echo "              safetensors imageio imageio-ffmpeg pillow"
    echo "  python /mnt/nvme/run_inference.py \\"
    echo "    --prompt 'A cat walks on grass' \\"
    echo "    --num-inference-steps 40"

else
    # -------------------------------------------------------------------------
    # Option B: Extract DLC artifacts + local venv
    # -------------------------------------------------------------------------
    echo ""
    echo "[4/6] Extracting DLC artifacts..."
    docker pull "${DLC_IMAGE}"
    IMAGE_ID=$(docker images -q --filter "reference=${DLC_IMAGE}")
    cd "$HOME"
    docker create --name tmp_beta3 "${IMAGE_ID}"
    docker cp tmp_beta3:/workspace .
    docker rm tmp_beta3
    echo "  Artifacts extracted to $HOME/workspace"

    echo ""
    echo "[5/6] Installing Neuron runtime .deb packages..."
    sudo apt-get update -q
    sudo apt-get install -y dkms build-essential
    sudo dpkg -i "$HOME/workspace/runtime_artifacts/"*.deb

    echo ""
    echo "[6/6] Creating Beta 3 virtual environment..."
    sudo apt-get install -y python3.12-venv
    VENV="$HOME/workspace/native_venv"
    python3.12 -m venv "${VENV}"
    source "${VENV}/bin/activate"

    pip install uv
    export UV_PROJECT_ENVIRONMENT="${VENV}"

    uv pip install "$HOME/workspace/nki_wheels/nki-0.4.0"*-cp312-cp312-linux_x86_64.whl
    uv pip install "$HOME/workspace/neuronx_cc_wheels/neuronx_cc-2."*-cp312-cp312-linux_x86_64.whl
    cd "$HOME/workspace/torch_neuron_eager"
    uv pip install -e ".[dev]"

    # Install diffusion deps
    uv pip install diffusers>=0.38.0 transformers>=4.44.0 accelerate \
                   safetensors imageio imageio-ffmpeg pillow tqdm einops

    echo "  Venv ready: ${VENV}"
    echo "  Activate:  source ${VENV}/bin/activate"
fi

# --- Verify NeuronCores -------------------------------------------------------
echo ""
echo "========================================================================"
echo "  Verifying NeuronCores"
echo "========================================================================"
neuron-ls 2>/dev/null || echo "  neuron-ls not in PATH — run from inside DLC container"

# --- Print Beta 3 env vars ---------------------------------------------------
echo ""
echo "========================================================================"
echo "  Beta 3 Environment Variables (set these before running inference)"
echo "========================================================================"
cat <<'EOF'
# LNC2 mode (2 physical cores per logical core — required for trn2.48xlarge)
export NEURON_RT_VIRTUAL_CORE_SIZE=2
export NEURON_RT_NUM_CORES=64
export NEURON_RT_VISIBLE_CORES=0-63
export NEURON_ENABLE_NATIVE_KERNEL=1

# Compiler flags
export NEURON_CC_FLAGS="-O1 --auto-cast=none --enable-native-kernel=1 --remat --enable-ccop-compute-overlap"

# Async NRT (default in Beta 3, explicit here)
export TORCH_NEURONX_ENABLE_ASYNC_NRT=1

# Host collective comms — enables compute/communication overlap
export TORCH_NEURONX_ENABLE_HOST_CC=1

# Persistent NEFF cache — eliminates recompilation on container restart
export NEURON_COMPILE_CACHE_URL="file:///mnt/nvme/neff_cache"
export NEURONX_CACHE="/mnt/nvme/neff_cache"
EOF

echo ""
echo "========================================================================"
echo "  Setup complete. Next steps:"
echo "========================================================================"
echo "  1. python download_model.py"
echo "  2. # Eager mode (fast validation, no compilation):"
echo "     python run_inference.py --prompt 'A cat walks on grass' --eager \\"
echo "       --num-inference-steps 5 --height 384 --width 640 --num-frames 1"
echo "  3. # Full compile mode (production, ~16 min cold / ~3 min warm cache):"
echo "     python run_inference.py --prompt 'A cat walks on grass'"
echo "========================================================================"
