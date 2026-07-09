#!/bin/bash
# =============================================================================
# WAN 2.2 T2V-A14B Setup & Compilation on trn2.48xlarge
# 
# Reference: https://github.com/whn09/aws-neuron-samples/tree/master/torch-neuronx/inference/hf_pretrained_wan2.2_t2v_a14b
#
# Architecture:
#   - 2 x WanTransformer3DModel (14B params each, MoE switching at timestep 875)
#   - TP=4, CP=2 for 480P (world_size=8, 8 NCs = 2 chips)
#   - UMT5-XXL text encoder
#   - AutoencoderKLWan VAE (z_dim=16)
#   - Resolution: 480x832, 81 frames
#
# Expected performance (480P):
#   - Text Encoding: ~22s
#   - Denoising: ~457s (40 steps, MoE switch at step 13)
#   - VAE Decode: ~44s
#   - Total: ~544s
#
# Persistent mode (all models co-resident, 32/64 NCs):
#   - Total: ~355s (35% faster, no model loading between phases)
# =============================================================================

set -e

echo "=============================================="
echo "WAN 2.2 T2V-A14B Setup for trn2.48xlarge"
echo "=============================================="

# Configuration
VENV=/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference
WORK_DIR=/home/ubuntu/aws-neuron-samples/torch-neuronx/inference/hf_pretrained_wan2.2_t2v_a14b
NVME=/opt/dlami/nvme
COMPILED_MODELS_DIR=${NVME}/compiled_models_t2v_a14b
COMPILER_WORKDIR=${NVME}/compiler_workdir_t2v_a14b
CACHE_DIR=${NVME}/wan2.2_t2v_a14b_hf_cache_dir

# Step 1: Activate the Neuron venv
echo "[Step 1] Activating Neuron virtual environment..."
source ${VENV}/bin/activate

# Step 2: Clone the aws-neuron-samples repo if not present
if [ ! -d "/home/ubuntu/aws-neuron-samples" ]; then
    echo "[Step 2] Cloning aws-neuron-samples repository..."
    cd /home/ubuntu
    git clone https://github.com/whn09/aws-neuron-samples.git
else
    echo "[Step 2] aws-neuron-samples already exists, pulling latest..."
    cd /home/ubuntu/aws-neuron-samples
    git pull || true
fi

# Step 3: Navigate to the WAN 2.2 inference directory
cd ${WORK_DIR}
export PYTHONPATH=$(pwd):$PYTHONPATH

# Step 4: Install dependencies
echo "[Step 3] Installing dependencies..."
pip install -r requirements.txt

# Step 5: Patch diffusers for Trainium2 compatibility (nearest-exact -> nearest)
echo "[Step 4] Patching diffusers for Trainium2 compatibility..."
DIFFUSERS_PATH=$(python -c "import diffusers; import os; print(os.path.dirname(diffusers.__file__))")
VAE_FILE="${DIFFUSERS_PATH}/models/autoencoders/autoencoder_kl_wan.py"
if grep -q 'nearest-exact' "${VAE_FILE}" 2>/dev/null; then
    echo "  Patching autoencoder_kl_wan.py: nearest-exact -> nearest"
    sed -i 's/nearest-exact/nearest/g' "${VAE_FILE}"
else
    echo "  Already patched or patch not needed"
fi

# Step 6: Create directories
mkdir -p "${COMPILED_MODELS_DIR}"
mkdir -p "${COMPILER_WORKDIR}"

echo ""
echo "=============================================="
echo "Setup complete!"
echo ""
echo "To compile all models (first time, ~1-2 hours):"
echo "  cd ${WORK_DIR}"
echo "  bash compile.sh"
echo ""
echo "To run 480P inference:"
echo "  python run_wan2.2_t2v_a14b.py \\"
echo "    --compiled_models_dir ${COMPILED_MODELS_DIR} \\"
echo "    --prompt 'A cat walks on the grass, realistic' \\"
echo "    --output output_t2v_480p.mp4"
echo ""
echo "To run persistent mode (faster, all models co-resident):"
echo "  python run_wan2.2_t2v_a14b_persistent.py \\"
echo "    --compiled_models_dir ${COMPILED_MODELS_DIR} \\"
echo "    --num_runs 3 \\"
echo "    --prompt 'A cat walks on the grass, realistic'"
echo "=============================================="
