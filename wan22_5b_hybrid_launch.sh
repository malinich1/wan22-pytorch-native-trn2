#!/bin/bash
# =============================================================================
# WAN 2.2 TI2V-5B — Beta 3 Native torch.compile Launcher
#
# Runs WAN 2.2 TI2V-5B (5B dense, T2V+I2V) inside the Beta 3 DLC container
# on trn2 instances using native torch.compile(backend="neuron").
#
# The 5B model fits on a single NeuronCore pair (LNC=2 = 48 GB HBM).
# No tensor parallelism needed — simple single-process inference.
#
# Prerequisites:
#   - trn2.48xlarge (or any trn2 with at least 1 NeuronCore)
#   - Beta 3 DLC image pulled: 421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest
#   - /mnt/nvme mounted (NVMe storage)
#
# Usage:
#   ./wan22_5b_hybrid_launch.sh "A cat walking through a garden"
#   HEIGHT=704 WIDTH=1280 FRAMES=81 STEPS=50 ./wan22_5b_hybrid_launch.sh "Ocean waves"
#   EAGER=1 FRAMES=1 STEPS=5 ./wan22_5b_hybrid_launch.sh "Quick test"
#
# Environment (Beta 3 DLC):
#   PyTorch: 2.11.0
#   torch-neuronx: 2.11.3
#   neuronx-cc: 2.25
#   Python: 3.12
#   Backend: "neuron" registered for torch.compile
# =============================================================================

set -e

PROMPT=${1:-"A beautiful fluffy orange tabby cat walking gracefully through a sunlit garden, detailed fur, natural lighting, cinematic quality, photorealistic"}

# Configuration (override via environment variables)
HEIGHT=${HEIGHT:-480}
WIDTH=${WIDTH:-832}
FRAMES=${FRAMES:-33}
STEPS=${STEPS:-30}
GUIDANCE=${GUIDANCE:-5.0}
SEED=${SEED:-42}
OUTPUT=${OUTPUT:-/mnt/nvme/outputs/wan22_5b_t2v.mp4}
NEFF_CACHE=${NEFF_CACHE:-/mnt/nvme/neff_cache_5b}
EAGER=${EAGER:-0}

# Container config
CONTAINER_NAME="wan22_5b_compile"
DLC_IMAGE="421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest"
SCRIPT_NAME="wan22_5b_hybrid_compile.py"

echo "=============================================="
echo "WAN 2.2 TI2V-5B — Beta 3 torch.compile"
echo "=============================================="
echo "  Model:        Wan-AI/Wan2.2-TI2V-5B-Diffusers (5B dense)"
echo "  Resolution:   ${WIDTH}x${HEIGHT}, ${FRAMES} frames"
echo "  Steps:        ${STEPS}, Guidance: ${GUIDANCE}"
echo "  Eager Mode:   ${EAGER}"
echo "  Output:       ${OUTPUT}"
echo "  NEFF Cache:   ${NEFF_CACHE}"
echo "  Prompt:       ${PROMPT:0:80}..."
echo "=============================================="

# Build eager flag
EAGER_FLAG=""
if [ "${EAGER}" = "1" ]; then
    EAGER_FLAG="--eager"
fi

# Ensure directories exist
mkdir -p $(dirname ${OUTPUT})
mkdir -p ${NEFF_CACHE}

# Copy script to shared volume
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cp "${SCRIPT_DIR}/${SCRIPT_NAME}" /mnt/nvme/${SCRIPT_NAME}

# Stop existing container if any
docker rm -f ${CONTAINER_NAME} 2>/dev/null || true

echo ""
echo "Launching inference in Beta 3 DLC container..."
echo "  (First run compiles NEFFs ~3-5 min; subsequent runs use cache)"
echo ""

# Run inside Beta 3 DLC — single NeuronCore is sufficient for 5B
docker run --rm --name ${CONTAINER_NAME} \
    --privileged \
    --device /dev/neuron0 \
    -v /mnt/nvme:/mnt/nvme \
    -e NEURON_RT_VIRTUAL_CORE_SIZE=2 \
    -e NEURON_RT_NUM_CORES=1 \
    ${DLC_IMAGE} \
    bash -c "
        pip install -q diffusers transformers accelerate imageio imageio-ffmpeg sentencepiece 2>/dev/null
        torchrun --nproc-per-node 1 \
            /mnt/nvme/${SCRIPT_NAME} \
            --prompt '${PROMPT}' \
            --height ${HEIGHT} \
            --width ${WIDTH} \
            --num-frames ${FRAMES} \
            --num-steps ${STEPS} \
            --guidance ${GUIDANCE} \
            --seed ${SEED} \
            --output ${OUTPUT} \
            --neff-cache ${NEFF_CACHE} \
            ${EAGER_FLAG}
    "

echo ""
echo "=============================================="
if [ -f "${OUTPUT}" ]; then
    echo "Video generated successfully!"
    ls -lh ${OUTPUT}
else
    # Check for image output (single frame)
    IMG_OUTPUT="${OUTPUT%.*}.png"
    if [ -f "${IMG_OUTPUT}" ]; then
        echo "Image generated successfully!"
        ls -lh ${IMG_OUTPUT}
    else
        echo "ERROR: Output not found at ${OUTPUT}"
        exit 1
    fi
fi
echo "=============================================="
