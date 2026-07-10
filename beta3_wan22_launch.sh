#!/bin/bash
# =============================================================================
# WAN 2.2 T2V-A14B — Beta 3 Distributed Inference Launcher
#
# Runs inside the Beta 3 DLC Docker container on trn2.48xlarge.
# Uses native torch.compile(backend="neuron") with tensor parallelism.
#
# Usage:
#   ./beta3_wan22_launch.sh <TP_DEGREE> "<PROMPT>"
#
# Examples:
#   ./beta3_wan22_launch.sh 2 "A cat walks through a garden"
#   HEIGHT=768 WIDTH=1280 FRAMES=81 STEPS=40 ./beta3_wan22_launch.sh 4 "Ocean waves"
#   EAGER_MODE=1 FRAMES=1 STEPS=5 ./beta3_wan22_launch.sh 2 "A cat"  # Quick test
# =============================================================================

set -e

TP_DEGREE=${1:-2}
PROMPT=${2:-"A close-up photograph of a beautiful fluffy orange tabby cat sitting in a sunlit garden, photorealistic, sharp focus, 8k"}

# Configuration (override via environment variables)
HEIGHT=${HEIGHT:-384}
WIDTH=${WIDTH:-640}
FRAMES=${FRAMES:-17}
STEPS=${STEPS:-20}
GUIDANCE=${GUIDANCE:-4.0}
GUIDANCE_2=${GUIDANCE_2:-3.0}
SEED=${SEED:-42}
OUTPUT=${OUTPUT:-/mnt/nvme/outputs/beta3_wan22_tp${TP_DEGREE}.mp4}
NEFF_CACHE=${NEFF_CACHE:-/mnt/nvme/neff_cache_beta3_wan22}
EAGER_MODE=${EAGER_MODE:-0}

# Container config
CONTAINER_NAME="beta3_wan22_run"
DLC_IMAGE="421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest"
SCRIPT_PATH="/mnt/nvme/beta3_wan22_distributed_inference.py"

echo "=============================================="
echo "WAN 2.2 T2V-A14B — Beta 3 Distributed Inference"
echo "=============================================="
echo "  TP Degree:    ${TP_DEGREE}"
echo "  Resolution:   ${WIDTH}x${HEIGHT}, ${FRAMES} frames"
echo "  Steps:        ${STEPS}"
echo "  Guidance:     ${GUIDANCE}/${GUIDANCE_2}"
echo "  Eager Mode:   ${EAGER_MODE}"
echo "  Output:       ${OUTPUT}"
echo "  NEFF Cache:   ${NEFF_CACHE}"
echo "  Prompt:       ${PROMPT}"
echo "=============================================="

# Validate
if [ ${TP_DEGREE} -lt 1 ] || [ ${TP_DEGREE} -gt 8 ]; then
    echo "ERROR: TP_DEGREE must be 1-8 (got ${TP_DEGREE})"
    exit 1
fi

# Build device list for Docker
DEVICES=""
for i in $(seq 0 $((TP_DEGREE - 1))); do
    DEVICES="${DEVICES} --device /dev/neuron${i}"
done

# Stop existing container if any
docker rm -f ${CONTAINER_NAME} 2>/dev/null || true

# Build eager flag
EAGER_FLAG=""
if [ "${EAGER_MODE}" = "1" ]; then
    EAGER_FLAG="--eager"
fi

# Ensure output directory exists
mkdir -p $(dirname ${OUTPUT})
mkdir -p ${NEFF_CACHE}

# Copy script to shared volume
cp /mnt/nvme/beta3_wan22_distributed_inference.py ${SCRIPT_PATH} 2>/dev/null || true

echo ""
echo "Launching Docker container with ${TP_DEGREE} NeuronCores..."
echo ""

# Run inference inside Beta 3 DLC container
docker run --rm --name ${CONTAINER_NAME} \
    --privileged \
    ${DEVICES} \
    -v /mnt/nvme:/mnt/nvme \
    -e NEURON_RT_VIRTUAL_CORE_SIZE=2 \
    -e NEURON_RT_NUM_CORES=${TP_DEGREE} \
    ${DLC_IMAGE} \
    bash -c "
        pip install -q diffusers transformers accelerate imageio imageio-ffmpeg 2>/dev/null
        cd /mnt/nvme
        torchrun --nproc-per-node ${TP_DEGREE} \
            ${SCRIPT_PATH} \
            --prompt '${PROMPT}' \
            --negative-prompt 'blurry, low quality, deformed, ugly, cartoon, anime, painting, text, watermark' \
            --height ${HEIGHT} \
            --width ${WIDTH} \
            --num-frames ${FRAMES} \
            --num-steps ${STEPS} \
            --guidance ${GUIDANCE} \
            --guidance-2 ${GUIDANCE_2} \
            --seed ${SEED} \
            --output ${OUTPUT} \
            --neff-cache ${NEFF_CACHE} \
            ${EAGER_FLAG}
    "

echo ""
echo "=============================================="
echo "Inference complete!"
echo "Output: ${OUTPUT}"
ls -lh ${OUTPUT} 2>/dev/null
echo "=============================================="
