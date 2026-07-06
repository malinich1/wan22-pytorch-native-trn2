#!/bin/bash
# =============================================================================
# Create GitHub Repository: wan22-pytorch-native-trn2
# Run this from the directory containing the project files
# =============================================================================

# 1. Initialize git repo
git init

# 2. Add all files
git add .

# 3. Initial commit
git commit -m "feat: WAN 2.2 T2V-A14B inference with PyTorch Native on Trainium 2

- PyTorch Native approach using device='neuron' and torch.compile()
- Replaces NXD (neuronx-distributed) with standard torch.distributed
- Expert weight swap via tensor.copy_() (single-process, no subprocess)
- Batched CFG (batch=2) for optimized denoising
- Performance targets: match/beat NXD baseline on trn2.48xlarge
- Supports eager mode fallback for debugging
- Full benchmarking suite with NXD comparison"

# 4. Create the repo on GitHub (requires 'gh' CLI to be authenticated)
gh repo create malinich1/wan22-pytorch-native-trn2 \
    --public \
    --description "WAN 2.2 T2V-A14B Video Generation — PyTorch Native (device='neuron') on Trainium 2. Comparison study vs NXD approach." \
    --source . \
    --push

echo ""
echo "✅ Repository created: https://github.com/malinich1/wan22-pytorch-native-trn2"
echo ""
echo "If 'gh' is not installed, use these manual steps instead:"
echo "  1. Go to https://github.com/new"
echo "  2. Create repo: malinich1/wan22-pytorch-native-trn2"
echo "  3. Then run:"
echo "     git remote add origin git@github.com:malinich1/wan22-pytorch-native-trn2.git"
echo "     git branch -M main"
echo "     git push -u origin main"
