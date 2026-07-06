# WAN 2.2 T2V-A14B Video Generation — PyTorch Native on Trainium 2

Generate 768×1280 (81 frames, ~5s at 16fps) video from text prompts using the **Wan 2.2 T2V-A14B** Mixture-of-Experts diffusion model on a `trn2.48xlarge` instance with **PyTorch Native** (`device='neuron'`).

> **Key difference from NXD approach**: This project uses PyTorch's native device abstraction (`torch.compile()` with Neuron backend + `device='neuron'`) instead of `neuronx-distributed` (NxD). This aligns with the recommended "TorchNeuron Native" path for Trn2/Trn3 (SDK 2.29+).

## Model

| Property | Value |
| --- | --- |
| Name | [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) |
| Architecture | 27B parameter MoE with 14B active per denoising step |
| Experts | 2 independent experts (high-noise / low-noise), zero shared weights |
| Weights | ~118 GB (Hugging Face) |

## Approach: PyTorch Native vs NXD

| Aspect | NXD Approach (existing) | PyTorch Native (this project) |
| --- | --- | --- |
| Device | XLA via `torch_neuronx` | `device='neuron'` native |
| Compilation | `torch_neuronx.trace()` → static NEFF | `torch.compile(backend='neuronx')` |
| Parallelism | `neuronx_distributed` (TP/CP explicit) | `torch.distributed` + DTensor |
| Expert Swap | `tensor.copy_()` on NxDModel weights | `tensor.copy_()` on native model params |
| Model Loading | `NxDModel` + `initialize()` | Standard `model.to('neuron')` |
| Eager Support | No (trace-only) | Yes — eager fallback for debugging |

## Performance Targets

Baseline to match/beat (from NXD implementation on trn2.48xlarge):

| Metric | NXD Optimized | Target (PyTorch Native) |
| --- | --- | --- |
| Per forward pass | 2,520 ms | ≤ 2,600 ms |
| Total denoising (40 steps) | ~202s | ≤ 210s |
| Expert swap | 64.1s | ≤ 65s |
| End-to-end | ~618s | ≤ 650s |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    trn2.48xlarge                              │
│  16 NeuronDevices × 4 NeuronCores = 64 NeuronCores (LNC=2) │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Text Encoder │    │  DiT Expert  │    │ VAE Decoder  │  │
│  │  (TP=4)      │    │ (TP=4,CP=16) │    │ (Tiled, 8NC) │  │
│  │  4 cores     │    │  64 cores    │    │  8 cores     │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                              │
│  PyTorch Native Path:                                        │
│  • model.to('neuron')                                        │
│  • torch.compile(backend='neuronx')                          │
│  • torch.distributed for TP/CP                               │
│  • DTensor for sharding                                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘

```

## Files

| File | Description |
| --- | --- |
| `README.md` | This file |
| `setup_env.sh` | Environment setup (NVMe, deps, venv) |
| `run_inference.py` | Main E2E inference script (PyTorch Native) |
| `compile_model.py` | Model compilation with torch.compile |
| `expert_swap.py` | Expert weight swapping via copy_() |
| `benchmarks.py` | Performance measurement & comparison |
| `configs/` | Configuration files (model, parallelism) |
| `utils/` | Shared utilities (logging, profiling) |

## Quick Start

```bash
# 1. Launch trn2.48xlarge with Neuron DLAMI (Ubuntu 24.04, SDK 2.29.1+)

# 2. Setup environment
./setup_env.sh

# 3. Activate venv
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate

# 4. Download model weights (~118 GB)
python download_model.py

# 5. Run inference
python run_inference.py --prompt "A cat walks on the grass, realistic style"

```

## Requirements

- **Instance**: `trn2.48xlarge` (16 NeuronDevices required)
- **LNC**: 2 (default, gives 64 logical cores with 24 GB HBM each)
- **AMI**: Deep Learning AMI Neuron (Ubuntu 24.04) 20260502+ (SDK 2.29.1+)
- **Python**: 3.10+
- **Key packages**:- `torch` >= 2.9 (with Neuron backend)
- `torch-neuronx` >= 2.9
- `neuronx-cc` >= 2.24
- `diffusers` >= 0.38.0 (WanPipeline MoE support)
- `transformers`, `accelerate`, `safetensors`

## References

- [NXD Implementation](https://github.com/malinich1/NeuronStuff/tree/main/Wan2.2-T2V-A14B) — Original NxD-based approach
- [PyTorch Native Workshop](https://catalog.us-east-1.prod.workshops.aws/workshops/f8ecb0ea-42ac-4480-924f-7b9149f9671e/en-US/3-hands-on-labs/31-basic-examples-with-pytorch-native) — Basic examples with PyTorch Native
- [TorchNeuron Native Intro](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuron-native/) — Official documentation
- [torch_neuronx.trace API](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/api-reference-guide/inference/api-torch-neuronx-trace.html) — Tracing API reference
- [AWS Neuron SDK](https://github.com/aws-neuron/aws-neuron-sdk) — SDK repository

