# FSDP Benchmark: Memory-Efficient Distributed Training for Vision Models

A comprehensive benchmarking suite for PyTorch's Fully Sharded Data Parallel (FSDP), comparing 10 distributed training configurations across vision models on NVIDIA GPUs.

## Key Findings

| Finding | Detail |
|---------|--------|
| FSDP+MP Pareto-dominates DDP | 54% less memory **and** 2.3x higher throughput for ViT-B/16 |
| 82% memory reduction | FSDP+MP+AC reduces ViT-B/16 from 5.16 GB to 0.92 GB per GPU |
| Scaling efficiency is model-size dependent | 93.6% for ViT-B/16 (86.6M) vs 81.8% for ResNet-50 (25.6M) |
| 3.2x peak throughput advantage | FSDP+MP peaks at 1472 s/s vs DDP's 460 s/s for ViT-B/16 |

## Configurations Benchmarked

| Configuration | Description | Optimizes |
|---------------|-------------|-----------|
| Single GPU | Baseline, no distribution | Baseline |
| DDP (2 GPU) | Standard gradient all-reduce | Throughput |
| FSDP Full Shard | Shard params/grads/optimizer states | Memory |
| FSDP Shard Grad Op | Shard grads + optimizer only | Both |
| FSDP No Shard | No sharding (DDP-equivalent via FSDP) | Baseline |
| FSDP + Mixed Precision | Full shard + FP16 compute | Both |
| FSDP + Activation Checkpointing | Full shard + recompute activations | Memory |
| FSDP + MP + AC | Full shard + MP + AC combined | Memory |
| FSDP + CPU Offload | Full shard + offload params to CPU | Memory |
| FSDP + Gradient Accumulation | Full shard + 4x grad accumulation | Batch size |

## Results (NVIDIA L40S, 2 GPUs)

### ResNet-50 (25.6M parameters)

| Config | Step (ms) | Throughput (s/s) | Memory (GB) |
|--------|-----------|-----------------|-------------|
| Single GPU | 43.58 | 734 | 2.92 |
| DDP (2GPU) | 47.86 | 1337 | 3.02 |
| FSDP Full | 54.81 | 1168 | 2.78 |
| FSDP GradOp | 47.29 | **1353** | 2.86 |
| FSDP + MP | 90.74 | 705 | 2.20 |
| FSDP + AC | 78.08 | 820 | 1.25 |
| FSDP + MP + AC | 136.70 | 468 | **1.00** |

### ViT-B/16 (86.6M parameters)

| Config | Step (ms) | Throughput (s/s) | Memory (GB) |
|--------|-----------|-----------------|-------------|
| Single GPU | 139.86 | 229 | 4.84 |
| DDP (2GPU) | 145.48 | 440 | 5.16 |
| FSDP Full | 150.27 | 426 | 4.39 |
| FSDP GradOp | 143.25 | 447 | 4.68 |
| FSDP + MP | 62.00 | **1032** | 2.38 |
| FSDP + AC | 194.24 | 330 | 1.15 |
| FSDP + MP + AC | 90.53 | 707 | **0.92** |

## Repository Structure

```
fsdp-benchmark/
├── scripts/
│   ├── utils.py                     # Shared model/param utilities
│   ├── benchmark_single_gpu.py      # Single-GPU baseline
│   ├── benchmark_ddp.py             # DDP benchmark
│   ├── benchmark_fsdp.py            # FSDP with all optimization flags
│   ├── benchmark_gpu_scaling.py     # GPU scaling study (1, 2 GPUs)
│   ├── benchmark_batch_scaling.py   # Batch size sweep
│   ├── oom_test.py                  # Out-of-memory boundary test
│   ├── profile_fsdp.py              # CUDA kernel profiling + Chrome traces
│   ├── analyze_scaling.py           # Scaling efficiency analysis
│   ├── liveplot.py                  # Report-ready figure generator
│   └── env_info.py                  # Hardware/software info capture
├── results/                         # JSON results + PNG figures
├── report/                          # LaTeX report (ACM sigconf format)
├── slurm_full.sh                    # Full benchmark suite (4hr, Lovelace)
└── requirements.txt
```

> **Note:** `slurm_full.sh` has a hardcoded virtualenv path (`/scratch/ss516/fsdp_env/`). Update this to your own environment path before running.

## Quick Start

### Prerequisites

```bash
pip install torch>=2.2.0 torchvision>=0.17.0 matplotlib pandas tabulate
```

### Run on SLURM Cluster

```bash
# Full suite (recommended: 4 hours, 2 GPUs)
sbatch slurm_full.sh

# Or submit to specific partition
sbatch --partition=scavenge --time=01:00:00 slurm_full.sh
```

### Run Individual Benchmarks

```bash
# Single GPU baseline
python scripts/benchmark_single_gpu.py --model vit_b_16 --batch_size 32 --steps 50

# DDP (2 GPUs)
torchrun --nproc_per_node=2 scripts/benchmark_ddp.py --model vit_b_16 --batch_size 32 --steps 50

# FSDP with mixed precision + activation checkpointing
torchrun --nproc_per_node=2 scripts/benchmark_fsdp.py \
    --model vit_b_16 --batch_size 32 --steps 50 \
    --sharding_strategy FULL_SHARD \
    --mixed_precision --activation_checkpointing

# FSDP with CPU offloading
torchrun --nproc_per_node=2 scripts/benchmark_fsdp.py \
    --model vit_b_16 --batch_size 32 --steps 50 \
    --sharding_strategy FULL_SHARD --cpu_offload

# GPU scaling study
torchrun --nproc_per_node=2 scripts/benchmark_gpu_scaling.py \
    --model vit_b_16 --batch_size 32 --steps 50 --mode fsdp

# Batch size sweep
torchrun --nproc_per_node=2 scripts/benchmark_batch_scaling.py \
    --model vit_b_16 --mode fsdp_mp --batch_sizes 8 16 32 64 128 256
```

### Generate Figures

```bash
# All report-ready figures (no GPU needed)
python scripts/liveplot.py

# Scaling analysis
python scripts/analyze_scaling.py
```

## Generated Figures

| Figure | Description |
|--------|-------------|
| `fig_comparison_{model}.png` | Bar charts: memory, throughput, latency across all configs |
| `fig_memory_waterfall_{model}.png` | Memory reduction ranked with % savings |
| `fig_tradeoff.png` | Memory vs throughput scatter (Pareto frontier) |
| `fig_scaling_{model}.png` | GPU scaling: throughput, efficiency, memory per GPU |
| `fig_batch_scaling_{model}.png` | Throughput and memory vs batch size |
| `fig_oom_test.png` | OOM test: single GPU vs FSDP for ViT-L/16 |

## Hardware Tested

| GPU | VRAM | Notes |
|-----|------|-------|
| NVIDIA L40S | 44.3 GB | Primary results (Ada Lovelace) |
| NVIDIA V100 | 32 GB | Also supported via `slurm_v100.sh` |

## Citation

If you find this benchmark useful, please cite:

```bibtex
@misc{siddiqua2026fsdp,
  title={Benchmarking PyTorch FSDP: Memory-Efficient Distributed Training for Vision Models},
  author={Siddiqua, Shanzita},
  year={2026},
  howpublished={\url{https://github.com/Shanzita/fsdp-benchmark}}
}
```

## License

MIT
