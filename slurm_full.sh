#!/bin/bash
#SBATCH --job-name=fsdp-full
#SBATCH --output=results/slurm_%j.out
#SBATCH --error=results/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:lovelace:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00

module purge
module load GCCcore/13.2.0
module load Python/3.11.5
module load CUDA/12.6.0
source /scratch/ss516/fsdp_env/bin/activate

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=4

NUM_GPUS=$(nvidia-smi -L | wc -l)
STEPS=50
BS=32

cd $SLURM_SUBMIT_DIR
mkdir -p results

echo "============================================================"
echo " FSDP Full Benchmark Suite - Job $SLURM_JOB_ID"
echo " Node: $(hostname) - GPUs: $NUM_GPUS - $(date)"
echo " Time limit: 4 hours"
echo "============================================================"

# ─── Step 0: Environment ───────────────────────────────────────
echo ">>> Step 0: Environment"
python scripts/env_info.py | tee results/environment.txt
nvidia-smi | tee results/nvidia_smi.txt

# ─── Step 1: Single-GPU baselines ──────────────────────────────
echo ">>> Step 1: Single-GPU baselines"
python scripts/benchmark_single_gpu.py --model resnet50 --batch_size $BS --steps $STEPS
python scripts/benchmark_single_gpu.py --model vit_b_16 --batch_size $BS --steps $STEPS
echo ">>> Live plot update..."
python scripts/liveplot.py

# ─── Step 2: DDP ───────────────────────────────────────────────
echo ">>> Step 2: DDP"
torchrun --nproc_per_node=$NUM_GPUS scripts/benchmark_ddp.py --model resnet50 --batch_size $BS --steps $STEPS
torchrun --nproc_per_node=$NUM_GPUS scripts/benchmark_ddp.py --model vit_b_16 --batch_size $BS --steps $STEPS
echo ">>> Live plot update..."
python scripts/liveplot.py

# ─── Step 3: FSDP FULL_SHARD ──────────────────────────────────
echo ">>> Step 3: FSDP FULL_SHARD"
torchrun --nproc_per_node=$NUM_GPUS scripts/benchmark_fsdp.py --model resnet50 --batch_size $BS --steps $STEPS --sharding_strategy FULL_SHARD
torchrun --nproc_per_node=$NUM_GPUS scripts/benchmark_fsdp.py --model vit_b_16 --batch_size $BS --steps $STEPS --sharding_strategy FULL_SHARD

# ─── Step 4: FSDP SHARD_GRAD_OP ───────────────────────────────
echo ">>> Step 4: FSDP SHARD_GRAD_OP"
torchrun --nproc_per_node=$NUM_GPUS scripts/benchmark_fsdp.py --model resnet50 --batch_size $BS --steps $STEPS --sharding_strategy SHARD_GRAD_OP
torchrun --nproc_per_node=$NUM_GPUS scripts/benchmark_fsdp.py --model vit_b_16 --batch_size $BS --steps $STEPS --sharding_strategy SHARD_GRAD_OP

# ─── Step 5: FSDP NO_SHARD ───────────────────────────────────
echo ">>> Step 5: FSDP NO_SHARD"
torchrun --nproc_per_node=$NUM_GPUS scripts/benchmark_fsdp.py --model resnet50 --batch_size $BS --steps $STEPS --sharding_strategy NO_SHARD
torchrun --nproc_per_node=$NUM_GPUS scripts/benchmark_fsdp.py --model vit_b_16 --batch_size $BS --steps $STEPS --sharding_strategy NO_SHARD
echo ">>> Live plot update..."
python scripts/liveplot.py

# ─── Step 6: FSDP + Mixed Precision ──────────────────────────
echo ">>> Step 6: FSDP + Mixed Precision"
torchrun --nproc_per_node=$NUM_GPUS scripts/benchmark_fsdp.py --model resnet50 --batch_size $BS --steps $STEPS --sharding_strategy FULL_SHARD --mixed_precision
torchrun --nproc_per_node=$NUM_GPUS scripts/benchmark_fsdp.py --model vit_b_16 --batch_size $BS --steps $STEPS --sharding_strategy FULL_SHARD --mixed_precision

# ─── Step 7: FSDP + Activation Checkpointing ─────────────────
echo ">>> Step 7: FSDP + Activation Checkpointing"
torchrun --nproc_per_node=$NUM_GPUS scripts/benchmark_fsdp.py --model resnet50 --batch_size $BS --steps $STEPS --sharding_strategy FULL_SHARD --activation_checkpointing
torchrun --nproc_per_node=$NUM_GPUS scripts/benchmark_fsdp.py --model vit_b_16 --batch_size $BS --steps $STEPS --sharding_strategy FULL_SHARD --activation_checkpointing

# ─── Step 8: FSDP + MP + AC ─────────────────────────────────
echo ">>> Step 8: FSDP + MP + AC (maximum memory savings)"
torchrun --nproc_per_node=$NUM_GPUS scripts/benchmark_fsdp.py --model resnet50 --batch_size $BS --steps $STEPS --sharding_strategy FULL_SHARD --mixed_precision --activation_checkpointing
torchrun --nproc_per_node=$NUM_GPUS scripts/benchmark_fsdp.py --model vit_b_16 --batch_size $BS --steps $STEPS --sharding_strategy FULL_SHARD --mixed_precision --activation_checkpointing

# ─── Step 9: FSDP + CPU Offloading ───────────────────────────
echo ">>> Step 9: FSDP + CPU Offloading"
torchrun --nproc_per_node=$NUM_GPUS scripts/benchmark_fsdp.py --model resnet50 --batch_size $BS --steps $STEPS --sharding_strategy FULL_SHARD --cpu_offload
torchrun --nproc_per_node=$NUM_GPUS scripts/benchmark_fsdp.py --model vit_b_16 --batch_size $BS --steps $STEPS --sharding_strategy FULL_SHARD --cpu_offload

# ─── Step 10: Gradient Accumulation ──────────────────────────
echo ">>> Step 10: Gradient Accumulation (4x effective batch)"
torchrun --nproc_per_node=$NUM_GPUS scripts/benchmark_fsdp.py --model resnet50 --batch_size $BS --steps $STEPS --sharding_strategy FULL_SHARD --grad_accum_steps 4
torchrun --nproc_per_node=$NUM_GPUS scripts/benchmark_fsdp.py --model vit_b_16 --batch_size $BS --steps $STEPS --sharding_strategy FULL_SHARD --grad_accum_steps 4

echo ">>> Live plot update after all core benchmarks..."
python scripts/liveplot.py

# ─── Step 11: GPU Scaling Study ──────────────────────────────
echo ">>> Step 11: GPU Scaling Study"
for N in 1 2; do
    for MODE in ddp fsdp; do
        for MODEL in resnet50 vit_b_16; do
            echo "   Scaling: $MODE $MODEL ${N}GPU"
            torchrun --nproc_per_node=$N scripts/benchmark_gpu_scaling.py --model $MODEL --batch_size $BS --steps $STEPS --mode $MODE
        done
    done
done
echo ">>> Live plot update after scaling..."
python scripts/liveplot.py

# ─── Step 12: Batch Size Scaling Study ───────────────────────
echo ">>> Step 12: Batch Size Scaling Study"
for MODE in single ddp fsdp fsdp_mp; do
    for MODEL in resnet50 vit_b_16; do
        echo "   Batch sweep: $MODE $MODEL"
        if [ "$MODE" = "single" ]; then
            python scripts/benchmark_batch_scaling.py --model $MODEL --mode $MODE --batch_sizes 8 16 32 64 128 256
        else
            torchrun --nproc_per_node=$NUM_GPUS scripts/benchmark_batch_scaling.py --model $MODEL --mode $MODE --batch_sizes 8 16 32 64 128 256
        fi
    done
done
echo ">>> Live plot update after batch scaling..."
python scripts/liveplot.py

# ─── Step 13: OOM test ───────────────────────────────────────
echo ">>> Step 13: OOM test"
python scripts/oom_test.py --mode single --model vit_l_16 --batch_size 64 || true
torchrun --nproc_per_node=$NUM_GPUS scripts/oom_test.py --mode fsdp --model vit_l_16 --batch_size 64
echo ">>> Live plot update after OOM..."
python scripts/liveplot.py

# ─── Step 14: Profiling ──────────────────────────────────────
echo ">>> Step 14: Profiling"
torchrun --nproc_per_node=$NUM_GPUS scripts/profile_fsdp.py --model resnet50 --batch_size $BS

# ─── Step 15: Final Charts & Analysis ────────────────────────
echo ">>> Step 15: Final Charts & Analysis"
python scripts/liveplot.py
python scripts/analyze_scaling.py

echo "============================================================"
echo " ALL DONE - Job $SLURM_JOB_ID - $(date)"
echo "============================================================"
echo ""
echo "Report figures generated:"
ls -la results/fig_*.png
echo ""
echo "All result files:"
ls -la results/
