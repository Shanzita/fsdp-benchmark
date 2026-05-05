#!/bin/bash
#SBATCH --job-name=fsdp-rest
#SBATCH --output=results/slurm_%j.out
#SBATCH --error=results/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:lovelace:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00

module purge
module load GCCcore/13.2.0
module load Python/3.11.5
module load CUDA/12.6.0
source /scratch/ss516/fsdp_env/bin/activate

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=4

NUM_GPUS=$(nvidia-smi -L | wc -l)
STEPS=50
BS=32

cd $SLURM_SUBMIT_DIR
mkdir -p results

echo "============================================================"
echo " FSDP Benchmark - Remaining Steps - $(date)"
echo " Node: $(hostname) - GPUs: $NUM_GPUS"
echo "============================================================"

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
echo ">>> Generating charts after scaling..."
python scripts/visualize_results.py
python scripts/analyze_scaling.py

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
echo ">>> Generating charts after batch scaling..."
python scripts/analyze_scaling.py

# ─── Step 13: OOM test ───────────────────────────────────────
echo ">>> Step 13: OOM test"
python scripts/oom_test.py --mode single --model vit_l_16 --batch_size 64 || true
torchrun --nproc_per_node=$NUM_GPUS scripts/oom_test.py --mode fsdp --model vit_l_16 --batch_size 64

# ─── Step 14: Profiling ──────────────────────────────────────
echo ">>> Step 14: Profiling"
torchrun --nproc_per_node=$NUM_GPUS scripts/profile_fsdp.py --model resnet50 --batch_size $BS

# ─── Step 15: Final Charts & Analysis ────────────────────────
echo ">>> Step 15: Final Charts & Analysis"
python scripts/visualize_results.py
python scripts/analyze_scaling.py

echo "============================================================"
echo " ALL REMAINING STEPS DONE - $(date)"
echo "============================================================"
ls -la results/
