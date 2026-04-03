#!/bin/bash
#SBATCH --job-name=fsdp-bench
#SBATCH --output=results/slurm_%j.out
#SBATCH --error=results/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:volta:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00

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

echo "============================================================"
echo " FSDP Benchmark - Rice NOTS - Job $SLURM_JOB_ID"
echo " Node: $(hostname) - GPUs: $NUM_GPUS - $(date)"
echo "============================================================"

cd $SLURM_SUBMIT_DIR
mkdir -p results

echo ">>> Step 0: Environment"
python scripts/env_info.py | tee results/environment.txt
nvidia-smi | tee results/nvidia_smi.txt

echo ">>> Step 1: Single-GPU baselines"
python scripts/benchmark_single_gpu.py --model resnet50 --batch_size $BS --steps $STEPS
python scripts/benchmark_single_gpu.py --model vit_b_16 --batch_size $BS --steps $STEPS

echo ">>> Step 2: DDP"
torchrun --nproc_per_node=$NUM_GPUS scripts/benchmark_ddp.py --model resnet50 --batch_size $BS --steps $STEPS
torchrun --nproc_per_node=$NUM_GPUS scripts/benchmark_ddp.py --model vit_b_16 --batch_size $BS --steps $STEPS

echo ">>> Step 3: FSDP FULL_SHARD"
torchrun --nproc_per_node=$NUM_GPUS scripts/benchmark_fsdp.py --model resnet50 --batch_size $BS --steps $STEPS --sharding_strategy FULL_SHARD
torchrun --nproc_per_node=$NUM_GPUS scripts/benchmark_fsdp.py --model vit_b_16 --batch_size $BS --steps $STEPS --sharding_strategy FULL_SHARD

echo ">>> Step 4: FSDP SHARD_GRAD_OP"
torchrun --nproc_per_node=$NUM_GPUS scripts/benchmark_fsdp.py --model resnet50 --batch_size $BS --steps $STEPS --sharding_strategy SHARD_GRAD_OP

echo ">>> Step 5: FSDP + Mixed Precision"
torchrun --nproc_per_node=$NUM_GPUS scripts/benchmark_fsdp.py --model resnet50 --batch_size $BS --steps $STEPS --sharding_strategy FULL_SHARD --mixed_precision
torchrun --nproc_per_node=$NUM_GPUS scripts/benchmark_fsdp.py --model vit_b_16 --batch_size $BS --steps $STEPS --sharding_strategy FULL_SHARD --mixed_precision

echo ">>> Step 6: OOM test"
python scripts/oom_test.py --mode single --model vit_l_16 --batch_size 64 || true
torchrun --nproc_per_node=$NUM_GPUS scripts/oom_test.py --mode fsdp --model vit_l_16 --batch_size 64

echo ">>> Step 7: Profiling"
torchrun --nproc_per_node=$NUM_GPUS scripts/profile_fsdp.py --model resnet50 --batch_size $BS

echo ">>> Step 8: Charts"
python scripts/visualize_results.py

echo "============================================================"
echo " ALL DONE - Job $SLURM_JOB_ID - $(date)"
echo "============================================================"
ls -la results/