"""GPU Scaling Study - measures strong and weak scaling efficiency across GPU counts.

This script is launched by the SLURM script at different GPU counts (1, 2, 4).
It runs both DDP and FSDP at each count and records scaling efficiency metrics.
"""
import argparse, json, os, time
from functools import partial
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import get_model, count_parameters, MODEL_CHOICES


def run_benchmark(model_name, batch_size, num_steps, mode, warmup_steps=5):
    """Run a single benchmark configuration and return results dict."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"\n{'='*60}")
        print(f" GPU Scaling: {mode.upper()} | {model_name} | {world_size} GPUs")
        print(f" Batch/GPU: {batch_size}")
        print(f"{'='*60}")

    torch.cuda.reset_peak_memory_stats(device)
    model = get_model(model_name).to(device)
    num_params = count_parameters(model)

    if mode == "ddp":
        par_model = DDP(model, device_ids=[local_rank])
    else:  # fsdp
        auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=1_000_000)
        par_model = FSDP(model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=local_rank, use_orig_params=True)

    optimizer = torch.optim.Adam(par_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    target = torch.randint(0, 1000, (batch_size,), device=device)

    par_model.train()
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        loss = criterion(par_model(x), target)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.reset_peak_memory_stats(device)

    step_times = []
    for step in range(num_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss = criterion(par_model(x), target)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)

    dist.barrier()
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024**3)
    peak_mem_tensor = torch.tensor([peak_mem], device=device)
    peak_mem_list = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(peak_mem_list, peak_mem_tensor)

    if rank == 0:
        avg_time = sum(step_times) / len(step_times)
        throughput = (batch_size * world_size) / avg_time
        per_gpu_mem = [t.item() for t in peak_mem_list]
        results = {
            "experiment": "gpu_scaling",
            "mode": mode,
            "model": model_name,
            "num_params_M": round(num_params / 1e6, 1),
            "batch_size_per_gpu": batch_size,
            "effective_batch_size": batch_size * world_size,
            "num_steps": num_steps,
            "num_gpus": world_size,
            "avg_step_time_ms": round(avg_time * 1000, 2),
            "throughput_samples_per_sec": round(throughput, 2),
            "peak_memory_gb_per_gpu": [round(m, 3) for m in per_gpu_mem],
            "max_peak_memory_gb": round(max(per_gpu_mem), 3),
            "final_loss": round(loss.item(), 4),
            "gpu_name": torch.cuda.get_device_name(0),
        }
        print(f" RESULTS: {avg_time*1000:.2f}ms/step | {throughput:.1f} samples/s | {max(per_gpu_mem):.3f} GB peak")
        os.makedirs("results", exist_ok=True)
        path = f"results/scaling_{mode}_{model_name}_{world_size}gpu.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f" Saved: {path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50", choices=MODEL_CHOICES)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--mode", default="fsdp", choices=["ddp", "fsdp"])
    args = parser.parse_args()
    run_benchmark(args.model, args.batch_size, args.steps, args.mode)
