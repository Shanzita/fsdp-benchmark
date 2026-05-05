"""Batch Size Scaling Study - finds max batch size and optimal throughput per config.

Sweeps batch sizes for single GPU, DDP, and FSDP to show how FSDP enables
larger batch sizes through memory sharding.
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


def try_batch_size(model_name, batch_size, mode, num_steps=10, warmup_steps=3):
    """Try a single batch size. Returns results dict or None on OOM."""
    if mode != "single":
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world_size, local_rank = 0, 1, 0

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = get_model(model_name).to(device)
        num_params = count_parameters(model)

        if mode == "ddp":
            par_model = DDP(model, device_ids=[local_rank])
        elif mode == "fsdp":
            par_model = FSDP(model,
                auto_wrap_policy=partial(size_based_auto_wrap_policy, min_num_params=1_000_000),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                device_id=local_rank, use_orig_params=True)
        elif mode == "fsdp_mp":
            par_model = FSDP(model,
                auto_wrap_policy=partial(size_based_auto_wrap_policy, min_num_params=1_000_000),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                device_id=local_rank, use_orig_params=True,
                mixed_precision=MixedPrecision(
                    param_dtype=torch.float16, reduce_dtype=torch.float32, buffer_dtype=torch.float32))
        else:
            par_model = model

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
        if mode != "single":
            dist.barrier()
        torch.cuda.reset_peak_memory_stats(device)

        step_times = []
        for _ in range(num_steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            optimizer.zero_grad()
            loss = criterion(par_model(x), target)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)

        if mode != "single":
            dist.barrier()

        peak_mem = torch.cuda.max_memory_allocated(device) / (1024**3)
        avg_time = sum(step_times) / len(step_times)
        throughput = (batch_size * world_size) / avg_time

        result = {
            "batch_size_per_gpu": batch_size,
            "status": "SUCCESS",
            "avg_step_time_ms": round(avg_time * 1000, 2),
            "throughput_samples_per_sec": round(throughput, 2),
            "peak_memory_gb": round(peak_mem, 3),
        }

        # Clean up model
        del par_model, model, optimizer, x, target
        torch.cuda.empty_cache()
        return result

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if rank == 0:
            print(f"   OOM at batch_size={batch_size}")
        return {"batch_size_per_gpu": batch_size, "status": "OOM"}


def sweep(model_name, mode, batch_sizes):
    """Sweep batch sizes for a given mode."""
    is_distributed = mode != "single"
    if is_distributed:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank, world_size = 0, 1

    if rank == 0:
        print(f"\n{'='*60}")
        print(f" Batch Size Sweep: {mode.upper()} | {model_name} | {world_size} GPUs")
        print(f" Testing batch sizes: {batch_sizes}")
        print(f"{'='*60}")

    sweep_results = []
    for bs in batch_sizes:
        if rank == 0:
            print(f"\n   Trying batch_size={bs}...")
        result = try_batch_size(model_name, bs, mode)
        if result:
            sweep_results.append(result)
            if rank == 0 and result["status"] == "SUCCESS":
                print(f"   bs={bs}: {result['throughput_samples_per_sec']:.1f} samples/s | {result['peak_memory_gb']:.3f} GB")
        if result and result["status"] == "OOM":
            # Stop sweeping - larger sizes will also OOM
            break

    if rank == 0:
        successful = [r for r in sweep_results if r["status"] == "SUCCESS"]
        output = {
            "experiment": "batch_scaling",
            "mode": mode,
            "model": model_name,
            "num_gpus": world_size,
            "gpu_name": torch.cuda.get_device_name(0),
            "max_successful_batch_size": max(r["batch_size_per_gpu"] for r in successful) if successful else 0,
            "sweep_results": sweep_results,
        }
        if successful:
            best = max(successful, key=lambda r: r["throughput_samples_per_sec"])
            output["optimal_batch_size"] = best["batch_size_per_gpu"]
            output["peak_throughput"] = best["throughput_samples_per_sec"]

        os.makedirs("results", exist_ok=True)
        path = f"results/batch_scaling_{mode}_{model_name}_{world_size}gpu.json"
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n Saved: {path}")
        print(f" Max batch size: {output.get('max_successful_batch_size', 'N/A')}")
        if successful:
            print(f" Optimal batch size: {output['optimal_batch_size']} ({output['peak_throughput']:.1f} samples/s)")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50", choices=MODEL_CHOICES)
    parser.add_argument("--mode", default="fsdp", choices=["single", "ddp", "fsdp", "fsdp_mp"])
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    args = parser.parse_args()
    sweep(args.model, args.mode, args.batch_sizes)
