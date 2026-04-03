"""OOM Test - proves FSDP enables training models that crash on single GPU"""
from functools import partial
import argparse, json, os
import torch
import torch.distributed as dist
import torch.nn as nn
from torchvision import models

def get_model(name):
    return {"resnet50": models.resnet50, "vit_b_16": models.vit_b_16,
            "vit_l_16": models.vit_l_16}[name](weights=None)

def test_single(model_name, batch_size):
    device = torch.device("cuda:0")
    print(f"\n{'='*60}")
    print(f" SINGLE GPU TEST: {model_name}, bs={batch_size}")
    print(f" GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}")
    try:
        model = get_model(model_name).to(device)
        torch.cuda.reset_peak_memory_stats(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(batch_size, 3, 224, 224, device=device)
        target = torch.randint(0, 1000, (batch_size,), device=device)
        model.train()
        loss = nn.CrossEntropyLoss()(model(x), target)
        loss.backward()
        optimizer.step()
        peak = torch.cuda.max_memory_allocated(device) / (1024**3)
        print(f" SUCCESS - Peak memory: {peak:.2f} GB")
        result = {"mode": "single_gpu", "model": model_name, "batch_size": batch_size,
                  "status": "SUCCESS", "peak_memory_gb": round(peak, 3)}
    except torch.cuda.OutOfMemoryError:
        peak = torch.cuda.max_memory_allocated(device) / (1024**3)
        print(f" OOM! Cannot fit {model_name} bs={batch_size} on single GPU")
        result = {"mode": "single_gpu", "model": model_name, "batch_size": batch_size,
                  "status": "OOM", "peak_memory_gb": round(peak, 3)}
    os.makedirs("results", exist_ok=True)
    with open(f"results/oom_single_{model_name}_bs{batch_size}.json", "w") as f:
        json.dump(result, f, indent=2)

def test_fsdp(model_name, batch_size):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if rank == 0:
        print(f"\n{'='*60}")
        print(f" FSDP TEST: {model_name}, bs={batch_size}/GPU, {world_size} GPUs")
        print(f"{'='*60}")
    try:
        model = get_model(model_name).to(device)
        torch.cuda.reset_peak_memory_stats(device)
        fsdp_model = FSDP(model,
            auto_wrap_policy=partial(size_based_auto_wrap_policy, min_num_params=1_000_000),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=local_rank, use_orig_params=True)
        optimizer = torch.optim.Adam(fsdp_model.parameters(), lr=1e-3)
        x = torch.randn(batch_size, 3, 224, 224, device=device)
        target = torch.randint(0, 1000, (batch_size,), device=device)
        fsdp_model.train()
        loss = nn.CrossEntropyLoss()(fsdp_model(x), target)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        dist.barrier()
        peak = torch.cuda.max_memory_allocated(device) / (1024**3)
        peak_t = torch.tensor([peak], device=device)
        peak_list = [torch.zeros(1, device=device) for _ in range(world_size)]
        dist.all_gather(peak_list, peak_t)
        if rank == 0:
            per_gpu = [t.item() for t in peak_list]
            print(f" SUCCESS! Peak memory/GPU: {[f'{m:.2f} GB' for m in per_gpu]}")
            result = {"mode": "fsdp", "model": model_name, "batch_size_per_gpu": batch_size,
                      "num_gpus": world_size, "status": "SUCCESS",
                      "peak_memory_gb_per_gpu": [round(m, 3) for m in per_gpu],
                      "max_peak_memory_gb": round(max(per_gpu), 3)}
            os.makedirs("results", exist_ok=True)
            with open(f"results/oom_fsdp_{model_name}_bs{batch_size}_{world_size}gpu.json", "w") as f:
                json.dump(result, f, indent=2)
    except torch.cuda.OutOfMemoryError:
        if rank == 0:
            print(f" OOM even with FSDP")
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["single", "fsdp"])
    parser.add_argument("--model", default="vit_l_16")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    if args.mode == "single":
        test_single(args.model, args.batch_size)
    else:
        test_fsdp(args.model, args.batch_size)