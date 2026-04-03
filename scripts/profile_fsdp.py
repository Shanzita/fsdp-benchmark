"""FSDP Profiling — Chrome traces + kernel summary for screenshots"""
import argparse, os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.profiler import profile, ProfilerActivity, schedule
from torchvision import models

def get_model(name):
    return {"resnet50": models.resnet50, "vit_b_16": models.vit_b_16,
            "vit_l_16": models.vit_l_16}[name](weights=None)

def profile_fsdp(model_name, batch_size):
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"\n{'='*60}")
        print(f" Profiling FSDP: {model_name} on {world_size} GPUs")
        print(f"{'='*60}")

    model = get_model(model_name).to(device)
    fsdp_model = FSDP(model,
        auto_wrap_policy=size_based_auto_wrap_policy(min_num_params=1_000_000),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=local_rank, use_orig_params=True)
    optimizer = torch.optim.Adam(fsdp_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    target = torch.randint(0, 1000, (batch_size,), device=device)

    # Warmup
    fsdp_model.train()
    for _ in range(5):
        optimizer.zero_grad()
        loss = criterion(fsdp_model(x), target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    dist.barrier()

    os.makedirs("results", exist_ok=True)

    # Chrome trace
    if rank == 0:
        print(" Generating profiler trace...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=2, active=5, repeat=1),
        on_trace_ready=lambda p: p.export_chrome_trace(
            f"results/trace_fsdp_{model_name}_rank{rank}.json"),
        record_shapes=True, profile_memory=True, with_stack=True,
    ) as prof:
        for _ in range(8):
            optimizer.zero_grad()
            loss = criterion(fsdp_model(x), target)
            loss.backward()
            optimizer.step()
            prof.step()
    if rank == 0:
        print(f"   Saved: results/trace_fsdp_{model_name}_rank0.json")

    dist.barrier()

    # Kernel summary
    if rank == 0:
        print(" Generating kernel summary...")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, profile_memory=True) as prof2:
        for _ in range(5):
            optimizer.zero_grad()
            loss = criterion(fsdp_model(x), target)
            loss.backward()
            optimizer.step()
    if rank == 0:
        print(prof2.key_averages().table(sort_by="cuda_time_total", row_limit=15))
        with open(f"results/kernel_summary_{model_name}.txt", "w") as f:
            f.write(prof2.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        print(f"   Saved: results/kernel_summary_{model_name}.txt")

    # Memory summary
    if rank == 0:
        print(f"\n Memory Summary:")
        print(torch.cuda.memory_summary(device=device, abbreviated=True))

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("\n Profiling complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50", choices=["resnet50", "vit_b_16", "vit_l_16"])
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    profile_fsdp(args.model, args.batch_size)