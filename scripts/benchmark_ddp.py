"""DDP (DistributedDataParallel) Benchmark"""
import argparse, json, os, time
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models

def get_model(name):
    model_map = {
        "resnet50": models.resnet50,
        "vit_b_16": models.vit_b_16,
        "vit_l_16": models.vit_l_16,
    }
    return model_map[name](weights=None)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark(model_name, batch_size, num_steps, warmup_steps=5):
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"\n{'='*60}")
        print(f" DDP Benchmark: {model_name}")
        print(f" World size: {world_size} | Batch/GPU: {batch_size}")
        print(f"{'='*60}")

    torch.cuda.reset_peak_memory_stats(device)
    model = get_model(model_name).to(device)
    ddp_model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    target = torch.randint(0, 1000, (batch_size,), device=device)

    ddp_model.train()
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        loss = criterion(ddp_model(x), target)
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
        loss = criterion(ddp_model(x), target)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        if rank == 0 and (step + 1) % 10 == 0:
            print(f"   Step {step+1}/{num_steps} | Loss: {loss.item():.4f} | Time: {step_times[-1]*1000:.1f}ms")

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
            "mode": "ddp", "model": model_name,
            "num_params_M": round(count_parameters(model) / 1e6, 1),
            "batch_size_per_gpu": batch_size,
            "effective_batch_size": batch_size * world_size,
            "num_steps": num_steps, "num_gpus": world_size,
            "avg_step_time_ms": round(avg_time * 1000, 2),
            "throughput_samples_per_sec": round(throughput, 2),
            "peak_memory_gb_per_gpu": [round(m, 3) for m in per_gpu_mem],
            "max_peak_memory_gb": round(max(per_gpu_mem), 3),
            "final_loss": round(loss.item(), 4),
            "gpu_name": torch.cuda.get_device_name(0),
        }
        print(f"\n RESULTS: {avg_time*1000:.2f}ms/step | {throughput:.1f} samples/s | {max(per_gpu_mem):.3f} GB peak")
        os.makedirs("results", exist_ok=True)
        path = f"results/ddp_{model_name}_{world_size}gpu.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f" Saved: {path}")

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50", choices=["resnet50", "vit_b_16", "vit_l_16"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    benchmark(args.model, args.batch_size, args.steps)