"""Single-GPU Baseline Benchmark"""
import argparse, json, os, time
import torch
import torch.nn as nn
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f" Single-GPU Baseline: {model_name}")
    print(f" Batch size: {batch_size} | Steps: {num_steps}")
    print(f"{'='*60}")

    torch.cuda.reset_peak_memory_stats()
    model = get_model(model_name).to(device)
    print(f" Parameters: {count_parameters(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    target = torch.randint(0, 1000, (batch_size,), device=device)

    model.train()
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        loss = criterion(model(x), target)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    step_times = []
    for step in range(num_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss = criterion(model(x), target)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        if (step + 1) % 10 == 0:
            print(f"   Step {step+1}/{num_steps} | Loss: {loss.item():.4f} | Time: {step_times[-1]*1000:.1f}ms")

    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    avg_time = sum(step_times) / len(step_times)
    throughput = batch_size / avg_time

    results = {
        "mode": "single_gpu", "model": model_name,
        "num_params_M": round(count_parameters(model) / 1e6, 1),
        "batch_size": batch_size, "num_steps": num_steps, "num_gpus": 1,
        "avg_step_time_ms": round(avg_time * 1000, 2),
        "throughput_samples_per_sec": round(throughput, 2),
        "peak_memory_gb": round(peak_mem, 3),
        "max_peak_memory_gb": round(peak_mem, 3),
        "final_loss": round(loss.item(), 4),
        "gpu_name": torch.cuda.get_device_name(0),
    }

    print(f"\n RESULTS: {avg_time*1000:.2f}ms/step | {throughput:.1f} samples/s | {peak_mem:.3f} GB peak")
    os.makedirs("results", exist_ok=True)
    path = f"results/single_gpu_{model_name}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f" Saved: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50", choices=["resnet50", "vit_b_16", "vit_l_16"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    benchmark(args.model, args.batch_size, args.steps)