"""Generate comparison charts from benchmark results"""
import glob, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_results(d="results"):
    results = []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        with open(p) as f:
            r = json.load(f)
            r["_file"] = os.path.basename(p)
            results.append(r)
    return results

def make_label(r):
    mode = r.get("mode", "unknown")
    gpus = r.get("num_gpus", 1)
    if mode == "single_gpu":
        return "Single GPU"
    elif mode == "ddp":
        return f"DDP ({gpus}GPU)"
    elif "fsdp" in mode:
        strat = r.get("sharding_strategy", "FULL_SHARD")
        mp = "+MP" if r.get("mixed_precision") else ""
        short = {"FULL_SHARD": "Full", "SHARD_GRAD_OP": "GradOp"}.get(strat, strat)
        return f"FSDP-{short}{mp} ({gpus}GPU)"
    return mode

def plot(d="results"):
    results = load_results(d)
    if not results:
        print("No results found. Run benchmarks first.")
        sys.exit(1)
    # Filter out oom test results
    results = [r for r in results if "oom" not in r.get("_file", "")]
    groups = {}
    for r in results:
        groups.setdefault(r.get("model", "unknown"), []).append(r)

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

    for model_name, mrs in groups.items():
        labels = [make_label(r) for r in mrs]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Benchmark — {model_name}", fontsize=16, fontweight="bold")

        # Memory
        ax = axes[0]
        mems = [r.get("max_peak_memory_gb", r.get("peak_memory_gb", 0)) for r in mrs]
        bars = ax.bar(range(len(labels)), mems, color=colors[:len(labels)])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Peak GPU Memory (GB)")
        ax.set_title("Memory Usage")
        for b, v in zip(bars, mems):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{v:.2f}", ha="center", fontsize=9)

        # Throughput
        ax = axes[1]
        thrus = [r.get("throughput_samples_per_sec", 0) for r in mrs]
        bars = ax.bar(range(len(labels)), thrus, color=colors[:len(labels)])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Throughput (samples/sec)")
        ax.set_title("Training Throughput")
        for b, v in zip(bars, thrus):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f"{v:.1f}", ha="center", fontsize=9)

        # Step time
        ax = axes[2]
        times = [r.get("avg_step_time_ms", 0) for r in mrs]
        bars = ax.bar(range(len(labels)), times, color=colors[:len(labels)])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Avg Step Time (ms)")
        ax.set_title("Per-Step Latency")
        for b, v in zip(bars, times):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, f"{v:.1f}", ha="center", fontsize=9)

        plt.tight_layout()
        path = os.path.join(d, f"chart_comparison_{model_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f" Chart saved: {path}")

    # Text table
    print(f"\n{'='*90}")
    hdr = f"{'Mode':<28} {'Model':<14} {'GPUs':>5} {'Step(ms)':>10} {'Thru(s/s)':>10} {'Mem(GB)':>9}"
    print(hdr)
    print("-" * 90)
    for r in sorted(results, key=lambda x: (x.get("model",""), x.get("mode",""))):
        print(f"{make_label(r):<28} {r.get('model','?'):<14} {r.get('num_gpus',1):>5} "
              f"{r.get('avg_step_time_ms',0):>10.2f} {r.get('throughput_samples_per_sec',0):>10.2f} "
              f"{r.get('max_peak_memory_gb', r.get('peak_memory_gb',0)):>9.3f}")
    with open(os.path.join(d, "comparison_table.txt"), "w") as f:
        f.write(hdr + "\n" + "-"*90 + "\n")
        for r in sorted(results, key=lambda x: (x.get("model",""), x.get("mode",""))):
            f.write(f"{make_label(r):<28} {r.get('model','?'):<14} {r.get('num_gpus',1):>5} "
                    f"{r.get('avg_step_time_ms',0):>10.2f} {r.get('throughput_samples_per_sec',0):>10.2f} "
                    f"{r.get('max_peak_memory_gb', r.get('peak_memory_gb',0)):>9.3f}\n")
    print(f"\n Table saved: {d}/comparison_table.txt")

if __name__ == "__main__":
    plot()