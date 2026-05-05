"""Analyze GPU scaling efficiency and generate scaling plots.

Computes scaling efficiency, speedup, and communication overhead from
gpu_scaling experiment results.
"""
import glob, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_scaling_results(d="results"):
    """Load all scaling_*.json files."""
    results = []
    for p in sorted(glob.glob(os.path.join(d, "scaling_*.json"))):
        with open(p) as f:
            results.append(json.load(f))
    return results


def load_batch_scaling_results(d="results"):
    """Load all batch_scaling_*.json files."""
    results = []
    for p in sorted(glob.glob(os.path.join(d, "batch_scaling_*.json"))):
        with open(p) as f:
            results.append(json.load(f))
    return results


def analyze_gpu_scaling(d="results"):
    """Compute and plot scaling efficiency."""
    results = load_scaling_results(d)
    if not results:
        print("No scaling results found.")
        return

    # Group by (mode, model)
    groups = {}
    for r in results:
        key = (r["mode"], r["model"])
        groups.setdefault(key, []).append(r)

    # Sort each group by GPU count
    for key in groups:
        groups[key].sort(key=lambda r: r["num_gpus"])

    # Compute scaling metrics
    print(f"\n{'='*80}")
    print(f" GPU SCALING ANALYSIS")
    print(f"{'='*80}")

    scaling_data = {}
    for (mode, model), runs in groups.items():
        baseline = runs[0]  # smallest GPU count
        base_gpus = baseline["num_gpus"]
        base_throughput = baseline["throughput_samples_per_sec"]

        print(f"\n {mode.upper()} - {model}")
        print(f" {'GPUs':>5} {'Throughput':>12} {'Speedup':>10} {'Efficiency':>12} {'Memory/GPU':>12}")
        print(f" {'-'*55}")

        for r in runs:
            n = r["num_gpus"]
            tp = r["throughput_samples_per_sec"]
            ideal_speedup = n / base_gpus
            actual_speedup = tp / base_throughput
            efficiency = (actual_speedup / ideal_speedup) * 100
            mem = r["max_peak_memory_gb"]
            print(f" {n:>5} {tp:>10.1f} s/s {actual_speedup:>8.2f}x {efficiency:>10.1f}% {mem:>10.3f} GB")

            r["speedup"] = round(actual_speedup, 3)
            r["scaling_efficiency_pct"] = round(efficiency, 1)

        scaling_data[(mode, model)] = runs

    # Save analysis
    analysis = []
    for (mode, model), runs in scaling_data.items():
        for r in runs:
            analysis.append(r)

    with open(os.path.join(d, "scaling_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n Saved: {d}/scaling_analysis.json")

    # Plot scaling charts
    _plot_scaling(scaling_data, d)


def _plot_scaling(scaling_data, d):
    """Generate scaling efficiency plots."""
    models_seen = set()
    for (mode, model) in scaling_data:
        models_seen.add(model)

    for model in models_seen:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"GPU Scaling — {model}", fontsize=16, fontweight="bold")

        colors = {"ddp": "#4C72B0", "fsdp": "#C44E52"}
        markers = {"ddp": "o", "fsdp": "s"}

        for mode in ["ddp", "fsdp"]:
            key = (mode, model)
            if key not in scaling_data:
                continue
            runs = scaling_data[key]
            gpus = [r["num_gpus"] for r in runs]
            throughputs = [r["throughput_samples_per_sec"] for r in runs]
            speedups = [r["speedup"] for r in runs]
            efficiencies = [r["scaling_efficiency_pct"] for r in runs]
            mems = [r["max_peak_memory_gb"] for r in runs]
            c = colors.get(mode, "#55A868")
            m = markers.get(mode, "^")

            # Throughput
            axes[0].plot(gpus, throughputs, f"-{m}", color=c, label=mode.upper(), linewidth=2, markersize=8)
            axes[0].set_xlabel("Number of GPUs")
            axes[0].set_ylabel("Throughput (samples/sec)")
            axes[0].set_title("Throughput Scaling")
            axes[0].legend()

            # Scaling efficiency
            axes[1].plot(gpus, efficiencies, f"-{m}", color=c, label=mode.upper(), linewidth=2, markersize=8)
            axes[1].axhline(y=100, color="gray", linestyle="--", alpha=0.5, label="Ideal" if mode == "ddp" else None)
            axes[1].set_xlabel("Number of GPUs")
            axes[1].set_ylabel("Scaling Efficiency (%)")
            axes[1].set_title("Scaling Efficiency")
            axes[1].legend()

            # Memory per GPU
            axes[2].plot(gpus, mems, f"-{m}", color=c, label=mode.upper(), linewidth=2, markersize=8)
            axes[2].set_xlabel("Number of GPUs")
            axes[2].set_ylabel("Peak Memory per GPU (GB)")
            axes[2].set_title("Memory per GPU")
            axes[2].legend()

        for ax in axes:
            ax.set_xticks(gpus)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(d, f"chart_scaling_{model}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f" Chart saved: {path}")


def analyze_batch_scaling(d="results"):
    """Analyze and plot batch size sweep results."""
    results = load_batch_scaling_results(d)
    if not results:
        print("No batch scaling results found.")
        return

    print(f"\n{'='*80}")
    print(f" BATCH SIZE SCALING ANALYSIS")
    print(f"{'='*80}")

    # Group by model
    by_model = {}
    for r in results:
        by_model.setdefault(r["model"], []).append(r)

    for model, runs in by_model.items():
        print(f"\n {model}")
        print(f" {'Mode':<12} {'GPUs':>5} {'Max BS':>8} {'Optimal BS':>12} {'Peak Thru':>12}")
        print(f" {'-'*55}")
        for r in runs:
            print(f" {r['mode']:<12} {r['num_gpus']:>5} {r.get('max_successful_batch_size','?'):>8} "
                  f"{r.get('optimal_batch_size','?'):>12} {r.get('peak_throughput',0):>10.1f} s/s")

    # Plot batch scaling
    for model, runs in by_model.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Batch Size Scaling — {model}", fontsize=16, fontweight="bold")

        colors = {"single": "#4C72B0", "ddp": "#DD8452", "fsdp": "#55A868", "fsdp_mp": "#C44E52"}

        for r in runs:
            mode = r["mode"]
            successful = [s for s in r["sweep_results"] if s["status"] == "SUCCESS"]
            if not successful:
                continue
            bs_list = [s["batch_size_per_gpu"] for s in successful]
            tp_list = [s["throughput_samples_per_sec"] for s in successful]
            mem_list = [s["peak_memory_gb"] for s in successful]
            label = f"{mode.upper()} ({r['num_gpus']}GPU)"
            c = colors.get(mode, "#937860")

            axes[0].plot(bs_list, tp_list, "-o", color=c, label=label, linewidth=2)
            axes[1].plot(bs_list, mem_list, "-o", color=c, label=label, linewidth=2)

        axes[0].set_xlabel("Batch Size per GPU")
        axes[0].set_ylabel("Throughput (samples/sec)")
        axes[0].set_title("Throughput vs Batch Size")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Batch Size per GPU")
        axes[1].set_ylabel("Peak Memory (GB)")
        axes[1].set_title("Memory vs Batch Size")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(d, f"chart_batch_scaling_{model}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f" Chart saved: {path}")


if __name__ == "__main__":
    d = "results"
    analyze_gpu_scaling(d)
    analyze_batch_scaling(d)
