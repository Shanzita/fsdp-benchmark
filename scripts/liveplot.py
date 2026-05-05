"""Live plotting - generates all report-ready figures from whatever results exist so far.
Run this anytime to see current progress as charts."""
import glob, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results"
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
          "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD", "#B07AA1", "#FF9DA7"]


def load_json(pattern):
    results = []
    for p in sorted(glob.glob(os.path.join(RESULTS_DIR, pattern))):
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
        ac = "+AC" if r.get("activation_checkpointing") else ""
        offload = "+Offld" if r.get("cpu_offload") else ""
        accum = f"+GA{r['grad_accum_steps']}" if r.get("grad_accum_steps", 1) > 1 else ""
        short = {"FULL_SHARD": "Full", "SHARD_GRAD_OP": "GradOp", "NO_SHARD": "NoSh"}.get(strat, strat)
        return f"FSDP-{short}{mp}{ac}{offload}{accum} ({gpus}GPU)"
    return mode


# ─── Figure 1 & 2: Main comparison bar charts (one per model) ───
def plot_comparison():
    results = load_json("*.json")
    results = [r for r in results if not any(k in r.get("_file", "")
               for k in ("oom", "scaling_", "batch_scaling", "scaling_analysis"))]
    if not results:
        return
    groups = {}
    for r in results:
        groups.setdefault(r.get("model", "unknown"), []).append(r)

    for model_name, mrs in groups.items():
        labels = [make_label(r) for r in mrs]
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle(f"FSDP Benchmark Comparison — {model_name}", fontsize=16, fontweight="bold")

        # Memory
        ax = axes[0]
        mems = [r.get("max_peak_memory_gb", r.get("peak_memory_gb", 0)) for r in mrs]
        bars = ax.bar(range(len(labels)), mems, color=COLORS[:len(labels)])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Peak GPU Memory (GB)")
        ax.set_title("Memory Usage")
        for b, v in zip(bars, mems):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{v:.2f}", ha="center", fontsize=8)

        # Throughput
        ax = axes[1]
        thrus = [r.get("throughput_samples_per_sec", 0) for r in mrs]
        bars = ax.bar(range(len(labels)), thrus, color=COLORS[:len(labels)])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Throughput (samples/sec)")
        ax.set_title("Training Throughput")
        for b, v in zip(bars, thrus):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f"{v:.0f}", ha="center", fontsize=8)

        # Step time
        ax = axes[2]
        times = [r.get("avg_step_time_ms", 0) for r in mrs]
        bars = ax.bar(range(len(labels)), times, color=COLORS[:len(labels)])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Avg Step Time (ms)")
        ax.set_title("Per-Step Latency")
        for b, v in zip(bars, times):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, f"{v:.1f}", ha="center", fontsize=8)

        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f"fig_comparison_{model_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [LIVE] {path}")


# ─── Figure 3: Memory reduction waterfall ───────────────────────
def plot_memory_waterfall():
    results = load_json("*.json")
    results = [r for r in results if not any(k in r.get("_file", "")
               for k in ("oom", "scaling_", "batch_scaling", "scaling_analysis"))]
    if not results:
        return
    groups = {}
    for r in results:
        groups.setdefault(r.get("model", "unknown"), []).append(r)

    for model_name, mrs in groups.items():
        # Sort by memory usage descending
        mrs_sorted = sorted(mrs, key=lambda r: r.get("max_peak_memory_gb", r.get("peak_memory_gb", 0)), reverse=True)
        labels = [make_label(r) for r in mrs_sorted]
        mems = [r.get("max_peak_memory_gb", r.get("peak_memory_gb", 0)) for r in mrs_sorted]
        baseline = mems[0] if mems else 1

        fig, ax = plt.subplots(figsize=(12, 6))
        colors_mem = ["#C44E52" if m > baseline * 0.7 else "#DD8452" if m > baseline * 0.4 else "#55A868" for m in mems]
        bars = ax.barh(range(len(labels)), mems, color=colors_mem)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Peak GPU Memory (GB)")
        ax.set_title(f"Memory Reduction Across Configurations — {model_name}", fontsize=14, fontweight="bold")
        for b, v in zip(bars, mems):
            pct = (1 - v / baseline) * 100
            label = f"{v:.2f} GB ({pct:+.0f}%)" if pct != 0 else f"{v:.2f} GB (baseline)"
            ax.text(v + 0.05, b.get_y() + b.get_height()/2, label, va="center", fontsize=9)
        ax.invert_yaxis()
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f"fig_memory_waterfall_{model_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [LIVE] {path}")


# ─── Figure 4: Memory vs Throughput tradeoff scatter ────────────
def plot_tradeoff():
    results = load_json("*.json")
    results = [r for r in results if not any(k in r.get("_file", "")
               for k in ("oom", "scaling_", "batch_scaling", "scaling_analysis"))]
    if not results:
        return
    groups = {}
    for r in results:
        groups.setdefault(r.get("model", "unknown"), []).append(r)

    fig, axes = plt.subplots(1, len(groups), figsize=(8 * len(groups), 6))
    if len(groups) == 1:
        axes = [axes]

    for ax, (model_name, mrs) in zip(axes, groups.items()):
        for i, r in enumerate(mrs):
            mem = r.get("max_peak_memory_gb", r.get("peak_memory_gb", 0))
            thru = r.get("throughput_samples_per_sec", 0)
            label = make_label(r)
            ax.scatter(mem, thru, c=COLORS[i % len(COLORS)], s=120, zorder=5, edgecolors="black", linewidth=0.5)
            ax.annotate(label, (mem, thru), fontsize=7, ha="left", va="bottom",
                       xytext=(5, 5), textcoords="offset points")
        ax.set_xlabel("Peak GPU Memory (GB)")
        ax.set_ylabel("Throughput (samples/sec)")
        ax.set_title(f"Memory vs Throughput — {model_name}", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig_tradeoff.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [LIVE] {path}")


# ─── Figure 5: GPU Scaling efficiency ───────────────────────────
def plot_scaling():
    results = load_json("scaling_*.json")
    if not results:
        return
    groups = {}
    for r in results:
        key = (r["mode"], r["model"])
        groups.setdefault(key, []).append(r)
    for key in groups:
        groups[key].sort(key=lambda r: r["num_gpus"])

    models = set(r["model"] for r in results)
    for model in models:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"GPU Scaling Efficiency — {model}", fontsize=16, fontweight="bold")
        style = {"ddp": ("#4C72B0", "o", "DDP"), "fsdp": ("#C44E52", "s", "FSDP")}

        for mode in ["ddp", "fsdp"]:
            key = (mode, model)
            if key not in groups:
                continue
            runs = groups[key]
            gpus = [r["num_gpus"] for r in runs]
            thrus = [r["throughput_samples_per_sec"] for r in runs]
            base_thru = thrus[0]
            speedups = [t / base_thru for t in thrus]
            efficiencies = [(t / base_thru) / (g / gpus[0]) * 100 for t, g in zip(thrus, gpus)]
            mems = [r["max_peak_memory_gb"] for r in runs]
            c, m, lab = style[mode]

            axes[0].plot(gpus, thrus, f"-{m}", color=c, label=lab, linewidth=2, markersize=8)
            axes[1].plot(gpus, efficiencies, f"-{m}", color=c, label=lab, linewidth=2, markersize=8)
            axes[2].plot(gpus, mems, f"-{m}", color=c, label=lab, linewidth=2, markersize=8)

        axes[0].set_xlabel("GPUs"); axes[0].set_ylabel("Throughput (samples/sec)"); axes[0].set_title("Throughput Scaling")
        axes[1].set_xlabel("GPUs"); axes[1].set_ylabel("Efficiency (%)"); axes[1].set_title("Scaling Efficiency")
        axes[1].axhline(y=100, color="gray", linestyle="--", alpha=0.5)
        axes[2].set_xlabel("GPUs"); axes[2].set_ylabel("Memory/GPU (GB)"); axes[2].set_title("Memory per GPU")
        for ax in axes:
            ax.legend(); ax.grid(True, alpha=0.3); ax.set_xticks(gpus)

        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f"fig_scaling_{model}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [LIVE] {path}")


# ─── Figure 6: Batch size scaling ───────────────────────────────
def plot_batch_scaling():
    results = load_json("batch_scaling_*.json")
    if not results:
        return
    by_model = {}
    for r in results:
        by_model.setdefault(r["model"], []).append(r)

    style = {"single": ("#4C72B0", "Single"), "ddp": ("#DD8452", "DDP"),
             "fsdp": ("#55A868", "FSDP"), "fsdp_mp": ("#C44E52", "FSDP+MP")}

    for model, runs in by_model.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Batch Size Scaling — {model}", fontsize=16, fontweight="bold")

        for r in runs:
            mode = r["mode"]
            successful = [s for s in r["sweep_results"] if s["status"] == "SUCCESS"]
            if not successful:
                continue
            bs = [s["batch_size_per_gpu"] for s in successful]
            tp = [s["throughput_samples_per_sec"] for s in successful]
            mem = [s["peak_memory_gb"] for s in successful]
            c, lab = style.get(mode, ("#937860", mode))
            lab = f"{lab} ({r['num_gpus']}GPU)"

            axes[0].plot(bs, tp, "-o", color=c, label=lab, linewidth=2)
            axes[1].plot(bs, mem, "-o", color=c, label=lab, linewidth=2)

        axes[0].set_xlabel("Batch Size/GPU"); axes[0].set_ylabel("Throughput (samples/sec)")
        axes[0].set_title("Throughput vs Batch Size"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].set_xlabel("Batch Size/GPU"); axes[1].set_ylabel("Peak Memory (GB)")
        axes[1].set_title("Memory vs Batch Size"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f"fig_batch_scaling_{model}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [LIVE] {path}")


# ─── Figure 7: OOM comparison ──────────────────────────────────
def plot_oom():
    results = load_json("oom_*.json")
    if not results:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    labels, mems, colors = [], [], []
    for r in results:
        mode = r.get("mode", "?")
        model = r.get("model", "?")
        bs = r.get("batch_size", r.get("batch_size_per_gpu", "?"))
        gpus = r.get("num_gpus", 1)
        status = r.get("status", "?")
        lab = f"{mode.upper()} {model}\nbs={bs} {gpus}GPU"
        labels.append(lab)
        mem = r.get("max_peak_memory_gb", r.get("peak_memory_gb", 0))
        mems.append(mem)
        colors.append("#55A868" if status == "SUCCESS" else "#C44E52")

    bars = ax.bar(range(len(labels)), mems, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Peak Memory (GB)")
    ax.set_title("OOM Test — Can FSDP Train What Single GPU Cannot?", fontsize=14, fontweight="bold")
    for b, v, r in zip(bars, mems, results):
        status = r.get("status", "?")
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.1,
                f"{v:.1f}GB\n{status}", ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig_oom_test.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [LIVE] {path}")


# ─── Summary table ─────────────────────────────────────────────
def print_table():
    results = load_json("*.json")
    results = [r for r in results if not any(k in r.get("_file", "")
               for k in ("oom", "scaling_", "batch_scaling", "scaling_analysis", "environment", "nvidia"))]
    if not results:
        return
    print(f"\n{'='*95}")
    hdr = f"{'Config':<35} {'Model':<12} {'GPUs':>5} {'Step(ms)':>10} {'Thru(s/s)':>10} {'Mem(GB)':>9}"
    print(hdr)
    print("-" * 95)
    lines = [hdr, "-" * 95]
    for r in sorted(results, key=lambda x: (x.get("model",""), x.get("max_peak_memory_gb", x.get("peak_memory_gb",0)))):
        line = (f"{make_label(r):<35} {r.get('model','?'):<12} {r.get('num_gpus',1):>5} "
                f"{r.get('avg_step_time_ms',0):>10.2f} {r.get('throughput_samples_per_sec',0):>10.2f} "
                f"{r.get('max_peak_memory_gb', r.get('peak_memory_gb',0)):>9.3f}")
        print(line)
        lines.append(line)
    with open(os.path.join(RESULTS_DIR, "summary_table.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  [LIVE] {RESULTS_DIR}/summary_table.txt")


if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f" LIVE PLOT — generating all report figures")
    print(f"{'='*50}")
    plot_comparison()
    plot_memory_waterfall()
    plot_tradeoff()
    plot_scaling()
    plot_batch_scaling()
    plot_oom()
    print_table()
    print(f"\n Done! Check results/ for all fig_*.png files.\n")
