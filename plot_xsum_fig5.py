#!/usr/bin/env python3

import csv
import json
from pathlib import Path

import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


D = 1024
LAYERS = 12
N_BASE = 406_291_456

ROOT = Path(".")
OUTPUT = ROOT / "analysis_outputs"
OUTPUT.mkdir(exist_ok=True)

# Approximate values transcribed from the paper's Figure 5.
# These are comparison references; our x-values below are calculated
# using the repository's Appendix B parameter-counting formulas.
PAPER = {
    "Prefix attn": [(3.6, 20.46), (9.2, 20.40)],
    "PA attn": [(3.6, 20.31), (9.2, 20.83)],
    "PA-FFN": [(2.4, 20.66), (6.1, 20.98), (12.3, 21.24)],
    "LoRA attn": [(7.2, 20.29)],
    "LoRA-FFN": [(6.1, 21.31)],
}

COLORS = {
    "Prefix attn": "#1baf7a",
    "PA attn": "#2a78d6",
    "PA-FFN": "#eb6834",
    "LoRA attn": "#805ad5",
    "LoRA-FFN": "#eda100",
}

MARKERS = {
    "Prefix attn": "o",   # circle
    "PA attn": "v",       # upside-down triangle
    "PA-FFN": "^",             # triangle
    "LoRA attn": "s",     # square
    "LoRA-FFN": "D",           # diamond
}

# Exact capacity formulas following analysis/fig4_results.py.
def parameter_count(method, capacity):
    if method == "Prefix attn":
        return 2 * capacity * D * 3 * LAYERS
    if method == "PA attn":
        return 2 * capacity * D * 3 * LAYERS
    if method == "PA-FFN":
        return 2 * capacity * D * 2 * LAYERS
    if method == "LoRA attn":
        return 4 * capacity * D * 3 * LAYERS
    if method == "LoRA-FFN":
        return 10 * capacity * D * 2 * LAYERS
    raise ValueError(f"Unknown method: {method}")


def parameter_percent(method, capacity):
    return 100.0 * parameter_count(method, capacity) / N_BASE


# Full-capacity Figure 5 runs only.
# The low-budget l=6/r=6/r=8/r=3/r=2 runs are intentionally excluded.
RUNS = [
    {
        "method": "Prefix attn",
        "capacity_label": "l=200",
        "capacity": 200,
        "path": "checkpoints/xsum/20261007/fig5_seed42/prefix_attn/predict_results.json",
    },
    {
        "method": "Prefix attn",
        "capacity_label": "l=512",
        "capacity": 512,
        "path": "checkpoints/xsum/20261008/fig5_seed42/prefix_attn_l512/predict_results.json",
    },
    {
        "method": "PA attn",
        "capacity_label": "r=200",
        "capacity": 200,
        "path": "checkpoints/xsum/20261007/fig5_seed42/pa_attn/predict_results.json",
    },
    {
        "method": "PA attn",
        "capacity_label": "r=512",
        "capacity": 512,
        "path": "checkpoints/xsum/20261009/pa_attn_r512_retry/predict_results.json",
    },
    {
        "method": "PA-FFN",
        "capacity_label": "r=200",
        "capacity": 200,
        "path": "checkpoints/xsum/20261006/xsum.pa_ffn.seed42/predict_results.json",
    },
    {
        "method": "PA-FFN",
        "capacity_label": "r=512",
        "capacity": 512,
        "path": "checkpoints/xsum/20261008/fig5_seed42/pa_ffn_r512/predict_results.json",
    },
    {
        "method": "PA-FFN",
        "capacity_label": "r=1024",
        "capacity": 1024,
        "path": "checkpoints/xsum/20261008/fig5_seed42/pa_ffn_r1024/predict_results.json",
    },
    {
        "method": "LoRA attn",
        "capacity_label": "r=200",
        "capacity": 200,
        "path": "checkpoints/xsum/20261007/fig5_seed42/lora_attn/predict_results.json",
    },
    {
        "method": "LoRA-FFN",
        "capacity_label": "r=102",
        "capacity": 102,
        "path": "checkpoints/xsum/20261007/fig5_seed42/lora_ffn/predict_results.json",
    },
]

# Optional repeated PA-FFN seed runs. These are shown as a mean with an
# error bar and are not treated as additional capacity points.
PA_FFN_R512_SEEDS = {
    42: "checkpoints/xsum/20261008/fig5_seed42/pa_ffn_r512/predict_results.json",
    43: "checkpoints/xsum/20261010/low_budget_seed42/pa_ffn_r512_seed43/predict_results.json",
    44: "checkpoints/xsum/20261010/low_budget_seed42/pa_ffn_r512_seed44/predict_results.json",
}


def read_rouge2(path):
    path = Path(path)
    if not path.exists():
        print(f"WARNING: missing {path}; skipping")
        return None

    with path.open() as handle:
        data = json.load(handle)

    return float(data["predict_rouge2"])


def main():
    rows = []

    for run in RUNS:
        rouge2 = read_rouge2(run["path"])
        if rouge2 is None:
            continue

        pct = parameter_percent(run["method"], run["capacity"])
        row = {
            "method": run["method"],
            "capacity": run["capacity_label"],
            "parameters_percent": pct,
            "predict_rouge2": rouge2,
            "source": "reproduction",
        }
        rows.append(row)

        print(
            f"{run['method']:18s} {run['capacity_label']:>6s} "
            f"{pct:7.3f}%  ROUGE-2={rouge2:.4f}"
        )

    seed_scores = {}

    for seed, path in PA_FFN_R512_SEEDS.items():
        score = read_rouge2(path)
        if score is not None:
            seed_scores[seed] = score

    if seed_scores:
        seed_values = list(seed_scores.values())
        seed_mean = statistics.mean(seed_values)
        seed_std = statistics.stdev(seed_values) if len(seed_values) > 1 else 0.0
        seed_pct = parameter_percent("PA-FFN", 512)

        print(
            f"PA-FFN r=512 across {len(seed_values)} seeds: "
            f"mean={seed_mean:.4f}, sample SD={seed_std:.4f}"
        )


    with (OUTPUT / "xsum_fig5_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "capacity",
                "parameters_percent",
                "predict_rouge2",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    ink = "#1f1f1e"
    muted = "#6b6a64"
    grid = "#e6e5df"
    background = "#fcfcfb"

    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=200)
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)
    ax.grid(True, color=grid, linewidth=0.8)
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Paper references: open markers and dashed lines.
    for method, points in PAPER.items():
        xs, ys = zip(*points)
        color = COLORS[method]
        marker = MARKERS[method]

        ax.plot(
            xs,
            ys,
            linestyle="--",
            linewidth=1.4,
            color=color,
            marker=marker,
            markersize=7,
            markerfacecolor=background,
            markeredgecolor=color,
            markeredgewidth=1.5,
            alpha=0.75,
            label=f"{method} — paper",
        )

    # Our reproduction: filled markers and solid connecting lines.
    for method in COLORS:
        points = [
            row for row in rows
            if row["method"] == method
        ]
        points.sort(key=lambda row: row["parameters_percent"])

        if not points:
            continue

        xs = [row["parameters_percent"] for row in points]
        ys = [row["predict_rouge2"] for row in points]
        color = COLORS[method]
        marker = MARKERS[method]

        ax.plot(
            xs,
            ys,
            linestyle="-",
            linewidth=2.0,
            color=color,
            marker=marker,
            markersize=8,
            markerfacecolor=color,
            markeredgecolor=background,
            markeredgewidth=1.5,
            label=f"{method} — ours",
            zorder=3,
        )

        for row in points:
            ax.annotate(
                row["capacity"],
                (row["parameters_percent"], row["predict_rouge2"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color=ink,
            )

    # Three-seed variability for PA-FFN r=512.
    # The filled red triangle already represents the original seed-42 run.
    if seed_scores:
        seed_values = list(seed_scores.values())
        seed_mean = statistics.mean(seed_values)
        seed_std = statistics.stdev(seed_values) if len(seed_values) > 1 else 0.0
        seed_pct = parameter_percent("PA-FFN", 512)

        ax.errorbar(
            [seed_pct],
            [seed_mean],
            yerr=[seed_std],
            fmt="none",
            ecolor=COLORS["PA-FFN"],
            alpha=0.78,
            elinewidth=1.4,
            capsize=4,
            capthick=1.3,
            label="PA-FFN r=512, 3 seeds",
            zorder=4,
        )

    ax.set_xlabel("Fine-tuned parameters (%)", color=ink)
    ax.set_ylabel("XSum ROUGE-2 (test)", color=ink)
    #ax.set_title("XSum Figure 5: PETL capacity versus ROUGE-2", color=ink)

    ax.set_xlim(2, 14)
    ax.set_ylim(19, 22.2)
    # Legend begins near x = 10% and remains in the lower-right region.
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(9.0, 19.05),
        bbox_transform=ax.transData,
        ncol=2,
        fontsize=8,
        columnspacing=1.2,
        handletextpad=0.5,
        borderaxespad=0.0,
        frameon=True,
        facecolor="#fcfcfb",
        edgecolor="#6b6a64",
        framealpha=0.92,
        fancybox=False,
        borderpad=0.6,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT / "xsum_fig5_reproduction.png", dpi=300, facecolor=background)
    fig.savefig(OUTPUT / "xsum_fig5_reproduction.pdf", facecolor=background)

    print(f"\nSaved outputs in {OUTPUT}/")
    print("  xsum_fig5_results.csv")
    print("  xsum_fig5_reproduction.png")
    print("  xsum_fig5_reproduction.pdf")


if __name__ == "__main__":
    main()
