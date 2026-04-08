"""Generate benchmark comparison graphs from benchmarks.csv."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BENCHMARKS_CSV = Path(__file__).resolve().parent.parent / "benchmarks.csv"
GRAPHS_DIR = Path(__file__).resolve().parent.parent / "graphs"

# --- Modern color palette (muted, high-contrast, colorblind-friendly) ------
PALETTE = {
    "naive": "#E8634F",           # warm red
    "sequence-alignment": "#4A90D9",  # steel blue
    "sequence-alignment-widened": "#5EBB73",  # jade green
    "sequence-alignment-widened-auto": "#F5A623",  # amber
}
VARIANT_LABELS = {
    "naive": "Naive (scalar)",
    "sequence-alignment": "SSE\u2192RVV (sse2rvv)",
    "sequence-alignment-widened": "Widened (manual)",
    "sequence-alignment-widened-auto": "Widened (auto)",
}

DATASET_ORDER = ["1k.fa", "10k.fa", "100k.fa", "1M.fa", "10M.fa"]

# --- GCUPS computation ------------------------------------------------------
# GCUPS = (m * n) / (t * 1e9)
# m = total query residues = 100 queries * 54 residues each = 5400
# n = number of residues in the dataset
M_QUERY = 54 * 100  # 5400
DATASET_RESIDUES = {
    "1k.fa": 1001,
    "10k.fa": 10001,
    "100k.fa": 100001,
    "1M.fa": 1000001,
    "10M.fa": 10000001,
}


def time_to_gcups(t, dataset):
    """Convert time in seconds to GCUPS for a given dataset."""
    n = DATASET_RESIDUES[dataset]
    return (M_QUERY * n) / (t * 1e9)

# --- Global style -----------------------------------------------------------
BG_COLOR = "#FAFBFC"
GRID_COLOR = "#E0E4E8"
TEXT_COLOR = "#2C3E50"
SPINE_COLOR = "#CBD5E0"
FONT_FAMILY = "sans-serif"
DPI = 200


def _apply_style():
    """Set a clean, modern matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": SPINE_COLOR,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "axes.labelcolor": TEXT_COLOR,
        "axes.labelsize": 12,
        "axes.labelweight": "medium",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlepad": 14,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": GRID_COLOR,
        "legend.fontsize": 10,
        "legend.borderpad": 0.8,
        "legend.handlelength": 1.5,
        "font.family": FONT_FAMILY,
        "font.size": 11,
        "text.color": TEXT_COLOR,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
    })


def _strip_spines(ax, keep_left=True):
    """Remove top and right spines for a cleaner look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not keep_left:
        ax.spines["left"].set_visible(False)


def _add_scatter(ax, data_list, positions, colors, jitter=0.06):
    """Overlay individual data points on boxplots."""
    rng = np.random.default_rng(42)
    for pos, runs, c in zip(positions, data_list, colors):
        xs = pos + rng.uniform(-jitter, jitter, len(runs))
        ax.scatter(xs, runs, color=c, alpha=0.45, s=22, zorder=5,
                   edgecolors="white", linewidths=0.5)


def _styled_boxplot(ax, data, colors, positions=None, width=0.55):
    """Create a consistently styled boxplot with scatter overlay."""
    if positions is None:
        positions = list(range(1, len(data) + 1))
    bp = ax.boxplot(
        data, positions=positions, patch_artist=True, widths=width,
        whiskerprops=dict(color=SPINE_COLOR, linewidth=1.2),
        capprops=dict(color=SPINE_COLOR, linewidth=1.2),
        flierprops=dict(marker="o", markerfacecolor="#AAB2BD", markersize=4,
                        markeredgecolor="white", markeredgewidth=0.5, alpha=0.6),
        medianprops=dict(color="#2C3E50", linewidth=2),
        boxprops=dict(linewidth=0),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    _add_scatter(ax, data, positions, colors)
    return bp


# --- Data loading -----------------------------------------------------------

def load_data() -> list[dict]:
    rows = []
    with open(BENCHMARKS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_cols = [f"run_{i}" for i in range(1, 11)]
            row["runs"] = [float(row[c]) for c in run_cols]
            for key in ("mean", "median", "min", "max", "stdev", "q1", "q3", "iqr"):
                row[key] = float(row[key])
            rows.append(row)
    return rows


def get_variant_data(rows, variant):
    return {r["dataset"]: r for r in rows if r["code_variant"] == variant}


def dataset_sort_key(d):
    return DATASET_ORDER.index(d) if d in DATASET_ORDER else 999


def save(fig, name):
    fig.savefig(GRAPHS_DIR / name, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved {name}")


# ---------------------------------------------------------------------------
# Boxplot — one subplot per dataset, own y-scale, log axis
# ---------------------------------------------------------------------------
def boxplot_combined(rows):
    from matplotlib.patches import Patch

    variants = ["sequence-alignment", "sequence-alignment-widened",
                "sequence-alignment-widened-auto"]
    vdata = {v: get_variant_data(rows, v) for v in variants}
    datasets = sorted(
        set.union(*(set(vdata[v]) for v in variants)),
        key=dataset_sort_key,
    )

    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 6),
                             sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        data, colors = [], []
        for v in variants:
            if ds in vdata[v]:
                data.append([time_to_gcups(t, ds) for t in vdata[v][ds]["runs"]])
                colors.append(PALETTE[v])

        _styled_boxplot(ax, data, colors, width=0.55)

        # X-tick labels: short variant names
        short_labels = []
        for v in variants:
            if ds in vdata[v]:
                short_labels.append(VARIANT_LABELS[v])
        ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=8.5)
        ax.set_ylabel("GCUPS")
        ax.set_title(ds, fontsize=13)
        _strip_spines(ax)

    # Shared legend outside the subplots (centered below)
    legend_handles = [Patch(facecolor=PALETTE[v], alpha=0.75, label=VARIANT_LABELS[v])
                      for v in variants]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(variants), fontsize=10, frameon=True,
               framealpha=0.95, edgecolor=GRID_COLOR,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("GCUPS Distribution \u2014 Translated Variants by Dataset",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, "boxplot_combined.png")


# ---------------------------------------------------------------------------
# Graph 5: Grouped bar — mean time, all 3 translated variants per dataset
# ---------------------------------------------------------------------------
def grouped_bar_translated(rows):
    variants = ["sequence-alignment", "sequence-alignment-widened",
                "sequence-alignment-widened-auto"]
    vdata = {v: get_variant_data(rows, v) for v in variants}
    datasets = sorted(
        set.intersection(*(set(vdata[v]) for v in variants)),
        key=dataset_sort_key,
    )

    x = np.arange(len(datasets))
    n = len(variants)
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, v in enumerate(variants):
        gcups_vals = [time_to_gcups(vdata[v][ds]["mean"], ds) for ds in datasets]
        # Propagate stdev: GCUPS = k/t, so σ_GCUPS ≈ (k/t²)·σ_t = GCUPS·(σ_t/t)
        gcups_errs = [
            gcups_vals[j] * (vdata[v][ds]["stdev"] / vdata[v][ds]["mean"])
            for j, ds in enumerate(datasets)
        ]
        bars = ax.bar(
            x + i * width - width * (n - 1) / 2, gcups_vals, width,
            yerr=gcups_errs, capsize=3,
            label=VARIANT_LABELS[v], color=PALETTE[v], alpha=0.85,
            error_kw=dict(lw=1, capthick=1, color="#7F8C8D"),
            edgecolor="white", linewidth=0.6,
        )
        # Value labels on bars
        for bar, g in zip(bars, gcups_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{g:.4f}", ha="center", va="bottom", fontsize=8,
                    color=TEXT_COLOR, fontweight="medium")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.set_ylabel("GCUPS")
    ax.set_xlabel("Dataset")
    ax.legend(loc="upper left")
    ax.set_title("Translated Variants: GCUPS Performance")
    _strip_spines(ax)
    fig.tight_layout()
    save(fig, "bar_translated_variants.png")


# ---------------------------------------------------------------------------
# Graph 6: Speedup chart — all variants relative to naive
# ---------------------------------------------------------------------------
def speedup_vs_naive(rows):
    naive = get_variant_data(rows, "naive")
    sse = get_variant_data(rows, "sequence-alignment")
    datasets = sorted(set(naive) & set(sse), key=dataset_sort_key)

    x = np.arange(len(datasets))
    width = 0.45
    fig, ax = plt.subplots(figsize=(9, 5.5))

    v = "sequence-alignment"
    speedups = [naive[ds]["median"] / sse[ds]["median"] for ds in datasets]
    bars = ax.bar(
        x, speedups, width,
        label=VARIANT_LABELS[v], color=PALETTE[v], alpha=0.85,
        edgecolor="white", linewidth=0.6,
    )
    for bar, sp in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"{sp:.1f}x", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=TEXT_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.set_ylabel("Speedup (x)")
    ax.set_xlabel("Dataset")
    ax.axhline(y=1, color=PALETTE["naive"], linestyle="--", linewidth=1, alpha=0.5,
               label="Naive baseline (1x)")
    ax.legend(loc="upper left")
    ax.set_title("SSE\u2192RVV Speedup over Naive (scalar)  \u2014  using median times")
    _strip_spines(ax)
    fig.tight_layout()
    save(fig, "speedup_vs_naive.png")


# ---------------------------------------------------------------------------
# Speedup of widened variants over sequence-alignment (median times)
# ---------------------------------------------------------------------------
def speedup_widened_vs_sse(rows):
    sse = get_variant_data(rows, "sequence-alignment")
    variants = ["sequence-alignment-widened", "sequence-alignment-widened-auto"]
    vdata = {v: get_variant_data(rows, v) for v in variants}
    datasets = sorted(
        set(sse) & set.intersection(*(set(vdata[v]) for v in variants)),
        key=dataset_sort_key,
    )

    x = np.arange(len(datasets))
    n = len(variants)
    width = 0.3
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, v in enumerate(variants):
        speedups = [sse[ds]["median"] / vdata[v][ds]["median"] for ds in datasets]
        bars = ax.bar(
            x + i * width - width * (n - 1) / 2, speedups, width,
            label=VARIANT_LABELS[v], color=PALETTE[v], alpha=0.85,
            edgecolor="white", linewidth=0.6,
        )
        for bar, sp in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{sp:.2f}x", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=TEXT_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.set_ylabel("Speedup (x)")
    ax.set_xlabel("Dataset")
    ax.axhline(y=1, color=PALETTE["sequence-alignment"], linestyle="--",
               linewidth=1, alpha=0.5, label="SSE\u2192RVV baseline (1x)")
    ax.legend(loc="lower right")
    ax.set_title("Speedup over SSE\u2192RVV (sse2rvv)  \u2014  using median times")
    _strip_spines(ax)
    fig.tight_layout()
    save(fig, "speedup_widened_vs_sse.png")


# ---------------------------------------------------------------------------
# Line chart — scaling across datasets
# ---------------------------------------------------------------------------
def scaling_line_chart(rows):
    variants = list(VARIANT_LABELS.keys())
    vdata = {v: get_variant_data(rows, v) for v in variants}

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for v in variants:
        datasets = sorted(vdata[v].keys(), key=dataset_sort_key)
        gcups_means = [time_to_gcups(vdata[v][ds]["mean"], ds) for ds in datasets]
        gcups_errs = [
            time_to_gcups(vdata[v][ds]["mean"], ds) * (vdata[v][ds]["stdev"] / vdata[v][ds]["mean"])
            for ds in datasets
        ]
        ax.errorbar(
            datasets, gcups_means, yerr=gcups_errs, marker="o", capsize=4,
            label=VARIANT_LABELS[v], color=PALETTE[v], linewidth=2.5,
            markersize=7, markeredgecolor="white", markeredgewidth=1.5,
            capthick=1, ecolor=PALETTE[v],
        )
        # Fill between for confidence band
        ax.fill_between(
            datasets,
            [m - s for m, s in zip(gcups_means, gcups_errs)],
            [m + s for m, s in zip(gcups_means, gcups_errs)],
            color=PALETTE[v], alpha=0.08,
        )

    ax.set_ylabel("GCUPS")
    ax.set_xlabel("Dataset")
    ax.legend(loc="upper left", framealpha=0.95)
    ax.set_title("GCUPS Scaling Across Dataset Sizes")
    _strip_spines(ax)
    fig.tight_layout()
    save(fig, "scaling_line_chart.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    _apply_style()
    GRAPHS_DIR.mkdir(exist_ok=True)
    rows = load_data()
    print(f"Loaded {len(rows)} benchmark rows from {BENCHMARKS_CSV}")
    print(f"Generating graphs in {GRAPHS_DIR}/\n")

    boxplot_combined(rows)
    grouped_bar_translated(rows)
    speedup_vs_naive(rows)
    speedup_widened_vs_sse(rows)
    scaling_line_chart(rows)

    print(f"\nDone \u2014 {len(list(GRAPHS_DIR.glob('*.png')))} graphs generated.")


if __name__ == "__main__":
    main()
