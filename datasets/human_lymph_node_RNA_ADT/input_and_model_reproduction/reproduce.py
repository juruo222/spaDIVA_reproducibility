"""Reproduce manuscript Figure 4a-c from versioned panel source data.

The compact workflow redraws the manuscript panels. A separate
``model_reproduction.ipynb`` provides the current release spaDIVA model run,
while baseline panels use the deposited manuscript results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SOURCE_DIR = HERE / "source_data"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "Figure_4"

METHOD_ORDER = ["spaDIVA", "SpaMosaic", "SpatialGlue", "totalVI"]
DOMAIN_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#279e68",
    "#d62728",
    "#aa40fc",
    "#8c564b",
    "#e377c2",
    "#b5bd61",
    "#17becf",
    "#aec7e8",
]
GROUND_TRUTH_ORDER = [
    "capsule",
    "cortex",
    "follicle",
    "hilum",
    "medulla cords",
    "medulla sinuses",
    "medulla vessels",
    "pericapsular adipose tissue",
    "subcapsular sinus",
    "trabeculae",
]
JACCARD_ORDER = ["shared", "shared+ADT", "shared+RNA", "ADT", "RNA"]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def label_sort_key(value: str) -> tuple[int, float | str]:
    try:
        return 0, float(value)
    except ValueError:
        return 1, value


def map_colors(labels: pd.Series, ordered_labels: list[str] | None = None) -> list[str]:
    text = labels.astype(str)
    categories = ordered_labels or sorted(text.unique().tolist(), key=label_sort_key)
    lookup = {category: DOMAIN_COLORS[index % len(DOMAIN_COLORS)] for index, category in enumerate(categories)}
    missing = sorted(set(text) - set(lookup))
    if missing:
        raise ValueError(f"Labels missing from palette: {missing}")
    return [lookup[value] for value in text]


def plot_spatial(ax: mpl.axes.Axes, table: pd.DataFrame, column: str, title: str) -> None:
    order = GROUND_TRUTH_ORDER if column == "ground_truth" else None
    ax.scatter(
        table["x"],
        table["y"],
        c=map_colors(table[column], order),
        s=3.0,
        linewidths=0,
        rasterized=True,
    )
    xpad = 0.02 * np.ptp(table["x"].to_numpy())
    ypad = 0.02 * np.ptp(table["y"].to_numpy())
    ax.set_xlim(table["x"].min() - xpad, table["x"].max() + xpad)
    ax.set_ylim(table["y"].min() - ypad, table["y"].max() + ypad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.set_title(title, pad=2)


def save(fig: mpl.figure.Figure, path: Path) -> None:
    fig.savefig(path, format="pdf", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def panel_a(domains: pd.DataFrame, output: Path) -> None:
    fig = plt.figure(figsize=(7.05, 3.05))
    grid = fig.add_gridspec(
        2,
        6,
        width_ratios=[1, 1, 1, 1, 1, 1.25],
        left=0.035,
        right=0.995,
        bottom=0.02,
        top=0.94,
        wspace=0.16,
        hspace=0.24,
    )
    top = [
        ("spaDIVA", "spaDIVA_integrated"),
        ("SpaMosaic", "SpaMosaic"),
        ("SpatialGlue", "SpatialGlue"),
        ("totalVI", "totalVI"),
        ("ground truth", "ground_truth"),
    ]
    for column, (title, key) in enumerate(top):
        plot_spatial(fig.add_subplot(grid[0, column]), domains, key, title)
    legend_ax = fig.add_subplot(grid[0, 5])
    handles = [
        mpl.lines.Line2D(
            [], [], marker="o", linestyle="none", markersize=3.4, color=DOMAIN_COLORS[index], label=label
        )
        for index, label in enumerate(GROUND_TRUTH_ORDER)
    ]
    legend_ax.legend(handles=handles, loc="center left", frameon=False, handlelength=0.6, handletextpad=0.35)
    legend_ax.axis("off")
    bottom = [
        ("spaDIVA-shared", "spaDIVA_shared"),
        ("spaDIVA-ADT", "spaDIVA_ADT_specific"),
        ("spaDIVA-RNA", "spaDIVA_RNA_specific"),
    ]
    for column, (title, key) in enumerate(bottom):
        plot_spatial(fig.add_subplot(grid[1, column]), domains, key, title)
    fig.text(0.004, 0.985, "a", fontsize=7, fontweight="bold", va="top")
    save(fig, output / "panel_a.pdf")


def panel_b(benchmark: pd.DataFrame, output: Path) -> None:
    colors = ["#E64B35", "#4C78A8", "#5B8CC0", "#2F5D8A"]
    fig, axes = plt.subplots(1, 2, figsize=(4.05, 1.95))
    for ax, metric, ylim in zip(axes, ["NMI", "ARI"], [(0.25, 0.40), (0.12, 0.27)]):
        values = [benchmark.loc[benchmark["method"] == method, metric].dropna().to_numpy() for method in METHOD_ORDER]
        boxes = ax.boxplot(
            values,
            positions=np.arange(1, 5),
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 0.4},
            boxprops={"linewidth": 0.5},
            whiskerprops={"linewidth": 0.5},
            capprops={"linewidth": 0.5},
        )
        for box, color in zip(boxes["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(0.8)
        ax.set_xticks(np.arange(1, 5), METHOD_ORDER, rotation=30, ha="right")
        ax.set_ylabel(metric)
        ax.set_ylim(*ylim)
        ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.11, right=0.995, bottom=0.31, top=0.96, wspace=0.50)
    fig.text(0.004, 0.985, "b", fontsize=7, fontweight="bold", va="top")
    save(fig, output / "panel_b.pdf")


def panel_c(jaccard_summary: pd.DataFrame, output: Path) -> None:
    colors = dict(zip(JACCARD_ORDER, plt.get_cmap("tab10").colors[: len(JACCARD_ORDER)]))
    fig, ax = plt.subplots(figsize=(3.20, 1.82))
    x_adt = np.arange(len(JACCARD_ORDER)) * 0.48
    x_rna = x_adt + len(JACCARD_ORDER) * 0.48 + 0.75
    for modality, positions in [("ADT", x_adt), ("RNA", x_rna)]:
        block = jaccard_summary[jaccard_summary["input_modality"] == modality].set_index("representation")
        for x, representation in zip(positions, JACCARD_ORDER):
            ax.bar(x, block.loc[representation, "mean_jaccard"], width=0.36, color=colors[representation])
    ax.set_xticks([x_adt.mean(), x_rna.mean()], ["ADT", "RNA"])
    ax.set_ylabel("Jaccard similarity")
    ax.spines[["top", "right"]].set_visible(False)
    handles = [mpl.patches.Patch(facecolor=colors[name], label=name) for name in JACCARD_ORDER]
    ax.legend(handles=handles, frameon=False, bbox_to_anchor=(1.01, 0.5), loc="center left")
    fig.subplots_adjust(left=0.17, right=0.70, bottom=0.21, top=0.96)
    fig.text(0.004, 0.985, "c", fontsize=7, fontweight="bold", va="top")
    save(fig, output / "panel_c.pdf")


def validate(domains: pd.DataFrame, benchmark: pd.DataFrame, jaccard: pd.DataFrame) -> dict[str, object]:
    required_columns = {
        "ground_truth",
        "spaDIVA_integrated",
        "spaDIVA_shared",
        "spaDIVA_ADT_specific",
        "spaDIVA_RNA_specific",
        "SpaMosaic",
        "SpatialGlue",
        "totalVI",
    }
    if len(domains) != 3484 or domains[sorted(required_columns)].isna().any().any():
        raise ValueError("Spatial-domain source data are incomplete.")
    counts = benchmark.groupby("method").size().to_dict()
    if counts != {method: 10 for method in METHOD_ORDER}:
        raise ValueError(f"Expected 10 benchmark runs per method, observed {counts}")
    if set(jaccard["k"].astype(int)) != {50} or set(jaccard["n_spots"].astype(int)) != {3484}:
        raise ValueError("Jaccard source data do not match the manuscript k=50 / n=3484 setting.")
    return {
        "n_spots": int(len(domains)),
        "benchmark_runs_per_method": counts,
        "benchmark_medians": {
            method: {
                metric: float(benchmark.loc[benchmark["method"] == method, metric].median())
                for metric in ["ARI", "NMI"]
            }
            for method in METHOD_ORDER
        },
        "jaccard_k": 50,
        "jaccard_distance_metric": sorted(jaccard["distance_metric"].unique().tolist()),
        "jaccard_means": {
            f"{row.input_modality}:{row.representation}": float(row.mean_jaccard)
            for row in jaccard.itertuples()
        },
        "panel_a_spadiva_source": "current release corr graph, KL=[1,1,1,1], 200 epochs, seed 202608017",
        "panel_c_values": "current manuscript bar means",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure_style()

    paths = {
        "spatial_domains": SOURCE_DIR / "spatial_domains.csv",
        "benchmark_runs": SOURCE_DIR / "benchmark_runs.csv",
        "jaccard_original_values": SOURCE_DIR / "figure4c_jaccard_original_values.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing frozen source data:\n" + "\n".join(missing))

    domains = pd.read_csv(paths["spatial_domains"], dtype={"barcode": str})
    benchmark = pd.read_csv(paths["benchmark_runs"], dtype={"seed": str})
    jaccard = pd.read_csv(paths["jaccard_original_values"])
    validation = validate(domains, benchmark, jaccard)

    panel_a(domains, output)
    panel_b(benchmark, output)
    panel_c(jaccard, output)
    outputs = [output / name for name in ["panel_a.pdf", "panel_b.pdf", "panel_c.pdf"]]
    validation["output_sha256"] = {path.name: sha256(path) for path in outputs}
    (output / "model_and_domains_validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "entry_point": Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
        "source_sha256": {name: sha256(path) for name, path in paths.items()},
        "outputs": [path.name for path in outputs],
        "mode": "compact frozen-source reproduction",
    }
    (output / "model_and_domains_run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Reproduced Figure 4a-c in {output}")


if __name__ == "__main__":
    main()
