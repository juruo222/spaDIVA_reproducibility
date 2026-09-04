"""Reproduce the quantitative and spatial-domain panels of Figure 5.

This lightweight workflow reads the frozen per-spot labels and metric tables
used for the manuscript. All panels are regenerated with the same plotting
code and a 7 pt base font.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
E13_SOURCE = REPO_ROOT / "datasets" / "E13E15E18_multi_slice" / "source_data"
P21_SOURCE = REPO_ROOT / "datasets" / "P21P22_multi_slice" / "source_data"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def categorical_colors(labels: pd.Series) -> dict[str, str]:
    levels = sorted(labels.astype(str).unique(), key=lambda value: (len(value), value))
    cmap = mpl.colormaps["tab20"]
    return {level: to_hex(cmap(index % 20)) for index, level in enumerate(levels)}


def plot_spatial_grid(
    table: pd.DataFrame,
    methods: list[str],
    batches: list[str],
    batch_labels: list[str],
    output: Path,
) -> None:
    figure, axes = plt.subplots(
        len(batches), len(methods), figsize=(1.45 * len(methods), 1.4 * len(batches))
    )
    axes = np.asarray(axes).reshape(len(batches), len(methods))
    for column, method in enumerate(methods):
        method_table = table[table["method"] == method].copy()
        palette = categorical_colors(method_table["cluster"])
        for row, (batch, batch_label) in enumerate(zip(batches, batch_labels)):
            axis = axes[row, column]
            subset = method_table[method_table["batch"].astype(str) == str(batch)]
            axis.scatter(
                subset["spatial_x"],
                subset["spatial_y"],
                c=subset["cluster"].astype(str).map(palette),
                s=1.8,
                linewidths=0,
                rasterized=True,
            )
            axis.set_aspect("equal", adjustable="box")
            axis.invert_yaxis()
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.7)
            if row == 0:
                axis.set_title(method, pad=2)
            if column == 0:
                axis.set_ylabel(batch_label, rotation=0, ha="right", va="center")
    figure.tight_layout(pad=0.35, w_pad=0.35, h_pad=0.35)
    figure.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(figure)


def colored_boxplot(
    axis: plt.Axes,
    groups: list[np.ndarray],
    labels: list[str],
    colors: list[str],
) -> None:
    artists = axis.boxplot(groups, patch_artist=True, showfliers=True, widths=0.62)
    for patch, color in zip(artists["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_linewidth(0.7)
    for key in ("whiskers", "caps", "medians"):
        for artist in artists[key]:
            artist.set_linewidth(0.7)
            artist.set_color("#555555")
    for flier in artists["fliers"]:
        flier.set_markersize(2)
        flier.set_alpha(0.6)
    axis.set_xticks(range(1, len(labels) + 1), labels, rotation=28, ha="right")
    axis.spines[["top", "right"]].set_visible(False)


def plot_e13_ari(table: pd.DataFrame, output: Path) -> None:
    stages = ["E13", "E15", "E18"]
    methods = ["spaDIVA", "SpaMosaic", "MultiVI", "MIDAS"]
    colors = ["#E76F51", "#A8B5C4", "#7895B2", "#6B7C93"]
    figure, axes = plt.subplots(1, 4, figsize=(7.1, 1.8), sharey=True)
    for axis, stage in zip(axes[:3], stages):
        groups = [
            table.loc[(table["stage"] == stage) & (table["method"] == method), "ARI"].to_numpy()
            for method in methods
        ]
        colored_boxplot(axis, groups, methods, colors)
        axis.set_title(stage)
    mean_rows = []
    for method in methods:
        pivot = table[table["method"] == method].pivot(index="seed", columns="stage", values="ARI")
        mean_rows.append(pivot[stages].mean(axis=1).to_numpy())
    colored_boxplot(axes[3], mean_rows, methods, colors)
    axes[3].set_title("mean")
    axes[0].set_ylabel("ARI")
    figure.tight_layout(pad=0.35, w_pad=0.45)
    figure.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(figure)


def plot_jaccard(table: pd.DataFrame, batches: list[str], output: Path) -> None:
    representations = ["shared", "shared+W_ATAC", "shared+W_RNA", "W_ATAC", "W_RNA"]
    display = {
        "shared": "shared",
        "shared+W_ATAC": "shared+ATAC",
        "shared+W_RNA": "shared+RNA",
        "W_ATAC": "ATAC",
        "W_RNA": "RNA",
    }
    colors = dict(zip(representations, ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD"]))
    figure, axes = plt.subplots(1, 2, figsize=(3.45, 1.55), sharey=False)
    width = 0.13
    x = np.arange(len(batches), dtype=float)
    for axis, omics in zip(axes, ("ATAC", "RNA")):
        for index, representation in enumerate(representations):
            values = []
            for batch in batches:
                row = table[
                    (table["slice"] == batch)
                    & (table["omics"] == omics)
                    & (table["representation"] == representation)
                ]
                values.append(float(row["jaccard"].iloc[0]))
            axis.bar(
                x + (index - 2) * width,
                values,
                width=width,
                color=colors[representation],
                label=display[representation],
            )
        axis.set_xticks(x, batches, rotation=30, ha="right")
        axis.set_title(omics)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Jaccard similarity")
    axes[1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout(pad=0.35, w_pad=0.8)
    figure.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(figure)


def plot_moran(table: pd.DataFrame, output: Path) -> None:
    methods = ["spaDIVA", "spamosaic", "multiVI"]
    labels = ["spaDIVA", "SpaMosaic", "MultiVI"]
    colors = ["#E76F51", "#A8B5C4", "#7895B2"]
    groups = [table.loc[table["method"] == method, "I"].to_numpy() for method in methods]
    figure, axis = plt.subplots(figsize=(2.0, 1.65))
    colored_boxplot(axis, groups, labels, colors)
    axis.set_ylabel("Moran's I score")
    figure.tight_layout(pad=0.35)
    figure.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(figure)


def plot_single_joint(table: pd.DataFrame, output: Path) -> None:
    stages = ["E13", "E15", "E18"]
    colors = {"Single": "#B8B8B8", "Joint": "#E76F51"}
    figure, axis = plt.subplots(figsize=(3.0, 1.65))
    positions: list[float] = []
    groups: list[np.ndarray] = []
    group_colors: list[str] = []
    for stage_index, stage in enumerate(stages):
        for offset, condition in ((-0.18, "Single"), (0.18, "Joint")):
            positions.append(stage_index + offset)
            groups.append(
                table.loc[(table["stage"] == stage) & (table["condition"] == condition), "ARI"].to_numpy()
            )
            group_colors.append(colors[condition])
    artists = axis.boxplot(groups, positions=positions, widths=0.3, patch_artist=True, showfliers=True)
    for patch, color in zip(artists["boxes"], group_colors):
        patch.set_facecolor(color)
        patch.set_linewidth(0.7)
    for key in ("whiskers", "caps", "medians"):
        for artist in artists[key]:
            artist.set_linewidth(0.7)
            artist.set_color("#555555")
    axis.set_xticks(range(len(stages)), stages)
    axis.set_ylabel("ARI")
    axis.spines[["top", "right"]].set_visible(False)
    handles = [mpl.patches.Patch(color=colors[name], label=name) for name in ("Single", "Joint")]
    axis.legend(handles=handles, frameon=False, loc="best")
    figure.tight_layout(pad=0.35)
    figure.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "results" / "Figure_5"
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure_style()

    e13_spatial = pd.read_csv(E13_SOURCE / "spatial_domains.csv")
    p21_spatial = pd.read_csv(P21_SOURCE / "spatial_domains.csv")
    e13_ari = pd.read_csv(E13_SOURCE / "ari_benchmark.csv")
    e13_jaccard = pd.read_csv(E13_SOURCE / "jaccard_by_section.csv")
    p21_jaccard = pd.read_csv(P21_SOURCE / "jaccard_by_section.csv")
    p21_moran = pd.read_csv(P21_SOURCE / "moran_by_domain.csv")
    single_joint = pd.read_csv(E13_SOURCE / "single_vs_joint_ari.csv")

    plot_spatial_grid(
        e13_spatial,
        ["spaDIVA", "SpaMosaic", "MultiVI", "MIDAS"],
        ["0", "1", "2"],
        ["E13", "E15", "E18"],
        output / "panel_b_spatial_domains.pdf",
    )
    plot_e13_ari(e13_ari, output / "panel_c_ari.pdf")
    plot_jaccard(e13_jaccard, ["E13", "E15", "E18"], output / "panel_d_jaccard.pdf")
    plot_spatial_grid(
        p21_spatial,
        ["spaDIVA", "SpaMosaic", "MultiVI"],
        ["0", "1", "2"],
        ["section1", "section2", "section3"],
        output / "panel_f_spatial_domains.pdf",
    )
    plot_moran(p21_moran, output / "panel_g_moran.pdf")
    plot_jaccard(p21_jaccard, ["section1", "section2", "section3"], output / "panel_h_jaccard.pdf")
    plot_single_joint(single_joint, output / "supplementary_S4_single_vs_joint_ari.pdf")

    validation = {
        "E13_spatial_rows": int(len(e13_spatial)),
        "P21P22_spatial_rows": int(len(p21_spatial)),
        "E13_ARI_rows": int(len(e13_ari)),
        "P21P22_Moran_rows": int(len(p21_moran)),
        "E13_Jaccard_rows": int(len(e13_jaccard)),
        "P21P22_Jaccard_rows": int(len(p21_jaccard)),
        "single_joint_rows": int(len(single_joint)),
        "base_font_pt": 7,
        "baseline_policy": "versioned manuscript source data",
    }
    (output / "validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    source_paths = sorted([*E13_SOURCE.glob("*.csv"), *P21_SOURCE.glob("*.csv")])
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": "Figure 5 and quantitative Figure S4 panel reproduction",
        "sources": [
            {
                "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in source_paths
        ],
        "outputs": sorted(path.name for path in output.glob("*.pdf")),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Saved Figure 5/S4 panels to {output}")


if __name__ == "__main__":
    main()
