"""Validate and redraw the five recomputed kNN Jaccard panels.

The script reads the versioned per-spot and summary files stored next to it and
writes generated PDFs to ``results/recomputed_jaccard`` by default. Passing
``--output-dir`` creates the same subdirectory layout elsewhere.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT.parents[1] / "results" / "recomputed_jaccard"
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


@dataclass(frozen=True)
class PanelSpec:
    directory: str
    per_spot_file: str
    summary_file: str
    category_column: str
    output_file: str
    input_order: tuple[str, str]
    representation_order: tuple[str, str, str, str, str]
    representation_labels: tuple[str, str, str, str, str]
    figure_size: tuple[float, float]
    ylim: tuple[float, float]
    yticks: tuple[float, ...]
    panel_letter: str | None = None
    show_legend: bool = False


PANELS = (
    PanelSpec(
        directory="figure_2e",
        per_spot_file="figure2e_knn_jaccard_per_spot.csv.gz",
        summary_file="figure2e_knn_jaccard_summary.csv",
        category_column="representation",
        output_file="panel_e.pdf",
        input_order=("omics1", "omics2"),
        representation_order=("J_shared", "J_shared+W1", "J_shared+W2", "J_W1", "J_W2"),
        representation_labels=("shared", "shared+omics1", "shared+omics2", "omics1", "omics2"),
        figure_size=(3.0941, 1.4228),
        ylim=(0.0, 0.16),
        yticks=(0.0, 0.05, 0.10, 0.15),
        panel_letter="e",
        show_legend=True,
    ),
    PanelSpec(
        directory="figure_3e",
        per_spot_file="figure3e_knn_jaccard_per_spot.csv.gz",
        summary_file="figure3e_knn_jaccard_summary.csv",
        category_column="representation",
        output_file="panel_e.pdf",
        input_order=("ATAC", "RNA"),
        representation_order=("shared", "shared+ATAC", "shared+RNA", "ATAC", "RNA"),
        representation_labels=("shared", "shared+ATAC", "shared+RNA", "ATAC", "RNA"),
        figure_size=(2.2605, 1.3169),
        ylim=(0.0, 0.032),
        yticks=(0.0, 0.01, 0.02, 0.03),
    ),
    PanelSpec(
        directory="figure_4c",
        per_spot_file="figure4c_knn_jaccard_per_spot.csv.gz",
        summary_file="figure4c_knn_jaccard_summary.csv",
        category_column="embedding",
        output_file="panel_c.pdf",
        input_order=("ADT", "RNA"),
        representation_order=("shared", "shared+ADT", "shared+RNA", "ADT", "RNA"),
        representation_labels=("shared", "shared+ADT", "shared+RNA", "ADT", "RNA"),
        figure_size=(3.2422, 1.6390),
        ylim=(0.0, 0.23),
        yticks=(0.0, 0.05, 0.10, 0.15, 0.20),
        panel_letter="c",
        show_legend=True,
    ),
    PanelSpec(
        directory="supplementary_figure_s4d",
        per_spot_file="S4d_E18_knn_jaccard_per_spot.csv.gz",
        summary_file="S4d_E18_knn_jaccard_summary.csv",
        category_column="representation",
        output_file="panel_d.pdf",
        input_order=("ATAC", "RNA"),
        representation_order=("shared", "shared+ATAC", "shared+RNA", "ATAC", "RNA"),
        representation_labels=("shared", "shared+ATAC", "shared+RNA", "ATAC", "RNA"),
        figure_size=(2.1, 1.4),
        ylim=(0.0, 0.165),
        yticks=(0.0, 0.05, 0.10, 0.15),
    ),
    PanelSpec(
        directory="supplementary_figure_s4h",
        per_spot_file="S4h_E15_knn_jaccard_per_spot.csv.gz",
        summary_file="S4h_E15_knn_jaccard_summary.csv",
        category_column="representation",
        output_file="panel_h.pdf",
        input_order=("ATAC", "RNA"),
        representation_order=("shared", "shared+ATAC", "shared+RNA", "ATAC", "RNA"),
        representation_labels=("shared", "shared+ATAC", "shared+RNA", "ATAC", "RNA"),
        figure_size=(2.1, 1.4),
        ylim=(0.0, 0.19),
        yticks=(0.0, 0.05, 0.10, 0.15),
    ),
)


def configure_style() -> None:
    """Apply the compact 7-point style used by the manuscript panels."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_and_validate(spec: PanelSpec) -> pd.DataFrame:
    """Return the saved summary after validating it against per-spot values."""
    panel_dir = ROOT / spec.directory
    per_spot = pd.read_csv(panel_dir / spec.per_spot_file)
    summary = pd.read_csv(panel_dir / spec.summary_file)

    required_per_spot = {"input_modality", spec.category_column, "jaccard", "k", "metric"}
    required_summary = {
        "input_modality",
        spec.category_column,
        "n_spots",
        "mean_jaccard",
        "sd_jaccard",
        "k",
        "metric",
    }
    missing_per_spot = required_per_spot.difference(per_spot.columns)
    missing_summary = required_summary.difference(summary.columns)
    if missing_per_spot or missing_summary:
        raise ValueError(
            f"{spec.directory}: missing columns; per-spot={sorted(missing_per_spot)}, "
            f"summary={sorted(missing_summary)}"
        )

    keys = ["input_modality", spec.category_column]
    recalculated = (
        per_spot.groupby(keys, sort=False, observed=True)["jaccard"]
        .agg(n_spots="count", mean_jaccard="mean", sd_jaccard="std")
        .reset_index()
    )
    merged = summary.merge(
        recalculated,
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_saved", "_recalculated"),
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError(f"{spec.directory}: summary groups do not match per-spot groups")
    if not np.array_equal(
        merged["n_spots_saved"].to_numpy(), merged["n_spots_recalculated"].to_numpy()
    ):
        raise ValueError(f"{spec.directory}: n_spots does not match per-spot data")
    for column in ("mean_jaccard", "sd_jaccard"):
        if not np.allclose(
            merged[f"{column}_saved"],
            merged[f"{column}_recalculated"],
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(f"{spec.directory}: saved {column} does not match per-spot data")

    expected_groups = {
        (input_name, representation)
        for input_name in spec.input_order
        for representation in spec.representation_order
    }
    actual_groups = set(zip(summary["input_modality"], summary[spec.category_column]))
    if actual_groups != expected_groups:
        raise ValueError(f"{spec.directory}: unexpected or missing plotting groups")
    if not summary["k"].eq(50).all() or not summary["metric"].eq("euclidean").all():
        raise ValueError(f"{spec.directory}: expected k=50 and Euclidean distance")

    return summary


def plot_panel(summary: pd.DataFrame, spec: PanelSpec, output_root: Path) -> Path:
    """Draw one grouped-bar panel from its validated saved summary."""
    lookup = summary.set_index(["input_modality", spec.category_column])["mean_jaccard"]
    x = np.arange(len(spec.input_order), dtype=float)
    width = 0.13
    offsets = (np.arange(len(spec.representation_order)) - 2.0) * width

    fig, ax = plt.subplots(figsize=spec.figure_size)
    for index, (representation, label) in enumerate(
        zip(spec.representation_order, spec.representation_labels)
    ):
        values = [float(lookup.loc[(input_name, representation)]) for input_name in spec.input_order]
        ax.bar(x + offsets[index], values, width=width * 0.88, color=COLORS[index], label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(spec.input_order)
    ax.set_ylabel("Jaccard similarity")
    ax.set_ylim(*spec.ylim)
    ax.set_yticks(spec.yticks)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")

    if spec.show_legend:
        ax.legend(
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            handlelength=1.4,
            handletextpad=0.4,
            labelspacing=0.35,
        )
        fig.subplots_adjust(left=0.22, right=0.68, bottom=0.27, top=0.96)
    else:
        fig.subplots_adjust(left=0.29, right=0.98, bottom=0.27, top=0.97)

    if spec.panel_letter:
        fig.text(0.012, 0.98, spec.panel_letter, ha="left", va="top", fontsize=7, fontweight="bold")

    output_path = output_root / spec.directory / spec.output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        format="pdf",
        transparent=True,
        metadata={"Creator": "spaDIVA reproducibility", "Title": spec.directory},
    )
    plt.close(fig)
    if output_path.stat().st_size == 0:
        raise OSError(f"Empty PDF was written: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root (default: results/recomputed_jaccard).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate per-spot data against summaries, then exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_dir.expanduser().resolve()
    configure_style()

    written: list[Path] = []
    for spec in PANELS:
        summary = load_and_validate(spec)
        print(f"validated {spec.directory}: {len(summary)} groups")
        if not args.verify_only:
            written.append(plot_panel(summary, spec, output_root))

    for output_path in written:
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
