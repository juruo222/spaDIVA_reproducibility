"""Reproduce the lightweight simulation panels used in Figure 2 and S1.

The default ``plot`` mode uses versioned compact source tables. Figure 2h also
reads the deposited ADT input for spatial coordinates and the four displayed
expression vectors. Model training is available through
``model/run_spadiva.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

# Keep Matplotlib's font/cache files inside this reproducibility subtree.  This
# avoids relying on a writable user profile on a fresh machine.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / "results" / ".mplconfig"),
)

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.stats import mannwhitneyu, spearmanr


HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "source_data"
DEFAULT_RESULTS = HERE.parents[1] / "results"
FACTOR_ORDER = ["Background", "factor 1", "factor 2", "factor 3", "factor 4"]
FACTOR_COLORS = {
    "Background": "#1f77b4",
    "factor 1": "#ff7f0e",
    "factor 2": "#2ca02c",
    "factor 3": "#d62728",
    "factor 4": "#9467bd",
}


def find_repo_root() -> Path:
    for candidate in (HERE, *HERE.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "spaDIVA").is_dir():
            return candidate
    raise RuntimeError("Could not locate spaDIVA_reproducibility root.")


def resolve_data_root(repo_root: Path, requested: Path | None) -> Path:
    if requested is not None:
        return requested.expanduser().resolve()
    if os.environ.get("SPADIVA_DATA_ROOT"):
        return Path(os.environ["SPADIVA_DATA_ROOT"]).expanduser().resolve()
    return (repo_root / "data").resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_style() -> tuple[str, str]:
    for family in ("Arial", "Liberation Sans", "DejaVu Sans"):
        try:
            font_path = fm.findfont(fm.FontProperties(family=family), fallback_to_default=False)
            break
        except ValueError:
            continue
    else:
        font_path = fm.findfont(fm.FontProperties(family="sans-serif"))
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams.update(
        {
            "font.family": font_name,
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
    return font_name, str(Path(font_path).resolve())


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(width=0.7, length=2.5)


def format_pvalue(value: float) -> str:
    if not np.isfinite(value):
        return "p = NA"
    if value < 1e-300:
        return "p < 1e-300"
    if value < 1e-3:
        return f"p = {value:.1e}"
    return f"p = {value:.3f}"


def validate_source_tables(
    translation: pd.DataFrame,
    ablation: pd.DataFrame,
    representatives: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> None:
    translation_required = {
        "feature_name", "rank_order", "rank_fraction", "unique_factor_ratio",
        "translation_accuracy_totalVI", "true_factor_label", "ratio_group",
        "representative_in_panel_h",
    }
    ablation_required = {"model", "seed", "status", "ARI", "NMI", "epochs"}
    representative_required = {
        "feature_name", "feature_display", "selected_label", "unique_factor_ratio",
        "translation_accuracy_totalVI",
    }
    benchmark_required = {"method", "seed", "ARI"}
    for name, frame, required in (
        ("translation", translation, translation_required),
        ("ablation", ablation, ablation_required),
        ("representatives", representatives, representative_required),
        ("benchmark", benchmark, benchmark_required),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"{name} source table is missing: {sorted(missing)}")
    if len(translation) != 100 or translation["feature_name"].astype(str).duplicated().any():
        raise ValueError("Figure 2 source must contain 100 unique features.")
    if set(representatives["feature_name"].astype(str)) != {"0", "32", "61", "95"}:
        raise ValueError("Figure 2h representative feature set changed.")
    ok = ablation.loc[ablation["status"] == "ok"]
    expected_models = {"spaDIVA", "without_KL", "without_shared_latent_PoE"}
    if set(ok["model"]) != expected_models:
        raise ValueError("S1 ablation model set changed.")
    counts = ok.groupby("model")["seed"].nunique().to_dict()
    if any(counts.get(model) != 10 for model in expected_models):
        raise ValueError(f"S1 requires 10 successful seeds per model, found {counts}.")
    benchmark_methods = {"spaDIVA", "SpaMosaic", "SpatialGlue", "MultiVI"}
    if set(benchmark["method"]) != benchmark_methods:
        raise ValueError("Figure 2c benchmark method set changed.")
    benchmark_counts = benchmark.groupby("method")["seed"].nunique().to_dict()
    if any(benchmark_counts.get(method) != 10 for method in benchmark_methods):
        raise ValueError(
            f"Figure 2c requires 10 runs per method, found {benchmark_counts}."
        )


def plot_panel_c(frame: pd.DataFrame, output: Path) -> dict:
    order = ["spaDIVA", "SpaMosaic", "SpatialGlue", "MultiVI"]
    colors = ["#E64B35", "#4C78A8", "#5B8CC0", "#2F5D8A"]
    values = [frame.loc[frame["method"] == method, "ARI"].to_numpy(float) for method in order]
    fig, ax = plt.subplots(figsize=(1.6, 1.6))
    boxes = ax.boxplot(
        values,
        positions=np.arange(1, len(order) + 1),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        boxprops={"linewidth": 0.4},
        whiskerprops={"linewidth": 0.4},
        capprops={"linewidth": 0.4},
        medianprops={"linewidth": 0.4},
    )
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_xticks(np.arange(1, len(order) + 1), order, rotation=30, ha="right")
    ax.set_ylabel("ARI")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return {
        method: {
            "n": int(len(data)),
            "median_ARI": float(np.median(data)),
            "q1_ARI": float(np.quantile(data, 0.25)),
            "q3_ARI": float(np.quantile(data, 0.75)),
        }
        for method, data in zip(order, values)
    }


def plot_panel_f(frame: pd.DataFrame, output: Path) -> dict:
    frame = frame.sort_values("rank_order").copy()
    rho, pvalue = spearmanr(frame["unique_factor_ratio"], frame["translation_accuracy_totalVI"])
    fig, ax = plt.subplots(figsize=(3.25, 2.05))
    for factor in FACTOR_ORDER:
        part = frame.loc[frame["true_factor_label"] == factor]
        if len(part):
            ax.scatter(
                part["rank_fraction"], part["translation_accuracy_totalVI"],
                s=11, color=FACTOR_COLORS[factor], alpha=0.88, linewidths=0, label=factor,
            )
    x = frame["rank_fraction"].to_numpy(float)
    y = frame["translation_accuracy_totalVI"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    ax.plot(x, slope * x + intercept, color="#B0B0B0", linestyle="--", linewidth=1.0)
    reps = frame.loc[frame["representative_in_panel_h"].astype(str).str.upper() == "TRUE"]
    offsets = {"0": (-0.02, 0.032), "32": (-0.05, -0.055), "61": (-0.05, 0.032), "95": (0.01, -0.035)}
    for row in reps.itertuples():
        feature = str(row.feature_name)
        ax.scatter(
            [row.rank_fraction], [row.translation_accuracy_totalVI], s=28,
            facecolor=FACTOR_COLORS[row.true_factor_label], edgecolor="#555555", linewidth=0.7, zorder=5,
        )
        dx, dy = offsets.get(feature, (0.01, 0.01))
        ax.text(row.rank_fraction + dx, row.translation_accuracy_totalVI + dy, f"feature {feature}", color="#555555")
    ax.text(0.03, 0.06, f"Spearman r = {rho:.3f}\n{format_pvalue(pvalue)}", transform=ax.transAxes, color="#555555")
    ax.set_xticks([])
    ax.set_xlabel("Features ranked by spaDIVA unique factor ratio\nlow private  ->  high private")
    ax.set_ylabel("Translation accuracy (totalVI)")
    style_axes(ax)
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 0.5), loc="center left", handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return {"spearman_r": float(rho), "spearman_p": float(pvalue), "n_features": int(len(frame))}


def plot_panel_g(frame: pd.DataFrame, output: Path) -> dict:
    ordered = frame.sort_values("rank_order")
    n_group = int(np.floor(len(ordered) * 0.20))
    low = ordered.iloc[:n_group]["translation_accuracy_totalVI"].to_numpy(float)
    high = ordered.iloc[-n_group:]["translation_accuracy_totalVI"].to_numpy(float)
    test = mannwhitneyu(low, high, alternative="greater")
    fig, ax = plt.subplots(figsize=(1.75, 1.85))
    boxes = ax.boxplot(
        [low, high], positions=[1, 2], widths=0.55, patch_artist=True, showfliers=False,
        medianprops={"color": "#555555", "linewidth": 0.7},
        whiskerprops={"color": "#555555", "linewidth": 0.6},
        capprops={"color": "#555555", "linewidth": 0.6},
        boxprops={"color": "#555555", "linewidth": 0.6},
    )
    for patch, color in zip(boxes["boxes"], ["#DCEAF7", "#FBE2D5"]):
        patch.set_facecolor(color)
    rng = np.random.default_rng(0)
    ax.scatter(rng.normal(1, 0.035, len(low)), low, s=10, color="#4C78A8", alpha=0.55, linewidths=0)
    ax.scatter(rng.normal(2, 0.035, len(high)), high, s=10, color="#F58518", alpha=0.55, linewidths=0)
    ax.set_xticks([1, 2], ["Low-private\nbottom 20%", "High-private\ntop 20%"])
    ax.set_ylabel("Translation accuracy (totalVI)")
    ax.text(0.97, 0.97, format_pvalue(test.pvalue), transform=ax.transAxes, ha="right", va="top", color="#555555")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    delta = 2 * test.statistic / (len(low) * len(high)) - 1
    return {
        "n_low": int(len(low)), "n_high": int(len(high)),
        "mannwhitney_u": float(test.statistic), "one_sided_p_low_greater_high": float(test.pvalue),
        "cliffs_delta": float(delta), "median_low": float(np.median(low)), "median_high": float(np.median(high)),
    }


def dense(values) -> np.ndarray:
    return values.toarray() if sparse.issparse(values) else np.asarray(values)


def plot_panel_h(adt_path: Path, representatives: pd.DataFrame, output: Path) -> dict:
    adata = ad.read_h5ad(adt_path)
    features = representatives["feature_name"].astype(str).tolist()
    missing = [feature for feature in features if feature not in adata.var_names]
    if missing:
        raise KeyError(f"Representative ADT features are absent: {missing}")
    matrix = dense(adata.layers["counts"] if "counts" in adata.layers else adata.X).astype(float)
    indices = [int(adata.var_names.get_loc(feature)) for feature in features]
    expression = np.log1p(matrix[:, indices])
    vmin, vmax = np.nanpercentile(expression, [1, 99])
    spatial = np.asarray(adata.obsm["spatial"])
    fig, axes = plt.subplots(1, len(features), figsize=(6.25, 1.8), sharex=True, sharey=True)
    for axis, row, values in zip(axes, representatives.itertuples(), expression.T):
        axis.scatter(
            spatial[:, 0], spatial[:, 1], c=values, cmap="viridis", vmin=vmin, vmax=vmax,
            s=7, linewidths=0, rasterized=True,
        )
        axis.set_title(f"{row.selected_label}\n{row.feature_display}", pad=3)
        axis.set_xlabel(
            f"UF ratio = {float(row.unique_factor_ratio):.2f}\n"
            f"TotalVI acc = {float(row.translation_accuracy_totalVI):.2f}", labelpad=3
        )
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(left=0.01, right=0.92, top=0.88, bottom=0.23, wspace=0.08)
    color_axis = fig.add_axes([0.94, 0.27, 0.012, 0.55])
    scalar = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    colorbar = fig.colorbar(scalar, cax=color_axis)
    colorbar.set_label("log1p expression")
    colorbar.outline.set_visible(False)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return {"features": features, "shared_vmin": float(vmin), "shared_vmax": float(vmax), "n_spots": int(adata.n_obs)}


def plot_s1_panel_d(frame: pd.DataFrame, output: Path) -> dict:
    frame = frame.loc[frame["status"] == "ok"].copy()
    order = ["spaDIVA", "without_KL", "without_shared_latent_PoE"]
    labels = ["spaDIVA", "spaDIVA_KL_ablation", "spaDIVA w/o PoE"]
    colors = ["#E64B35", "#4C78A8", "#5B8CC0"]
    values = [frame.loc[frame["model"] == model, "ARI"].to_numpy(float) for model in order]
    fig, ax = plt.subplots(figsize=(2.35, 1.8))
    boxes = ax.boxplot(
        values, positions=np.arange(1, 4), widths=0.55, patch_artist=True, showfliers=False,
        boxprops={"linewidth": 0.6}, whiskerprops={"linewidth": 0.6},
        capprops={"linewidth": 0.6}, medianprops={"linewidth": 0.6, "color": "#F58518"},
    )
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks(np.arange(1, 4), labels, rotation=30, ha="right")
    ax.set_ylabel("ARI")
    ax.set_ylim(0.48, 1.02)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return {
        model: {
            "n": int(len(data)), "median_ARI": float(np.median(data)),
            "q1_ARI": float(np.quantile(data, 0.25)), "q3_ARI": float(np.quantile(data, 0.75)),
        }
        for model, data in zip(order, values)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("plot", "validate"), default="plot")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root()
    data_root = resolve_data_root(repo_root, args.data_root)
    adt_path = data_root / "datasets" / "simulation" / "input_data" / "ADT_100.h5ad"
    if not adt_path.exists():
        raise FileNotFoundError(adt_path)
    translation_path = SOURCE_DIR / "figure2_translation_features.csv"
    ablation_path = SOURCE_DIR / "s1_ablation_benchmark.csv"
    representative_path = SOURCE_DIR / "figure2_representative_features.csv"
    benchmark_path = SOURCE_DIR / "figure2c_ari_benchmark.csv"
    translation = pd.read_csv(translation_path)
    translation["feature_name"] = translation["feature_name"].astype(str)
    ablation = pd.read_csv(ablation_path)
    representatives = pd.read_csv(representative_path, dtype={"feature_name": str})
    benchmark = pd.read_csv(benchmark_path)
    validate_source_tables(translation, ablation, representatives, benchmark)

    results_root = args.results_root.expanduser().resolve()
    figure2_dir = results_root / "Figure_2"
    s1_dir = results_root / "Supplementary_Figure_S1"
    figure2_dir.mkdir(parents=True, exist_ok=True)
    s1_dir.mkdir(parents=True, exist_ok=True)
    font_name, font_path = configure_style()

    validation = {
        "source_rows": {
            "figure2": len(translation),
            "figure2c_benchmark": len(benchmark),
            "s1": len(ablation),
            "representatives": len(representatives),
        },
        "source_sha256": {
            translation_path.relative_to(HERE).as_posix(): sha256(translation_path),
            ablation_path.relative_to(HERE).as_posix(): sha256(ablation_path),
            representative_path.relative_to(HERE).as_posix(): sha256(representative_path),
            benchmark_path.relative_to(HERE).as_posix(): sha256(benchmark_path),
        },
    }
    outputs: list[Path] = []
    if args.mode == "plot":
        panel_c = figure2_dir / "panel_c.pdf"
        panel_f = figure2_dir / "panel_f.pdf"
        panel_g = figure2_dir / "panel_g.pdf"
        panel_h = figure2_dir / "panel_h.pdf"
        panel_s1d = s1_dir / "panel_d.pdf"
        validation["Figure_2c"] = plot_panel_c(benchmark, panel_c)
        validation["Figure_2f"] = plot_panel_f(translation, panel_f)
        validation["Figure_2g"] = plot_panel_g(translation, panel_g)
        validation["Figure_2h"] = plot_panel_h(adt_path, representatives, panel_h)
        validation["Supplementary_Figure_S1d"] = plot_s1_panel_d(ablation, panel_s1d)
        outputs = [panel_c, panel_f, panel_g, panel_h, panel_s1d]

    figure_validation = {key: value for key, value in validation.items() if key.startswith("Figure_2")}
    s1_validation = {key: value for key, value in validation.items() if key.startswith("Supplementary_Figure_S1")}
    (figure2_dir / "validation.json").write_text(json.dumps(figure_validation, indent=2), encoding="utf-8")
    (s1_dir / "validation.json").write_text(json.dumps(s1_validation, indent=2), encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "entry_point": Path(__file__).resolve().relative_to(repo_root).as_posix(),
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            "anndata": importlib.metadata.version("anndata"), "matplotlib": matplotlib.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        },
        "font": {"name": font_name, "path": font_path, "size_pt": 7},
        "data_input": {
            "path_within_data_root": adt_path.relative_to(data_root).as_posix(),
            "sha256": sha256(adt_path),
        },
        "source_sha256": validation["source_sha256"],
        "outputs": {
            path.relative_to(results_root).as_posix(): sha256(path)
            for path in outputs
        },
        "validation": validation,
    }
    (figure2_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Validated source data and wrote results under {results_root}")


if __name__ == "__main__":
    main()
