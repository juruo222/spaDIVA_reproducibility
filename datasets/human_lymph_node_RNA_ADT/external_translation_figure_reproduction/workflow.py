"""Reproduce the lymph-node translation figure panels.

The deposited ratio table supplies the manuscript UF statistics and records
``z_source=Z_poe``. Zenodo inputs recreate the expression, spatial, and
RNA--ADT concordance panels, while the archived totalVI prediction provides an
independent validation of translation accuracy.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

# Must be set before optional totalVI/scikit-learn imports on Windows.
if sys.platform == "win32":
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import anndata as ad
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.font_manager import findfont
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import mannwhitneyu, pearsonr, spearmanr


EXPECTED_N_FEATURES = 31
EXPECTED_SPEARMAN_R = -0.4903225806451614
EXPECTED_SPEARMAN_P = 0.00510626603450084
EXPECTED_MANNWHITNEY_P = 0.012987012987012988
EXPECTED_HIGH_MATCHES = ["SDC1", "CD274", "CD19", "KRT5", "CEACAM8", "EPCAM"]
EXPECTED_LOW_MATCHES = ["VIM", "CD3E", "BCL2", "ACTA2", "CCR7", "CD4"]
PANEL_F_HIGH = ["CD19", "EPCAM", "CEACAM8"]
PANEL_F_LOW = ["ACTA2", "BCL2", "CCR7"]

EXTERNAL_METRIC = "totalvi_feature_pearson"
RATIO_COLUMN = "unique_factor_ratio"
SPLIT_COUNTS = {"train": 2438, "val": 349, "test": 697}
FONT_SIZE = 7
SPOT_SIZE = 3.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dense(matrix, dtype=np.float32):
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=dtype)


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def configure_plot_style() -> str:
    for family in ("Arial", "Liberation Sans", "DejaVu Sans"):
        try:
            findfont(family, fallback_to_default=False)
            break
        except ValueError:
            continue
    else:
        family = "DejaVu Sans"

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [family],
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    return family


def load_split_table(path: Path) -> tuple[pd.DataFrame, dict[str, pd.Index]]:
    table = pd.read_csv(path)
    required = {"obs_name", "split", "original_index"}
    missing = required.difference(table.columns)
    if missing:
        raise KeyError(f"Split table is missing columns: {sorted(missing)}")

    table = table.loc[:, ["obs_name", "split", "original_index"]].copy()
    table["obs_name"] = table["obs_name"].astype(str)
    table["split"] = table["split"].astype(str)
    table["original_index"] = table["original_index"].astype(int)
    if table["obs_name"].duplicated().any():
        raise ValueError("Split table contains duplicate observation names.")
    if table["original_index"].duplicated().any():
        raise ValueError("Split table contains duplicate original indices.")

    counts = table["split"].value_counts().to_dict()
    if counts != SPLIT_COUNTS:
        raise ValueError(f"Unexpected split counts: {counts}; expected {SPLIT_COUNTS}.")

    names = {}
    for label in ("train", "val", "test"):
        part = table.loc[table["split"] == label].sort_values("original_index")
        names[label] = pd.Index(part["obs_name"].tolist(), dtype="object")
    return table.sort_values("original_index").reset_index(drop=True), names


def _validate_input_observations(
    adata_adt: ad.AnnData,
    adata_rna: ad.AnnData,
    split_table: pd.DataFrame,
) -> None:
    expected = set(split_table["obs_name"])
    for label, obj in (("ADT", adata_adt), ("RNA", adata_rna)):
        observed = set(obj.obs_names.astype(str))
        if observed != expected:
            missing = sorted(expected - observed)[:5]
            extra = sorted(observed - expected)[:5]
            raise ValueError(
                f"{label} observations disagree with the fixed split; "
                f"missing={missing}, extra={extra}."
            )


def _read_inputs(
    data_root: Path,
    artifact_root: Path,
) -> tuple[ad.AnnData, ad.AnnData, pd.DataFrame, dict[str, pd.Index], Path]:
    dataset_root = data_root / "datasets" / "human_lymph_node_RNA_ADT"
    adt_path = dataset_root / "input_data" / "human lymph node adata_ADT.h5ad"
    rna_path = dataset_root / "input_data" / "human lymph node adata_RNA.h5ad"
    annotation_path = dataset_root / "annotation" / "annotation.csv"
    split_path = artifact_root / "artifacts" / "split_assignments.csv"

    for path in (adt_path, rna_path, annotation_path, split_path):
        if not path.exists():
            raise FileNotFoundError(path)

    adata_adt = ad.read_h5ad(adt_path)
    adata_rna = ad.read_h5ad(rna_path)
    for obj in (adata_adt, adata_rna):
        obj.obs_names = obj.obs_names.astype(str)
        obj.var_names_make_unique()
        obj.var_names = obj.var_names.astype(str)

    split_table, split_names = load_split_table(split_path)
    _validate_input_observations(adata_adt, adata_rna, split_table)

    annotation = pd.read_csv(annotation_path)
    required_annotation = {"Barcode", "manual-anno"}
    missing = required_annotation.difference(annotation.columns)
    if missing:
        raise KeyError(f"Annotation is missing columns: {sorted(missing)}")
    annotation["Barcode"] = annotation["Barcode"].astype(str)
    annotation = annotation.set_index("Barcode")
    if not set(adata_adt.obs_names).issubset(annotation.index):
        raise ValueError("Annotation does not cover all ADT spots.")

    return adata_adt, adata_rna, annotation, split_names, dataset_root


def _assert_fixed_split(obs_names, split_names, split_indices, *, stage: str) -> None:
    """Require ordered names, disjoint integer indices, and full coverage."""
    labels = ("train", "val", "test")
    names = pd.Index(obs_names).astype(str)
    if not names.is_unique:
        raise ValueError(f"{stage}: model observation names are not unique.")
    if set(split_names) != set(labels) or len(split_indices) != 3:
        raise ValueError(f"{stage}: exactly train, val, and test splits are required.")

    arrays = []
    for label, indices in zip(labels, split_indices):
        expected = pd.Index(split_names[label]).astype(str)
        indices = np.asarray(indices)
        if not expected.is_unique or expected.empty:
            raise ValueError(f"{stage}: {label} names must be unique and nonempty.")
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"{stage}: {label} indices must be a 1D integer array.")
        if np.any(indices < 0) or np.any(indices >= len(names)):
            raise ValueError(f"{stage}: {label} indices are outside model observations.")
        if not np.array_equal(names.to_numpy()[indices], expected.to_numpy()):
            raise ValueError(f"{stage}: {label} observation names/order differ from the fixed split.")
        arrays.append(indices)

    combined = np.concatenate(arrays)
    if len(combined) != len(names) or not np.array_equal(
        np.sort(combined), np.arange(len(names))
    ):
        raise ValueError(f"{stage}: splits must be disjoint and cover every observation once.")


def _prepare_fixed_split(model, split_names, *, batch_size: int) -> list[np.ndarray]:
    """Check the actual scvi splitter before any optimizer step can run."""
    names = pd.Index(model.adata_manager.adata.obs_names).astype(str)
    if not names.is_unique:
        raise ValueError("Model observation names are not unique.")
    if set(split_names) != {"train", "val", "test"}:
        raise ValueError("Exactly train, val, and test splits are required.")
    indices = [
        names.get_indexer(pd.Index(split_names[label]).astype(str)).astype(np.int64)
        for label in ("train", "val", "test")
    ]
    _assert_fixed_split(names, split_names, indices, stage="Before training")
    splitter = model._data_splitter_cls(
        model.adata_manager,
        external_indexing=[part.copy() for part in indices],
        shuffle_set_split=False,
        batch_size=batch_size,
    )
    splitter.setup()
    _assert_fixed_split(
        names,
        split_names,
        [splitter.train_idx, splitter.val_idx, splitter.test_idx],
        stage="scvi pre-training split",
    )
    return indices


def retrain_totalvi(
    adata_adt: ad.AnnData,
    adata_rna: ad.AnnData,
    split_names: dict[str, pd.Index],
    output_path: Path,
    device: str = "auto",
) -> Path:
    """Retrain totalVI on the fixed name-based split.

    This optional path uses a fixed 100 epochs and validates the
    actual scvi train/validation/test observation names before training and
    again before prediction.
    """

    # Set this before importing scvi/sklearn/joblib to bypass joblib's
    # physical-core helper probe.
    if sys.platform == "win32":
        os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

    try:
        import scvi
        import torch
        from mudata import MuData
    except ImportError as exc:
        raise RuntimeError(
            "--retrain-totalvi requires the external-translation environment "
            "with scvi-tools, torch and mudata."
        ) from exc

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    scvi.settings.seed = seed

    train_names = split_names["train"]
    val_names = split_names["val"]
    test_names = split_names["test"]
    ordered_names = train_names.append(val_names).append(test_names)

    rna = adata_rna[ordered_names].copy()
    adt_true = adata_adt[ordered_names].copy()
    rna.layers["counts"] = rna.X.copy()
    adt_true.layers["counts"] = adt_true.X.copy()

    n_test = len(test_names)
    adt_matrix = _dense(adt_true.layers["counts"])
    adt_matrix[-n_test:, :] = 0.0
    adt_masked = ad.AnnData(
        X=adt_matrix.copy(),
        obs=adt_true.obs.copy(),
        var=adt_true.var.copy(),
    )
    adt_masked.layers["counts"] = adt_matrix
    if "spatial" in adt_true.obsm:
        adt_masked.obsm["spatial"] = adt_true.obsm["spatial"].copy()

    if sparse.issparse(rna.layers["counts"]):
        rna.layers["counts"] = rna.layers["counts"].tocsr().astype(np.float32)
    else:
        rna.layers["counts"] = np.asarray(rna.layers["counts"], dtype=np.float32)
    rna.X = rna.X.astype(np.float32)

    split_labels = (
        ["train"] * len(train_names)
        + ["val"] * len(val_names)
        + ["test"] * len(test_names)
    )
    modality_labels = (
        ["paired"] * (len(train_names) + len(val_names))
        + ["expression"] * len(test_names)
    )
    for obj in (rna, adt_masked):
        obj.obs["split"] = split_labels
        obj.obs["modality"] = modality_labels

    mdata = MuData({"rna": rna, "adt": adt_masked})
    mdata.obs["split"] = split_labels
    mdata.obs["modality"] = modality_labels
    mdata.update()

    scvi.model.TOTALVI.setup_mudata(
        mdata,
        rna_layer="counts",
        protein_layer="counts",
        batch_key="modality",
        modalities={"rna_layer": "rna", "protein_layer": "adt"},
    )
    model = scvi.model.TOTALVI(mdata)

    cuda_available = torch.cuda.is_available()
    cuda_build_compatible = cuda_available
    if cuda_available and hasattr(torch.cuda, "get_arch_list"):
        major, minor = torch.cuda.get_device_capability()
        target_arches = {f"sm_{major}{minor}", f"compute_{major}{minor}"}
        build_arches = set(torch.cuda.get_arch_list())
        cuda_build_compatible = not build_arches or bool(target_arches & build_arches)

    if device == "auto":
        accelerator = "gpu" if cuda_available and cuda_build_compatible else "cpu"
        if cuda_available and not cuda_build_compatible:
            print(
                "CUDA is visible, but this PyTorch build does not support the "
                "GPU compute capability; falling back to CPU."
            )
    elif device == "cuda":
        if not cuda_available:
            raise RuntimeError("--device cuda was requested but CUDA is unavailable.")
        if not cuda_build_compatible:
            raise RuntimeError(
                "--device cuda was requested, but this PyTorch build does not "
                "support the detected GPU compute capability."
            )
        accelerator = "gpu"
    else:
        accelerator = "cpu"

    split_indices = _prepare_fixed_split(model, split_names, batch_size=256)
    model.train(
        max_epochs=100,
        accelerator=accelerator,
        devices=1,
        batch_size=256,
        external_indexing=split_indices,
        shuffle_set_split=False,
        early_stopping=False,
        check_val_every_n_epoch=1,
    )

    _assert_fixed_split(
        model.adata_manager.adata.obs_names,
        split_names,
        [model.train_indices, model.validation_indices, model.test_indices],
        stage="totalVI after training",
    )

    _, predicted_adt = model.get_normalized_expression(
        indices=model.test_indices,
        return_numpy=True,
    )
    prediction = ad.AnnData(
        X=np.asarray(predicted_adt, dtype=np.float32),
        obs=adata_rna[test_names].obs.copy(),
        var=adata_adt.var.copy(),
    )
    if "spatial" in adata_rna.obsm:
        prediction.obsm["spatial"] = adata_rna[test_names].obsm["spatial"].copy()
    prediction.uns["totalvi_seed"] = seed
    prediction.uns["totalvi_epochs"] = 100
    prediction.uns["totalvi_accelerator"] = accelerator
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction.write_h5ad(output_path)
    return output_path


def _featurewise_pearson(
    truth: ad.AnnData,
    prediction: ad.AnnData,
    features: list[str],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for feature in features:
        if feature not in truth.var_names or feature not in prediction.var_names:
            values[feature] = np.nan
            continue
        observed = _dense(truth[:, [feature]].X, dtype=np.float64).reshape(-1)
        estimated = _dense(prediction[:, [feature]].X, dtype=np.float64).reshape(-1)
        if np.std(observed) == 0 or np.std(estimated) == 0:
            values[feature] = np.nan
        else:
            values[feature] = float(pearsonr(observed, estimated).statistic)
    return values


def compute_analysis(
    adata_adt: ad.AnnData,
    adata_rna: ad.AnnData,
    split_names: dict[str, pd.Index],
    prediction_path: Path,
    source_path: Path,
    use_frozen_translation: bool,
) -> tuple[pd.DataFrame, float]:
    """Load deposited UF ratios and independently validate translation.

    The archived totalVI prediction and Zenodo ADT input reproduce the
    translation-accuracy column. The maximum difference from the deposited
    values is returned for validation.
    """

    result = pd.read_csv(source_path)
    required = {
        "feature_name",
        "unique_factor_ratio",
        "translation_accuracy_totalVI",
        "rna_adt_pearson",
    }
    missing = required.difference(result.columns)
    if missing:
        raise KeyError(f"Manuscript Source Data are missing columns: {sorted(missing)}")
    if result["feature_name"].duplicated().any():
        raise ValueError("Manuscript Source Data contain duplicate feature names.")
    result = result.rename(columns={"translation_accuracy_totalVI": EXTERNAL_METRIC})
    result["feature_name"] = result["feature_name"].astype(str)
    result["z_source"] = "Z_poe"

    prediction = ad.read_h5ad(prediction_path)
    prediction.obs_names = prediction.obs_names.astype(str)
    prediction.var_names_make_unique()
    prediction.var_names = prediction.var_names.astype(str)

    test_names = split_names["test"]
    if set(prediction.obs_names) != set(test_names):
        raise ValueError("totalVI prediction observations do not match the fixed test split.")
    prediction = prediction[test_names].copy()
    test_truth = adata_adt[test_names].copy()
    recomputed_translation = _featurewise_pearson(
        test_truth,
        prediction,
        result["feature_name"].tolist(),
    )
    recomputed = result["feature_name"].map(recomputed_translation).to_numpy(float)
    frozen = result[EXTERNAL_METRIC].to_numpy(float)
    translation_error = float(np.nanmax(np.abs(recomputed - frozen)))
    result["recomputed_totalvi_feature_pearson"] = recomputed
    if not use_frozen_translation:
        result[EXTERNAL_METRIC] = recomputed

    common_names = adata_adt.obs_names.intersection(adata_rna.obs_names)
    adt_aligned = adata_adt[common_names]
    rna_aligned = adata_rna[common_names]
    rna_aligned = rna_aligned[adt_aligned.obs_names]
    correlations = {}
    for protein in result["feature_name"].astype(str):
        if protein not in adata_adt.var_names or protein not in adata_rna.var_names:
            correlations[protein] = np.nan
            continue
        adt_values = np.log1p(_dense(adt_aligned[:, [protein]].X).reshape(-1))
        rna_values = np.log1p(_dense(rna_aligned[:, [protein]].X).reshape(-1))
        if np.std(adt_values) == 0 or np.std(rna_values) == 0:
            correlations[protein] = np.nan
        else:
            correlations[protein] = float(pearsonr(adt_values, rna_values).statistic)
    result["rna_adt_pearson"] = result["feature_name"].map(correlations)
    return result.sort_values(RATIO_COLUMN).reset_index(drop=True), translation_error


def select_exact_matches(
    result: pd.DataFrame,
    adata_adt: ad.AnnData,
    adata_rna: ad.AnnData,
    ascending: bool,
    n: int = 6,
) -> list[str]:
    selected = []
    ranked = result.sort_values(RATIO_COLUMN, ascending=ascending)
    for feature in ranked["feature_name"].astype(str):
        if feature in adata_adt.var_names and feature in adata_rna.var_names:
            selected.append(feature)
        if len(selected) == n:
            break
    if len(selected) != n:
        raise ValueError(f"Only {len(selected)} exact RNA/ADT matches were found; expected {n}.")
    return selected


def _feature_values(adata: ad.AnnData, feature: str, log1p: bool = True) -> np.ndarray:
    values = _dense(adata[:, [feature]].X, dtype=np.float64).reshape(-1)
    return np.log1p(values) if log1p else values


def _linear_trend(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    slope, intercept = np.polyfit(x, y, 1)
    return slope * x + intercept


def plot_panel_d(
    result: pd.DataFrame,
    output_path: Path,
    label_low: list[str],
    label_high: list[str],
) -> dict:
    plot_df = result.dropna(subset=[RATIO_COLUMN, EXTERNAL_METRIC]).copy()
    plot_df = plot_df.sort_values(RATIO_COLUMN).reset_index(drop=True)
    x = np.linspace(0, 1, len(plot_df))
    y = plot_df[EXTERNAL_METRIC].to_numpy(float)
    rho, pvalue = spearmanr(plot_df[RATIO_COLUMN], y)
    n_each = max(1, int(np.floor(len(plot_df) * 0.2)))
    n_each = min(n_each, len(plot_df) // 2)
    low = plot_df.head(n_each)[EXTERNAL_METRIC].to_numpy(float)
    high = plot_df.tail(n_each)[EXTERNAL_METRIC].to_numpy(float)
    mannwhitney_p = float(mannwhitneyu(low, high, alternative="greater").pvalue)

    fig = plt.figure(figsize=(5.1, 2.15))
    grid = fig.add_gridspec(1, 2, width_ratios=[2.45, 1.5], wspace=0.52)
    ax = fig.add_subplot(grid[0, 0])
    ax.scatter(x, y, s=12, color="#B7C7DD", alpha=0.75, linewidths=0)
    ax.plot(x, _linear_trend(x, y), "--", color="#BDBDBD", linewidth=1.0)
    low_color, high_color = "#4C78A8", "#F58518"
    label_offsets = {
        "ACTA2": (3, 5),
        "BCL2": (3, 7),
        "CCR7": (5, -5),
        "CD19": (-3, 3),
        "EPCAM": (-3, 4),
        "CEACAM8": (-3, 3),
    }
    for names, color in ((label_low, low_color), (label_high, high_color)):
        for feature in names:
            row_idx = plot_df.index[plot_df["feature_name"].astype(str) == feature]
            if len(row_idx) != 1:
                continue
            idx = int(row_idx[0])
            ax.scatter(
                [x[idx]], [y[idx]], s=34, facecolors="none", edgecolors=color,
                linewidths=0.8, zorder=4,
            )
            dx, dy = label_offsets.get(feature, (-3 if x[idx] > 0.62 else 3, 4))
            ax.annotate(
                feature,
                (x[idx], y[idx]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="right" if dx < 0 else "left",
                va="bottom" if dy >= 0 else "top",
            )
    ax.set_xticks([])
    ax.set_xlabel("Proteins ranked by spaDIVA unique factor ratio\nlow private  →  high private")
    ax.set_ylabel("Translation accuracy (totalVI)")
    ax.text(
        0.03, 0.05, f"Spearman r = {rho:.3f}\np = {pvalue:.3g}",
        transform=ax.transAxes, ha="left", va="bottom",
    )
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[0, 1])
    box = ax.boxplot(
        [low, high],
        tick_labels=["Low-private\nbottom 20%", "High-private\ntop 20%"],
        widths=0.55,
        showfliers=False,
        patch_artist=True,
    )
    for patch, color in zip(box["boxes"], ("#D7E2F0", "#F8DCC7")):
        patch.set_facecolor(color)
        patch.set_edgecolor("#777777")
    rng = np.random.default_rng(0)
    for position, (values, color) in enumerate(((low, low_color), (high, high_color)), 1):
        ax.scatter(
            rng.normal(position, 0.035, len(values)), values,
            s=13, color=color, alpha=0.65, edgecolors="none",
        )
    ax.set_ylabel("Translation accuracy (totalVI)")
    ax.margins(y=0.12)
    ax.text(
        0.98, 0.96, f"p = {mannwhitney_p:.3g}",
        transform=ax.transAxes, ha="right", va="top",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "spearman_r": float(rho),
        "spearman_p": float(pvalue),
        "mannwhitney_p": mannwhitney_p,
        "n_low": int(n_each),
        "n_high": int(n_each),
    }


def _annotation_groups(
    adata_adt: ad.AnnData,
    annotation: pd.DataFrame,
    min_size: int = 80,
) -> tuple[pd.Series, list[str]]:
    labels = annotation.loc[adata_adt.obs_names, "manual-anno"].astype(str)
    counts = labels.value_counts()
    labels = labels.where(~labels.isin(counts[counts < min_size].index), "other")
    order = sorted(label for label in labels.unique() if label != "other")
    if "other" in labels.values:
        order.append("other")
    return labels, order


def plot_panel_e(
    adata_adt: ad.AnnData,
    adata_rna: ad.AnnData,
    annotation: pd.DataFrame,
    low_features: list[str],
    high_features: list[str],
    output_path: Path,
) -> pd.DataFrame:
    shared = adata_adt.obs_names.intersection(adata_rna.obs_names)
    adt = adata_adt[shared]
    rna = adata_rna[shared]
    rna = rna[adt.obs_names]
    labels, order = _annotation_groups(adt, annotation)
    features = low_features + high_features
    matrix = pd.DataFrame(index=order, columns=features, dtype=float)
    for group in order:
        names = labels.index[labels == group]
        for feature in features:
            x = _feature_values(adt[names], feature)
            y = _feature_values(rna[names], feature)
            if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
                matrix.loc[group, feature] = pearsonr(x, y).statistic

    fig, ax = plt.subplots(figsize=(3.4, 2.15))
    norm = TwoSlopeNorm(vmin=-0.2, vcenter=0.0, vmax=0.7)
    image = ax.imshow(matrix.to_numpy(float), cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(features)), features, rotation=90)
    ax.set_yticks(np.arange(len(order)), order)
    ax.axvline(len(low_features) - 0.5, color="black", linestyle="--", linewidth=0.9)
    ax.text((len(low_features) - 1) / 2, -0.85, "low-private", ha="center", va="bottom")
    ax.text(
        len(low_features) + (len(high_features) - 1) / 2,
        -0.85,
        "high-private",
        ha="center",
        va="bottom",
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("RNA–ADT Pearson r")
    cbar.outline.set_visible(False)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return matrix


def _percentile_scale(values: np.ndarray, low: float = 1, high: float = 99) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    vmin, vmax = np.nanpercentile(values, [low, high])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(values)
    return (np.clip(values, vmin, vmax) - vmin) / (vmax - vmin)


def _spatial_scatter(
    ax,
    coords: np.ndarray,
    values: np.ndarray,
    vmin: float,
    vmax: float,
):
    order = np.argsort(values)
    artist = ax.scatter(
        coords[order, 0], coords[order, 1], c=values[order],
        s=SPOT_SIZE, cmap="viridis", vmin=vmin, vmax=vmax,
        linewidths=0, rasterized=True,
    )
    # The manuscript assets were exported to square plotting frames.  Keeping
    # the axes aspect automatic reproduces that display without assigning a
    # different marker size to high- and low-ratio maps.
    ax.set_aspect("auto")
    ax.set_axis_off()
    return artist


def plot_panel_f(
    adata_adt: ad.AnnData,
    adata_rna: ad.AnnData,
    output_path: Path,
) -> None:
    features = PANEL_F_LOW + PANEL_F_HIGH
    coords_adt = np.asarray(adata_adt.obsm["spatial"])
    coords_rna = np.asarray(adata_rna.obsm["spatial"])
    fig, axes = plt.subplots(2, len(features), figsize=(7.1, 2.45))
    for column, feature in enumerate(features):
        adt_scaled = _percentile_scale(_feature_values(adata_adt, feature))
        rna_scaled = _percentile_scale(_feature_values(adata_rna, feature))
        adt_artist = _spatial_scatter(axes[0, column], coords_adt, adt_scaled, 0, 1)
        rna_artist = _spatial_scatter(axes[1, column], coords_rna, rna_scaled, 0, 1)
        axes[0, column].set_title(feature, pad=3)
    axes[0, 0].text(-0.18, 0.5, "ADT", transform=axes[0, 0].transAxes, rotation=90, va="center")
    axes[1, 0].text(-0.18, 0.5, "RNA", transform=axes[1, 0].transAxes, rotation=90, va="center")
    axes[0, 1].text(0.5, 1.25, "low private", transform=axes[0, 1].transAxes, ha="center")
    axes[0, 4].text(0.5, 1.25, "high private", transform=axes[0, 4].transAxes, ha="center")
    fig.colorbar(adt_artist, ax=axes[0, :], fraction=0.012, pad=0.015, label="scaled ADT expression")
    fig.colorbar(rna_artist, ax=axes[1, :], fraction=0.012, pad=0.015, label="scaled RNA expression")
    fig.subplots_adjust(left=0.04, right=0.90, bottom=0.04, top=0.86, wspace=0.04, hspace=0.08)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_s3_panel_a(
    adata_adt: ad.AnnData,
    annotation: pd.DataFrame,
    high_features: list[str],
    low_features: list[str],
    output_path: Path,
) -> None:
    labels, group_order = _annotation_groups(adata_adt, annotation)
    ordered_names = []
    centers = []
    boundaries = []
    start = 0
    for group in group_order:
        names = labels.index[labels == group].tolist()
        ordered_names.extend(names)
        centers.append(start + (len(names) - 1) / 2)
        start += len(names)
        boundaries.append(start - 0.5)
    features = high_features + low_features
    matrix = np.column_stack(
        [_feature_values(adata_adt[ordered_names], feature) for feature in features]
    )
    vmin, vmax = np.nanpercentile(matrix, [1, 99])
    fig, ax = plt.subplots(figsize=(3.5, 2.65))
    image = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto", rasterized=True)
    ax.set_xticks(np.arange(len(features)), features, rotation=90)
    ax.set_yticks(centers, group_order)
    ax.axvline(len(high_features) - 0.5, color="black", linestyle="--", linewidth=0.9)
    for boundary in boundaries[:-1]:
        ax.axhline(boundary, color="black", linewidth=0.55)
    ax.text((len(high_features) - 1) / 2, -130, "high-private", ha="center", va="bottom")
    ax.text(
        len(high_features) + (len(low_features) - 1) / 2,
        -130,
        "low-private",
        ha="center",
        va="bottom",
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("log1p ADT expression")
    cbar.outline.set_visible(False)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_s3_panel_b(result: pd.DataFrame, output_path: Path) -> None:
    plot_df = result.dropna(subset=[RATIO_COLUMN, EXTERNAL_METRIC]).copy()
    plot_df = plot_df.sort_values(RATIO_COLUMN).reset_index(drop=True)
    x = np.linspace(0, 1, len(plot_df))
    y = plot_df[EXTERNAL_METRIC].to_numpy(float)
    colors = plot_df["rna_adt_pearson"].to_numpy(float)
    rho, pvalue = spearmanr(plot_df[RATIO_COLUMN], y)
    fig, ax = plt.subplots(figsize=(2.55, 1.95))
    valid_color = np.isfinite(colors)
    ax.scatter(x[~valid_color], y[~valid_color], color="#D9D9D9", s=14, linewidths=0)
    artist = ax.scatter(
        x[valid_color], y[valid_color], c=colors[valid_color],
        s=14, cmap="viridis", linewidths=0,
    )
    ax.plot(x, _linear_trend(x, y), "--", color="#BDBDBD", linewidth=1.0)
    ax.set_xticks([])
    ax.set_xlabel("Proteins ranked by spaDIVA unique factor ratio\nlow private  →  high private")
    ax.set_ylabel("Translation accuracy (totalVI)")
    ax.text(
        0.03, 0.05, f"Spearman r = {rho:.3f}\np = {pvalue:.3g}",
        transform=ax.transAxes, ha="left", va="bottom",
    )
    ax.spines[["top", "right"]].set_visible(False)
    cbar = fig.colorbar(artist, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("RNA–ADT Pearson r")
    cbar.outline.set_visible(False)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_s3_panel_c(
    adata_adt: ad.AnnData,
    adata_rna: ad.AnnData,
    high_features: list[str],
    low_features: list[str],
    output_path: Path,
) -> dict:
    selected = high_features + low_features
    adt_values = {feature: _feature_values(adata_adt, feature) for feature in selected}
    rna_values = {feature: _feature_values(adata_rna, feature) for feature in selected}
    adt_vmin, adt_vmax = np.nanpercentile(np.column_stack(list(adt_values.values())), [1, 99])
    rna_vmin, rna_vmax = np.nanpercentile(np.column_stack(list(rna_values.values())), [1, 99])
    coords_adt = np.asarray(adata_adt.obsm["spatial"])
    coords_rna = np.asarray(adata_rna.obsm["spatial"])

    fig, axes = plt.subplots(4, 6, figsize=(7.15, 4.9))
    for column, feature in enumerate(high_features):
        adt_artist = _spatial_scatter(axes[0, column], coords_adt, adt_values[feature], adt_vmin, adt_vmax)
        rna_artist = _spatial_scatter(axes[1, column], coords_rna, rna_values[feature], rna_vmin, rna_vmax)
        axes[0, column].set_title(feature, pad=3)
    for column, feature in enumerate(low_features):
        _spatial_scatter(axes[2, column], coords_adt, adt_values[feature], adt_vmin, adt_vmax)
        _spatial_scatter(axes[3, column], coords_rna, rna_values[feature], rna_vmin, rna_vmax)
        axes[2, column].set_title(feature, pad=3)
    row_labels = ["High-ratio ADT", "Matched RNA", "Low-ratio ADT", "Matched RNA"]
    for row, label in enumerate(row_labels):
        axes[row, 0].text(-0.22, 0.5, label, transform=axes[row, 0].transAxes, ha="right", va="center")

    adt_sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(adt_vmin, adt_vmax), cmap="viridis")
    rna_sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(rna_vmin, rna_vmax), cmap="viridis")
    fig.subplots_adjust(left=0.11, right=0.89, bottom=0.03, top=0.95, wspace=0.04, hspace=0.18)
    adt_cax = fig.add_axes([0.915, 0.60, 0.012, 0.25])
    rna_cax = fig.add_axes([0.915, 0.16, 0.012, 0.25])
    adt_cbar = fig.colorbar(adt_sm, cax=adt_cax)
    adt_cbar.set_label("log1p ADT expression")
    adt_cbar.outline.set_visible(False)
    rna_cbar = fig.colorbar(rna_sm, cax=rna_cax)
    rna_cbar.set_label("log1p RNA expression")
    rna_cbar.outline.set_visible(False)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "adt_common_vmin": float(adt_vmin),
        "adt_common_vmax": float(adt_vmax),
        "rna_common_vmin": float(rna_vmin),
        "rna_common_vmax": float(rna_vmax),
    }


def _compare_expected_table(result: pd.DataFrame, expected_path: Path) -> dict:
    expected = pd.read_csv(expected_path)
    expected = expected.rename(columns={"translation_accuracy_totalVI": EXTERNAL_METRIC})
    compare = result.merge(
        expected[["feature_name", RATIO_COLUMN, EXTERNAL_METRIC]],
        on="feature_name",
        suffixes=("_actual", "_expected"),
        how="outer",
        indicator=True,
    )
    if set(compare["_merge"]) != {"both"}:
        raise AssertionError("Feature identities differ from the frozen expected table.")
    ratio_error = np.max(
        np.abs(compare[f"{RATIO_COLUMN}_actual"] - compare[f"{RATIO_COLUMN}_expected"])
    )
    translation_error = np.max(
        np.abs(compare[f"{EXTERNAL_METRIC}_actual"] - compare[f"{EXTERNAL_METRIC}_expected"])
    )
    return {
        "max_abs_unique_factor_ratio_error": float(ratio_error),
        "max_abs_translation_accuracy_error": float(translation_error),
    }


def reproduce(
    data_root: Path,
    artifact_root: Path,
    results_root: Path,
    retrain: bool = False,
    device: str = "auto",
) -> dict:
    data_root = data_root.expanduser().resolve()
    artifact_root = artifact_root.resolve()
    results_root = results_root.resolve()
    figure4_dir = results_root / "Figure_4"
    s3_dir = results_root / "Supplementary_Figure_S3"
    tables_dir = results_root / "tables"
    for directory in (figure4_dir, s3_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)

    font_family = configure_plot_style()
    adata_adt, adata_rna, annotation, split_names, dataset_root = _read_inputs(
        data_root, artifact_root
    )

    frozen_prediction = artifact_root / "artifacts" / "totalvi_test_pred_protein.h5ad"
    if retrain:
        prediction_path = retrain_totalvi(
            adata_adt,
            adata_rna,
            split_names,
            results_root / "external_predictions" / "test_pred_protein.h5ad",
            device=device,
        )
        prediction_source = "retrained"
    else:
        if not frozen_prediction.exists():
            raise FileNotFoundError(frozen_prediction)
        prediction_path = frozen_prediction
        prediction_source = "frozen"

    source_path = artifact_root / "source_data" / "Figure_4_manuscript_source.csv"
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    result, translation_recompute_error = compute_analysis(
        adata_adt,
        adata_rna,
        split_names,
        prediction_path,
        source_path,
        use_frozen_translation=not retrain,
    )
    high_features = select_exact_matches(result, adata_adt, adata_rna, ascending=False)
    low_features = select_exact_matches(result, adata_adt, adata_rna, ascending=True)

    panel_d_stats = plot_panel_d(
        result,
        figure4_dir / "panel_d.pdf",
        label_low=PANEL_F_LOW,
        label_high=PANEL_F_HIGH,
    )
    panel_e_matrix = plot_panel_e(
        adata_adt,
        adata_rna,
        annotation,
        low_features,
        high_features,
        figure4_dir / "panel_e.pdf",
    )
    plot_panel_f(adata_adt, adata_rna, figure4_dir / "panel_f.pdf")
    plot_s3_panel_a(
        adata_adt,
        annotation,
        high_features,
        low_features,
        s3_dir / "panel_a.pdf",
    )
    plot_s3_panel_b(result, s3_dir / "panel_b.pdf")
    s3_scales = plot_s3_panel_c(
        adata_adt,
        adata_rna,
        high_features,
        low_features,
        s3_dir / "panel_c.pdf",
    )

    result.to_csv(tables_dir / "unique_factor_ratio.csv", index=False)
    panel_e_matrix.to_csv(tables_dir / "panel_e_rna_adt_pearson.csv")
    selected = {
        "high_exact_matches": high_features,
        "low_exact_matches": low_features,
        "figure4f_high": PANEL_F_HIGH,
        "figure4f_low": PANEL_F_LOW,
    }
    (tables_dir / "selected_features.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )

    expected_comparison = _compare_expected_table(
        result, source_path
    )
    checks: dict[str, bool] = {
        "n_features": bool(len(result) == EXPECTED_N_FEATURES),
        "high_features": high_features == EXPECTED_HIGH_MATCHES,
        "low_features": low_features == EXPECTED_LOW_MATCHES,
        "expected_ratio_table": bool(expected_comparison["max_abs_unique_factor_ratio_error"] < 1e-6),
        "z_source": set(result["z_source"].astype(str)) == {"Z_poe"},
    }
    if prediction_source == "frozen":
        checks.update(
            {
                "spearman_r": bool(np.isclose(panel_d_stats["spearman_r"], EXPECTED_SPEARMAN_R, atol=1e-10, rtol=0)),
                "spearman_p": bool(np.isclose(panel_d_stats["spearman_p"], EXPECTED_SPEARMAN_P, atol=1e-10, rtol=0)),
                "mannwhitney_p": bool(np.isclose(panel_d_stats["mannwhitney_p"], EXPECTED_MANNWHITNEY_P, atol=1e-12, rtol=0)),
                "expected_translation_table": bool(expected_comparison["max_abs_translation_accuracy_error"] < 1e-6),
                "frozen_totalvi_prediction": bool(translation_recompute_error < 1e-6),
            }
        )
    else:
        checks["retrained_translation_is_finite"] = bool(
            np.isfinite(result[EXTERNAL_METRIC].to_numpy(float)).all()
        )
    validation = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "observed": {
            "n_features": int(len(result)),
            **panel_d_stats,
            **selected,
            **s3_scales,
            **expected_comparison,
            "max_abs_frozen_totalvi_recompute_error": translation_recompute_error,
        },
        "expected": {
            "n_features": EXPECTED_N_FEATURES,
            "spearman_r": EXPECTED_SPEARMAN_R,
            "spearman_p": EXPECTED_SPEARMAN_P,
            "mannwhitney_p": EXPECTED_MANNWHITNEY_P,
            "high_exact_matches": EXPECTED_HIGH_MATCHES,
            "low_exact_matches": EXPECTED_LOW_MATCHES,
        },
    }
    validation_path = figure4_dir / "translation_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")

    artifact_files = {
        "split_assignments": artifact_root / "artifacts" / "split_assignments.csv",
        "totalvi_prediction": prediction_path,
        "manuscript_source_data": source_path,
        "ADT_input": dataset_root / "input_data" / "human lymph node adata_ADT.h5ad",
        "RNA_input": dataset_root / "input_data" / "human lymph node adata_RNA.h5ad",
        "annotation": dataset_root / "annotation" / "annotation.csv",
    }
    output_files = [
        figure4_dir / "panel_d.pdf",
        figure4_dir / "panel_e.pdf",
        figure4_dir / "panel_f.pdf",
        s3_dir / "panel_a.pdf",
        s3_dir / "panel_b.pdf",
        s3_dir / "panel_c.pdf",
        tables_dir / "unique_factor_ratio.csv",
        tables_dir / "panel_e_rna_adt_pearson.csv",
        tables_dir / "selected_features.json",
        validation_path,
    ]
    manifest = {
        "workflow": "human_lymph_node_RNA_ADT manuscript-exact translation figures",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_source": prediction_source,
        "z_source": "Z_poe",
        "uf_ratio_provenance": (
            "deposited manuscript Source Data table recording z_source=Z_poe"
        ),
        "font_family": font_family,
        "font_size_pt": FONT_SIZE,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: _version(name)
            for name in ("anndata", "matplotlib", "numpy", "pandas", "scanpy", "scikit-learn", "scipy", "spaDIVA")
        },
        "inputs": {
            name: {"file_name": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for name, path in artifact_files.items()
        },
        "outputs": {
            str(path.relative_to(results_root)).replace("\\", "/"): {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in output_files
        },
    }
    (figure4_dir / "translation_run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if validation["status"] != "passed":
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Reproduction validation failed: {failed}")
    return validation
