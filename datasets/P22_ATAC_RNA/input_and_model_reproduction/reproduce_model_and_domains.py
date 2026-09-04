"""Reproduce the P22 model/domain panels from one entry point.

The default compact mode redraws Figure 3b-e and Supplementary Figure S2
from frozen, panel-level source data.

Full mode reruns spaDIVA with the public release API and combines the refreshed
spaDIVA rows/maps with the versioned baseline and ablation source data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
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
DEFAULT_OUTPUT = REPO_ROOT / "results" / "Figure_3"

FONT_SIZE = 7
DEFAULT_20 = [
    "#1f77b4", "#ff7f0e", "#279e68", "#d62728", "#aa40fc",
    "#8c564b", "#e377c2", "#b5bd61", "#17becf", "#aec7e8",
    "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94",
    "#f7b6d2", "#dbdb8d", "#9edae5", "#ad494a", "#8c6d31",
]
METHOD_COLORS = {
    "spaDIVA": "#E64B35",
    "SpaMosaic": "#4C78A8",
    "SpatialGlue": "#5B8CC0",
    "MultiVI": "#2F5D8A",
    "spaDIVA w/o PoE": "#F09A7E",
    "spaDIVA_KL_ablation": "#EFAAA0",
}
JACCARD_ORDER = ["shared", "shared+ATAC", "shared+RNA", "ATAC", "RNA"]
JACCARD_COLORS = dict(zip(JACCARD_ORDER, mpl.colormaps["tab10"].colors[:5]))


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    maps = pd.read_csv(SOURCE_DIR / "domain_map_values.csv.gz", dtype={"cluster": str})
    metadata = pd.read_csv(SOURCE_DIR / "domain_map_metadata.csv")
    moran_main = pd.read_csv(SOURCE_DIR / "figure3c_morans_i_per_domain.csv")
    moran_s2 = pd.read_csv(SOURCE_DIR / "supplementary_s2_morans_i_per_domain.csv")
    jaccard = pd.read_csv(SOURCE_DIR / "figure3e_jaccard_original_values.csv")
    return maps, metadata, moran_main, moran_s2, jaccard


def _numeric_cluster(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="raise").astype(int).to_numpy()


def plot_spatial(ax: plt.Axes, values: pd.DataFrame, title: str = "") -> None:
    clusters = _numeric_cluster(values["cluster"])
    colors = [DEFAULT_20[(int(value) - 1) % len(DEFAULT_20)] for value in clusters]
    ax.scatter(
        values["x"], values["y"], c=colors, s=0.75,
        linewidths=0, edgecolors="none", rasterized=True,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("black")


def select_map(maps: pd.DataFrame, map_id: str) -> pd.DataFrame:
    result = maps.loc[maps["map_id"] == map_id]
    if result.empty:
        raise KeyError(f"Missing frozen map: {map_id}")
    return result


def align_clusters_to_frozen(
    frozen_map: pd.DataFrame,
    spot_ids: pd.Index,
    new_labels: np.ndarray,
) -> tuple[np.ndarray, float, dict[str, str]]:
    """Relabel a full-mode clustering to the frozen panel by maximum overlap.

    Cluster identifiers are arbitrary.  Aligning identifiers preserves the
    anatomical legend and color semantics without changing cluster membership.
    """

    from scipy.optimize import linear_sum_assignment

    reference = frozen_map.set_index("spot_id")["cluster"].astype(str)
    current = pd.Series(np.asarray(new_labels).astype(str), index=spot_ids.astype(str))
    common = reference.index.intersection(current.index)
    table = pd.crosstab(current.loc[common], reference.loc[common])
    row_idx, col_idx = linear_sum_assignment(-table.to_numpy())
    mapping = {str(table.index[i]): str(table.columns[j]) for i, j in zip(row_idx, col_idx)}
    aligned = current.map(lambda value: mapping.get(value, value)).to_numpy(dtype=str)
    matched = sum(int(table.iat[i, j]) for i, j in zip(row_idx, col_idx))
    fraction = float(matched / len(common))
    return aligned, fraction, mapping


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.close(fig)


def plot_figure3b(maps: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(3.75, 2.45))
    specs = [
        ("spadiva_integrated_k12", "spaDIVA"),
        ("spamosaic_integrated_k12", "SpaMosaic"),
        ("spatialglue_integrated_k12", "SpatialGlue"),
        ("multivi_integrated_k12", "MultiVI"),
    ]
    for ax, (map_id, title) in zip(axes.flat, specs):
        plot_spatial(ax, select_map(maps, map_id), title)
    annotations = pd.read_csv(SOURCE_DIR / "figure3b_spadiva_cluster_annotations.csv", dtype={"cluster": str})
    handles = [
        mpl.lines.Line2D([], [], marker="o", linestyle="", markersize=3,
                         markerfacecolor=DEFAULT_20[int(row.cluster) - 1], markeredgewidth=0,
                         label=f"{row.cluster}:{row.anatomical_label}")
        for row in annotations.itertuples()
    ]
    fig.legend(handles=handles, title="spaDIVA\ncluster", loc="center left",
               bbox_to_anchor=(0.76, 0.50), frameon=False, handlelength=0.6,
               handletextpad=0.35, labelspacing=0.25, title_fontsize=FONT_SIZE)
    fig.subplots_adjust(left=0.02, right=0.75, bottom=0.03, top=0.94, wspace=0.10, hspace=0.20)
    save_figure(fig, output)


def _boxplot(ax: plt.Axes, frame: pd.DataFrame, order: list[str], display: dict[str, str]) -> None:
    data = [frame.loc[frame["method"] == method, "I"].dropna().to_numpy() for method in order]
    labels = [display.get(method, method) for method in order]
    bp = ax.boxplot(
        data, widths=0.62, showfliers=True, patch_artist=True,
        medianprops={"color": "black", "linewidth": 0.55},
        whiskerprops={"color": "#555555", "linewidth": 0.55},
        capprops={"color": "#555555", "linewidth": 0.55},
        boxprops={"edgecolor": "#555555", "linewidth": 0.55},
        flierprops={"marker": "o", "markersize": 1.5, "markerfacecolor": "#777777",
                    "markeredgewidth": 0, "alpha": 0.65},
    )
    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(METHOD_COLORS.get(label, "#7C9BC4"))
        patch.set_alpha(0.86)
    ax.set_xticks(np.arange(1, len(labels) + 1), labels, rotation=25, ha="right")
    ax.set_ylabel("Moran's $I$ score")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_figure3c(moran: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(2.0, 1.75))
    order = ["spaDIVA", "SpaMosaic", "SpatialGlue", "MultiVI"]
    _boxplot(ax, moran, order, {})
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.34, top=0.98)
    save_figure(fig, output)


def plot_figure3d(maps: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(3.85, 2.55))
    specs = [
        ("spadiva_integrated_k12", "spaDIVA-integrated"),
        ("spadiva_atac_specific_k6", "spaDIVA-ATAC-specific"),
        ("spadiva_rna_specific_k6", "spaDIVA-RNA-specific"),
        ("spadiva_shared_k12", "spaDIVA-shared"),
        ("atac_input_k6", "ATAC-input"),
        ("rna_input_k6", "RNA-input"),
    ]
    for ax, (map_id, title) in zip(axes.flat, specs):
        plot_spatial(ax, select_map(maps, map_id), title)
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.02, top=0.93, wspace=0.12, hspace=0.30)
    save_figure(fig, output)


def summarize_jaccard(jaccard: pd.DataFrame) -> pd.DataFrame:
    if "mean_jaccard" in jaccard.columns:
        return jaccard.copy()
    return (
        jaccard.groupby(["input_modality", "representation"], sort=False)["jaccard"]
        .agg(n_spots="count", mean_jaccard="mean", sd_jaccard="std")
        .reset_index()
    )


def plot_figure3e(jaccard: pd.DataFrame, output: Path) -> None:
    summary = summarize_jaccard(jaccard)
    fig, ax = plt.subplots(figsize=(2.25, 1.35))
    width = 0.26
    x_atac = np.arange(5) * 0.38
    x_rna = x_atac + x_atac[-1] + 0.72
    for positions, modality in ((x_atac, "ATAC"), (x_rna, "RNA")):
        for x, representation in zip(positions, JACCARD_ORDER):
            value = summary.loc[
                (summary["input_modality"] == modality)
                & (summary["representation"] == representation), "mean_jaccard"
            ].iloc[0]
            ax.bar(x, value, width=width, color=JACCARD_COLORS[representation], linewidth=0)
    ax.set_xticks([x_atac.mean(), x_rna.mean()], ["ATAC", "RNA"])
    ax.set_ylabel("Jaccard similarity")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles = [mpl.patches.Patch(facecolor=JACCARD_COLORS[key], label=key) for key in JACCARD_ORDER]
    ax.legend(handles=handles, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.55),
              borderaxespad=0, handlelength=1.0, labelspacing=0.35)
    fig.subplots_adjust(left=0.20, right=0.68, bottom=0.24, top=0.98)
    save_figure(fig, output)


def plot_s2a(maps: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(5.9, 1.28))
    specs = [
        ("spadiva_integrated_k18", "spaDIVA"),
        ("spadiva_no_poe_integrated_k18", "spaDIVA w/o PoE"),
        ("spamosaic_integrated_k18", "SpaMosaic"),
        ("spatialglue_integrated_k18", "SpatialGlue"),
        ("multivi_integrated_k18", "MultiVI"),
    ]
    for ax, (map_id, title) in zip(axes, specs):
        plot_spatial(ax, select_map(maps, map_id), title)
    fig.text(0.01, 0.46, "$K=18$", ha="left", va="center")
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.04, top=0.88, wspace=0.15)
    save_figure(fig, output)


def plot_s2b(maps: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 6, figsize=(7.15, 1.32))
    specs = [
        ("spadiva_kl_ablation_integrated_k18", "Integrated\n$K=18$"),
        ("spadiva_kl_ablation_shared_k18", "Shared\n$K=18$"),
        ("spadiva_kl_ablation_integrated_k12", "Integrated\n$K=12$"),
        ("spadiva_kl_ablation_shared_k12", "Shared\n$K=12$"),
        ("spadiva_kl_ablation_atac_specific_k6", "ATAC-specific\n$K=6$"),
        ("spadiva_kl_ablation_rna_specific_k6", "RNA-specific\n$K=6$"),
    ]
    for ax, (map_id, title) in zip(axes, specs):
        plot_spatial(ax, select_map(maps, map_id), title)
    fig.text(0.01, 0.45, "spaDIVA_KL_ablation", ha="left", va="center")
    fig.subplots_adjust(left=0.15, right=0.995, bottom=0.03, top=0.78, wspace=0.14)
    save_figure(fig, output)


def plot_s2c(moran: pd.DataFrame, output: Path) -> None:
    rename = {
        "spamosaic": "SpaMosaic",
        "spatialGlue": "SpatialGlue",
        "multiVI": "MultiVI",
        "spaDIVA_poe_ablation": "spaDIVA w/o PoE",
    }
    frame = moran.copy()
    frame["method"] = frame["method"].replace(rename)
    order = ["spaDIVA", "spaDIVA w/o PoE", "spaDIVA_KL_ablation", "SpaMosaic", "SpatialGlue", "MultiVI"]
    fig, axes = plt.subplots(1, 2, figsize=(5.15, 1.85))
    for ax, k in zip(axes, (18, 12)):
        _boxplot(ax, frame.loc[frame["K"] == k], order, {})
        ax.text(0.5, -0.48, f"$K={k}$", transform=ax.transAxes, ha="center", va="top")
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.43, top=0.98, wspace=0.30)
    save_figure(fig, output)


def row_standardize(weights):
    from scipy import sparse

    weights = weights.tocsr()
    row_sum = np.asarray(weights.sum(axis=1)).ravel()
    inverse = np.zeros_like(row_sum, dtype=float)
    inverse[row_sum > 0] = 1.0 / row_sum[row_sum > 0]
    return sparse.diags(inverse) @ weights


def moran_rows(labels: np.ndarray, spatial: np.ndarray, method: str, column: str, k_value: int) -> pd.DataFrame:
    from sklearn.neighbors import kneighbors_graph

    weights = kneighbors_graph(spatial, n_neighbors=6, mode="connectivity", include_self=False).tocsr()
    weights = weights.maximum(weights.T).tocsr()
    weights.setdiag(0)
    weights.eliminate_zeros()
    weights = row_standardize(weights)
    rows = []
    labels = np.asarray(labels).astype(str)
    for cluster in pd.unique(labels):
        indicator = (labels == cluster).astype(float)
        deviation = indicator - indicator.mean()
        denominator = float(deviation @ deviation)
        score = float((len(indicator) / weights.sum()) * ((deviation @ (weights @ deviation)) / denominator))
        rows.append({"K": k_value, "cluster": cluster, "I": score,
                     "method": method, "cluster_col_used": column})
    return pd.DataFrame(rows)


def knn_indices(values: np.ndarray, k: int = 50) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(values)
    return nn.kneighbors(values, return_distance=False)[:, 1:]


def jaccard_from_embeddings(
    spot_ids: pd.Index,
    representations: dict[str, np.ndarray],
    inputs: dict[str, np.ndarray],
) -> pd.DataFrame:
    all_embeddings = {**representations, **{f"input::{key}": value for key, value in inputs.items()}}
    neighbors = {key: knn_indices(np.asarray(value), 50) for key, value in all_embeddings.items()}
    rows = []
    for input_name in ("ATAC", "RNA"):
        for representation in JACCARD_ORDER:
            left = neighbors[representation]
            right = neighbors[f"input::{input_name}"]
            overlap = np.empty(left.shape[0], dtype=float)
            for i, (aa, bb) in enumerate(zip(left, right)):
                sa, sb = set(aa), set(bb)
                overlap[i] = len(sa & sb) / len(sa | sb)
            rows.append(
                pd.DataFrame(
                    {
                        "spot_id": spot_ids.astype(str),
                        "input_modality": input_name,
                        "representation": representation,
                        "k": 50,
                        "metric": "euclidean",
                        "jaccard": overlap,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def resolve_input_dir(data_root: Path) -> Path:
    candidates = [
        data_root / "datasets" / "P22_ATAC_RNA" / "input_data",
        data_root / "P22_ATAC_RNA" / "input_data",
        data_root / "input_data",
    ]
    for candidate in candidates:
        if (candidate / "adata_peaks_normalized.h5ad").exists() and (candidate / "adata_RNA.h5ad").exists():
            return candidate
    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Could not find the two P22 input H5AD files. Searched:\n{searched}")


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def full_spadiva_refresh(
    maps: pd.DataFrame,
    moran_main: pd.DataFrame,
    moran_s2: pd.DataFrame,
    data_root: Path,
    output: Path,
    epochs: int,
    use_cuda: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Retrain only spaDIVA using the current public release API."""

    import scanpy as sc
    import torch
    from anndata import AnnData
    from sklearn.decomposition import PCA
    from spaDIVA import cal_spatial, cal_weight, cluster, infer_latents, lsi, train_spadiva

    seed = 42
    poe_sample_seed = 42
    set_seed(seed)
    input_dir = resolve_input_dir(data_root)
    atac = sc.read_h5ad(input_dir / "adata_peaks_normalized.h5ad")
    rna = sc.read_h5ad(input_dir / "adata_RNA.h5ad")
    atac.var_names_make_unique()
    rna.var_names_make_unique()
    atac.X = atac.X.astype("float32")
    rna.X = rna.X.astype("float32")

    sc.pp.filter_genes(rna, min_cells=10)
    sc.pp.filter_cells(rna, min_genes=200)
    sc.pp.highly_variable_genes(rna, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.scale(rna)
    rna = rna[:, rna.var.highly_variable].copy()
    atac = atac[rna.obs_names].copy()
    atac.obsm["lsi"] = lsi(atac, use_highly_variable=False, n_components=65, random_state=seed)
    rna.obsm["pca"] = PCA(n_components=64).fit_transform(rna.to_df())

    spatial = np.asarray(atac.obsm["spatial"])
    edge_index = cal_spatial(spatial, k=6)
    x1 = np.asarray(atac.obsm["lsi"], dtype=np.float32)
    x2 = np.asarray(rna.obsm["pca"], dtype=np.float32)
    model, loss = train_spadiva(
        x1, x2, x1, x2, edge_index=edge_index,
        learning_rate=1e-3, weight=1.0, max_epochs=epochs,
        use_cuda=use_cuda, hidden_dim1=64, hidden_dim2=64,
        w1_dim=30, w2_dim=30, z_dim=30, KL_weight=[1, 1, 1, 1],
    )
    z_poe, z1, z2, w1, w2, _, _ = infer_latents(
        model, x1, x2, edge_index=edge_index, use_cuda=use_cuda,
        sample_seed=poe_sample_seed,
    )
    alpha1, alpha2 = cal_weight(z1, z2, k=20)
    z = alpha1.reshape(-1, 1) * z1 + alpha2.reshape(-1, 1) * z2
    z_w = np.concatenate([w1, z, w2], axis=1)

    labels = {
        "mclust_Z_W_12": cluster(z_w, 12, spatial, random_seed=seed, show=False, return_labels=True),
        "mclust_Z_12": cluster(z, 12, spatial, random_seed=seed, show=False, return_labels=True),
        "mclust_Z_W_18": cluster(z_w, 18, spatial, random_seed=seed, show=False, return_labels=True),
        "mclust_Z_18": cluster(z, 18, spatial, random_seed=seed, show=False, return_labels=True),
        "mclust_W1": cluster(w1, 6, spatial, random_seed=seed, show=False, return_labels=True),
        "mclust_W2": cluster(w2, 6, spatial, random_seed=seed, show=False, return_labels=True),
    }
    set_seed(seed)
    x1_30 = lsi(atac, use_highly_variable=False, n_components=31, random_state=seed)
    x2_30 = PCA(n_components=30).fit_transform(rna.to_df())
    labels["ATAC"] = cluster(x1_30, 6, spatial, random_seed=seed, show=False, return_labels=True)
    labels["RNA"] = cluster(x2_30, 6, spatial, random_seed=seed, show=False, return_labels=True)

    model_adata = AnnData(X=z.astype(np.float32))
    model_adata.obs_names = atac.obs_names.copy()
    model_adata.obsm["spatial"] = spatial
    model_adata.obsm["Z"] = z
    model_adata.obsm["Z_W"] = z_w
    model_adata.obsm["Z_poe"] = z_poe
    model_adata.obsm["W_ATAC"] = w1
    model_adata.obsm["W_RNA"] = w2
    for key, value in labels.items():
        model_adata.obs[key] = pd.Series(value, index=model_adata.obs_names).astype(str)
    model_adata.uns["model_seed"] = seed
    model_adata.uns["poe_sample_seed"] = poe_sample_seed
    model_adata.uns["epochs"] = int(epochs)
    output.mkdir(parents=True, exist_ok=True)
    model_adata.write_h5ad(output / "P22_spaDIVA_release_full.h5ad")
    pd.DataFrame({"epoch": np.arange(1, len(loss) + 1), "loss": np.asarray(loss, dtype=float)}).to_csv(
        output / "spaDIVA_training_loss.csv", index=False
    )

    replacement_specs = {
        "spadiva_integrated_k12": "mclust_Z_W_12",
        "spadiva_atac_specific_k6": "mclust_W1",
        "spadiva_rna_specific_k6": "mclust_W2",
        "spadiva_shared_k12": "mclust_Z_12",
        "atac_input_k6": "ATAC",
        "rna_input_k6": "RNA",
        "spadiva_integrated_k18": "mclust_Z_W_18",
    }
    refreshed = maps.loc[~maps["map_id"].isin(replacement_specs)].copy()
    replacement_rows = []
    label_alignment = {}
    for map_id, label_key in replacement_specs.items():
        aligned, overlap_fraction, mapping = align_clusters_to_frozen(
            select_map(maps, map_id), model_adata.obs_names, model_adata.obs[label_key].to_numpy()
        )
        label_alignment[map_id] = {
            "maximum_overlap_fraction": overlap_fraction,
            "new_to_frozen_label": mapping,
        }
        replacement_rows.append(
            pd.DataFrame(
                {
                    "map_id": map_id,
                    "spot_id": model_adata.obs_names.astype(str),
                    "x": spatial[:, 0],
                    "y": spatial[:, 1],
                    "cluster": aligned,
                }
            )
        )
    refreshed = pd.concat([refreshed, *replacement_rows], ignore_index=True)

    new_main = moran_rows(labels["mclust_Z_W_12"], spatial, "spaDIVA", "mclust_Z_W_12", 12)
    moran_main = pd.concat([moran_main.loc[moran_main["method"] != "spaDIVA"], new_main], ignore_index=True)
    new_s2 = pd.concat(
        [
            moran_rows(labels["mclust_Z_W_18"], spatial, "spaDIVA", "mclust_Z_W_18", 18),
            new_main,
        ],
        ignore_index=True,
    )
    moran_s2 = pd.concat([moran_s2.loc[moran_s2["method"] != "spaDIVA"], new_s2], ignore_index=True)
    jaccard = jaccard_from_embeddings(
        model_adata.obs_names,
        {
            "shared": z,
            "shared+ATAC": np.concatenate([z, w1], axis=1),
            "shared+RNA": np.concatenate([z, w2], axis=1),
            "ATAC": w1,
            "RNA": w2,
        },
        {"ATAC": x1, "RNA": x2},
    )
    details = {
        "input_dir": str(input_dir),
        "spaDIVA_epochs": int(epochs),
        "model_seed": seed,
        "poe_sample_seed": poe_sample_seed,
        "use_cuda_requested": bool(use_cuda),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "baseline_results": "frozen",
        "ablation_results": "frozen",
        "cluster_label_alignment": label_alignment,
    }
    return refreshed, moran_main, moran_s2, jaccard, details


def validate(
    maps: pd.DataFrame,
    metadata: pd.DataFrame,
    moran_main: pd.DataFrame,
    moran_s2: pd.DataFrame,
    jaccard: pd.DataFrame,
    outputs: list[Path],
) -> dict:
    expected_map_ids = set(metadata["map_id"])
    actual_map_ids = set(maps["map_id"])
    summarized_jaccard = summarize_jaccard(jaccard)
    jaccard_summary = summarized_jaccard.to_dict(orient="records")
    report = {
        "status": "pass",
        "font_size_pt": FONT_SIZE,
        "domain_maps": {
            "expected_map_ids": len(expected_map_ids),
            "actual_map_ids": len(actual_map_ids),
            "missing_map_ids": sorted(expected_map_ids - actual_map_ids),
            "rows": int(len(maps)),
        },
        "figure3c": {
            "rows": int(len(moran_main)),
            "methods": sorted(moran_main["method"].unique().tolist()),
        },
        "supplementary_s2": {
            "moran_rows": int(len(moran_s2)),
            "K": sorted(pd.to_numeric(moran_s2["K"]).astype(int).unique().tolist()),
        },
        "figure3e": {"rows": int(len(jaccard)), "summary": jaccard_summary},
        "outputs": {path.name: {"exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0} for path in outputs},
    }
    if report["domain_maps"]["missing_map_ids"]:
        report["status"] = "fail"
    if any(not item["exists"] or item["bytes"] == 0 for item in report["outputs"].values()):
        report["status"] = "fail"
    if set(report["figure3c"]["methods"]) != {"MultiVI", "SpaMosaic", "SpatialGlue", "spaDIVA"}:
        report["status"] = "fail"
    expected_jaccard_rows = 10 if "mean_jaccard" in jaccard.columns else 91960
    if report["figure3e"]["rows"] != expected_jaccard_rows:
        report["status"] = "fail"
    if len(summarized_jaccard) != 10:
        report["status"] = "fail"
    if set(pd.to_numeric(summarized_jaccard["n_spots"]).astype(int)) != {9196}:
        report["status"] = "fail"
    return report


def reproduce(mode: str, output: Path, data_root: Path | None, epochs: int, use_cuda: bool) -> None:
    configure_style()
    maps, metadata, moran_main, moran_s2, jaccard = load_source_data()
    full_details = None
    if mode == "full":
        if data_root is None:
            raw = os.environ.get("SPADIVA_DATA_ROOT")
            if not raw:
                raise RuntimeError("Full mode requires --data-root or SPADIVA_DATA_ROOT.")
            data_root = Path(raw)
        maps, moran_main, moran_s2, jaccard, full_details = full_spadiva_refresh(
            maps, moran_main, moran_s2, data_root.expanduser().resolve(), output, epochs, use_cuda
        )

    outputs = [
        output / "panel_b.pdf",
        output / "panel_c.pdf",
        output / "panel_d.pdf",
        output / "panel_e.pdf",
        output / "Supplementary_Figure_S2_panel_a.pdf",
        output / "Supplementary_Figure_S2_panel_b.pdf",
        output / "Supplementary_Figure_S2_panel_c.pdf",
    ]
    plot_figure3b(maps, outputs[0])
    plot_figure3c(moran_main, outputs[1])
    plot_figure3d(maps, outputs[2])
    plot_figure3e(jaccard, outputs[3])
    plot_s2a(maps, outputs[4])
    plot_s2b(maps, outputs[5])
    plot_s2c(moran_s2, outputs[6])

    report = validate(maps, metadata, moran_main, moran_s2, jaccard, outputs)
    output.mkdir(parents=True, exist_ok=True)
    validation_path = output / "model_and_domains_validation.json"
    validation_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    source_files = sorted(path for path in SOURCE_DIR.iterdir() if path.is_file())
    manifest = {
        "mode": mode,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "entry_point": Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
        "source_data": {path.name: sha256(path) for path in source_files},
        "frozen_baselines": True,
        "frozen_ablation_results": True,
        "full_mode": full_details,
        "validation_status": report["status"],
    }
    (output / "model_and_domains_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if report["status"] != "pass":
        raise RuntimeError(f"Validation failed; inspect {validation_path}")
    print(f"P22 model/domain reproduction complete: {output}")
    print("Generated Figure 3b-e and Supplementary Figure S2a-c.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("compact", "full"), default="compact")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=1600,
                        help="spaDIVA epochs in full mode (default: 1600, matching the P22 tutorial).")
    parser.add_argument("--use-cuda", action="store_true", help="Request CUDA in full mode.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    reproduce(args.mode, args.output_dir.resolve(), args.data_root, args.epochs, args.use_cuda)
