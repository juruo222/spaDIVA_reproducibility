"""Implement ``--mode full`` for the P22 external-translation analysis.

Compact reproduction reads versioned figure tables in the plotting
environment. Full mode adds scvi-tools, Scanpy, GSEA libraries, the
gene-activity matrix, and GMT snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import random
import re
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_P22_GA_SIZE = 3_666_760_836
EXPECTED_P22_GA_SHA256 = "2940b50bd687d967b625e6cf86cd2c486bc122787a36e0ae0733d43e1cad8f4b"
EXPECTED_GMT_STEMS = (
    "GO_Biological_Process_2023",
    "KEGG_2019_Mouse",
    "WikiPathways_2019_Mouse",
)
EXPECTED_GMT_SHA256 = {
    "GO_Biological_Process_2023.gmt": "058dd244b69f1011c922c2cee7ccefbf70b23b794fe08399513fdfa312d5f354",
    "KEGG_2019_Mouse.gmt": "8baaf4dbaa86ebac975c65728054c0d8b2cc9f73f384d9b53ef66ce0b7fc0830",
    "WikiPathways_2019_Mouse.gmt": "dcb8e46e2009b628ad9ffa968453b4044f310611dd5819cf185caa5dfd0d065c",
}
SELECTED_GENES = ("Pde10a", "Rgs9", "Tmeff2", "Nab2")


class FullModeUnavailable(RuntimeError):
    """Raised when an external asset required by full mode is missing."""


@dataclass(frozen=True)
class FullInputs:
    input_dir: Path
    atac_path: Path
    rna_path: Path
    split_csv: Path
    latent_path: Path
    public_ga_path: Path
    gmt_paths: tuple[Path, ...]


def _resolve_optional_path(value: str | Path | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(value).expanduser().resolve()


def _discover_gmts(gmt_dir: Path | None) -> tuple[Path, ...]:
    if gmt_dir is None or not gmt_dir.is_dir():
        return ()
    return tuple(sorted(path.resolve() for path in gmt_dir.glob("*.gmt") if path.is_file()))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight_full(
    *,
    workflow_root: Path,
    data_root: Path,
    public_ga_path: Path | None,
    gmt_dir: Path | None,
) -> FullInputs:
    """Resolve and validate every non-generated full-mode input before training."""
    workflow_root = workflow_root.resolve()
    data_root = data_root.expanduser().resolve()
    input_dir = data_root / "datasets" / "P22_ATAC_RNA" / "input_data"
    atac_path = input_dir / "adata_peaks_normalized.h5ad"
    rna_path = input_dir / "adata_RNA.h5ad"
    split_csv = workflow_root / "fixed_split" / "split_assignments.csv"
    latent_path = workflow_root / "latent" / "P22_spaDIVA_sampled_Z_poe.h5ad"

    if public_ga_path is None:
        public_ga_path = _resolve_optional_path(os.environ.get("SPADIVA_P22_GA_PATH"))
    if public_ga_path is None:
        public_ga_path = workflow_root / "external_predictions" / "P22_GA.txt"
    public_ga_path = public_ga_path.expanduser().resolve()

    if gmt_dir is None:
        gmt_dir = _resolve_optional_path(os.environ.get("SPADIVA_P22_GMT_DIR"))
    if gmt_dir is None:
        gmt_dir = workflow_root / "external_predictions" / "gene_sets"
    gmt_dir = gmt_dir.expanduser().resolve()
    gmt_paths = _discover_gmts(gmt_dir)

    problems: list[str] = []
    for label, path in (
        ("P22 normalized ATAC input", atac_path),
        ("P22 RNA input", rna_path),
        ("portable fixed split", split_csv),
        ("frozen spaDIVA sampled Z_poe latent", latent_path),
        ("P22 public gene-activity matrix", public_ga_path),
    ):
        if not path.is_file():
            problems.append(f"- Missing {label}: {path}")

    if public_ga_path.is_file() and public_ga_path.stat().st_size != EXPECTED_P22_GA_SIZE:
        problems.append(
            "- P22_GA.txt size mismatch: "
            f"expected {EXPECTED_P22_GA_SIZE:,} bytes, found "
            f"{public_ga_path.stat().st_size:,} bytes ({public_ga_path})"
        )

    observed_stems = {path.stem for path in gmt_paths}
    missing_gmts = [stem for stem in EXPECTED_GMT_STEMS if stem not in observed_stems]
    if missing_gmts:
        problems.append(
            "- Missing frozen GMT snapshot(s) in "
            f"{gmt_dir}: {', '.join(stem + '.gmt' for stem in missing_gmts)}"
        )

    # Hash once at preflight, after cheap presence/size checks, before training.
    if not problems:
        expected_hashes = [(public_ga_path, EXPECTED_P22_GA_SHA256)] + [
            (gmt_dir / f"{stem}.gmt", EXPECTED_GMT_SHA256[f"{stem}.gmt"])
            for stem in EXPECTED_GMT_STEMS
        ]
        for path, expected_hash in expected_hashes:
            actual_hash = _sha256_file(path)
            if actual_hash != expected_hash:
                problems.append(
                    f"- SHA256 mismatch for {path}: expected {expected_hash}, "
                    f"found {actual_hash}. Use the deposited paper-version file."
                )

    if problems:
        details = "\n".join(problems)
        raise FullModeUnavailable(
            "P22 full reproduction requires the deposited assets listed below.\n"
            f"{details}\n"
            "For full mode, download P22_GA.txt and the three versioned GMT files, "
            "then set SPADIVA_P22_GA_PATH and SPADIVA_P22_GMT_DIR (or pass the "
            "corresponding command-line options). Compact mode uses the bundled "
            "figure tables through `python reproduce.py`."
        )

    return FullInputs(
        input_dir=input_dir,
        atac_path=atac_path,
        rna_path=rna_path,
        split_csv=split_csv,
        latent_path=latent_path,
        public_ga_path=public_ga_path,
        gmt_paths=tuple(
            gmt_dir / f"{stem}.gmt" for stem in EXPECTED_GMT_STEMS
        ),
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _load_fixed_split(split_csv: Path, ordered_obs_names: pd.Index) -> pd.DataFrame:
    split = pd.read_csv(
        split_csv,
        dtype={"obs_name": str, "split": str, "original_index": int},
    )
    required = {"obs_name", "split", "original_index"}
    missing = sorted(required.difference(split.columns))
    if missing:
        raise KeyError(f"Fixed split is missing columns: {missing}")
    if split["obs_name"].duplicated().any():
        raise ValueError("Fixed split contains duplicate spot names.")
    if set(split["split"]) != {"train", "val", "test"}:
        raise ValueError("Fixed split must contain train, val and test labels.")
    if len(split) != len(ordered_obs_names):
        raise ValueError(
            f"Fixed split has {len(split):,} rows but aligned data have "
            f"{len(ordered_obs_names):,} spots."
        )
    observed_indices = ordered_obs_names.get_indexer(split["obs_name"])
    if np.any(observed_indices < 0):
        raise KeyError("Fixed split contains spots absent from the aligned inputs.")
    if not np.array_equal(
        observed_indices, split["original_index"].to_numpy(dtype=int)
    ):
        raise ValueError("Fixed split original_index no longer matches spot order.")
    expected_counts = {"train": 6436, "val": 920, "test": 1840}
    counts = split.groupby("split").size().to_dict()
    if counts != expected_counts:
        raise ValueError(f"Fixed split counts changed: {counts}")
    return split


def _indices_for(split: pd.DataFrame, label: str, obs_names: pd.Index) -> np.ndarray:
    names = split.loc[split["split"] == label, "obs_name"]
    indices = obs_names.get_indexer(names)
    if np.any(indices < 0):
        raise KeyError(f"Some {label} spot names were not found after alignment.")
    return indices


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


def train_multivi(
    inputs: FullInputs,
    intermediate_dir: Path,
    *,
    seed: int = 0,
    max_epochs: int = 100,
    accelerator: str = "auto",
) -> tuple[Path, Path, Path]:
    """Train MultiVI with the deposited split and save held-out ATAC outputs."""
    import anndata as ad
    from mudata import MuData
    import scanpy as sc
    import scvi
    from scipy.sparse import csr_matrix, issparse

    _seed_everything(seed)
    scvi.settings.seed = seed
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    atac = sc.read_h5ad(inputs.atac_path)
    rna = sc.read_h5ad(inputs.rna_path)
    latent = sc.read_h5ad(inputs.latent_path)
    for adata in (atac, rna, latent):
        adata.obs_names = adata.obs_names.astype(str)
    atac.var_names_make_unique()
    rna.var_names_make_unique()

    common = latent.obs_names.intersection(rna.obs_names).intersection(atac.obs_names)
    if len(common) != latent.n_obs:
        raise ValueError(
            "P22 RNA/ATAC inputs do not contain every spot in the frozen latent."
        )
    rna = rna[latent.obs_names].copy()
    atac = atac[latent.obs_names].copy()
    rna.X = rna.X.astype(np.float32)
    atac.X = atac.X.astype(np.float32)

    split = _load_fixed_split(inputs.split_csv, rna.obs_names)
    train_indices = _indices_for(split, "train", rna.obs_names)
    val_indices = _indices_for(split, "val", rna.obs_names)
    test_indices = _indices_for(split, "test", rna.obs_names)

    rna_train = rna[train_indices].copy()
    rna_val = rna[val_indices].copy()
    rna_test = rna[test_indices].copy()
    atac_train = atac[train_indices].copy()
    atac_val = atac[val_indices].copy()
    atac_test_true = atac[test_indices].copy()

    if issparse(atac_test_true.X):
        masked_x = csr_matrix(
            (atac_test_true.n_obs, atac_test_true.n_vars), dtype=np.float32
        )
    else:
        masked_x = np.zeros(
            (atac_test_true.n_obs, atac_test_true.n_vars), dtype=np.float32
        )
    atac_test_masked = ad.AnnData(
        X=masked_x,
        obs=atac_test_true.obs.copy(),
        var=atac_test_true.var.copy(),
    )
    if "spatial" in atac_test_true.obsm:
        atac_test_masked.obsm["spatial"] = atac_test_true.obsm["spatial"].copy()

    rna_all = ad.concat([rna_train, rna_val, rna_test], join="inner", merge="same")
    atac_all = ad.concat(
        [atac_train, atac_val, atac_test_masked], join="inner", merge="same"
    )
    split_labels = (
        ["train"] * len(train_indices)
        + ["val"] * len(val_indices)
        + ["test"] * len(test_indices)
    )
    modality_labels = (
        ["paired"] * (len(train_indices) + len(val_indices))
        + ["expression"] * len(test_indices)
    )
    for adata in (rna_all, atac_all):
        adata.obs["split"] = split_labels
        adata.obs["modality"] = modality_labels
        if issparse(adata.X):
            adata.X = adata.X.tocsr()

    mdata = MuData({"rna": rna_all, "atac": atac_all})
    mdata.obs["split"] = split_labels
    mdata.obs["modality"] = modality_labels
    mdata.update()
    scvi.model.MULTIVI.setup_mudata(
        mdata,
        batch_key="modality",
        modalities={
            "rna_layer": "rna",
            "atac_layer": "atac",
            "batch_key": "rna",
        },
    )
    model = scvi.model.MULTIVI(
        mdata,
        n_genes=rna.n_vars,
        n_regions=atac.n_vars,
    )
    split_names = {
        "train": rna_train.obs_names,
        "val": rna_val.obs_names,
        "test": rna_test.obs_names,
    }
    split_indices = _prepare_fixed_split(model, split_names, batch_size=128)
    model.train(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        batch_size=128,
        datasplitter_kwargs={"external_indexing": split_indices},
        shuffle_set_split=False,
        early_stopping=True,
        check_val_every_n_epoch=1,
    )
    _assert_fixed_split(
        model.adata_manager.adata.obs_names,
        split_names,
        [model.train_indices, model.validation_indices, model.test_indices],
        stage="MultiVI after training",
    )

    if hasattr(model, "get_normalized_accessibility"):
        predicted = model.get_normalized_accessibility(
            indices=model.test_indices, return_numpy=True
        )
    else:
        predicted = np.asarray(
            model.get_accessibility_estimates(indices=model.test_indices)
        )
    predicted = np.asarray(predicted, dtype=np.float32)

    predicted_adata = ad.AnnData(
        X=predicted,
        obs=rna_test.obs.copy(),
        var=atac.var.copy(),
    )
    if "spatial" in rna_test.obsm:
        predicted_adata.obsm["spatial"] = rna_test.obsm["spatial"].copy()

    pred_path = intermediate_dir / "test_pred_atac.h5ad"
    true_path = intermediate_dir / "test_true_atac.h5ad"
    rna_test_path = intermediate_dir / "test_input_rna.h5ad"
    predicted_adata.write_h5ad(pred_path)
    atac_test_true.write_h5ad(true_path)
    rna_test.write_h5ad(rna_test_path)
    return pred_path, true_path, rna_test_path


def _as_float32_matrix(matrix: Any) -> Any:
    from scipy.sparse import issparse

    if issparse(matrix):
        return matrix.tocsr().astype(np.float32)
    return np.asarray(matrix, dtype=np.float32)


def _matrix_feature_stats(matrix: Any) -> tuple[np.ndarray, np.ndarray]:
    from scipy.sparse import issparse

    n_observations = matrix.shape[0]
    if issparse(matrix):
        matrix = matrix.tocsr()
        nonzero_fraction = matrix.getnnz(axis=0) / n_observations
        mean = np.asarray(matrix.mean(axis=0)).ravel()
        variance = np.asarray(matrix.power(2).mean(axis=0)).ravel() - mean**2
    else:
        matrix = np.asarray(matrix)
        nonzero_fraction = np.mean(matrix != 0, axis=0)
        variance = matrix.var(axis=0)
    return variance.astype(np.float32), nonzero_fraction.astype(np.float32)


def _standardize_from_train(
    train: np.ndarray, *others: np.ndarray
) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=0, keepdims=True)
    standard_deviation = train.std(axis=0, keepdims=True)
    standard_deviation[standard_deviation < 1e-8] = 1.0
    return tuple(
        ((array - mean) / standard_deviation).astype(np.float32)
        for array in (train, *others)
    )


def _fit_closed_form_ridge(
    design: np.ndarray, response: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    design64 = np.asarray(design, dtype=np.float64)
    response32 = np.asarray(response, dtype=np.float32)
    response_mean = response32.mean(axis=0).astype(np.float64)
    right_hand_side = design64.T @ response32
    right_hand_side -= np.outer(design64.sum(axis=0), response_mean)
    system = design64.T @ design64
    system += alpha * np.eye(design64.shape[1], dtype=np.float64)
    coefficients = np.linalg.solve(system, right_hand_side)
    return coefficients.astype(np.float32), response_mean.astype(np.float32)


def _column_pearson(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    left64 -= left64.mean(axis=0, keepdims=True)
    right64 -= right64.mean(axis=0, keepdims=True)
    numerator = np.sum(left64 * right64, axis=0)
    denominator = np.sqrt(
        np.sum(left64**2, axis=0) * np.sum(right64**2, axis=0)
    )
    result = np.full(left64.shape[1], np.nan, dtype=np.float64)
    valid = denominator > 1e-12
    result[valid] = numerator[valid] / denominator[valid]
    return result.astype(np.float32)


def _raw_r2(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    truth64 = np.asarray(truth, dtype=np.float64)
    prediction64 = np.asarray(prediction, dtype=np.float64)
    residual_sum = np.sum((truth64 - prediction64) ** 2, axis=0)
    centered = truth64 - truth64.mean(axis=0, keepdims=True)
    total_sum = np.sum(centered**2, axis=0)
    result = np.full(truth64.shape[1], np.nan, dtype=np.float64)
    valid = total_sum > 1e-12
    result[valid] = 1 - residual_sum[valid] / total_sum[valid]
    return result


def _load_public_ga(path: Path) -> pd.DataFrame:
    public_ga = pd.read_csv(path, sep="\t", index_col=0, low_memory=False)
    public_ga.index = public_ga.index.astype(str)
    public_ga.columns = public_ga.columns.astype(str)
    public_ga = public_ga[~public_ga.index.duplicated(keep="first")]
    public_ga = public_ga.apply(pd.to_numeric, errors="coerce")
    public_ga = public_ga.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    public_ga = public_ga.astype(np.float32)
    if (
        public_ga.index[:5]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.isnumeric()
        .mean()
        > 0.5
    ):
        raise ValueError("P22_GA.txt gene names look numeric; check its orientation.")
    transposed = public_ga.T.copy()
    transposed.index = transposed.index.astype(str)
    transposed.columns = transposed.columns.astype(str)
    return transposed


def _clean_gsea_term(term: str) -> str:
    term = re.sub(r"^(?:GO_Biological_Process_2023|KEGG_2019_Mouse|WikiPathways_2019_Mouse)__", "", str(term))
    term = re.sub(r"\s*\(GO:\d+\)$", "", term)
    term = re.sub(r"\s+WP\d+$", "", term)
    replacements = {
        "Gamma-Aminobutyric Acid Signaling Pathway": "GABA signaling",
        "Adenylate Cyclase-Activating Adrenergic Receptor Signaling Pathway": "AC-activating adrenergic signaling",
        "Adrenergic Receptor Signaling Pathway": "Adrenergic receptor signaling",
        "Synaptic Transmission, GABAergic": "GABAergic synaptic transmission",
    }
    return replacements.get(term, term)


def _read_local_gmts(paths: tuple[Path, ...]) -> dict[str, list[str]]:
    gene_sets: dict[str, list[str]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 3:
                    raise ValueError(f"Malformed GMT line {line_number}: {path}")
                term, genes = fields[0], [gene for gene in fields[2:] if gene]
                gene_sets[f"{path.stem}__{term}"] = genes
    if not gene_sets:
        raise ValueError("Frozen GMT files contained no gene sets.")
    return gene_sets


def _run_gsea(
    figure_f_source: pd.DataFrame,
    gmt_paths: tuple[Path, ...],
    full_result_path: Path,
) -> pd.DataFrame:
    import gseapy as gp

    ranking = figure_f_source[["gene", "unique_factor_ratio"]].copy()
    ranking = ranking.sort_values("unique_factor_ratio", ascending=False).reset_index(drop=True)
    ranking["unique_factor_ratio"] += np.linspace(1e-12, 0, len(ranking))
    result = gp.prerank(
        rnk=ranking,
        gene_sets=_read_local_gmts(gmt_paths),
        min_size=10,
        max_size=500,
        permutation_num=1000,
        weight=1.0,
        ascending=False,
        threads=4,
        outdir=None,
        no_plot=True,
        seed=0,
        verbose=False,
    ).res2d.copy()
    if "Term" not in result:
        result = result.reset_index().rename(columns={"index": "Term"})
    result["NES"] = pd.to_numeric(result["NES"], errors="coerce")
    result["FDR q-val"] = pd.to_numeric(result["FDR q-val"], errors="coerce")
    result = result.dropna(subset=["Term", "NES", "FDR q-val"])
    result.to_csv(full_result_path, index=False)

    return _select_displayed_gsea(result)


def _select_displayed_gsea(result: pd.DataFrame) -> pd.DataFrame:
    """Select up to five significant pathways per direction for a new run."""
    significant = result.loc[
        np.isfinite(result["NES"])
        & np.isfinite(result["FDR q-val"])
        & (result["FDR q-val"] >= 0)
        & (result["FDR q-val"] < 0.25)
    ]
    positive = significant[significant["NES"] > 0].nsmallest(5, "FDR q-val")
    negative = significant[significant["NES"] < 0].nsmallest(5, "FDR q-val")
    positive = positive.sort_values("NES", ascending=False)
    negative = negative.sort_values("NES", ascending=False)
    displayed = pd.concat([positive, negative], ignore_index=True)

    output = pd.DataFrame(
        {
            "plot_order_top_to_bottom": np.arange(1, len(displayed) + 1),
            "gene_set_library": displayed["Term"].astype(str).str.split("__").str[0],
            "pathway": displayed["Term"].map(_clean_gsea_term),
            "original_term": displayed["Term"].astype(str),
            "NES": displayed["NES"].to_numpy(dtype=float),
            "FDR_q_value": displayed["FDR q-val"].to_numpy(dtype=float),
            "enriched_group": np.where(
                displayed["NES"].to_numpy(dtype=float) > 0,
                "high-private",
                "low-private",
            ),
        }
    )
    return output


def build_full_source_data(
    inputs: FullInputs,
    prediction_path: Path,
    truth_path: Path,
    source_dir: Path,
    intermediate_dir: Path,
) -> None:
    """Build the five compact source tables consumed by the shared plotter."""
    import scanpy as sc
    from scipy.sparse import issparse
    from sklearn.decomposition import TruncatedSVD
    from sklearn.linear_model import Ridge

    source_dir.mkdir(parents=True, exist_ok=True)
    public_ga = _load_public_ga(inputs.public_ga_path)
    full_atac = sc.read_h5ad(inputs.atac_path)
    true_atac = sc.read_h5ad(truth_path)
    predicted_atac = sc.read_h5ad(prediction_path)
    latent = sc.read_h5ad(inputs.latent_path)
    for adata in (full_atac, true_atac, predicted_atac, latent):
        adata.obs_names = adata.obs_names.astype(str)
    for adata in (full_atac, true_atac, predicted_atac):
        adata.var_names_make_unique()
        adata.var_names = adata.var_names.astype(str)
    for key in ("Z_poe", "W_ATAC", "spatial"):
        if key not in latent.obsm:
            raise KeyError(f'Frozen latent is missing obsm["{key}"].')

    common_spots = latent.obs_names.intersection(full_atac.obs_names)
    latent = latent[common_spots].copy()
    full_atac = full_atac[latent.obs_names].copy()
    split = _load_fixed_split(inputs.split_csv, latent.obs_names)
    train_names = pd.Index(split.loc[split["split"] == "train", "obs_name"])
    test_names = pd.Index(split.loc[split["split"] == "test", "obs_name"])
    available_train = set(public_ga.index).intersection(latent.obs_names)
    available_test = (
        set(public_ga.index)
        .intersection(latent.obs_names)
        .intersection(true_atac.obs_names)
        .intersection(predicted_atac.obs_names)
    )
    train_names = pd.Index([name for name in train_names if name in available_train])
    test_names = pd.Index([name for name in test_names if name in available_test])
    if len(train_names) != 6436 or len(test_names) != 1840:
        raise ValueError(
            "Full analysis lost fixed-split spots after alignment: "
            f"train={len(train_names)}, test={len(test_names)}."
        )

    full_train = full_atac[train_names].copy()
    true_test = true_atac[test_names].copy()
    pred_test = predicted_atac[test_names].copy()
    latent_train = latent[train_names].copy()
    latent_test = latent[test_names].copy()
    y_train_public = public_ga.loc[train_names].to_numpy(dtype=np.float32)
    y_test_public = public_ga.loc[test_names].to_numpy(dtype=np.float32)
    gene_names = public_ga.columns.astype(str).to_numpy()

    true_peak_set = set(true_test.var_names)
    pred_peak_set = set(pred_test.var_names)
    common_peaks = pd.Index(
        [
            peak
            for peak in full_train.var_names
            if peak in true_peak_set and peak in pred_peak_set
        ]
    )
    if common_peaks.empty:
        raise ValueError("No common peaks across full, true and predicted ATAC.")
    full_train = full_train[:, common_peaks].copy()
    true_test = true_test[:, common_peaks].copy()
    pred_test = pred_test[:, common_peaks].copy()

    x_train_all = _as_float32_matrix(full_train.X)
    peak_variance, peak_nonzero = _matrix_feature_stats(x_train_all)
    selected_indices = np.where(
        np.isfinite(peak_variance)
        & np.isfinite(peak_nonzero)
        & (peak_variance > 1e-10)
        & (peak_nonzero >= 0.001)
    )[0]
    if len(selected_indices) == 0:
        raise ValueError("No eligible train-only peaks for the public-GA bridge.")
    x_train = x_train_all[:, selected_indices]
    x_true = _as_float32_matrix(true_test.X[:, selected_indices])
    x_pred = _as_float32_matrix(pred_test.X[:, selected_indices])

    n_components = min(100, x_train.shape[0] - 1, x_train.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, n_iter=7, random_state=0)
    z_train = np.asarray(svd.fit_transform(x_train), dtype=np.float32)
    z_true = np.asarray(x_true @ svd.components_.T, dtype=np.float32)
    z_pred = np.asarray(x_pred @ svd.components_.T, dtype=np.float32)
    z_train, z_true, z_pred = _standardize_from_train(z_train, z_true, z_pred)
    bridge_beta, bridge_intercept = _fit_closed_form_ridge(
        z_train, y_train_public, alpha=1.0
    )

    latent_z_train = np.asarray(latent_train.obsm["Z_poe"], dtype=np.float32)
    latent_w_train = np.asarray(latent_train.obsm["W_ATAC"], dtype=np.float32)
    latent_zw_train = np.concatenate([latent_z_train, latent_w_train], axis=1)
    latent_z_test = np.asarray(latent_test.obsm["Z_poe"], dtype=np.float32)
    latent_w_test = np.asarray(latent_test.obsm["W_ATAC"], dtype=np.float32)
    latent_zw_test = np.concatenate([latent_z_test, latent_w_test], axis=1)

    rows: list[dict[str, Any]] = []
    selected_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    chunk_size = 1000
    for start in range(0, len(gene_names), chunk_size):
        end = min(start + chunk_size, len(gene_names))
        public_train_chunk = y_train_public[:, start:end]
        public_test_chunk = y_test_public[:, start:end]
        bridge_true = (
            z_true @ bridge_beta[:, start:end] + bridge_intercept[None, start:end]
        ).astype(np.float32)
        bridge_pred = (
            z_pred @ bridge_beta[:, start:end] + bridge_intercept[None, start:end]
        ).astype(np.float32)
        bridge_qc = _column_pearson(bridge_true, public_test_chunk)
        translation = _column_pearson(bridge_pred, public_test_chunk)

        predictions: dict[str, np.ndarray] = {}
        for label, train_design, test_design in (
            ("z", latent_z_train, latent_z_test),
            ("w", latent_w_train, latent_w_test),
            ("zw", latent_zw_train, latent_zw_test),
        ):
            model = Ridge(alpha=1.0, fit_intercept=True, solver="auto")
            model.fit(train_design, public_train_chunk)
            predictions[label] = model.predict(test_design)
        r2_z = _raw_r2(public_test_chunk, predictions["z"])
        r2_w = _raw_r2(public_test_chunk, predictions["w"])
        r2_zw = _raw_r2(public_test_chunk, predictions["zw"])
        gain = np.maximum(r2_zw - r2_z, 0)
        variance = public_test_chunk.var(axis=0)

        for local_index, gene in enumerate(gene_names[start:end]):
            rows.append(
                {
                    "gene": gene,
                    "r2_z": float(r2_z[local_index]),
                    "r2_w": float(r2_w[local_index]),
                    "r2_zw": float(r2_zw[local_index]),
                    "unique_factor_gain": float(gain[local_index]),
                    "bridge_qc": float(bridge_qc[local_index]),
                    "translation_accuracy": float(translation[local_index]),
                    "public_ga_variance": float(variance[local_index]),
                }
            )
            if gene in SELECTED_GENES:
                selected_values[gene] = (
                    public_test_chunk[:, local_index].astype(np.float32),
                    bridge_pred[:, local_index].astype(np.float32),
                )

    analysis = pd.DataFrame(rows)
    valid_ratio = analysis["r2_zw"] > 0.001
    analysis["unique_factor_ratio"] = np.nan
    analysis.loc[valid_ratio, "unique_factor_ratio"] = (
        analysis.loc[valid_ratio, "unique_factor_gain"]
        / analysis.loc[valid_ratio, "r2_zw"]
    )
    figure_f = analysis.dropna(
        subset=["unique_factor_ratio", "translation_accuracy", "bridge_qc"]
    )
    figure_f = figure_f[
        (figure_f["bridge_qc"] > 0) & (figure_f["public_ga_variance"] > 1e-8)
    ].copy()
    figure_f = figure_f.sort_values(
        ["unique_factor_ratio", "gene"], kind="mergesort"
    ).reset_index(drop=True)
    figure_f["rank_order"] = np.arange(1, len(figure_f) + 1)
    figure_f["rank_fraction"] = np.linspace(0, 1, len(figure_f))
    n_extreme = int(np.floor(len(figure_f) * 0.20))
    figure_f["ratio_group"] = "middle 60%"
    figure_f.loc[: n_extreme - 1, "ratio_group"] = "low-private bottom 20%"
    figure_f.loc[len(figure_f) - n_extreme :, "ratio_group"] = "high-private top 20%"
    figure_f["displayed_in_panel_h"] = figure_f["gene"].isin(SELECTED_GENES)
    figure_f = figure_f.rename(
        columns={
            "translation_accuracy": "translation_accuracy_MultiVI_derived_GA"
        }
    )
    figure_f[
        [
            "gene",
            "rank_order",
            "rank_fraction",
            "unique_factor_ratio",
            "translation_accuracy_MultiVI_derived_GA",
            "ratio_group",
            "displayed_in_panel_h",
        ]
    ].to_csv(source_dir / "figure3f_ratio_translation.csv", index=False)

    missing_selected = set(SELECTED_GENES).difference(selected_values)
    if missing_selected:
        raise KeyError(f"Full bridge did not produce selected genes: {missing_selected}")
    spatial_test = np.asarray(latent_test.obsm["spatial"], dtype=float)
    spatial_all = np.asarray(latent.obsm["spatial"], dtype=float)
    h_values = pd.DataFrame(
        {
            "obs_name": latent_test.obs_names.astype(str),
            "spatial_x": spatial_test[:, 0],
            "spatial_y": spatial_test[:, 1],
        }
    )
    h_metadata_rows: list[dict[str, Any]] = []
    ratio_by_gene = figure_f.set_index("gene")["unique_factor_ratio"]
    for order, gene in enumerate(SELECTED_GENES, start=1):
        truth, prediction = selected_values[gene]
        h_values[f"true_{gene}"] = truth
        h_values[f"pred_{gene}"] = prediction
        correlation = float(np.corrcoef(truth, prediction)[0, 1])
        combined = np.concatenate([truth, prediction])
        h_metadata_rows.append(
            {
                "display_order": order,
                "display_group": "low_private" if order <= 2 else "private_elevated",
                "gene": gene,
                "unique_factor_ratio": float(ratio_by_gene.loc[gene]),
                "expected_pred_vs_public_ga": correlation,
                "recomputed_pred_vs_public_ga": correlation,
                "pred_correlation_difference": 0.0,
                "expected_true_bridge_vs_public_ga": np.nan,
                "recomputed_true_bridge_vs_public_ga": np.nan,
                "true_bridge_correlation_difference": np.nan,
                "n_test_spots": len(h_values),
                "shared_color_vmin_percentile_1": float(np.nanpercentile(combined, 1)),
                "shared_color_vmax_percentile_99": float(np.nanpercentile(combined, 99)),
            }
        )
    h_values.to_csv(source_dir / "figure3h_selected_gene_activity.csv", index=False)
    pd.DataFrame(h_metadata_rows).to_csv(
        source_dir / "figure3h_selected_gene_activity_metadata.csv", index=False
    )
    pd.DataFrame(
        {
            "obs_name": latent.obs_names.astype(str),
            "spatial_x": spatial_all[:, 0],
            "spatial_y": spatial_all[:, 1],
        }
    ).to_csv(source_dir / "figure3h_spatial_background.csv", index=False)

    displayed_gsea = _run_gsea(
        figure_f,
        inputs.gmt_paths,
        intermediate_dir / "gsea_all_results.csv",
    )
    displayed_gsea.to_csv(source_dir / "figure3g_gsea_displayed.csv", index=False)
    analysis.to_csv(intermediate_dir / "gene_level_full_analysis.csv", index=False)


def run_full_pipeline(
    *,
    workflow_root: Path,
    data_root: Path,
    output_dir: Path,
    public_ga_path: Path | None,
    gmt_dir: Path | None,
    accelerator: str,
) -> tuple[Path, dict[str, Any]]:
    """Run full analysis and return its generated source-data directory."""
    inputs = preflight_full(
        workflow_root=workflow_root,
        data_root=data_root,
        public_ga_path=public_ga_path,
        gmt_dir=gmt_dir,
    )
    intermediate_dir = output_dir / "full_intermediates"
    generated_source_dir = output_dir / "full_source_data"
    prediction, truth, _ = train_multivi(
        inputs,
        intermediate_dir,
        seed=0,
        max_epochs=100,
        accelerator=accelerator,
    )
    build_full_source_data(
        inputs,
        prediction,
        truth,
        generated_source_dir,
        intermediate_dir,
    )
    context = {
        "mode": "full",
        "multivi_seed": 0,
        "multivi_max_epochs": 100,
        "accelerator": accelerator,
        "z_source": "Z_poe",
        "w_source": "W_ATAC",
        "public_ga_path": str(inputs.public_ga_path),
        "gmt_paths": [str(path) for path in inputs.gmt_paths],
        "fixed_split_path": str(inputs.split_csv),
        "latent_path": str(inputs.latent_path),
    }
    return generated_source_dir, context
