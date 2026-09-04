"""Run the canonical release spaDIVA simulation workflow.

This entry point follows the release tutorial, uses the standardized Zenodo
layout, saves all latent representations (including sampled Z_poe), and writes
the model-level quantities needed by Figure 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def find_repo_root() -> Path:
    for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "spaDIVA").is_dir():
            return candidate
    raise RuntimeError("Could not locate the spaDIVA_reproducibility root.")


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


def set_all_seeds(seed: int, torch_module: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cudnn.benchmark = False
    torch_module.use_deterministic_algorithms(True, warn_only=True)


def choose_cuda(device: str, torch_module: Any) -> bool:
    if device == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device == "cuda" or (device == "auto" and torch_module.cuda.is_available())


def neighbor_jaccard(a: np.ndarray, b: np.ndarray, k: int = 50) -> float:
    from sklearn.neighbors import NearestNeighbors

    if a.shape[0] != b.shape[0]:
        raise ValueError("Representations must contain the same number of spots.")
    n_neighbors = min(k + 1, a.shape[0])

    def neighborhoods(x: np.ndarray) -> np.ndarray:
        indices = NearestNeighbors(
            n_neighbors=n_neighbors, metric="euclidean", algorithm="auto"
        ).fit(x).kneighbors(x, return_distance=False)
        return np.asarray([row[row != i][:k] for i, row in enumerate(indices)])

    a_nn = neighborhoods(np.asarray(a))
    b_nn = neighborhoods(np.asarray(b))
    values = []
    for left, right in zip(a_nn, b_nn):
        left_set, right_set = set(left), set(right)
        values.append(len(left_set & right_set) / len(left_set | right_set))
    return float(np.mean(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--poe-sample-seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Keep ``--help`` lightweight by deferring scientific-stack and spaDIVA
    # imports until model execution.
    import scanpy as sc
    import torch
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    from spaDIVA.analysis import collect_spadiva_outputs, fuse_shared_latent
    from spaDIVA.train import infer_latents, train_spadiva
    from spaDIVA.utils import cal_spatial, clr_normalize_each_cell, cluster

    repo_root = find_repo_root()
    data_root = resolve_data_root(repo_root, args.data_root)
    input_dir = data_root / "datasets" / "simulation" / "input_data"
    adt_path = input_dir / "ADT_100.h5ad"
    rna_path = input_dir / "RNA_ZINB.h5ad"
    for path in (adt_path, rna_path):
        if not path.exists():
            raise FileNotFoundError(path)

    default_output = repo_root / "datasets" / "simulation" / "results" / "model"
    output_dir = (args.output_dir or default_output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    use_cuda = choose_cuda(args.device, torch)
    set_all_seeds(args.seed, torch)

    adata_adt = sc.read_h5ad(adt_path)
    adata_rna = sc.read_h5ad(rna_path)
    adata_adt.var_names_make_unique()
    adata_rna.var_names_make_unique()
    if not adata_adt.obs_names.equals(adata_rna.obs_names):
        common = adata_adt.obs_names.intersection(adata_rna.obs_names)
        if len(common) != adata_adt.n_obs or len(common) != adata_rna.n_obs:
            raise ValueError("The two simulation modalities do not contain the same spots.")
        adata_adt = adata_adt[common].copy()
        adata_rna = adata_rna[common].copy()
    # The deposited simulation files retain log1p values in ``X`` and the
    # original observations in ``layers["counts"]``.  Always restart from
    # counts so that the documented normalization is applied exactly once.
    if "counts" not in adata_adt.layers or "counts" not in adata_rna.layers:
        raise KeyError(
            "Both simulation inputs must contain layers['counts']; refusing "
            "to preprocess an already transformed X matrix a second time."
        )
    adata_adt.X = adata_adt.layers["counts"].copy().astype("float32")
    adata_rna.X = adata_rna.layers["counts"].copy().astype("float32")

    adt_category = adata_adt.obs["category"].astype(str)
    rna_category = adata_rna.obs["category"].astype(str)
    combined = pd.Series("Background", index=adata_adt.obs_names, dtype=object)
    only_adt = (adt_category != "Background") & (rna_category == "Background")
    only_rna = (adt_category == "Background") & (rna_category != "Background")
    shared = (adt_category != "Background") & (rna_category != "Background")
    combined[only_adt] = adt_category[only_adt]
    combined[only_rna] = rna_category[only_rna]
    combined[shared] = adt_category[shared]

    truth = {
        "Z": combined.where(combined.isin(["factor 1", "factor 2"]), "Background"),
        "Z_W": combined,
        "W_ADT": combined.where(combined == "factor 3", "Background"),
        "W_RNA": combined.where(combined == "factor 4", "Background"),
    }

    adata_adt = clr_normalize_each_cell(adata_adt)
    sc.pp.scale(adata_adt)
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.scale(adata_rna)
    # Match the release tutorial's automatic solver choice while fixing the
    # randomized branch for cross-machine repeatability.
    x_adt = PCA(n_components=64, svd_solver="auto", random_state=args.seed).fit_transform(
        adata_adt.to_df()
    ).astype("float32")
    x_rna = PCA(n_components=64, svd_solver="auto", random_state=args.seed).fit_transform(
        adata_rna.to_df()
    ).astype("float32")
    spatial = np.asarray(adata_adt.obsm["spatial"])
    edge_index = cal_spatial(spatial, k=4)

    model, losses = train_spadiva(
        x_adt,
        x_rna,
        x_adt,
        x_rna,
        edge_index=edge_index,
        learning_rate=1e-3,
        weight=1.0,
        max_epochs=args.epochs,
        KL_weight=[1.0, 1.0, 1.0, 1.0],
        use_cuda=use_cuda,
    )
    z_poe, z_adt, z_rna, w_adt, w_rna, xhat_adt, xhat_rna = infer_latents(
        model,
        x_adt,
        x_rna,
        edge_index=edge_index,
        use_cuda=use_cuda,
        sample_seed=args.poe_sample_seed,
    )
    z_wnn = fuse_shared_latent(z_adt, z_rna, k=20)
    zdata = collect_spadiva_outputs(
        Z=z_wnn,
        W1=w_adt,
        W2=w_rna,
        spatial=spatial,
        obs_names=adata_adt.obs_names,
        modality_names=("ADT", "RNA"),
        Z_poe=z_poe,
        Z1=z_adt,
        Z2=z_rna,
        X1_hat=xhat_adt,
        X2_hat=xhat_rna,
        model_seed=args.seed,
        poe_sample_seed=args.poe_sample_seed,
    )
    zdata.obs["truth_integrated"] = pd.Categorical(combined.reindex(zdata.obs_names))

    cluster_counts = {"Z": 3, "Z_W": 5, "W_ADT": 2, "W_RNA": 2}
    metric_rows = []
    for key, n_clusters in cluster_counts.items():
        labels = cluster(
            zdata.obsm[key],
            num_cluster=n_clusters,
            spatial=spatial,
            random_seed=args.seed,
            show=False,
            return_labels=True,
        )
        zdata.obs[f"mclust_{key}"] = pd.Categorical(labels)
        target = truth[key].reindex(zdata.obs_names).to_numpy()
        metric_rows.append(
            {
                "representation": key,
                "n_clusters": n_clusters,
                "ARI": adjusted_rand_score(target, labels),
                "NMI": normalized_mutual_info_score(target, labels),
            }
        )

    representation_map = {
        "shared": zdata.obsm["Z"],
        "shared+ADT": np.concatenate((zdata.obsm["Z"], zdata.obsm["W_ADT"]), axis=1),
        "shared+RNA": np.concatenate((zdata.obsm["Z"], zdata.obsm["W_RNA"]), axis=1),
        "ADT": zdata.obsm["W_ADT"],
        "RNA": zdata.obsm["W_RNA"],
    }
    jaccard_rows = []
    for modality, reference in (("ADT", x_adt), ("RNA", x_rna)):
        for representation, values in representation_map.items():
            jaccard_rows.append(
                {
                    "modality": modality,
                    "representation": representation,
                    "k": 50,
                    "metric": "euclidean",
                    "jaccard_similarity": neighbor_jaccard(values, reference, k=50),
                }
            )

    latent_path = output_dir / "spadiva_simulation.h5ad"
    zdata.write_h5ad(latent_path)
    pd.DataFrame(metric_rows).to_csv(output_dir / "clustering_metrics.csv", index=False)
    pd.DataFrame(jaccard_rows).to_csv(output_dir / "jaccard_similarity.csv", index=False)
    pd.DataFrame(
        {"epoch": np.arange(1, len(losses) + 1), "training_loss": np.asarray(losses, dtype=float)}
    ).to_csv(output_dir / "training_loss.csv", index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "entry_point": Path(__file__).resolve().relative_to(repo_root).as_posix(),
        "python": sys.version,
        "platform": platform.platform(),
        "device_requested": args.device,
        "cuda_used": use_cuda,
        "epochs": args.epochs,
        "model_seed": args.seed,
        "poe_sample_seed": args.poe_sample_seed,
        "inputs": {
            adt_path.relative_to(data_root).as_posix(): sha256(adt_path),
            rna_path.relative_to(data_root).as_posix(): sha256(rna_path),
        },
        "output_latent": latent_path.relative_to(output_dir).as_posix(),
        "output_latent_sha256": sha256(latent_path),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Saved canonical simulation model outputs to {output_dir}")


if __name__ == "__main__":
    main()
