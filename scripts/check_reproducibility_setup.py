"""Check saved-result plotting and model environments with separate profiles.

The default profile checks selected figure workflows. Explicit profiles
validate model-training and external-translation requirements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import os
import sys
from pathlib import Path

try:
    from .figure_workflows import Workflow, resolve_data_root as resolve_plot_data_root, selected_workflows
except ImportError:
    from figure_workflows import Workflow, resolve_data_root as resolve_plot_data_root, selected_workflows


REPLOT_IMPORTS = ("numpy", "pandas", "scipy", "matplotlib", "anndata", "h5py")


CORE_IMPORTS = (
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "skmisc",
    "matplotlib",
    "scanpy",
    "anndata",
    "torch",
    "torch_geometric",
    "hnswlib",
    "rpy2",
    "spaDIVA",
)

EXTERNAL_TRANSLATION_IMPORTS = (
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "matplotlib",
    "scanpy",
    "anndata",
    "torch",
    "torchvision",
    "scvi",
    "mudata",
    "seaborn",
    "gseapy",
    "h5py",
)


# Minimal repository inventory. Versioned baseline source data support compact
# redraws; external-method training uses its dedicated workflow.
REPOSITORY_GROUPS: dict[str, tuple[str, ...]] = {
    "root entry and package metadata": (
        "reproduce.py",
        "pyproject.toml",
        "DATA_AVAILABILITY.md",
    ),
    "spaDIVA release snapshot": (
        "spaDIVA/__init__.py",
        "spaDIVA/analysis.py",
        "spaDIVA/cal_wnn_weight.py",
        "spaDIVA/compute_poe.py",
        "spaDIVA/interpretation.py",
        "spaDIVA/model.py",
        "spaDIVA/train.py",
        "spaDIVA/utils.py",
    ),
    "simulation compact/model/source": (
        "datasets/simulation/reproduce.py",
        "datasets/simulation/model/run_spadiva.py",
        "datasets/simulation/data_generation/generate_simulation_data.py",
        "datasets/simulation/source_data/figure2_representative_features.csv",
        "datasets/simulation/source_data/figure2_translation_features.csv",
        "datasets/simulation/source_data/figure2c_ari_benchmark.csv",
        "datasets/simulation/source_data/s1_ablation_benchmark.csv",
    ),
    "P22 compact external-translation workflow": (
        "datasets/P22_ATAC_RNA/external_translation_figure_reproduction/reproduce.py",
        "datasets/P22_ATAC_RNA/external_translation_figure_reproduction/_plotting.py",
        "datasets/P22_ATAC_RNA/external_translation_figure_reproduction/source_data/figure3f_ratio_translation.csv",
        "datasets/P22_ATAC_RNA/external_translation_figure_reproduction/source_data/figure3g_gsea_displayed.csv",
        "datasets/P22_ATAC_RNA/external_translation_figure_reproduction/source_data/figure3h_selected_gene_activity.csv",
        "datasets/P22_ATAC_RNA/external_translation_figure_reproduction/source_data/figure3h_selected_gene_activity_metadata.csv",
        "datasets/P22_ATAC_RNA/external_translation_figure_reproduction/source_data/figure3h_spatial_background.csv",
    ),
    "P22 compact model/domain workflow": (
        "datasets/P22_ATAC_RNA/input_and_model_reproduction/reproduce_model_and_domains.py",
        "datasets/P22_ATAC_RNA/input_and_model_reproduction/source_data/domain_map_metadata.csv",
        "datasets/P22_ATAC_RNA/input_and_model_reproduction/source_data/domain_map_values.csv.gz",
        "datasets/P22_ATAC_RNA/input_and_model_reproduction/source_data/figure3c_morans_i_per_domain.csv",
        "datasets/P22_ATAC_RNA/input_and_model_reproduction/source_data/figure3e_jaccard_original_values.csv",
        "datasets/P22_ATAC_RNA/input_and_model_reproduction/source_data/supplementary_s2_morans_i_per_domain.csv",
    ),
    "lymph-node compact external-translation workflow": (
        "datasets/human_lymph_node_RNA_ADT/external_translation_figure_reproduction/reproduce.py",
        "datasets/human_lymph_node_RNA_ADT/external_translation_figure_reproduction/workflow.py",
        "datasets/human_lymph_node_RNA_ADT/external_translation_figure_reproduction/source_data/Figure_4_manuscript_source.csv",
        "datasets/human_lymph_node_RNA_ADT/external_translation_figure_reproduction/artifacts/split_assignments.csv",
        "datasets/human_lymph_node_RNA_ADT/external_translation_figure_reproduction/artifacts/totalvi_test_pred_protein.h5ad",
        "datasets/human_lymph_node_RNA_ADT/external_translation_figure_reproduction/expected/metrics.json",
    ),
    "lymph-node compact model/domain workflow": (
        "datasets/human_lymph_node_RNA_ADT/input_and_model_reproduction/reproduce.py",
        "datasets/human_lymph_node_RNA_ADT/input_and_model_reproduction/model_reproduction.ipynb",
        "datasets/human_lymph_node_RNA_ADT/input_and_model_reproduction/source_data/spatial_domains.csv",
        "datasets/human_lymph_node_RNA_ADT/input_and_model_reproduction/source_data/benchmark_runs.csv",
        "datasets/human_lymph_node_RNA_ADT/input_and_model_reproduction/source_data/figure4c_jaccard_original_values.csv",
    ),
    "E13/E15/E18 model and frozen source": (
        "datasets/E13E15E18_multi_slice/model_reproduction.ipynb",
        "datasets/E13E15E18_multi_slice/source_data/ari_benchmark.csv",
        "datasets/E13E15E18_multi_slice/source_data/jaccard_by_section.csv",
        "datasets/E13E15E18_multi_slice/source_data/jaccard_summary.csv",
        "datasets/E13E15E18_multi_slice/source_data/single_vs_joint_ari.csv",
        "datasets/E13E15E18_multi_slice/source_data/spatial_domains.csv",
    ),
    "P21/P22 model and frozen source": (
        "datasets/P21P22_multi_slice/model_reproduction.ipynb",
        "datasets/P21P22_multi_slice/source_data/jaccard_by_section.csv",
        "datasets/P21P22_multi_slice/source_data/jaccard_summary.csv",
        "datasets/P21P22_multi_slice/source_data/moran_by_domain.csv",
        "datasets/P21P22_multi_slice/source_data/spatial_domains.csv",
    ),
    "Figure 5 compact plotter": ("scripts/reproduce_figure5.py",),
    "environments": (
        "environments/environment_spadiva_release.yml",
        "environments/environment_external_translation_paper.yml",
    ),
}


# This mirrors the five input-data sections in DATA_AVAILABILITY.md.
ZENODO_DATA_FILES: dict[str, tuple[str, ...]] = {
    "simulation": (
        "datasets/simulation/input_data/ADT_100.h5ad",
        "datasets/simulation/input_data/RNA_ZINB.h5ad",
    ),
    "P22 ATAC-RNA": (
        "datasets/P22_ATAC_RNA/input_data/adata_peaks_normalized.h5ad",
        "datasets/P22_ATAC_RNA/input_data/adata_RNA.h5ad",
    ),
    "human lymph node RNA-ADT": (
        "datasets/human_lymph_node_RNA_ADT/input_data/human lymph node adata_ADT.h5ad",
        "datasets/human_lymph_node_RNA_ADT/input_data/human lymph node adata_RNA.h5ad",
        "datasets/human_lymph_node_RNA_ADT/annotation/annotation.csv",
    ),
    "E13/E15/E18 multi-slice": (
        "datasets/E13E15E18_multi_slice/input_data/Mouse_embryonic_brain/E13/E13_adata_atac.h5ad",
        "datasets/E13E15E18_multi_slice/input_data/Mouse_embryonic_brain/E13/E13_adata_rna.h5ad",
        "datasets/E13E15E18_multi_slice/input_data/Mouse_embryonic_brain/E15/E15_adata_atac.h5ad",
        "datasets/E13E15E18_multi_slice/input_data/Mouse_embryonic_brain/E15/E15_adata_rna.h5ad",
        "datasets/E13E15E18_multi_slice/input_data/Mouse_embryonic_brain/E18/E18_adata_atac.h5ad",
        "datasets/E13E15E18_multi_slice/input_data/Mouse_embryonic_brain/E18/E18_adata_rna.h5ad",
        "datasets/E13E15E18_multi_slice/annotation/E13_5-S1_annotation.csv",
        "datasets/E13E15E18_multi_slice/annotation/E15_5-S1_annotation.csv",
        "datasets/E13E15E18_multi_slice/annotation/E18_5-S1_annotation.csv",
    ),
    "P21/P22 multi-slice": (
        "datasets/P21P22_multi_slice/input_data/Mouse_postnatal_brain/slice1/s1_adata_atac.h5ad",
        "datasets/P21P22_multi_slice/input_data/Mouse_postnatal_brain/slice1/s1_adata_rna.h5ad",
        "datasets/P21P22_multi_slice/input_data/Mouse_postnatal_brain/slice2/s2_adata_atac.h5ad",
        "datasets/P21P22_multi_slice/input_data/Mouse_postnatal_brain/slice2/s2_adata_rna.h5ad",
        "datasets/P21P22_multi_slice/input_data/Mouse_postnatal_brain/slice3/s3_adata_atac.h5ad",
        "datasets/P21P22_multi_slice/input_data/Mouse_postnatal_brain/slice3/s3_adata_rna.h5ad",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the minimal spaDIVA figure-reproduction setup."
    )
    parser.add_argument(
        "--profile", choices=("replot", "model", "external"), default=None,
        help="Default: replot. Model/external profiles check training dependencies.",
    )
    parser.add_argument(
        "--figure", choices=("all", "2", "3", "4", "5"), default="all",
        help="Saved-result figure(s) to check in the replot profile.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Extracted Zenodo data root for inputs selected by the replot profile.",
    )
    parser.add_argument(
        "--require-all-data",
        action="store_true",
        help="Check all model inputs (also selects model profile unless --profile is given).",
    )
    parser.add_argument(
        "--import-check",
        action="store_true",
        help="Import packages and report their versions during validation.",
    )
    parser.add_argument(
        "--external-translation-env",
        action="store_true",
        help="Compatibility alias for --profile external.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_replot_setup(
    repo_root: Path,
    workflows: tuple[Workflow, ...],
    data_root: Path | None,
    *,
    import_check: bool = False,
) -> int:
    """Validate selected inputs before plotting using standard-library checks."""
    errors: list[str] = []
    manifest_path = repo_root / "MANIFEST.tsv"
    hashes: dict[str, str] = {}
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
            hashes = {row["relative_path"]: row["sha256"] for row in csv.DictReader(stream, delimiter="\t")}
    else:
        errors.append("MANIFEST.tsv is required to verify versioned sources.")

    checked: set[str] = set()
    external: set[str] = set()
    for workflow in workflows:
        print(f"[WORKFLOW] {workflow.key}: {', '.join(workflow.panels)}")
        for relative in (workflow.script, *workflow.code_files, *workflow.source_files):
            if relative in checked:
                continue
            checked.add(relative)
            path = repo_root / relative
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"Missing/empty repository file: {relative}")
            elif relative in workflow.source_files:
                if relative not in hashes:
                    errors.append(f"Frozen source is not recorded in MANIFEST.tsv: {relative}")
                elif file_sha256(path) != hashes[relative]:
                    errors.append(f"Frozen source differs from MANIFEST.tsv: {relative}")
        external.update(workflow.data_files)

    if external and data_root is None:
        errors.append("Set SPADIVA_DATA_ROOT or pass --data-root to the extracted data directory.")
    for relative in sorted(external):
        path = data_root / relative if data_root is not None else None
        if path is None or not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing/empty plotting input: {path if path is not None else relative}")

    print(f"\nSaved-result files: {len(checked)}; required external inputs: {len(external)}")
    if not external:
        print("[OK] The selected figures need no Zenodo inputs or external predictions beyond bundled files.")
    missing_packages = check_packages(REPLOT_IMPORTS, import_check)
    for message in errors:
        print(f"[FAIL] {message}")
    if errors or missing_packages:
        print("\nPreflight failed; no figure workflow was started.")
        return 1
    print("\nSaved-result preflight passed. No models will be trained.")
    return 0


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__).resolve()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "spaDIVA").is_dir() and (candidate / "reproduce.py").is_file():
            return candidate
    raise RuntimeError("Could not locate the spaDIVA_reproducibility root.")


def resolve_data_root(repo_root: Path, args: argparse.Namespace) -> tuple[Path | None, bool]:
    env_value = os.environ.get("SPADIVA_DATA_ROOT")
    explicitly_selected = args.data_root is not None or bool(env_value)

    if args.data_root is not None:
        root = args.data_root.expanduser()
    elif env_value:
        root = Path(env_value).expanduser()
    elif args.require_all_data:
        root = repo_root / "data"
    else:
        return None, False

    if not root.is_absolute():
        root = (repo_root / root).resolve()
    else:
        root = root.resolve()
    return root, explicitly_selected or args.require_all_data


def check_repository(repo_root: Path) -> list[str]:
    missing: list[str] = []
    for group, relative_paths in REPOSITORY_GROUPS.items():
        group_missing = []
        for relative_path in relative_paths:
            path = repo_root / relative_path
            if not path.is_file() or path.stat().st_size == 0:
                group_missing.append(relative_path)
        if group_missing:
            print(f"[FAIL] {group}")
            for relative_path in group_missing:
                print(f"  missing/empty: {relative_path}")
            missing.extend(group_missing)
        else:
            print(f"[OK]   {group} ({len(relative_paths)} files)")
    return missing


def check_packages(names: tuple[str, ...], import_check: bool) -> list[str]:
    missing: list[str] = []
    for name in names:
        try:
            if importlib.util.find_spec(name) is None:
                raise ModuleNotFoundError(name)
            if import_check:
                importlib.import_module(name)
        except Exception as exc:  # imports may fail after discovery (for example DLL/R mismatch)
            action = "import" if import_check else "package"
            print(f"[FAIL] {action:7s} {name}: {type(exc).__name__}: {exc}")
            missing.append(name)
        else:
            action = "import" if import_check else "package"
            print(f"[OK]   {action:7s} {name}")
    return missing


def check_data(data_root: Path) -> list[str]:
    missing: list[str] = []
    if not data_root.is_dir():
        print(f"[FAIL] data root is not a directory: {data_root}")
        return [str(data_root)]

    for dataset, relative_paths in ZENODO_DATA_FILES.items():
        dataset_missing = [
            relative_path
            for relative_path in relative_paths
            if not (data_root / relative_path).is_file()
        ]
        if dataset_missing:
            print(f"[FAIL] data: {dataset}")
            for relative_path in dataset_missing:
                print(f"  missing: {relative_path}")
            missing.extend(dataset_missing)
        else:
            print(f"[OK]   data: {dataset} ({len(relative_paths)} files)")
    return missing


def main() -> int:
    args = parse_args()
    repo_root = find_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    profile = "external" if args.external_translation_env else (
        args.profile or ("model" if args.require_all_data else "replot")
    )
    if profile == "replot":
        return check_replot_setup(
            repo_root,
            selected_workflows(args.figure),
            resolve_plot_data_root(repo_root, args.data_root),
            import_check=args.import_check,
        )

    data_root, require_data = resolve_data_root(repo_root, args)
    package_set = (
        EXTERNAL_TRANSLATION_IMPORTS
        if profile == "external"
        else CORE_IMPORTS
    )
    environment_name = (
        "external translation" if profile == "external" else "spaDIVA release"
    )

    print(f"Repository root: {repo_root}")
    print(f"Environment:     {environment_name}")
    print(f"Package mode:    {'import' if args.import_check else 'discoverability'}")
    print(f"Zenodo data:     {data_root if require_data else 'not required'}")

    print("\nRepository assets")
    missing_repository = check_repository(repo_root)

    print(f"\nPython packages ({environment_name})")
    missing_packages = check_packages(package_set, args.import_check)

    if require_data and data_root is not None:
        print("\nZenodo inputs")
        missing_data = check_data(data_root)
    else:
        missing_data = []
        print(
            "\nZenodo inputs\n"
            "[SKIP] Set SPADIVA_DATA_ROOT, pass --data-root, or use "
            "--require-all-data to validate all five datasets."
        )

    if missing_repository:
        print(f"\nSetup check failed: {len(missing_repository)} repository asset(s) missing.")
        return 3
    if missing_packages:
        print(f"\nSetup check failed: {len(missing_packages)} package(s) unavailable.")
        return 1
    if missing_data:
        print(f"\nSetup check failed: {len(missing_data)} Zenodo input(s) missing.")
        return 2

    print("\nSetup check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
