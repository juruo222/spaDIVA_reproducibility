"""Define contracts for saved-result figure workflows using the standard library.

Paths identify repository, extracted-data, and results-root inputs through
their field names. Model runs use the dedicated entry points documented in
MODEL_RUNS.md.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Workflow:
    """One existing plotting command and the files it actually requires."""

    key: str
    figure: str
    panels: tuple[str, ...]
    script: str
    arguments: tuple[str, ...]
    source_files: tuple[str, ...]
    code_files: tuple[str, ...]
    data_files: tuple[str, ...]
    outputs: tuple[str, ...]
    output_option: str
    output_subdir: str


_SIM = "datasets/simulation"
_P22_MODEL = "datasets/P22_ATAC_RNA/input_and_model_reproduction"
_P22_TRANSLATION = "datasets/P22_ATAC_RNA/external_translation_figure_reproduction"
_LYMPH_MODEL = "datasets/human_lymph_node_RNA_ADT/input_and_model_reproduction"
_LYMPH_TRANSLATION = "datasets/human_lymph_node_RNA_ADT/external_translation_figure_reproduction"
_E13 = "datasets/E13E15E18_multi_slice/source_data"
_P21 = "datasets/P21P22_multi_slice/source_data"


WORKFLOWS: tuple[Workflow, ...] = (
    Workflow(
        key="simulation",
        figure="2",
        panels=("Figure 2c", "Figure 2f", "Figure 2g", "Figure 2h", "S1d"),
        script=f"{_SIM}/reproduce.py",
        arguments=("--mode", "plot"),
        source_files=tuple(
            f"{_SIM}/source_data/{name}"
            for name in (
                "figure2_translation_features.csv",
                "s1_ablation_benchmark.csv",
                "figure2_representative_features.csv",
                "figure2c_ari_benchmark.csv",
            )
        ),
        code_files=(f"{_SIM}/reproduce.py",),
        data_files=("datasets/simulation/input_data/ADT_100.h5ad",),
        outputs=(
            "Figure_2/panel_c.pdf",
            "Figure_2/panel_f.pdf",
            "Figure_2/panel_g.pdf",
            "Figure_2/panel_h.pdf",
            "Supplementary_Figure_S1/panel_d.pdf",
        ),
        output_option="--results-root",
        output_subdir="",
    ),
    Workflow(
        key="p22_domains",
        figure="3",
        panels=("Figure 3b", "Figure 3c", "Figure 3d", "Figure 3e", "S2a", "S2b", "S2c"),
        script=f"{_P22_MODEL}/reproduce_model_and_domains.py",
        arguments=("--mode", "compact"),
        source_files=tuple(
            f"{_P22_MODEL}/source_data/{name}"
            for name in (
                "domain_map_values.csv.gz",
                "domain_map_metadata.csv",
                "figure3b_spadiva_cluster_annotations.csv",
                "figure3c_morans_i_per_domain.csv",
                "supplementary_s2_morans_i_per_domain.csv",
                "figure3e_jaccard_original_values.csv",
            )
        ),
        code_files=(f"{_P22_MODEL}/reproduce_model_and_domains.py",),
        data_files=(),
        outputs=(
            "Figure_3/panel_b.pdf",
            "Figure_3/panel_c.pdf",
            "Figure_3/panel_d.pdf",
            "Figure_3/panel_e.pdf",
            "Figure_3/Supplementary_Figure_S2_panel_a.pdf",
            "Figure_3/Supplementary_Figure_S2_panel_b.pdf",
            "Figure_3/Supplementary_Figure_S2_panel_c.pdf",
        ),
        output_option="--output-dir",
        output_subdir="Figure_3",
    ),
    Workflow(
        key="p22_translation",
        figure="3",
        panels=("Figure 3f", "Figure 3g", "Figure 3h"),
        script=f"{_P22_TRANSLATION}/reproduce.py",
        arguments=("--mode", "compact"),
        source_files=tuple(
            f"{_P22_TRANSLATION}/source_data/{name}"
            for name in (
                "figure3f_ratio_translation.csv",
                "figure3g_gsea_displayed.csv",
                "figure3h_selected_gene_activity.csv",
                "figure3h_selected_gene_activity_metadata.csv",
                "figure3h_spatial_background.csv",
            )
        ),
        code_files=(f"{_P22_TRANSLATION}/reproduce.py", f"{_P22_TRANSLATION}/_plotting.py"),
        data_files=(),
        outputs=("Figure_3/panel_f.pdf", "Figure_3/panel_g.pdf", "Figure_3/panel_h.pdf"),
        output_option="--output-dir",
        output_subdir="Figure_3",
    ),
    Workflow(
        key="lymph_domains",
        figure="4",
        panels=("Figure 4a", "Figure 4b", "Figure 4c"),
        script=f"{_LYMPH_MODEL}/reproduce.py",
        arguments=(),
        source_files=tuple(
            f"{_LYMPH_MODEL}/source_data/{name}"
            for name in (
                "spatial_domains.csv",
                "benchmark_runs.csv",
                "figure4c_jaccard_original_values.csv",
            )
        ),
        code_files=(f"{_LYMPH_MODEL}/reproduce.py",),
        data_files=(),
        outputs=("Figure_4/panel_a.pdf", "Figure_4/panel_b.pdf", "Figure_4/panel_c.pdf"),
        output_option="--output-dir",
        output_subdir="Figure_4",
    ),
    Workflow(
        key="lymph_translation",
        figure="4",
        panels=("Figure 4d", "Figure 4e", "Figure 4f", "S3a", "S3b", "S3c"),
        script=f"{_LYMPH_TRANSLATION}/reproduce.py",
        arguments=(),
        source_files=(
            f"{_LYMPH_TRANSLATION}/source_data/Figure_4_manuscript_source.csv",
            f"{_LYMPH_TRANSLATION}/artifacts/split_assignments.csv",
            f"{_LYMPH_TRANSLATION}/artifacts/totalvi_test_pred_protein.h5ad",
        ),
        code_files=(f"{_LYMPH_TRANSLATION}/reproduce.py", f"{_LYMPH_TRANSLATION}/workflow.py"),
        data_files=(
            "datasets/human_lymph_node_RNA_ADT/input_data/human lymph node adata_ADT.h5ad",
            "datasets/human_lymph_node_RNA_ADT/input_data/human lymph node adata_RNA.h5ad",
            "datasets/human_lymph_node_RNA_ADT/annotation/annotation.csv",
        ),
        outputs=(
            "Figure_4/panel_d.pdf",
            "Figure_4/panel_e.pdf",
            "Figure_4/panel_f.pdf",
            "Supplementary_Figure_S3/panel_a.pdf",
            "Supplementary_Figure_S3/panel_b.pdf",
            "Supplementary_Figure_S3/panel_c.pdf",
        ),
        output_option="--results-dir",
        output_subdir="",
    ),
    Workflow(
        key="figure5",
        figure="5",
        panels=("Figure 5b", "Figure 5c", "Figure 5d", "Figure 5f", "Figure 5g", "Figure 5h", "S4i"),
        script="scripts/reproduce_figure5.py",
        arguments=(),
        source_files=(
            f"{_E13}/spatial_domains.csv",
            f"{_P21}/spatial_domains.csv",
            f"{_E13}/ari_benchmark.csv",
            f"{_E13}/jaccard_by_section.csv",
            f"{_P21}/jaccard_by_section.csv",
            f"{_P21}/moran_by_domain.csv",
            f"{_E13}/single_vs_joint_ari.csv",
        ),
        code_files=("scripts/reproduce_figure5.py",),
        data_files=(),
        outputs=(
            "Figure_5/panel_b_spatial_domains.pdf",
            "Figure_5/panel_c_ari.pdf",
            "Figure_5/panel_d_jaccard.pdf",
            "Figure_5/panel_f_spatial_domains.pdf",
            "Figure_5/panel_g_moran.pdf",
            "Figure_5/panel_h_jaccard.pdf",
            "Figure_5/supplementary_S4_single_vs_joint_ari.pdf",
        ),
        output_option="--output-dir",
        output_subdir="Figure_5",
    ),
)


def selected_workflows(figure: str = "all") -> tuple[Workflow, ...]:
    """Return the fixed saved-result workflows for a main-figure selection."""
    if figure == "all":
        return WORKFLOWS
    if figure not in {workflow.figure for workflow in WORKFLOWS}:
        raise ValueError(f"Unknown figure {figure!r}; choose all, 2, 3, 4, or 5.")
    return tuple(workflow for workflow in WORKFLOWS if workflow.figure == figure)


def resolve_data_root(repo_root: Path, requested: Path | str | None = None) -> Path:
    """Resolve an explicit path, the environment, or ``repo/data``.

    An explicitly selected path is authoritative. Otherwise use the
    ``SPADIVA_DATA_ROOT`` environment setting when present, followed by
    ``repo/data`` as the expected default path.
    """
    if requested is not None:
        return Path(requested).expanduser().resolve()
    if os.environ.get("SPADIVA_DATA_ROOT"):
        return Path(os.environ["SPADIVA_DATA_ROOT"]).expanduser().resolve()
    repo_root = Path(repo_root).expanduser().resolve()
    return repo_root / "data"


def required_files(
    workflows: Iterable[Workflow],
    repo_root: Path,
    data_root: Path | None,
) -> list[Path]:
    """Return deduplicated concrete code and input paths."""
    repo_root = Path(repo_root).expanduser().resolve()
    data_root = resolve_data_root(repo_root) if data_root is None else Path(data_root).expanduser().resolve()
    paths: list[Path] = []
    for workflow in workflows:
        paths.extend(repo_root / relative for relative in workflow.code_files)
        paths.extend(repo_root / relative for relative in workflow.source_files)
        paths.extend(data_root / relative for relative in workflow.data_files)
    return list(dict.fromkeys(paths))
