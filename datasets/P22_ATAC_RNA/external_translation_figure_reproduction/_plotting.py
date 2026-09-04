"""Shared validation and plotting for the P22 Figure 3f-h workflow.

Both compact and full reproduction modes write the same five source tables and
call :func:`render_panels`.  This keeps statistical definitions and figure
styling identical between the deposited-source and from-scratch workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


FONT_SIZE = 7.0
EXTREME_FRACTION = 0.20
DISPLAY_GENES = ("Pde10a", "Rgs9", "Tmeff2", "Nab2")
EXPECTED_GSEA_PATHWAYS = (
    "Staphylococcus aureus infection",
    "Graft-versus-host disease",
    "Allograft rejection",
    "Ascorbate and aldarate metabolism",
    "Type I diabetes mellitus",
    "Monoamine GPCRs",
    "GABA signaling",
    "Adrenergic receptor signaling",
    "GABAergic synaptic transmission",
    "AC-activating adrenergic signaling",
)
EXPECTED_SPEARMAN = -0.7025019959722367


SOURCE_FILES = {
    "panel_f": "figure3f_ratio_translation.csv",
    "panel_g": "figure3g_gsea_displayed.csv",
    "panel_h_values": "figure3h_selected_gene_activity.csv",
    "panel_h_metadata": "figure3h_selected_gene_activity_metadata.csv",
    "panel_h_background": "figure3h_spatial_background.csv",
}


def _require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def configure_style() -> str:
    """Use Arial when installed and otherwise a stable sans-serif fallback."""
    selected = "sans-serif"
    for family in ("Arial", "Liberation Sans", "DejaVu Sans"):
        try:
            findfont(FontProperties(family=family), fallback_to_default=False)
            selected = family
            break
        except ValueError:
            continue

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [selected],
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "legend.title_fontsize": FONT_SIZE,
            "figure.titlesize": FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "pdf.compression": 9,
            "axes.unicode_minus": False,
        }
    )
    return selected


def _force_seven_point(figure: plt.Figure) -> None:
    figure.canvas.draw()
    for text in figure.findobj(match=mpl.text.Text):
        if text.get_visible() and text.get_text():
            text.set_fontsize(FONT_SIZE)
    figure.canvas.draw()


def _save_pdf(figure: plt.Figure, path: Path, title: str) -> None:
    _force_seven_point(figure)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format="pdf",
        facecolor="white",
        edgecolor="none",
        metadata={
            "Title": title,
            "Creator": "spaDIVA reproducibility workflow",
        },
    )
    plt.close(figure)


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.tick_params(axis="both", width=0.8, length=3, pad=1)


def _format_pvalue(value: float) -> str:
    if not np.isfinite(value):
        return "p = NA"
    if value == 0 or value < 1e-300:
        return "p < 1e-300"
    if value < 1e-3:
        return f"p = {value:.1e}"
    return f"p = {value:.3f}"


def _validate_gsea_sources(
    panel_g: pd.DataFrame, *, strict_reference: bool
) -> pd.DataFrame:
    ordered = panel_g.sort_values("plot_order_top_to_bottom").reset_index(drop=True)
    pathways = tuple(ordered["pathway"].astype(str))
    if strict_reference:
        if pathways != EXPECTED_GSEA_PATHWAYS:
            raise AssertionError(
                "Figure 3g displayed pathways or their order changed.\n"
                f"Observed: {pathways}"
            )
        return ordered

    numeric = ordered[["NES", "FDR_q_value"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Full-mode GSEA contains non-finite NES or FDR values.")
    nes = numeric["NES"].to_numpy(dtype=float)
    fdr = numeric["FDR_q_value"].to_numpy(dtype=float)
    if np.any((fdr < 0) | (fdr >= 0.25)):
        raise ValueError("Full-mode GSEA displays pathways with 0 <= FDR < 0.25.")
    if np.any(nes == 0) or np.sum(nes > 0) > 5 or np.sum(nes < 0) > 5:
        raise ValueError("Full-mode GSEA allows up to five pathways per NES direction.")
    if not np.array_equal(
        ordered["plot_order_top_to_bottom"].to_numpy(),
        np.arange(1, len(ordered) + 1),
    ):
        raise ValueError("Full-mode GSEA plot order must be consecutive from one.")
    ordered[["NES", "FDR_q_value"]] = numeric
    return ordered


def load_and_validate_sources(
    source_dir: Path, *, strict_reference: bool = True
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    source_dir = source_dir.expanduser().resolve()
    paths = {
        key: _require_file(source_dir / filename, key)
        for key, filename in SOURCE_FILES.items()
    }

    panel_f = pd.read_csv(paths["panel_f"])
    panel_g = pd.read_csv(paths["panel_g"])
    panel_h_values = pd.read_csv(paths["panel_h_values"], dtype={"obs_name": str})
    panel_h_metadata = pd.read_csv(paths["panel_h_metadata"])
    panel_h_background = pd.read_csv(
        paths["panel_h_background"], dtype={"obs_name": str}
    )

    required_f = {
        "gene",
        "rank_order",
        "rank_fraction",
        "unique_factor_ratio",
        "translation_accuracy_MultiVI_derived_GA",
        "ratio_group",
        "displayed_in_panel_h",
    }
    required_g = {
        "plot_order_top_to_bottom",
        "pathway",
        "original_term",
        "NES",
        "FDR_q_value",
        "enriched_group",
    }
    required_h_values = {"obs_name", "spatial_x", "spatial_y"}
    required_h_meta = {
        "display_order",
        "gene",
        "unique_factor_ratio",
        "recomputed_pred_vs_public_ga",
        "n_test_spots",
        "shared_color_vmin_percentile_1",
        "shared_color_vmax_percentile_99",
    }
    required_background = {"obs_name", "spatial_x", "spatial_y"}

    for label, frame, required in (
        ("Figure 3f", panel_f, required_f),
        ("Figure 3g", panel_g, required_g),
        ("Figure 3h values", panel_h_values, required_h_values),
        ("Figure 3h metadata", panel_h_metadata, required_h_meta),
        ("Figure 3h background", panel_h_background, required_background),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise KeyError(f"{label} source is missing columns: {missing}")

    numeric_f = panel_f[
        ["unique_factor_ratio", "translation_accuracy_MultiVI_derived_GA"]
    ].replace([np.inf, -np.inf], np.nan)
    if numeric_f.isna().any().any():
        raise ValueError("Figure 3f contains non-finite ratio or translation values.")
    if len(panel_f) != 24027:
        raise ValueError(f"Expected 24,027 Figure 3f genes, found {len(panel_f):,}.")
    if panel_f["gene"].astype(str).duplicated().any():
        raise ValueError("Figure 3f contains duplicate gene names.")

    ratio = panel_f["unique_factor_ratio"].to_numpy(dtype=float)
    translation = panel_f[
        "translation_accuracy_MultiVI_derived_GA"
    ].to_numpy(dtype=float)
    spearman_r, spearman_p = spearmanr(ratio, translation)
    if strict_reference and abs(float(spearman_r) - EXPECTED_SPEARMAN) > 1e-10:
        raise AssertionError(
            "Figure 3f Spearman correlation changed: "
            f"{spearman_r} != {EXPECTED_SPEARMAN}."
        )

    ordered_f = panel_f.sort_values(
        ["unique_factor_ratio", "rank_order"], kind="mergesort"
    ).reset_index(drop=True)
    n_extreme = int(np.floor(len(ordered_f) * EXTREME_FRACTION))
    low_values = ordered_f.iloc[:n_extreme][
        "translation_accuracy_MultiVI_derived_GA"
    ].to_numpy(dtype=float)
    high_values = ordered_f.iloc[-n_extreme:][
        "translation_accuracy_MultiVI_derived_GA"
    ].to_numpy(dtype=float)
    mw = mannwhitneyu(low_values, high_values, alternative="greater")
    cliffs_delta = 2 * float(mw.statistic) / (n_extreme * n_extreme) - 1

    ordered_g = _validate_gsea_sources(panel_g, strict_reference=strict_reference)
    pathways = tuple(ordered_g["pathway"].astype(str))

    panel_h_metadata = panel_h_metadata.sort_values("display_order").reset_index(drop=True)
    h_genes = tuple(panel_h_metadata["gene"].astype(str))
    if h_genes != DISPLAY_GENES:
        raise AssertionError(f"Figure 3h genes or order changed: {h_genes}")
    if panel_h_values["obs_name"].duplicated().any():
        raise ValueError("Figure 3h values contain duplicate spot names.")
    if len(panel_h_values) != 1840:
        raise ValueError(
            f"Expected 1,840 Figure 3h test spots, found {len(panel_h_values):,}."
        )
    if panel_h_background["obs_name"].duplicated().any():
        raise ValueError("Figure 3h background contains duplicate spot names.")
    if len(panel_h_background) != 9196:
        raise ValueError(
            "Expected 9,196 Figure 3h background spots, found "
            f"{len(panel_h_background):,}."
        )
    missing_test_spots = set(panel_h_values["obs_name"]).difference(
        panel_h_background["obs_name"]
    )
    if missing_test_spots:
        raise ValueError(
            f"Figure 3h background is missing {len(missing_test_spots)} test spots."
        )

    h_correlations: dict[str, float] = {}
    for gene in DISPLAY_GENES:
        true_column = f"true_{gene}"
        pred_column = f"pred_{gene}"
        if true_column not in panel_h_values or pred_column not in panel_h_values:
            raise KeyError(f"Figure 3h is missing true/pred values for {gene}.")
        observed = float(
            np.corrcoef(panel_h_values[true_column], panel_h_values[pred_column])[0, 1]
        )
        expected = float(
            panel_h_metadata.loc[
                panel_h_metadata["gene"].astype(str) == gene,
                "recomputed_pred_vs_public_ga",
            ].iloc[0]
        )
        if not np.isfinite(observed) or abs(observed - expected) > 2e-6:
            raise AssertionError(
                f"Figure 3h correlation changed for {gene}: {observed} != {expected}."
            )
        h_correlations[gene] = observed

    validation = {
        "status": "passed",
        "figure3f": {
            "n_genes": int(len(panel_f)),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "n_low_private": n_extreme,
            "n_high_private": n_extreme,
            "mannwhitney_u": float(mw.statistic),
            "mannwhitney_p_one_sided_low_greater_high": float(mw.pvalue),
            "cliffs_delta_low_vs_high": float(cliffs_delta),
        },
        "figure3g": {
            "n_displayed_pathways": int(len(panel_g)),
            "pathways_top_to_bottom": list(pathways),
            "selection_status": (
                "significant_pathways" if len(panel_g) else "no_significant_pathways"
            ),
        },
        "figure3h": {
            "n_test_spots": int(len(panel_h_values)),
            "n_background_spots": int(len(panel_h_background)),
            "genes": list(DISPLAY_GENES),
            "prediction_correlations": h_correlations,
        },
    }
    frames = {
        "panel_f": panel_f,
        "panel_g": ordered_g,
        "panel_h_values": panel_h_values,
        "panel_h_metadata": panel_h_metadata,
        "panel_h_background": panel_h_background,
    }
    return frames, validation


def plot_panel_f(panel_f: pd.DataFrame, output_path: Path) -> None:
    ordered = panel_f.sort_values(
        ["unique_factor_ratio", "rank_order"], kind="mergesort"
    ).reset_index(drop=True)
    ordered["plot_rank_fraction"] = np.linspace(0, 1, len(ordered))
    x = ordered["plot_rank_fraction"].to_numpy(dtype=float)
    y = ordered["translation_accuracy_MultiVI_derived_GA"].to_numpy(dtype=float)
    ratio = ordered["unique_factor_ratio"].to_numpy(dtype=float)
    spearman_r, spearman_p = spearmanr(ratio, y)

    figure = plt.figure(figsize=(4.15, 1.72))
    grid = figure.add_gridspec(1, 2, width_ratios=(2.45, 1.45), wspace=0.44)
    scatter_axis = figure.add_subplot(grid[0, 0])
    box_axis = figure.add_subplot(grid[0, 1])

    scatter_axis.scatter(
        x,
        y,
        s=3,
        color="#4C78A8",
        alpha=0.18,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    slope, intercept = np.polyfit(x, y, deg=1)
    scatter_axis.plot(
        x,
        slope * x + intercept,
        color="#B0B0B0",
        linestyle="--",
        linewidth=1.2,
        zorder=2,
    )

    selected = ordered[
        ordered["gene"].astype(str).isin(DISPLAY_GENES)
    ].set_index("gene")
    label_settings = {
        "Pde10a": ((10, 4), "left", "top"),
        "Rgs9": ((5, 5), "right", "bottom"),
        "Tmeff2": ((-10, -10), "left", "bottom"),
        "Nab2": ((10, -12), "left", "top"),
    }
    for gene in DISPLAY_GENES:
        row = selected.loc[gene]
        is_low = gene in {"Pde10a", "Rgs9"}
        scatter_axis.scatter(
            row["plot_rank_fraction"],
            row["translation_accuracy_MultiVI_derived_GA"],
            s=30,
            facecolor="#4C78A8" if is_low else "#F58518",
            edgecolor="black",
            linewidth=0.7,
            zorder=5,
            clip_on=False,
        )
        offset, horizontal, vertical = label_settings[gene]
        scatter_axis.annotate(
            gene,
            xy=(
                row["plot_rank_fraction"],
                row["translation_accuracy_MultiVI_derived_GA"],
            ),
            xytext=offset,
            textcoords="offset points",
            ha=horizontal,
            va=vertical,
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
            },
            arrowprops=(
                None
                if gene in {"Rgs9", "Tmeff2"}
                else {
                    "arrowstyle": "-",
                    "color": "#333333",
                    "lw": 0.5,
                    "shrinkA": 0,
                    "shrinkB": 4,
                }
            ),
            zorder=6,
            annotation_clip=False,
        )

    scatter_axis.set_xticks([])
    scatter_axis.set_ylim(-0.05, 1.10)
    scatter_axis.set_xlabel("GA genes ranked by spaDIVA unique factor ratio", labelpad=2)
    scatter_axis.set_ylabel("Translation accuracy\n(MultiVI-derived GA)")
    scatter_axis.text(
        0.03,
        0.06,
        f"Spearman r = {spearman_r:.3f}\n{_format_pvalue(float(spearman_p))}",
        transform=scatter_axis.transAxes,
        ha="left",
        va="bottom",
        color="#333333",
    )
    scatter_axis.text(
        0.5,
        -0.20,
        "low private   →   high private",
        transform=scatter_axis.transAxes,
        ha="center",
        va="top",
    )
    _style_axis(scatter_axis)

    n_extreme = int(np.floor(len(ordered) * EXTREME_FRACTION))
    low = ordered.iloc[:n_extreme][
        "translation_accuracy_MultiVI_derived_GA"
    ].to_numpy(dtype=float)
    high = ordered.iloc[-n_extreme:][
        "translation_accuracy_MultiVI_derived_GA"
    ].to_numpy(dtype=float)
    mw = mannwhitneyu(low, high, alternative="greater")
    boxes = box_axis.boxplot(
        [low, high],
        positions=[1, 2],
        widths=0.55,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#333333", "linewidth": 0.6},
        whiskerprops={"color": "#333333", "linewidth": 0.5},
        capprops={"color": "#333333", "linewidth": 0.5},
        boxprops={"color": "#333333", "linewidth": 0.5},
    )
    for patch, color in zip(boxes["boxes"], ("#DCEAF7", "#FBE2D5")):
        patch.set_facecolor(color)
    rng = np.random.default_rng(0)
    box_axis.scatter(
        rng.normal(1, 0.04, len(low)),
        low,
        s=1.5,
        color="#4C78A8",
        alpha=0.12,
        linewidths=0,
        rasterized=True,
        zorder=4,
    )
    box_axis.scatter(
        rng.normal(2, 0.04, len(high)),
        high,
        s=1.5,
        color="#F58518",
        alpha=0.12,
        linewidths=0,
        rasterized=True,
        zorder=4,
    )
    box_axis.set_xticks([1, 2])
    box_axis.set_ylim(-0.05, 1.10)
    box_axis.set_xticklabels(
        ["Low-private\nbottom 20%", "High-private\ntop 20%"]
    )
    box_axis.set_ylabel("Translation accuracy\n(MultiVI-derived GA)")
    box_axis.text(
        0.97,
        0.97,
        _format_pvalue(float(mw.pvalue)),
        transform=box_axis.transAxes,
        ha="right",
        va="top",
    )
    _style_axis(box_axis)
    figure.subplots_adjust(left=0.11, right=0.99, top=0.97, bottom=0.26)
    _save_pdf(figure, output_path, "spaDIVA Figure 3f")


def plot_panel_g(panel_g: pd.DataFrame, output_path: Path) -> None:
    panel_g = panel_g.sort_values("plot_order_top_to_bottom").reset_index(drop=True)
    y_positions = np.arange(len(panel_g))
    values = panel_g["NES"].to_numpy(dtype=float)
    colors = np.where(values > 0, "#D43D32", "#2E78B0")

    figure, axis = plt.subplots(figsize=(4.0, 2.25))
    if panel_g.empty:
        axis.set_axis_off()
        axis.text(
            0.5,
            0.5,
            "No pathways with FDR < 0.25",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
        _save_pdf(figure, output_path, "spaDIVA Figure 3g")
        return
    axis.barh(
        y_positions,
        values,
        color=colors,
        edgecolor="none",
        height=0.66,
        alpha=0.96,
    )
    axis.axvline(0, color="black", linestyle="--", linewidth=0.7)
    axis.set_yticks(y_positions)
    axis.set_yticklabels([])
    axis.tick_params(axis="y", length=0)
    axis.invert_yaxis()
    axis.set_xlabel("NES", labelpad=1)
    axis.set_title("low private   →   high private", pad=3)

    x_abs = max(float(np.max(np.abs(values))), 1.0)
    axis.set_xlim(-1.28 * x_abs, 1.28 * x_abs)
    offset = 0.04 * x_abs
    for index, row in panel_g.iterrows():
        value = float(row["NES"])
        axis.text(
            -offset if value > 0 else offset,
            index,
            str(row["pathway"]),
            va="center",
            ha="right" if value > 0 else "left",
            clip_on=True,
        )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="x", length=2, pad=1)
    axis.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.35)
    figure.subplots_adjust(left=0.05, right=0.97, top=0.91, bottom=0.14)
    _save_pdf(figure, output_path, "spaDIVA Figure 3g")


def _spatial_limits(
    values: pd.DataFrame, background: pd.DataFrame
) -> tuple[tuple[float, float], tuple[float, float]]:
    x = np.concatenate(
        [
            values["spatial_x"].to_numpy(dtype=float),
            background["spatial_x"].to_numpy(dtype=float),
        ]
    )
    y = np.concatenate(
        [
            values["spatial_y"].to_numpy(dtype=float),
            background["spatial_y"].to_numpy(dtype=float),
        ]
    )
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    padding = 0.015 * max(x_max - x_min, y_max - y_min, 1.0)
    return (x_min - padding, x_max + padding), (y_min - padding, y_max + padding)


def _clean_spatial_axis(
    axis: plt.Axes,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def plot_panel_h(
    values: pd.DataFrame,
    metadata: pd.DataFrame,
    background: pd.DataFrame,
    output_path: Path,
) -> None:
    x_limits, y_limits = _spatial_limits(values, background)
    figure, axes = plt.subplots(2, 4, figsize=(7.2, 3.25))
    figure.subplots_adjust(
        left=0.095,
        right=0.995,
        top=0.91,
        bottom=0.235,
        wspace=0.08,
        hspace=0.12,
    )

    for column, gene in enumerate(DISPLAY_GENES):
        row = metadata.loc[metadata["gene"].astype(str) == gene].iloc[0]
        vmin = float(row["shared_color_vmin_percentile_1"])
        vmax = float(row["shared_color_vmax_percentile_99"])
        for plot_row, kind in enumerate(("true", "pred")):
            axis = axes[plot_row, column]
            axis.scatter(
                background["spatial_x"],
                background["spatial_y"],
                s=3.2,
                color="#D0D0D0",
                linewidths=0,
                rasterized=True,
                zorder=0,
            )
            axis.scatter(
                values["spatial_x"],
                values["spatial_y"],
                c=values[f"{kind}_{gene}"],
                s=3.6,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                linewidths=0,
                rasterized=True,
                zorder=1,
            )
            _clean_spatial_axis(axis, x_limits, y_limits)
        axes[0, column].set_title(gene, pad=2.0)
        axes[1, column].text(
            0.5,
            -0.105,
            (
                f"UF ratio = {float(row['unique_factor_ratio']):.2f}\n"
                f"MultiVI acc = {float(row['recomputed_pred_vs_public_ga']):.2f}"
            ),
            transform=axes[1, column].transAxes,
            ha="center",
            va="top",
            linespacing=1.15,
        )

    top = axes[0, 0].get_position()
    bottom = axes[1, 0].get_position()
    figure.text(0.082, (top.y0 + top.y1) / 2, "True GA", ha="right", va="center")
    figure.text(
        0.082, (bottom.y0 + bottom.y1) / 2, "pred GA", ha="right", va="center"
    )
    left_center = (axes[1, 0].get_position().x0 + axes[1, 1].get_position().x1) / 2
    right_center = (axes[1, 2].get_position().x0 + axes[1, 3].get_position().x1) / 2
    figure.text(left_center, 0.035, "low-private", ha="center", va="bottom")
    figure.text(right_center, 0.035, "high-private", ha="center", va="bottom")
    _save_pdf(figure, output_path, "spaDIVA Figure 3h")


def render_panels(
    source_dir: Path,
    output_dir: Path,
    *,
    strict_reference: bool = True,
) -> dict[str, Any]:
    """Validate one source-data bundle and render all Figure 3f-h panels."""
    frames, validation = load_and_validate_sources(
        source_dir, strict_reference=strict_reference
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    font_family = configure_style()

    plot_panel_f(frames["panel_f"], output_dir / "panel_f.pdf")
    plot_panel_g(frames["panel_g"], output_dir / "panel_g.pdf")
    plot_panel_h(
        frames["panel_h_values"],
        frames["panel_h_metadata"],
        frames["panel_h_background"],
        output_dir / "panel_h.pdf",
    )
    validation["font_family"] = font_family
    validation["font_size_pt"] = FONT_SIZE
    validation["strict_reference_validation"] = strict_reference
    validation["outputs"] = ["panel_f.pdf", "panel_g.pdf", "panel_h.pdf"]
    return validation
