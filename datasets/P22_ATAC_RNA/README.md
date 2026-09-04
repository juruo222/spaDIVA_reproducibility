# P22 ATAC-RNA figures

Run `python reproduce.py --figure 3` from the repository root to redraw
Figure 3b-h and S2a-c under `results/` from the versioned figure tables.

- [Domains and benchmarks](input_and_model_reproduction/README.md): frozen
  spatial labels, Moran's I, neighborhood overlap, and ablation results.
- [Translation](external_translation_figure_reproduction/README.md): frozen
  gene-level statistics, selected gene-activity maps, and GSEA pathways.

Figure 3a is a separately assembled anatomical reference. Figure 3e reads the
ten manuscript bar values stored in
`input_and_model_reproduction/source_data/figure3e_jaccard_original_values.csv`,
so the compact workflow redraws the displayed bar heights exactly. The
deterministic per-spot recalculation is retained separately under
`recomputed_metrics/jaccard/figure_3e/`.

Independent spaDIVA/MultiVI commands are in [Model runs](../../MODEL_RUNS.md).
The MultiVI/UF/GSEA entry uses the archived manuscript `Z_poe`. New spaDIVA
model runs and manuscript translation reproduction are separate workflows.
