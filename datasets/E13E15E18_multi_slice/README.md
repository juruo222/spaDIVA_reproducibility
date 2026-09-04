# E13/E15/E18 figures

From the repository root, `python reproduce.py --figure 5` redraws frozen
Figure 5b-d and S4i, together with P21/P22 panels, under `results/`.
The inputs are domain labels, benchmark runs, and neighborhood metrics in
`source_data/`. Joint-model training is available through the separate model
workflow.

S4a-h present independent E18/E15 single-section analyses. S4i compares
single-section and joint runs using ARI. The compact plotting entry covers
Figure 5b-d and S4i; Figure 5a and S4a-h use separately assembled figure
materials.

The S4d and S4h manuscript Jaccard bar values are stored under
`single_section_supplement/source_data/`. Deterministic recalculations from the
saved E18 and E15 latents are stored separately under
`recomputed_metrics/jaccard/supplementary_figure_s4d/` and
`recomputed_metrics/jaccard/supplementary_figure_s4h/`.

The separate `model_reproduction.ipynb` runs the release joint model and
returns its latent in memory. The frozen figure tables remain versioned inputs. See
[Model runs](../../MODEL_RUNS.md) for inputs and commands.
