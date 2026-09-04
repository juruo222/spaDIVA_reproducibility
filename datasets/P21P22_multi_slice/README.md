# P21/P22 figures

From the repository root, `python reproduce.py --figure 5` redraws frozen
Figure 5f-h, together with E13/E15/E18 panels, under `results/`. Inputs in
`source_data/` contain spatial-domain labels, per-domain Moran's I, and
neighborhood metrics. The compact entry covers Figure 5f-h; Figure 5e uses the
separately assembled input-layout panel.

The separate `model_reproduction.ipynb` runs the release joint model and
returns its latent in memory. Metrics and frozen-table updates are separate
analysis steps. See [Model runs](../../MODEL_RUNS.md) for inputs and commands.
