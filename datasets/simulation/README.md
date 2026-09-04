# Simulation figure reproduction

The plot entry redraws Figure 2c,f-h and S1d from frozen tables. A separate
model entry runs the release package and writes new model outputs.

The frozen plot uses the bundled `source_data/` tables. Figure 2h additionally
reads the deposited `datasets/simulation/input_data/ADT_100.h5ad` to draw the
representative spatial-expression maps. The separate model run requires both
deposited simulation H5ADs.

## Simulation data generation

`data_generation/generate_simulation_data.py` implements the simulation
design and imports Scanpy directly. Both modalities use a 36 x 36 grid,
loading/non-spatial-factor seeds 101, `expr_mean=20`,
`mix_frac_spat=0.55`, `bkg_mean=0.2`, and `nb_shape=10`. RNA contains 1,000
mixed features, active factors 1/2/4, and `zero_prob=0.5`; ADT contains 100
mixed features, active factors 1/2/3, and `zero_prob=0`. Factors 1 and 2 are
shared, factor 4 is RNA-specific, and factor 3 is ADT-specific. The generated
AnnData files retain raw counts in `layers["counts"]` and log1p values in `X`.

The default `COUNT_SEED=101` reproduces the deposited RNA and ADT counts,
log1p expression values, observation/feature annotations, factor arrays,
and spatial coordinates exactly. This data-generation seed is separate
from the model-training and PoE-sample seeds.

## Lightweight figure reproduction

Run from the repository root after setting the data directory as described in
the [quick start](../../README.md#quick-start):

```bash
python reproduce.py --figure 2
```

Outputs: `results/Figure_2/panel_{c,f,g,h}.pdf` and
`results/Supplementary_Figure_S1/panel_d.pdf`, with validation and hash
manifests. Versioned inputs remain under `source_data/`.

Independent generation/training commands and output details are in
[Model runs](../../MODEL_RUNS.md). The model starts from `layers["counts"]`
and uses model/PoE-sample seeds 42; it writes a separate set of model outputs.

The Figure 2c source contains 40 benchmark values and S1d contains 30
ablation values. The compact plot entry covers Figure 2c,f-h and S1d. The
separate model entry produces one release-model run; the versioned repeated-run
benchmarks and UF-ratio/totalVI results remain the inputs for these panels.

The ten Figure 2e manuscript bar values are stored in
`source_data/figure2e_jaccard_original_values.csv`. The deterministic
recalculation and its per-spot values are stored separately under
`recomputed_metrics/jaccard/figure_2e/`. The recomputed Figure 2e is available through the
Jaccard redraw workflow described there.
