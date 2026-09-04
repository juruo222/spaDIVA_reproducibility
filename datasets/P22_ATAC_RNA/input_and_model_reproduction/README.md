# P22 model and spatial-domain reproduction

Run from the repository root:

```bash
python reproduce.py --figure 3
```

The domain workflow reads frozen `source_data/` and
writes the following to `results/Figure_3/`. Model training is available as a
separate workflow:

- Figure 3b: integrated domain maps for spaDIVA and three baselines;
- Figure 3c: per-domain Moran's I distributions;
- Figure 3d: spaDIVA integrated, shared, modality-specific, and input maps;
- Figure 3e: the ten kNN Jaccard bar values used in the manuscript;
- Supplementary Figure S2a-c: K=18 maps, KL/PoE ablations, and K=12/K=18
  Moran's I distributions;
- validation and source/output hash manifests.

Panel text is 7 pt. The sibling translation workflow writes panels f-h.

Independent spaDIVA training, output scope, and color-label matching are
explained in [Model runs](../../../MODEL_RUNS.md). Training writes new model
outputs; the bundled tables and sibling translation workflow retain their
versioned manuscript inputs.

## Frozen source data

Domain tables contain per-spot labels; Moran's I tables contain one value for
each inferred domain. Figure 3e reads the ten means in
`figure3e_jaccard_original_values.csv` (k=50, Euclidean distance). A separate
deterministic recalculation, including 91,960 per-spot values, is stored under
`recomputed_metrics/jaccard/figure_3e/`.
