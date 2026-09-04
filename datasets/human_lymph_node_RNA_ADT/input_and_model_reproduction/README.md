# Human lymph node: Figure 4a-c

```bash
python reproduce.py --figure 4
```

Run from the repository root. The domain workflow's frozen `source_data/` produces 7 pt panels a-c,
validation, and a hash manifest in `results/Figure_4/`. Model training is
available as a separate workflow.

## Source-data scope

- `spatial_domains.csv`: barcode, spatial coordinates, ground-truth labels,
  final release spaDIVA labels, and the three frozen baseline label vectors.
- `benchmark_runs.csv`: ten ARI/NMI values per method. The generated Figure 4b
  uses the bundled totalVI snapshot (ARI median 0.187815; NMI median 0.319743),
  while the current manuscript panel uses ARI median 0.173777 and NMI median
  0.309463. The remaining three methods match the manuscript.
- `figure4c_jaccard_original_values.csv`: the ten k=50 means displayed in
  Figure 4c.

Figure 4a spaDIVA labels use the release correlation-weighted graph,
`KL_weight=[1,1,1,1]`, 200 epochs, and seed `202608017`.

Figure 4c uses the manuscript bar values with k=50 Euclidean neighbors. A
separate deterministic recalculation from the final release lymph model,
including per-spot values, is stored under
`recomputed_metrics/jaccard/figure_4c/`.

The same figure command also dispatches the sibling translation workflow,
which requires deposited RNA/ADT inputs and annotation. For independent
spaDIVA training see [Model runs](../../../MODEL_RUNS.md). Its notebook
returns latent results in memory. Figure 4c and the UF analysis use their
versioned manuscript source values.
