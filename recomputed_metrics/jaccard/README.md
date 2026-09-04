# Recomputed kNN Jaccard metrics

This directory contains deterministic recalculations for five kNN Jaccard panels:

- Figure 2e
- Figure 3e
- Figure 4c
- Supplementary Figure S4d
- Supplementary Figure S4h

Each panel directory contains the per-spot values, their ten-row summary, and the corresponding PDF. All calculations use Euclidean `k=50` nearest-neighbor sets. The plotting style uses 7-point text.

From the repository root, validate all per-spot values against the saved summaries and redraw the PDFs
under the ignored generated-results directory
`results/recomputed_jaccard/`:

```bash
python recomputed_metrics/jaccard/redraw_recomputed_jaccard.py
```

Write the same directory layout elsewhere:

```bash
python recomputed_metrics/jaccard/redraw_recomputed_jaccard.py --output-dir path/to/output
```

Run validation alone:

```bash
python recomputed_metrics/jaccard/redraw_recomputed_jaccard.py --verify-only
```

The script resolves every input relative to its own location and is independent
of the original workspace layout. By default, it writes to
`results/recomputed_jaccard/`; pass `recomputed_metrics/jaccard/` explicitly
with `--output-dir` to refresh the versioned PDFs.
