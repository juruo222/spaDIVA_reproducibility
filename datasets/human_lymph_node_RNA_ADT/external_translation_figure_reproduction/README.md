# Human lymph node translation figure reproduction

This workflow redraws Figure 4d-f and S3a-c using:

- `source_data/Figure_4_manuscript_source.csv` for the deposited 31-protein
  unique factor (UF) ratios and manuscript translation statistics;
- Zenodo RNA, ADT, and annotation files for expression, spatial maps, and
  RNA-ADT concordance;
- `artifacts/totalvi_test_pred_protein.h5ad` to independently verify the
  deposited totalVI accuracy column on the fixed test split.

For the manuscript-version UF analysis, the archived artifact is the deposited
ratio table, which records `z_source=Z_poe`. The frozen prediction and held-out
ADT data provide an independent check of translation accuracy.

## Run

Run from the repository root in the core environment, with `SPADIVA_DATA_ROOT`
set as described in [Data availability](../../../DATA_AVAILABILITY.md):

```bash
python reproduce.py --figure 4
```

## Outputs

Under `results/`: Figure 4 panels d-f, S3 panels a-c, analysis tables,
validation, and a hash manifest. Default checks are 31 proteins, Spearman
`r=-0.4903225806`, `p=0.0051062660`, and one-sided low-ratio > high-ratio
Mann-Whitney `p=0.0129870130`. Panels use 7 pt text and equal map-marker sizes;
font fallback is recorded.

## Independent model run

See [Model runs](../../../MODEL_RUNS.md) for totalVI retraining with the fixed
split and external environment. New predictions support an independent
translation analysis, while UF ratios remain linked to the versioned manuscript
table. The Figure 4b benchmark is covered by the domain workflow.
