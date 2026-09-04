# Data availability

Saved figure tables, cluster labels, fixed splits, the frozen lymph totalVI
prediction, and the compact P22 sampled latent are bundled under `datasets/`.
The Zenodo archive supplies processed input `.h5ad` files and annotations.
After extraction, set:

Linux (Bash):

```bash
export SPADIVA_DATA_ROOT="/path/to/spaDIVA_data"
```

Windows (PowerShell):

```powershell
$env:SPADIVA_DATA_ROOT = "D:\path\to\spaDIVA_data"
```

## Figure-redraw inputs

- Figure 2: `datasets/simulation/input_data/ADT_100.h5ad` for the four displayed
  expression maps and their coordinates.
- Figure 4/S3: the two lymph H5ADs and `annotation.csv` listed below for expression
  maps and RNA-ADT concordance.
- Figures 3 and 5: all plotting inputs are bundled. Figure 3g is redrawn from
  saved GSEA results.

Thus `python reproduce.py --figure all --check-only` requires four external
files. The remaining files below support the independent model runs in
[MODEL_RUNS.md](MODEL_RUNS.md).

## Simulation

```text
datasets/simulation/input_data/ADT_100.h5ad
datasets/simulation/input_data/RNA_ZINB.h5ad
```

The simulation-design generator is retained at
`datasets/simulation/data_generation/generate_simulation_data.py`.
See the [simulation instructions](datasets/simulation/README.md) for the
parameters, including count seed 101, which reproduce the deposited input data.

## P22 mouse brain ATAC-RNA

Accession: `GSE205055`.

```text
datasets/P22_ATAC_RNA/input_data/adata_peaks_normalized.h5ad
datasets/P22_ATAC_RNA/input_data/adata_RNA.h5ad
```

The compact Figure 3 workflow is self-contained. A full MultiVI-to-GSEA rerun
also needs the separate 3,666,760,836-byte derived matrix `P22_GA.txt`.
This matrix is distributed separately from the core-input archive; its public
download location is pending.
The three paper-version GMT snapshots are bundled in
`datasets/P22_ATAC_RNA/external_translation_figure_reproduction/external_predictions/gene_sets/`.
See that workflow's `README.md` and `external_predictions/ARTIFACT_MANIFEST.tsv`
for provenance, file hashes, and preflight instructions. Preflight verifies
these assets before training.

## Human lymph node RNA-ADT

Accession: `GSE263617`.

```text
datasets/human_lymph_node_RNA_ADT/input_data/human lymph node adata_ADT.h5ad
datasets/human_lymph_node_RNA_ADT/input_data/human lymph node adata_RNA.h5ad
datasets/human_lymph_node_RNA_ADT/annotation/annotation.csv
```

## E13/E15/E18 mouse embryonic brain

Accession: `OEP003285`.

```text
datasets/E13E15E18_multi_slice/input_data/Mouse_embryonic_brain/E13/E13_adata_atac.h5ad
datasets/E13E15E18_multi_slice/input_data/Mouse_embryonic_brain/E13/E13_adata_rna.h5ad
datasets/E13E15E18_multi_slice/input_data/Mouse_embryonic_brain/E15/E15_adata_atac.h5ad
datasets/E13E15E18_multi_slice/input_data/Mouse_embryonic_brain/E15/E15_adata_rna.h5ad
datasets/E13E15E18_multi_slice/input_data/Mouse_embryonic_brain/E18/E18_adata_atac.h5ad
datasets/E13E15E18_multi_slice/input_data/Mouse_embryonic_brain/E18/E18_adata_rna.h5ad
datasets/E13E15E18_multi_slice/annotation/E13_5-S1_annotation.csv
datasets/E13E15E18_multi_slice/annotation/E15_5-S1_annotation.csv
datasets/E13E15E18_multi_slice/annotation/E18_5-S1_annotation.csv
```

## P21/P22 postnatal mouse brain sections

Accession: `GSE205055`.

```text
datasets/P21P22_multi_slice/input_data/Mouse_postnatal_brain/slice1/s1_adata_atac.h5ad
datasets/P21P22_multi_slice/input_data/Mouse_postnatal_brain/slice1/s1_adata_rna.h5ad
datasets/P21P22_multi_slice/input_data/Mouse_postnatal_brain/slice2/s2_adata_atac.h5ad
datasets/P21P22_multi_slice/input_data/Mouse_postnatal_brain/slice2/s2_adata_rna.h5ad
datasets/P21P22_multi_slice/input_data/Mouse_postnatal_brain/slice3/s3_adata_atac.h5ad
datasets/P21P22_multi_slice/input_data/Mouse_postnatal_brain/slice3/s3_adata_rna.h5ad
```

Validate all model-input paths with:

```bash
python scripts/check_reproducibility_setup.py --profile model --require-all-data
```

The public data-archive DOI and the additional P22 GA download link will be
added when those deposits are finalized. The listed accessions identify the
original studies; the processed files above define the reproduction inputs.
