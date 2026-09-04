# Independent model runs

These entries provide independent model runs alongside the frozen figure
redraws in `reproduce.py`. Bundled `source_data/` remains the input for the
manuscript panels, while new runs write separate outputs for comparison.

Run commands from the repository root. Training uses the deposited input H5ADs
and annotations where applicable. See
[Data availability](DATA_AVAILABILITY.md) for the extracted archive layout.

## spaDIVA environment and inputs

```bash
conda env create -f environments/environment_spadiva_release.yml
conda activate spaDIVA
python -m pip install -e . --no-deps
```

If this `spaDIVA` environment is already installed, start with `conda activate spaDIVA`.

Set the extracted data directory in the current terminal:

Linux (Bash):

```bash
export SPADIVA_DATA_ROOT="/path/to/spaDIVA_data"
```

Windows (PowerShell):

```powershell
$env:SPADIVA_DATA_ROOT = "D:\path\to\spaDIVA_data"
```

The remaining commands work in both shells:

```bash
python scripts/check_reproducibility_setup.py --require-all-data --import-check
```

### Simulation

```bash
python datasets/simulation/model/run_spadiva.py --epochs 1000 --device auto
```

This single run starts from `layers["counts"]`, preprocesses each modality once,
and uses model/PoE-sample seeds 42. It saves `spadiva_simulation.h5ad` with
`Z`, `Z_poe`, `Z_W`, `W_ADT`, and `W_RNA`, clustering metrics, k=50 Euclidean
Jaccard, loss, and a manifest under `datasets/simulation/results/model/`.
The 40 benchmark values, 30 ablation values, and totalVI/UF analysis remain the
frozen inputs for the manuscript figures.

To independently regenerate the simulation input data in a new directory:

```bash
python datasets/simulation/data_generation/generate_simulation_data.py --output-dir results/generated_simulation
```

Count seed 101 reproduces the deposited expression arrays and annotations;
it is distinct from the model seeds. See the [design](datasets/simulation/README.md).

### P22 spaDIVA domains

```bash
python datasets/P22_ATAC_RNA/input_and_model_reproduction/reproduce_model_and_domains.py --mode full --use-cuda --output-dir results/model_runs/P22_domains
```

Omit `--use-cuda` for CPU. This uses 1,600 epochs and model/PoE-sample seeds 42,
saves the latent, and recomputes spaDIVA clustering, Moran's I, and Jaccard.
Baselines and S2 ablations remain frozen. Maximum-overlap cluster-label matching
controls display colors and is excluded from metric calculation. The sibling
translation entry uses the archived manuscript-version latent.

### Lymph node and joint sections

Open the relevant notebook using the core environment:

```bash
jupyter lab datasets/human_lymph_node_RNA_ADT/input_and_model_reproduction/model_reproduction.ipynb
jupyter lab datasets/E13E15E18_multi_slice/model_reproduction.ipynb
jupyter lab datasets/P21P22_multi_slice/model_reproduction.ipynb
```

Each uses `SPADIVA_DATA_ROOT` and the release snapshot. These notebooks leave
`z_adata` in memory and provide spaDIVA results independently of the frozen
figure tables and baselines. The archived lymph UF/Jaccard values remain the
source for their manuscript panels. The E13/E15/E18 notebook covers joint
training; S4a-h report the separate E18/E15 single-section analyses.

## External translation environment

In the same terminal, activate the external-model environment; the data-root
setting above remains available:

```bash
conda env create -f environments/environment_external_translation_paper.yml
conda activate spaDIVA-external-translation
python -m pip install -e . --no-deps
```

### P22 MultiVI, UF ratio, and GSEA

The exact derived `P22_GA.txt` matrix is additionally required; its public
download is pending and is distributed separately from the core H5AD archive.
The three manuscript-version GMT libraries, fixed splits, and sampled
`Z_poe`/`W_ATAC` latent are bundled. Preflight checks the GA/GMT sizes and
SHA-256; training validates the exact 6,436/920/1,840 train/validation/test
observation IDs.

The examples use `data/P22_GA.txt` relative to the repository root; replace this
path with the downloaded file's location.

```bash
python datasets/P22_ATAC_RNA/external_translation_figure_reproduction/reproduce.py --mode full --preflight-only --p22-ga "data/P22_GA.txt" --output-dir results/model_runs/P22_translation

python datasets/P22_ATAC_RNA/external_translation_figure_reproduction/reproduce.py --mode full --p22-ga "data/P22_GA.txt" --output-dir results/model_runs/P22_translation
```

Full mode trains MultiVI and recomputes derived gene activity, UF ratios, and
GSEA using the bundled manuscript-version spaDIVA latent. See
[asset details](datasets/P22_ATAC_RNA/external_translation_figure_reproduction/README.md).

### Lymph totalVI

```bash
python datasets/human_lymph_node_RNA_ADT/external_translation_figure_reproduction/reproduce.py --retrain-totalvi --device auto --results-dir results/model_runs/lymph_translation
```

This trains totalVI with seed 0 and at most 100 epochs, using the validated
2,438/349/697 train/validation/test split. It creates new predictions and
translation statistics paired with the frozen 31-protein UF ratios. Figure 4b
uses its separate repeated-run benchmark.

Use separate output directories when comparing new analyses with frozen plots.
Commands that share default figure output paths can overwrite generated PDFs;
bundled source tables remain unchanged. Before updating a manuscript panel,
review the new feature membership, metrics, and enrichment results. Hardware and
library changes can alter neural-network results.
