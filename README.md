# spaDIVA reproducibility

This repository contains the scripts and frozen analysis outputs used to
reproduce selected panels in the spaDIVA manuscript. The installable software,
API, and tutorials are maintained in the
[spaDIVA release repository](https://github.com/juruo222/spaDIVA).

The default workflow redraws saved results. Independent spaDIVA and
external-translation runs are documented in [MODEL_RUNS.md](MODEL_RUNS.md).

## Quick start

The environment targets Linux and Windows on x86_64. Run from the repository root:

```bash
conda env create -f environments/environment_spadiva_release.yml
conda activate spaDIVA
python -m pip install -e . --no-deps
```

If this `spaDIVA` environment is already installed, start with `conda activate spaDIVA`.

Figures 2 and 4 require the expression and annotation files described in
[DATA_AVAILABILITY.md](DATA_AVAILABILITY.md). Set the directory containing the
extracted `datasets/` folder before running them:

Linux (Bash):

```bash
export SPADIVA_DATA_ROOT="/path/to/spaDIVA_data"
```

Windows (PowerShell):

```powershell
$env:SPADIVA_DATA_ROOT = "D:\path\to\spaDIVA_data"
```

Then run the same commands on either platform:

```bash
python reproduce.py --list
python reproduce.py --figure all --check-only --import-check
python reproduce.py --figure all
```

Data-root priority is `--data-root PATH`, then `SPADIVA_DATA_ROOT`, then the
repository's `data/` directory. Generated PDFs and run records are written to
`results/`; use `--output-dir PATH` to select another destination.
PDF generation runs without a display and uses 7 pt Arial, with Liberation Sans
and DejaVu Sans as font fallbacks.

## Supported outputs

| Dataset | Panels redrawn from saved results |
| --- | --- |
| Simulation | Figure 2c, f-h; Supplementary Figure S1d |
| P22 ATAC-RNA | Figure 3b-h; Supplementary Figure S2a-c |
| Human lymph node | Figure 4a-f; Supplementary Figure S3a-c |
| Multi-section datasets | Figure 5b-d, f-h; Supplementary Figure S4i |

`--figure all` produces these 31 panels. The remaining manuscript panels draw
on anatomical references, input-layout materials, or analyses maintained
outside this plotting entry: Figure 1; Figure 2a, b, d, e; Supplementary Figure
S1a-c; and Supplementary Figure S4a-h.

## Scope

- Versioned numerical inputs are stored under `datasets/**/source_data/`.
- Recomputed per-spot Jaccard results are stored under
  `recomputed_metrics/jaccard/`, separately from the manuscript bar values.
- Dataset-specific README files document source-version details, including the
  archived Figure 4b totalVI snapshot.
- [MODEL_RUNS.md](MODEL_RUNS.md) describes the selected model-running workflows.

Dataset-specific details are in each dataset README. [MANIFEST.tsv](MANIFEST.tsv)
records the files and SHA-256 hashes in this snapshot.
