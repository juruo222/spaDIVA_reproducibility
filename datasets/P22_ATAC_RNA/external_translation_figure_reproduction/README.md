# P22 external translation and Figure 3f-h

## Compact plots

```bash
python reproduce.py --figure 3
```

Run from the repository root with the core environment. This reads frozen
`source_data/` and writes 7 pt `panel_{f,g,h}.pdf`, validation, and a hash
manifest to `results/Figure_3/`. Checks cover 24,027 genes, Spearman
`r=-0.702502`, 4,805 genes per extreme group, ten displayed GSEA pathways,
9,196 spatial coordinates, and four example-gene correlations.

## Independent analytical run

[Model runs](../../../MODEL_RUNS.md) contains full-mode and preflight commands.
Full mode trains MultiVI with validated train/validation/test observation IDs,
then recomputes gene activity, UF ratio, and GSEA using the bundled
manuscript-version `latent/P22_spaDIVA_sampled_Z_poe.h5ad` (`Z_poe` and
`W_ATAC`). The spaDIVA model run and the manuscript translation workflow use
independent, versioned inputs.

Full-mode GSEA displays up to five pathways per NES direction at FDR < 0.25.
An empty selection is labeled "No pathways with FDR < 0.25"; the full GSEA
results are saved in either case.

### Required gene-activity matrix

`P22_GA.txt` is a 3,666,760,836-byte derived gene-activity bridge matrix,
separate from the RNA/ATAC H5ADs. Full mode requires this exact file through
`--p22-ga` or `SPADIVA_P22_GA_PATH`; its public deposit link will accompany the
data release.

### Frozen GSEA libraries

The three paper-version GMT snapshots are included in
`external_predictions/gene_sets/` and are used by default:

- `GO_Biological_Process_2023.gmt`: 5,407 gene sets; [official Enrichr resource](https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=GO_Biological_Process_2023).
- `KEGG_2019_Mouse.gmt`: 303 gene sets; [official Enrichr resource](https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=KEGG_2019_Mouse).
- `WikiPathways_2019_Mouse.gmt`: 176 gene sets; [official Enrichr resource](https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=WikiPathways_2019_Mouse).

These snapshots fix the library versions and byte hashes used for the manuscript.
Attribution belongs to the named resources. Full-mode preflight
checks GA/GMT sizes and SHA-256 against
[ARTIFACT_MANIFEST.tsv](external_predictions/ARTIFACT_MANIFEST.tsv).
`SPADIVA_P22_GMT_DIR` may override the directory and expects the same files.
Compact mode uses the versioned figure tables and runs independently of
the GA matrix and external-model training.

Fixed splits are in `fixed_split/split_assignments.csv`. Generated files are
written separately from bundled inputs. Feature counts, correlations, and FDRs
provide the validation criteria for any manuscript-panel refresh.
