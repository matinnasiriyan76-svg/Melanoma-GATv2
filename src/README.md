# Melanoma-GATv2 (v1.1.0): Integrative multi-omics analysis identifies an immunometabolic signature associated with melanoma survival Pipeline

Official code repository for the manuscript:
> **Integrative multi-omics analysis identifies an immunometabolic signature associated with melanoma survival**.

## What's New in v1.1.0
- Aligned `k = 15` symmetric union k-NN graph construction in `src/01_sc_preprocessing_and_lopo_graphs.py` with the audited `cell47_state_snapshot.json` manifest and manuscript Methods.
- Clarified GSE120575 TPM matrix QC filtering (label-blind median + 3 MAD total-expression filter + CP10k/log1p standardization).
- Added baseline comparison models (**GCN**, **CellMLP_no_edges**, and **Logistic Regression** with inner 3-fold CV) and full 10-replicate clustering-coefficient auditing in `src/02_train_realgatv2_and_rewiring.py`.
- Implemented post-hoc patient-level PCA group-contrast extraction (`outer_01_final`) prior to the 12-fold stability audit in `src/03_pca_signature_and_12fold_audit.py`.
- Added clinical biomarker benchmarks (**Ayers IFN-γ**, **CYT**, **CD274**), Harrell's **C-index** (OS/PFS), **RECIST ORR AUROC**, and **anti-PD-1 monotherapy sensitivity** meta-analysis in `src/04_external_bulk_survival_and_meta.py`.
- Added **Uniform Random 10-Gene-Set Permutation** alongside **Expression- & Variance-Matched Permutation** in `src/05_tme_part_whole_and_matched_permutations.py`.

## Repository Structure

```text
Melanoma-GATv2-v1.1.0/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── run_all.sh
└── src/
    ├── __init__.py
    ├── config.py
    ├── 01_sc_preprocessing_and_lopo_graphs.py
    ├── 02_train_realgatv2_and_rewiring.py
    ├── 03_pca_signature_and_12fold_audit.py
    ├── 04_external_bulk_survival_and_meta.py
    ├── 05_tme_part_whole_and_matched_permutations.py
    ├── 06_spatial_visium_and_depmap_crispr.py
    └── 07_generate_figures_and_tables.py
```

## Public Data Resources & Expected Directory Layout

Download the six public datasets from their primary repositories and place them under `$MELANOMA_DATA_DIR` (default: `./data`):

| Layer | Dataset | Accession / Source | Expected Relative Path under `data/` |
|---|---|---|---|
| 1–2 | Sade-Feldman et al. (2018) | GEO: `GSE120575` | `sc_gse120575/GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz` & `sc_gse120575/baseline_12patients_metadata.csv` |
| 3 | Liu et al. (2019) | cBioPortal: `mel_dfci_2019` | `bulk_liu2019/data_mrna_seq_tpm.txt` & `bulk_liu2019/data_clinical_patient.txt` |
| 3 | Gide et al. (2019) | ENA: `PRJEB23709` | `bulk_gide2019/cancercell_normalized_counts_genenames.txt` & `bulk_gide2019/clinical_metadata.csv` |
| 3 | TCGA-SKCM | GDC / cBioPortal | `bulk_tcga_skcm/tcga_skcm_expression.csv` & `bulk_tcga_skcm/tcga_skcm_clinical.csv` |
| 4 | 10x Visium FFPE Melanoma | 10x Genomics Public Datasets | `spatial_visium/CytAssist_FFPE_Human_Skin_Melanoma_filtered_feature_bc_matrix.h5` & `spatial_visium/tissue_positions.csv` |
| 4 | DepMap Public 26Q3 | Broad DepMap Portal | `depmap_26q3/DepMap_Custom_10Genes_Chronos.csv` |

## Reproduction

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export MELANOMA_DATA_DIR=/path/to/data
export MELANOMA_OUT_DIR=/path/to/results
bash run_all.sh
```

## Zero-Fabrication Policy
All scripts read strictly from real public datasets and audited intermediate CSV/NPZ artifacts. Missing input files raise explicit `FileNotFoundError` exceptions rather than generating synthetic data.
