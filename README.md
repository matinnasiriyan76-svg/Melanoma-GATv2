# Integrative multi-omics analysis identifies an immunometabolic signature associated with melanoma survival.

Official code repository for the manuscript:
> **Integrative multi-omics analysis identifies an immunometabolic signature associated with melanoma survival**.

## Overview

This repository provides the complete, leakage-controlled computational pipeline integrating **six public data resources** without synthetic or imputed data:

1. **Layer 1 — Single-Cell Discovery & Exploratory Graph Modeling (`src/01_sc_preprocessing_and_lopo_graphs.py`, `src/02_train_realgatv2_and_rewiring.py`)**
   - Baseline melanoma scRNA-seq (GSE120575; 12 anti-PD-1 patients, 3,795 post-QC cells: 4 Durable Benefit, 8 Progressive Disease).
   - Strict outer-fold Leave-One-Patient-Out (LOPO) isolation (`outer_01_final` to `outer_12_final`) for HVG selection, Z-score scaling, 32-PC PCA, and per-patient union k-NN cell–cell graphs.
   - Exploratory 3-seed `RealGATv2` training (seeds `42, 142, 242`) and 10-replicate test-time degree-preserving Maslov–Sneppen rewiring audit.

2. **Layer 2 — Post-Hoc PCA Signature Derivation & 12-Fold Stability Audit (`src/03_pca_signature_and_12fold_audit.py`)**
   - Independently of GATv2 attention weights, derives and locks the **10-gene immunometabolic signature** from patient-level PCA group contrasts in `outer_01_final`:
     - **Risk Arm (3 genes):** `GAPDH`, `PGAM1`, `FKBP1A` (PC1+)
     - **Protective Arm (7 genes):** `CCR7`, `IL7R`, `S1PR1` (PC1−), `IFI30`, `IRF8` (PC2+), `NR4A1` (exploratory PC8+), and `CD8A` (biologically prioritized cytotoxic T-cell anchor)
   - Audits 12-fold loading stability via gene-agnostic dot-product sign alignment across 1,861 shared HVGs.

3. **Layer 3 — Zero-Shot External Bulk Validation, Survival Meta-Analysis & Sensitivity Controls (`src/04_external_bulk_survival_and_meta.py`, `src/05_tme_part_whole_and_matched_permutations.py`)**
   - Zero-shot application (no retraining or reweighting; cohort-relative gene-wise Z-score standardization) in:
     - **Liu et al. (2019)** (cBioPortal `mel_dfci_2019`; N = 121 total, N = 118 adjusted)
     - **Gide et al. (2019)** (ENA `PRJEB23709`; N = 73: 41 anti-PD-1 monotherapy, 32 combination)
     - **TCGA-SKCM** (non-ICB general prognostic cohort; N = 245 total, N = 166 adjusted; strictly separated from ICB pooling)
   - Pre-specified DerSimonian–Laird random-effects ICB meta-analysis (K = 2, N = 191) with conservative truncated Hartung–Knapp–Sidik–Jonkman (HKSJ, df = 1) and anti-PD-1 monotherapy sensitivity analyses.
   - Bulk TME pathway part–whole sensitivity (removing all signature-overlapping genes) and clean 1,000-iteration expression- and variance-matched permutation controls excluding all 10 signature genes.

4. **Layer 4 — Orthogonal Exploratory Context (`src/06_spatial_visium_and_depmap_crispr.py`)**
   - **10x Genomics Visium FFPE Human Cutaneous Melanoma** (1 section, 3,458 spots; exploratory 8-gene spatial risk score without `GAPDH`/`FKBP1A`; global Moran's I and descriptive spot-level Spearman rho before/after overlap removal).
   - **DepMap Public 26Q3 CRISPR Chronos Screens** (N = 73 melanoma cell lines with complete 9-gene coverage; exact two-sided Wilcoxon signed-rank test, Hodges–Lehmann pseudomedian, 2,000-resample bootstrap 95% CI, and single-gene decompositions).

---

## Repository Structure

```text
melanoma-immunometabolic-signature/
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

---

## Public Data Resources & Expected Directory Layout

No raw patient data are redistributed in this repository. Download the six public datasets from their primary repositories and place them under `$MELANOMA_DATA_DIR` (default: `./data`):

| Layer | Dataset | Source / Accession | Expected Relative Path under `data/` |
|---|---|---|---|
| 1–2 | Sade-Feldman et al. (2018) | GEO: `GSE120575` | `sc_gse120575/GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz` & `sc_gse120575/baseline_12patients_metadata.csv` |
| 3 | Liu et al. (2019) | cBioPortal: `mel_dfci_2019` | `bulk_liu2019/data_mrna_seq_tpm.txt` & `bulk_liu2019/data_clinical_patient.txt` |
| 3 | Gide et al. (2019) | ENA: `PRJEB23709` | `bulk_gide2019/cancercell_normalized_counts_genenames.txt` & `bulk_gide2019/clinical_metadata.csv` |
| 3 | TCGA-SKCM | GDC / cBioPortal | `bulk_tcga_skcm/tcga_skcm_expression.csv` & `bulk_tcga_skcm/tcga_skcm_clinical.csv` |
| 4 | 10x Visium FFPE Melanoma | 10x Genomics Public Datasets | `spatial_visium/CytAssist_FFPE_Human_Skin_Melanoma_filtered_feature_bc_matrix.h5` & `spatial_visium/tissue_positions.csv` |
| 4 | DepMap Public 26Q3 | Broad DepMap Portal | `depmap_26q3/DepMap_Custom_10Genes_Chronos.csv` |

---

## Installation & Reproduction

```bash
git clone https://github.com/<your-username>/melanoma-immunometabolic-signature.git
cd melanoma-immunometabolic-signature
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your data and output directories (or pass `--data-dir` / `--out-dir` to each script):

```bash
export MELANOMA_DATA_DIR=/path/to/data
export MELANOMA_OUT_DIR=/path/to/results
bash run_all.sh
```

## Methodological Integrity & Zero-Fabrication Policy

- **Zero synthetic data:** All analysis and plotting scripts read exclusively from real public datasets and real audited intermediate outputs. Missing input files raise explicit `FileNotFoundError` exceptions rather than falling back to simulated data.
- **No patient-level leakage:** All HVG selection, scaling, PCA, and k-NN graph construction are strictly confined to the 11 outer-training patients in each LOPO fold.
- **Transparent reporting:** Both DerSimonian–Laird and conservative truncated HKSJ (df = 1) meta-analysis estimates, part–whole pathway overlap removals, and positively shifted matched permutation null distributions are explicitly computed and exported.

## Citation

If you use this code, please cite the accompanying manuscript in *npj Precision Oncology* (citation details will be updated upon publication).
