#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${MELANOMA_DATA_DIR:-./data}"
OUT_DIR="${MELANOMA_OUT_DIR:-./results}"

python3 src/01_sc_preprocessing_and_lopo_graphs.py --data-dir "${DATA_DIR}" --out-dir "${OUT_DIR}"
python3 src/02_train_realgatv2_and_rewiring.py     --out-dir "${OUT_DIR}"
python3 src/03_pca_signature_and_12fold_audit.py   --out-dir "${OUT_DIR}"
python3 src/04_external_bulk_survival_and_meta.py  --data-dir "${DATA_DIR}" --out-dir "${OUT_DIR}"
python3 src/05_tme_part_whole_and_matched_permutations.py --data-dir "${DATA_DIR}" --out-dir "${OUT_DIR}"
python3 src/06_spatial_visium_and_depmap_crispr.py --data-dir "${DATA_DIR}" --out-dir "${OUT_DIR}"
python3 src/07_generate_figures_and_tables.py      --out-dir "${OUT_DIR}"
