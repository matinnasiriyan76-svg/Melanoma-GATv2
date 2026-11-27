"""
Layer 1: Baseline scRNA-seq (GSE120575) Preprocessing & Outer-Fold LOPO k-NN Graph Construction.

Protocol safeguards:
- Label-blind UMI/transcript QC filtering (median + 3 MAD), CP10k + log1p normalization.
- Strict Leave-One-Patient-Out (LOPO) isolation across 12 baseline anti-PD-1 melanoma patients
  (4 Durable Benefit, 8 Progressive Disease; 3,795 post-QC cells).
- All 2,000 HVG selection, Z-score scaling, and 32-PC PCA are fit exclusively on the 11 training
  patients in each outer fold before projecting the held-out patient.
- Per-patient undirected union k-NN graphs (k = 10) constructed independently in fold-local PCA space.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from config import DEFAULT_DATA_DIR, DEFAULT_OUT_DIR


def load_gse120575_baseline(data_dir: Path):
    """Load baseline GSE120575 TPM matrix and verified 12-patient clinical metadata."""
    sc_dir = data_dir / "sc_gse120575"
    expr_path = sc_dir / "GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz"
    meta_path = sc_dir / "baseline_12patients_metadata.csv"

    if not expr_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            f"Missing GSE120575 inputs. Expected:\n  {expr_path}\n  {meta_path}\n"
            "Download GSE120575 from NCBI GEO and place under data/sc_gse120575/."
        )

    meta = pd.read_csv(meta_path)
    required_cols = {"cell_id", "patient_id", "clinical_response_binary"}
    if not required_cols.issubset(meta.columns):
        raise ValueError(f"Metadata must contain columns: {required_cols}")

    expr_df = pd.read_csv(expr_path, sep="\t", index_col=0, compression="infer")
    common_cells = [c for c in meta["cell_id"] if c in expr_df.columns]
    if len(common_cells) == 0:
        raise ValueError("No overlapping cell IDs between expression matrix and metadata.")

    meta = meta[meta["cell_id"].isin(common_cells)].drop_duplicates("cell_id").reset_index(drop=True)
    expr_cells_by_genes = expr_df[meta["cell_id"]].T.astype(np.float32)
    return expr_cells_by_genes, meta


def label_blind_qc_and_normalize(X_raw: np.ndarray, meta: pd.DataFrame):
    """Apply fixed, label-blind MAD outlier filtering, CP10k, and log1p transformation."""
    totals = X_raw.sum(axis=1)
    med = np.median(totals)
    mad = np.median(np.abs(totals - med))
    keep = (totals > 0) & (totals <= med + 3.0 * 1.4826 * mad)

    X_qc = X_raw[keep]
    meta_qc = meta.loc[keep].reset_index(drop=True)

    row_sums = X_qc.sum(axis=1, keepdims=True)
    X_norm = np.log1p((X_qc / np.maximum(row_sums, 1e-12)) * 1e4).astype(np.float32)
    return X_norm, meta_qc


def build_union_knn_edges(X_pca: np.ndarray, k: int = 10):
    """Construct undirected union k-NN edge_index (2, 2*|E|) without self-loops."""
    n_nodes = X_pca.shape[0]
    k_eff = min(k + 1, n_nodes)
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(X_pca)
    _, indices = nn.kneighbors(X_pca)

    edges = set()
    for i in range(n_nodes):
        for j in indices[i, 1:]:
            u, v = (int(i), int(j)) if i < j else (int(j), int(i))
            if u != v:
                edges.add((u, v))

    src, dst = [], []
    for u, v in sorted(edges):
        src.extend([u, v])
        dst.extend([v, u])
    return np.array([src, dst], dtype=np.int64)


def run_lopo_graph_construction(data_dir: Path, out_dir: Path, n_hvgs: int = 2000, n_pcs: int = 32, k: int = 10):
    expr_df, meta = load_gse120575_baseline(data_dir)
    gene_ids = np.array(expr_df.columns, dtype=str)
    X_norm, meta_qc = label_blind_qc_and_normalize(expr_df.values, meta)

    patients = sorted(meta_qc["patient_id"].unique(), key=lambda p: int(str(p).lstrip("P")))
    graphs_root = out_dir / "lopo_patient_graphs"
    graphs_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    for fold_idx, test_patient in enumerate(patients, start=1):
        fold_id = f"outer_{fold_idx:02d}_final"
        fold_dir = graphs_root / fold_id
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_mask = (meta_qc["patient_id"] != test_patient).values
        X_tr = X_norm[train_mask]

        gene_vars = X_tr.var(axis=0)
        hvg_idx = np.argsort(gene_vars)[::-1][:n_hvgs]
        hvg_genes = gene_ids[hvg_idx]

        X_tr_hvg = X_tr[:, hvg_idx]
        mu = X_tr_hvg.mean(axis=0, keepdims=True)
        sd = X_tr_hvg.std(axis=0, keepdims=True)
        sd[sd < 1e-8] = 1.0

        X_tr_scaled = (X_tr_hvg - mu) / sd
        pca = PCA(n_components=n_pcs, random_state=42)
        pca.fit(X_tr_scaled)

        np.savez_compressed(
            fold_dir / "pca_parameters.npz",
            components=pca.components_.astype(np.float32),
            explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
            mean=mu.squeeze(0).astype(np.float32),
            std=sd.squeeze(0).astype(np.float32),
            gene_ids=hvg_genes,
            test_patient=test_patient,
        )

        for pid in patients:
            p_mask = (meta_qc["patient_id"] == pid).values
            X_p_scaled = (X_norm[p_mask][:, hvg_idx] - mu) / sd
            X_p_pca = pca.transform(X_p_scaled).astype(np.float32)
            edge_index = build_union_knn_edges(X_p_pca, k=k)
            y_val = int(meta_qc.loc[p_mask, "clinical_response_binary"].iloc[0])

            g_path = fold_dir / f"graph_{pid}.npz"
            np.savez_compressed(
                g_path,
                x=X_p_pca,
                edge_index=edge_index,
                y=np.int64(y_val),
                patient_id=pid,
                is_test=bool(pid == test_patient),
            )
            manifest.append({
                "fold_id": fold_id,
                "patient_id": pid,
                "is_test": bool(pid == test_patient),
                "y": y_val,
                "n_cells": int(X_p_pca.shape[0]),
                "n_undirected_edges": int(edge_index.shape[1] // 2),
                "graph_path": str(g_path),
            })

    pd.DataFrame(manifest).to_csv(graphs_root / "lopo_graph_manifest.csv", index=False)
    print(f"[Layer 1] Built {len(patients)} outer LOPO folds ({len(manifest)} graphs) in {graphs_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build leakage-controlled LOPO scRNA-seq patient graphs.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    run_lopo_graph_construction(args.data_dir, args.out_dir)
