"""
Layer 2: Post-Hoc Patient-Level PCA Group Contrasts & 12-Fold Stability Audit.

Part 1 (Initial Derivation in Reference Fold outer_01_final):
- Computes patient-level mean PCA coordinates for all 32 PCs in outer_01_final.
- Contrasts Durable Benefit (y=1) vs. Progressive Disease (y=0) patients (Cohen's d & Mann-Whitney U)
  to identify the top discriminative axes (PC1, PC2, exploratory PC8) and extracts the top-loading
  genes on each opposing pole independently of GATv2 attention weights.

Part 2 (12-Fold Cross-Validation Stability Audit):
- Aligns PC1, PC2, and PC8 across all 12 outer folds via gene-agnostic dot-product sign alignment
  over the shared HVG intersection, exporting pairwise cosine similarities and Top-10 rank rates.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from config import ALL_SIGNATURE_GENES, DEFAULT_OUT_DIR

GENE_ASSIGNMENTS = {
    "GAPDH":  (0, +1, "Risk",       "PC1+"),
    "PGAM1":  (0, +1, "Risk",       "PC1+"),
    "FKBP1A": (0, +1, "Risk",       "PC1+"),
    "CCR7":   (0, -1, "Protective", "PC1-"),
    "IL7R":   (0, -1, "Protective", "PC1-"),
    "S1PR1":  (0, -1, "Protective", "PC1-"),
    "IFI30":  (1, +1, "Protective", "PC2+"),
    "IRF8":   (1, +1, "Protective", "PC2+"),
    "CD8A":   (1, -1, "Protective", "PC2- (Biologically Prioritized Anchor)"),
    "NR4A1":  (7, +1, "Protective", "PC8+ (Exploratory)"),
}


def run_reference_fold_group_contrast(ref_fold_dir: Path, audit_dir: Path):
    """Part 1: Patient-level PCA group contrast in outer_01_final (independent of GATv2)."""
    pca = np.load(ref_fold_dir / "pca_parameters.npz", allow_pickle=True)
    comps = pca["components"]
    gids = [str(g) for g in pca["gene_ids"]]

    pat_means, pat_labels = [], []
    for g_file in sorted(ref_fold_dir.glob("graph_*.npz")):
        d = np.load(g_file, allow_pickle=True)
        if bool(d["is_test"]):
            continue  # outer_01_final training patients only (no test-patient leakage)
        pat_means.append(d["x"].mean(axis=0))
        pat_labels.append(int(d["y"]))

    P = np.vstack(pat_means)
    y = np.asarray(pat_labels)

    contrast_rows = []
    for pc_i in range(P.shape[1]):
        g1 = P[y == 1, pc_i]
        g0 = P[y == 0, pc_i]
        pooled_sd = np.sqrt(((len(g1) - 1) * g1.var(ddof=1) + (len(g0) - 1) * g0.var(ddof=1)) / (len(g1) + len(g0) - 2))
        cohen_d = float((g1.mean() - g0.mean()) / (pooled_sd + 1e-12))
        u_res = stats.mannwhitneyu(g1, g0, alternative="two-sided")
        s_load = pd.Series(comps[pc_i], index=gids)
        contrast_rows.append({
            "pc": f"PC{pc_i + 1}",
            "mean_benefit": float(g1.mean()),
            "mean_pd": float(g0.mean()),
            "cohen_d_benefit_vs_pd": cohen_d,
            "mannwhitney_p": float(u_res.pvalue),
            "top5_pos_genes": ",".join(s_load.nlargest(5).index.tolist()),
            "top5_neg_genes": ",".join(s_load.nsmallest(5).index.tolist()),
        })

    pd.DataFrame(contrast_rows).to_csv(audit_dir / "outer01_pca_patient_group_contrasts.csv", index=False)


def run_pca_12fold_audit(out_dir: Path):
    graphs_root = out_dir / "lopo_patient_graphs"
    fold_dirs = sorted(graphs_root.glob("outer_*_final"))
    if len(fold_dirs) != 12:
        raise FileNotFoundError(f"Expected 12 outer folds in {graphs_root}, found {len(fold_dirs)}.")

    audit_dir = out_dir / "pca_12fold_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Part 1: Documented post-hoc PCA group contrast in outer_01_final
    run_reference_fold_group_contrast(fold_dirs[0], audit_dir)

    # Part 2: 12-fold loading stability audit
    pcs_to_audit = [0, 1, 7]
    raw_series = {pc: [] for pc in pcs_to_audit}
    var_ratios = {pc: [] for pc in pcs_to_audit}

    for fd in fold_dirs:
        pca = np.load(fd / "pca_parameters.npz", allow_pickle=True)
        comps = pca["components"]
        gids = [str(g) for g in pca["gene_ids"]]
        evr = pca["explained_variance_ratio"]
        for pc in pcs_to_audit:
            raw_series[pc].append(pd.Series(comps[pc], index=gids))
            var_ratios[pc].append(float(evr[pc]))

    shared_hvgs = sorted(set.intersection(*(set(s.index) for s in raw_series[0])))

    aligned_series = {pc: [] for pc in pcs_to_audit}
    axis_summary, pairwise_rows = [], []

    for pc in pcs_to_audit:
        ref = raw_series[pc][0].loc[shared_hvgs].values
        sign_flips, aligned_mat = 0, []
        for s in raw_series[pc]:
            vec = s.loc[shared_hvgs].values
            sign = -1.0 if np.dot(vec, ref) < 0 else 1.0
            if sign < 0:
                sign_flips += 1
            aligned_series[pc].append(s * sign)
            aligned_mat.append(vec * sign)

        mat = np.vstack(aligned_mat)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        cos_mat = (mat @ mat.T) / (norms @ norms.T + 1e-12)
        cos_vals = cos_mat[np.triu_indices(len(mat), k=1)]

        pc_name = f"PC{pc + 1}"
        axis_summary.append({
            "pc": pc_name,
            "mean_var_explained": float(np.mean(var_ratios[pc])),
            "n_shared_hvgs": len(shared_hvgs),
            "sign_flips": sign_flips,
            "mean_cosine": float(np.mean(cos_vals)),
            "sd_cosine": float(np.std(cos_vals, ddof=1)),
            "min_cosine": float(np.min(cos_vals)),
            "max_cosine": float(np.max(cos_vals)),
        })
        for pair_idx, c_val in enumerate(cos_vals, start=1):
            pairwise_rows.append({"pc": pc_name, "pair_idx": pair_idx, "cosine_similarity": float(c_val)})

    gene_rows = []
    for gene in ALL_SIGNATURE_GENES:
        pc_idx, pole_sign, arm, pole_lbl = GENE_ASSIGNMENTS[gene]
        loadings, top10_hits = [], 0
        for s in aligned_series[pc_idx]:
            if gene in s.index:
                loadings.append(float(s.loc[gene]))
                ordered = s.sort_values(ascending=(pole_sign < 0))
                if int(ordered.index.get_loc(gene)) + 1 <= 10:
                    top10_hits += 1

        gene_rows.append({
            "gene": gene, "arm": arm, "assigned_pole": pole_lbl,
            "mean_loading": float(np.mean(loadings)) if loadings else np.nan,
            "sd_loading": float(np.std(loadings, ddof=1)) if len(loadings) > 1 else np.nan,
            "folds_present": len(loadings),
            "top10_folds": top10_hits,
            "top10_rate_pct": (100.0 * top10_hits / len(loadings)) if loadings else 0.0,
        })

    pd.DataFrame(axis_summary).to_csv(audit_dir / "pca_12fold_axis_stability.csv", index=False)
    pd.DataFrame(pairwise_rows).to_csv(audit_dir / "pca_12fold_pairwise_cosines.csv", index=False)
    pd.DataFrame(gene_rows).to_csv(audit_dir / "pca_12fold_gene_stability.csv", index=False)
    print(f"[Layer 2] Group contrasts + 12-fold audit ({len(shared_hvgs)} shared HVGs) saved to {audit_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run post-hoc PCA group contrast and 12-fold stability audit.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    run_pca_12fold_audit(args.out_dir)
