"""
Layer 4: Exploratory Spatial Transcriptomics (10x Visium FFPE) & DepMap 26Q3 CRISPR Dependency.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.neighbors import NearestNeighbors

from config import (
    BOOTSTRAP_RESAMPLES,
    CURATED_TME_PATHWAYS,
    DEFAULT_DATA_DIR,
    DEFAULT_OUT_DIR,
    PROTECTIVE_GENES,
    SPATIAL_PROTECTIVE_GENES,
    SPATIAL_RISK_GENES,
    get_non_overlapping_pathways,
)


def hodges_lehmann_pseudomedian(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    iu = np.triu_indices(len(x), k=0)
    return float(np.median(0.5 * (x[iu[0]] + x[iu[1]])))


def morans_i_permutation(values: np.ndarray, coords: np.ndarray, k: int = 6, n_perm: int = 999, seed: int = 42):
    n = len(values)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    W = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    W = 0.5 * (W + W.T)
    S0 = float(W.sum())

    def calc_i(v):
        vc = v - v.mean()
        denom = float(np.dot(vc, vc))
        return 0.0 if denom == 0 else float((n / S0) * (vc @ (W @ vc)) / denom)

    obs_i = calc_i(values)
    rng = np.random.default_rng(seed)
    ge = sum(abs(calc_i(rng.permutation(values))) >= abs(obs_i) for _ in range(n_perm))
    return obs_i, float((ge + 1) / (n_perm + 1))


def run_visium_spatial(data_dir: Path, out_dir: Path):
    vis_dir = data_dir / "spatial_visium"
    h5_path = vis_dir / "CytAssist_FFPE_Human_Skin_Melanoma_filtered_feature_bc_matrix.h5"
    pos_path = vis_dir / "tissue_positions.csv"
    if not h5_path.is_file() or not pos_path.is_file():
        raise FileNotFoundError(f"Missing Visium FFPE files:\n  {h5_path}\n  {pos_path}")

    with h5py.File(h5_path, "r") as f:
        grp = f["matrix"]
        mat = sparse.csc_matrix((grp["data"][:], grp["indices"][:], grp["indptr"][:]), shape=grp["shape"][:]).T.tocsr()
        barcodes = [b.decode("utf-8") for b in grp["barcodes"][:]]
        genes = [g.decode("utf-8") for g in grp["features"]["name"][:]]

    totals = np.asarray(mat.sum(axis=1)).ravel()
    mat_norm = mat.multiply((1e4 / np.maximum(totals, 1e-12))[:, None]).tocsr()
    mat_norm.data = np.log1p(mat_norm.data)
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    def z_vec(g_list):
        present = [gene_to_idx[g] for g in g_list if g in gene_to_idx]
        arr = mat_norm[:, present].toarray()
        sd = arr.std(axis=0, keepdims=True)
        sd[sd < 1e-8] = 1.0
        return ((arr - arr.mean(axis=0, keepdims=True)) / sd).mean(axis=1), len(present)

    z_pgam1, _ = z_vec(SPATIAL_RISK_GENES)
    z_prot7, _ = z_vec(SPATIAL_PROTECTIVE_GENES)
    score_8g = z_pgam1 - z_prot7

    pos = pd.read_csv(pos_path).set_index("barcode").loc[barcodes]
    moran_i, moran_p = morans_i_permutation(score_8g, pos[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values)

    clean_pws = get_non_overlapping_pathways()
    rows = []
    for niche in ["Glycolysis", "T_cell_Stemness_Memory", "CD8_Cytotoxic", "Curated_IFNG"]:
        zf, nf = z_vec(CURATED_TME_PATHWAYS[niche])
        zc, nc = z_vec(clean_pws[niche])
        rows.append({
            "spatial_niche": niche, "n_full_probes": nf, "n_clean_probes": nc,
            "score8_full_rho": float(stats.spearmanr(score_8g, zf).statistic),
            "score8_clean_rho": float(stats.spearmanr(score_8g, zc).statistic),
            "pgam1_clean_rho": float(stats.spearmanr(z_pgam1, zc).statistic),
            "prot7_full_rho": float(stats.spearmanr(z_prot7, zf).statistic),
            "prot7_clean_rho": float(stats.spearmanr(z_prot7, zc).statistic),
            "morans_I_8gene": moran_i, "morans_P_999perm": moran_p,
        })

    out_sp = out_dir / "spatial_visium"
    out_sp.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_sp / "visium_ffpe_part_whole_descriptive.csv", index=False)


def run_depmap_crispr(data_dir: Path, out_dir: Path, seed: int = 42):
    dep_path = data_dir / "depmap_26q3" / "DepMap_Custom_10Genes_Chronos.csv"
    if not dep_path.is_file():
        raise FileNotFoundError(f"Missing DepMap Chronos file: {dep_path}")

    dep = pd.read_csv(dep_path)
    pcols = [g for g in PROTECTIVE_GENES if g in dep.columns]
    comp = dep.dropna(subset=["GAPDH", "FKBP1A"] + pcols).copy()
    diffs = comp[["GAPDH", "FKBP1A"]].mean(axis=1).values - comp[pcols].mean(axis=1).values

    w_res = stats.wilcoxon(diffs, alternative="two-sided", method="exact")
    hl = hodges_lehmann_pseudomedian(diffs)
    rng = np.random.default_rng(seed)
    ci_lo, ci_hi = np.percentile([np.median(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(BOOTSTRAP_RESAMPLES)], [2.5, 97.5])

    rows = [{
        "comparison": "Primary_Paired_GAPDH_FKBP1A_vs_7Prot", "n_lines": int(len(comp)),
        "mean_val_or_diff": float(diffs.mean()), "hodges_lehmann": hl, "median_diff": float(np.median(diffs)),
        "ci_lower": float(ci_lo), "ci_upper": float(ci_hi),
        "wilcoxon_W": float(w_res.statistic), "exact_two_sided_p": float(w_res.pvalue),
    }]
    for g in ["GAPDH", "FKBP1A"]:
        sub = dep.dropna(subset=[g] + pcols)
        d_g = sub[g].values - sub[pcols].mean(axis=1).values
        wg = stats.wilcoxon(d_g, alternative="two-sided", method="exact")
        se_g = d_g.std(ddof=1) / np.sqrt(len(d_g))
        rows.append({
            "comparison": f"{g}_vs_7Prot", "n_lines": int(len(sub)),
            "mean_val_or_diff": float(sub[g].mean()), "hodges_lehmann": hodges_lehmann_pseudomedian(d_g),
            "median_diff": float(d_g.mean()),
            "ci_lower": float(d_g.mean() - 1.96 * se_g), "ci_upper": float(d_g.mean() + 1.96 * se_g),
            "wilcoxon_W": float(wg.statistic), "exact_two_sided_p": float(wg.pvalue),
        })
    if "PGAM1" in dep.columns:
        pg = dep["PGAM1"].dropna().values
        rows.append({"comparison": "PGAM1_Exploratory_Only", "n_lines": int(len(pg)),
                     "mean_val_or_diff": float(pg.mean()) if len(pg) else np.nan,
                     "hodges_lehmann": np.nan, "median_diff": np.nan, "ci_lower": np.nan,
                     "ci_upper": np.nan, "wilcoxon_W": np.nan, "exact_two_sided_p": np.nan})

    out_dep = out_dir / "depmap_crispr"
    out_dep.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dep / "depmap_paired_and_single_gene_summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run exploratory Visium FFPE spatial and DepMap CRISPR analyses.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    run_visium_spatial(args.data_dir, args.out_dir)
    run_depmap_crispr(args.data_dir, args.out_dir)
