"""
Layer 3 Sensitivity Controls:
1. Bulk TME Pathway Part-Whole Sensitivity (10 curated pathways before vs. after removing overlaps).
2. Uniform Random 10-Gene-Set Permutation Control (1,000 iterations excluding the 10 signature genes).
3. Clean Expression- & Variance-Matched Permutation Control (1,000 iterations excluding the 10 signature genes).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats
from statsmodels.stats.multitest import multipletests

from config import (
    ALL_SIGNATURE_GENES,
    CURATED_TME_PATHWAYS,
    DEFAULT_DATA_DIR,
    DEFAULT_OUT_DIR,
    PERMUTATION_ITERATIONS,
    PROTECTIVE_GENES,
    RISK_GENES,
    get_non_overlapping_pathways,
)


def prep_log2_z(expr_samples_by_genes: pd.DataFrame):
    X = expr_samples_by_genes.copy()
    if float(np.nanmax(X.values)) > 50.0:
        X = np.log2(np.maximum(X, 0.0) + 1.0)
    stds = X.std(axis=0, ddof=0)
    X = X.loc[:, stds > 1e-8]
    Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
    return X, Z


def run_tme_part_whole(Z: pd.DataFrame, risk_score: pd.Series, cohort_name: str):
    clean_pws = get_non_overlapping_pathways()
    rows = []
    for pw_name, full_genes in CURATED_TME_PATHWAYS.items():
        fg = [g for g in full_genes if g in Z.columns]
        cg = [g for g in clean_pws[pw_name] if g in Z.columns]
        rho_f, p_f = stats.spearmanr(risk_score, Z[fg].mean(axis=1))
        rho_c, p_c = stats.spearmanr(risk_score, Z[cg].mean(axis=1))
        rows.append({
            "cohort": cohort_name, "pathway": pw_name,
            "n_full": len(fg), "n_clean": len(cg),
            "rho_full": float(rho_f), "p_full": float(p_f),
            "rho_clean": float(rho_c), "p_clean": float(p_c),
        })
    df = pd.DataFrame(rows)
    df["fdr_full"] = multipletests(df["p_full"], method="fdr_bh")[1]
    df["fdr_clean"] = multipletests(df["p_clean"], method="fdr_bh")[1]
    return df


def build_matched_pools(X_log: pd.DataFrame, target_genes: list, min_candidates: int = 15):
    means = X_log.mean(axis=0)
    variances = X_log.var(axis=0, ddof=0)
    eligible = [g for g in X_log.columns if g not in set(ALL_SIGNATURE_GENES)]
    m_el, v_el = means.loc[eligible], variances.loc[eligible]

    pools = {}
    for g in target_genes:
        gm, gv = float(means[g]), float(variances[g])
        tol = 0.05
        while tol <= 2.0:
            mask = (np.abs(m_el - gm) <= tol * max(abs(gm), 1e-6)) & (
                np.abs(v_el - gv) <= tol * max(abs(gv), 1e-6)
            )
            cands = m_el.index[mask].tolist()
            if len(cands) >= min_candidates:
                pools[g] = cands
                break
            tol += 0.05
        else:
            dist = ((m_el - gm) / m_el.std()) ** 2 + ((v_el - gv) / v_el.std()) ** 2
            pools[g] = dist.nsmallest(min_candidates).index.tolist()
    return pools


def univariate_cox_z(score: np.ndarray, durations: np.ndarray, events: np.ndarray) -> float:
    df = pd.DataFrame({"T": durations, "E": events, "S": (score - score.mean()) / (score.std() + 1e-12)}).dropna()
    cph = CoxPHFitter()
    cph.fit(df, duration_col="T", event_col="E")
    return float(cph.summary.loc["S", "z"])


def run_permutation_suite(
    X_liu: pd.DataFrame, Z_liu: pd.DataFrame, clin_liu: pd.DataFrame,
    X_gide: pd.DataFrame, Z_gide: pd.DataFrame, clin_gide: pd.DataFrame,
    mode: str = "matched", n_iter: int = PERMUTATION_ITERATIONS, seed: int = 42,
):
    """Run either 'uniform' random 10-gene-set or 'matched' expression/variance permutation."""
    rng = np.random.default_rng(seed)
    common_genes = sorted(set(X_liu.columns).intersection(X_gide.columns))
    X_liu, Z_liu = X_liu[common_genes], Z_liu[common_genes]
    X_gide, Z_gide = X_gide[common_genes], Z_gide[common_genes]
    eligible_all = [g for g in common_genes if g not in set(ALL_SIGNATURE_GENES)]

    if mode == "matched":
        pools_l = build_matched_pools(X_liu, ALL_SIGNATURE_GENES)
        pools_g = build_matched_pools(X_gide, ALL_SIGNATURE_GENES)
        pools = {g: (sorted(set(pools_l[g]).intersection(pools_g[g])) or pools_l[g]) for g in ALL_SIGNATURE_GENES}
    else:
        pools = None

    def score_of(Z_df, r_genes, p_genes):
        return (Z_df[r_genes].mean(axis=1) - Z_df[p_genes].mean(axis=1)).values

    obs_liu = score_of(Z_liu, RISK_GENES, PROTECTIVE_GENES)
    obs_gide = score_of(Z_gide, RISK_GENES, PROTECTIVE_GENES)

    n_liu, n_gide = len(clin_liu), len(clin_gide)
    w_liu = np.sqrt(n_liu / (n_liu + n_gide))
    w_gide = np.sqrt(n_gide / (n_liu + n_gide))

    endpoints = [
        ("OS",  clin_liu["OS_MONTHS"].values,  clin_liu["OS_EVENT"].values,
                clin_gide["OS_Months"].values, clin_gide["OS_Event"].values),
        ("PFS", clin_liu["PFS_MONTHS"].values, clin_liu["PFS_EVENT"].values,
                clin_gide["PFS_Months"].values, clin_gide["PFS_Event"].values),
    ]

    summary_rows, null_records = [], []
    for ep, t_l, e_l, t_g, e_g in endpoints:
        z_obs_l = univariate_cox_z(obs_liu, t_l, e_l)
        z_obs_g = univariate_cox_z(obs_gide, t_g, e_g)
        z_obs_eq = (z_obs_l + z_obs_g) / np.sqrt(2.0)
        z_obs_nw = w_liu * z_obs_l + w_gide * z_obs_g

        null_l, null_g, null_eq, null_nw = [], [], [], []
        for it in range(n_iter):
            if mode == "matched":
                chosen = []
                for g in ALL_SIGNATURE_GENES:
                    avail = [c for c in pools[g] if c not in chosen]
                    chosen.append(str(rng.choice(avail if avail else pools[g])))
            else:
                chosen = [str(x) for x in rng.choice(eligible_all, size=10, replace=False)]

            r_sub, p_sub = chosen[:3], chosen[3:]
            zl = univariate_cox_z(score_of(Z_liu, r_sub, p_sub), t_l, e_l)
            zg = univariate_cox_z(score_of(Z_gide, r_sub, p_sub), t_g, e_g)
            zeq = (zl + zg) / np.sqrt(2.0)
            znw = w_liu * zl + w_gide * zg
            null_l.append(zl); null_g.append(zg); null_eq.append(zeq); null_nw.append(znw)
            null_records.append({"mode": mode, "endpoint": ep, "iter": it + 1, "z_liu": zl, "z_gide": zg, "z_eq": zeq, "z_nw": znw})

        for label, z_o, arr in [
            (f"{mode}_Pooled_ICB_{ep}_EqualWeight", z_obs_eq, np.array(null_eq)),
            (f"{mode}_Pooled_ICB_{ep}_NWeighted",   z_obs_nw, np.array(null_nw)),
            (f"{mode}_Gide_Alone_{ep}",             z_obs_g,  np.array(null_g)),
            (f"{mode}_Liu_Alone_{ep}",              z_obs_l,  np.array(null_l)),
        ]:
            p_emp = float((np.sum(np.abs(arr) >= abs(z_o)) + 1) / (len(arr) + 1))
            summary_rows.append({
                "mode": mode, "specification": label,
                "observed_z": float(z_o), "null_mean_z": float(arr.mean()),
                "null_sd_z": float(arr.std(ddof=1)), "empirical_two_sided_p": p_emp,
                "n_valid_iterations": len(arr),
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(null_records)


def main(data_dir: Path, out_dir: Path):
    sens_dir = out_dir / "sensitivity_controls"
    sens_dir.mkdir(parents=True, exist_ok=True)

    liu_expr = pd.read_csv(data_dir / "bulk_liu2019" / "data_mrna_seq_tpm.txt", sep="\t", index_col=0).T
    liu_clin = pd.read_csv(data_dir / "bulk_liu2019" / "data_clinical_patient.txt", sep="\t", comment="#").set_index("PATIENT_ID")
    liu_idx = liu_expr.index.intersection(liu_clin.index)
    X_liu, Z_liu = prep_log2_z(liu_expr.loc[liu_idx])
    liu_clin = liu_clin.loc[liu_idx]

    gide_expr = pd.read_csv(data_dir / "bulk_gide2019" / "cancercell_normalized_counts_genenames.txt", sep="\t", index_col=0).T
    gide_clin = pd.read_csv(data_dir / "bulk_gide2019" / "clinical_metadata.csv").set_index("Sample_ID")
    gide_idx = gide_expr.index.intersection(gide_clin.index)
    X_gide, Z_gide = prep_log2_z(gide_expr.loc[gide_idx])
    gide_clin = gide_clin.loc[gide_idx]

    r_liu = Z_liu[RISK_GENES].mean(axis=1) - Z_liu[PROTECTIVE_GENES].mean(axis=1)
    r_gide = Z_gide[RISK_GENES].mean(axis=1) - Z_gide[PROTECTIVE_GENES].mean(axis=1)

    tme_df = pd.concat([
        run_tme_part_whole(Z_liu, r_liu, "Liu2019"),
        run_tme_part_whole(Z_gide, r_gide, "Gide2019"),
    ], ignore_index=True)
    tme_df.to_csv(sens_dir / "tme_part_whole_sensitivity.csv", index=False)

    u_sum, u_null = run_permutation_suite(X_liu, Z_liu, liu_clin, X_gide, Z_gide, gide_clin, mode="uniform", seed=42)
    m_sum, m_null = run_permutation_suite(X_liu, Z_liu, liu_clin, X_gide, Z_gide, gide_clin, mode="matched", seed=42)
    pd.concat([u_sum, m_sum], ignore_index=True).to_csv(sens_dir / "permutation_controls_summary.csv", index=False)
    pd.concat([u_null, m_null], ignore_index=True).to_csv(sens_dir / "permutation_controls_nulls.csv", index=False)
    print(f"[Layer 3 Controls] Saved 10-pathway TME sensitivity + uniform & matched permutations to {sens_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run bulk TME sensitivity and both permutation controls.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    main(args.data_dir, args.out_dir)
