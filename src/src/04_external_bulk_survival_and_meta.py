"""
Layer 3: Zero-Shot External Bulk Validation, Clinical Biomarker Benchmarks,
Multivariable Cox Survival, and ICB Random-Effects Meta-Analysis (Primary + Monotherapy).

Includes:
- Locked 10-gene Risk Score (cohort-relative gene-wise Z-score standardization)
- Clinical benchmarks: Ayers IFN-gamma (6-gene), CYT (GZMA/PRF1), CD274 (PD-L1)
- Harrell's C-index (OS & PFS) and RECIST ORR AUROC
- Multivariable Cox models in Liu et al., Gide et al., and separate non-ICB TCGA-SKCM
- Pre-specified ICB DerSimonian-Laird + truncated HKSJ (df=1) meta-analysis (primary N=191
  and anti-PD-1 monotherapy sensitivity N=159)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy import stats
from sklearn.metrics import roc_auc_score

from config import (
    AYERS_IFNG_6,
    CYT_GENES,
    DEFAULT_DATA_DIR,
    DEFAULT_OUT_DIR,
    PDL1_GENE,
    PROTECTIVE_GENES,
    RISK_GENES,
)


def log2_and_z(expr_samples_by_genes: pd.DataFrame) -> pd.DataFrame:
    X = expr_samples_by_genes.copy()
    if float(np.nanmax(X.values)) > 50.0:
        X = np.log2(np.maximum(X, 0.0) + 1.0)
    return (X - X.mean(axis=0)) / X.std(axis=0, ddof=0).replace(0, np.nan)


def compute_signatures(expr_samples_by_genes: pd.DataFrame) -> pd.DataFrame:
    Z = log2_and_z(expr_samples_by_genes)
    missing = [g for g in RISK_GENES + PROTECTIVE_GENES if g not in Z.columns]
    if missing:
        raise KeyError(f"Missing signature genes in bulk expression matrix: {missing}")

    raw_10g = Z[RISK_GENES].mean(axis=1) - Z[PROTECTIVE_GENES].mean(axis=1)
    out = pd.DataFrame(index=Z.index)
    out["Risk_Score_Std"] = (raw_10g - raw_10g.mean()) / raw_10g.std(ddof=0)
    out["Ayers_IFNG"] = Z[[g for g in AYERS_IFNG_6 if g in Z.columns]].mean(axis=1)
    out["CYT"] = Z[[g for g in CYT_GENES if g in Z.columns]].mean(axis=1)
    out["CD274"] = Z[PDL1_GENE[0]] if PDL1_GENE[0] in Z.columns else np.nan
    return out


def benchmark_biomarkers(df: pd.DataFrame, cohort: str, os_t: str, os_e: str, pfs_t: str, pfs_e: str, orr_col: str):
    rows = []
    for sig, higher_is_worse in [("Risk_Score_Std", True), ("Ayers_IFNG", False), ("CYT", False), ("CD274", False)]:
        if sig not in df.columns or df[sig].isna().all():
            continue
        sub_os = df[[os_t, os_e, sig]].dropna()
        sub_pfs = df[[pfs_t, pfs_e, sig]].dropna()
        c_os = concordance_index(sub_os[os_t], -sub_os[sig] if higher_is_worse else sub_os[sig], sub_os[os_e])
        c_pfs = concordance_index(sub_pfs[pfs_t], -sub_pfs[sig] if higher_is_worse else sub_pfs[sig], sub_pfs[pfs_e])
        orr_auc = np.nan
        if orr_col in df.columns:
            sub_orr = df[[orr_col, sig]].dropna()
            if sub_orr[orr_col].nunique() == 2:
                orr_auc = roc_auc_score(sub_orr[orr_col], -sub_orr[sig] if higher_is_worse else sub_orr[sig])
        rows.append({
            "cohort": cohort, "biomarker": sig,
            "c_index_os": float(c_os), "c_index_pfs": float(c_pfs), "auroc_recist_orr": float(orr_auc),
        })
    return rows


def fit_multivariable_cox(df: pd.DataFrame, duration_col: str, event_col: str, covariates: list):
    cols = [duration_col, event_col, "Risk_Score_Std"] + covariates
    sub = df[cols].dropna().copy()
    cph = CoxPHFitter()
    cph.fit(sub, duration_col=duration_col, event_col=event_col)
    row = cph.summary.loc["Risk_Score_Std"]
    return {
        "n_adjusted": int(len(sub)),
        "n_events": int(sub[event_col].sum()),
        "log_hr": float(row["coef"]),
        "se": float(row["se(coef)"]),
        "hr": float(row["exp(coef)"]),
        "ci_lower": float(row["exp(coef) lower 95%"]),
        "ci_upper": float(row["exp(coef) upper 95%"]),
        "p_value": float(row["p"]),
    }


def dersimonian_laird_and_hksj(log_hrs: list, ses: list):
    y = np.asarray(log_hrs, dtype=float)
    s = np.asarray(ses, dtype=float)
    k = len(y)
    w_fe = 1.0 / (s ** 2)
    theta_fe = np.sum(w_fe * y) / np.sum(w_fe)
    Q = float(np.sum(w_fe * (y - theta_fe) ** 2))
    df = k - 1
    C = float(np.sum(w_fe) - np.sum(w_fe ** 2) / np.sum(w_fe))
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    I2 = max(0.0, (Q - df) / Q * 100.0) if Q > 0 else 0.0

    w_re = 1.0 / (s ** 2 + tau2)
    theta_dl = float(np.sum(w_re * y) / np.sum(w_re))
    se_dl = float(np.sqrt(1.0 / np.sum(w_re)))
    p_dl = float(2.0 * (1.0 - stats.norm.cdf(abs(theta_dl / se_dl))))
    ci_dl = (float(np.exp(theta_dl - 1.96 * se_dl)), float(np.exp(theta_dl + 1.96 * se_dl)))

    q_trunc = max(1.0, float(np.sum(w_re * (y - theta_dl) ** 2) / df))
    se_hksj = float(np.sqrt(q_trunc / np.sum(w_re)))
    t_crit = float(stats.t.ppf(0.975, df=df))
    p_hksj = float(2.0 * (1.0 - stats.t.cdf(abs(theta_dl / se_hksj), df=df)))
    ci_hksj = (float(np.exp(theta_dl - t_crit * se_hksj)), float(np.exp(theta_dl + t_crit * se_hksj)))

    return {
        "K": k, "I2_pct": I2, "Q": Q, "tau2": tau2, "hr": float(np.exp(theta_dl)),
        "dl_ci_lower": ci_dl[0], "dl_ci_upper": ci_dl[1], "dl_p": p_dl,
        "hksj_ci_lower": ci_hksj[0], "hksj_ci_upper": ci_hksj[1], "hksj_p": p_hksj,
    }


def run_bulk_validation(data_dir: Path, out_dir: Path):
    surv_dir = out_dir / "bulk_survival_meta"
    surv_dir.mkdir(parents=True, exist_ok=True)

    cohorts = {
        "Liu2019": (data_dir / "bulk_liu2019" / "data_mrna_seq_tpm.txt", data_dir / "bulk_liu2019" / "data_clinical_patient.txt"),
        "Gide2019": (data_dir / "bulk_gide2019" / "cancercell_normalized_counts_genenames.txt", data_dir / "bulk_gide2019" / "clinical_metadata.csv"),
        "TCGA_SKCM": (data_dir / "bulk_tcga_skcm" / "tcga_skcm_expression.csv", data_dir / "bulk_tcga_skcm" / "tcga_skcm_clinical.csv"),
    }
    for name, (ep, cp) in cohorts.items():
        if not ep.is_file() or not cp.is_file():
            raise FileNotFoundError(f"Missing {name} files:\n  {ep}\n  {cp}")

    liu_expr = pd.read_csv(cohorts["Liu2019"][0], sep="\t", index_col=0).T
    liu_clin = pd.read_csv(cohorts["Liu2019"][1], sep="\t", comment="#").set_index("PATIENT_ID")
    liu_idx = liu_expr.index.intersection(liu_clin.index)
    liu_df = liu_clin.loc[liu_idx].join(compute_signatures(liu_expr.loc[liu_idx]))

    gide_expr = pd.read_csv(cohorts["Gide2019"][0], sep="\t", index_col=0).T
    gide_clin = pd.read_csv(cohorts["Gide2019"][1]).set_index("Sample_ID")
    gide_idx = gide_expr.index.intersection(gide_clin.index)
    gide_df = gide_clin.loc[gide_idx].join(compute_signatures(gide_expr.loc[gide_idx]))

    tcga_expr = pd.read_csv(cohorts["TCGA_SKCM"][0], index_col=0).T
    tcga_clin = pd.read_csv(cohorts["TCGA_SKCM"][1]).set_index("PATIENT_ID")
    tcga_idx = tcga_expr.index.intersection(tcga_clin.index)
    tcga_df = tcga_clin.loc[tcga_idx].join(compute_signatures(tcga_expr.loc[tcga_idx]))

    # Clinical biomarker benchmarks (C-index & RECIST ORR AUROC)
    bench_rows = (
        benchmark_biomarkers(liu_df, "Liu et al. (2019)", "OS_MONTHS", "OS_EVENT", "PFS_MONTHS", "PFS_EVENT", "ORR_BINARY")
        + benchmark_biomarkers(gide_df, "Gide et al. (2019)", "OS_Months", "OS_Event", "PFS_Months", "PFS_Event", "ORR_Binary")
    )
    pd.DataFrame(bench_rows).to_csv(surv_dir / "clinical_biomarker_benchmarks.csv", index=False)

    # Multivariable Cox models
    liu_os = fit_multivariable_cox(liu_df, "OS_MONTHS", "OS_EVENT", ["log_TMB", "LDH_elevated", "prior_CTLA4", "sex_male"])
    liu_pfs = fit_multivariable_cox(liu_df, "PFS_MONTHS", "PFS_EVENT", ["log_TMB", "LDH_elevated", "prior_CTLA4", "sex_male"])
    gide_os = fit_multivariable_cox(gide_df, "OS_Months", "OS_Event", ["Age", "Sex_Male", "Is_Combo"])
    gide_pfs = fit_multivariable_cox(gide_df, "PFS_Months", "PFS_Event", ["Age", "Sex_Male", "Is_Combo"])
    tcga_os = fit_multivariable_cox(tcga_df, "OS_time_days", "OS_event", ["Age", "Sex_Male"])

    # Anti-PD-1 monotherapy sensitivity subgroup in Gide et al. (Is_Combo == 0)
    gide_mono = gide_df[gide_df["Is_Combo"] == 0].copy()
    gide_mono_os = fit_multivariable_cox(gide_mono, "OS_Months", "OS_Event", ["Age", "Sex_Male"])
    gide_mono_pfs = fit_multivariable_cox(gide_mono, "PFS_Months", "PFS_Event", ["Age", "Sex_Male"])

    indiv_rows = [
        {"cohort": "Liu et al. (2019)", "endpoint": "OS", **liu_os},
        {"cohort": "Liu et al. (2019)", "endpoint": "PFS", **liu_pfs},
        {"cohort": "Gide et al. (2019)", "endpoint": "OS", **gide_os},
        {"cohort": "Gide et al. (2019)", "endpoint": "PFS", **gide_pfs},
        {"cohort": "Gide Monotherapy Subgroup", "endpoint": "OS", **gide_mono_os},
        {"cohort": "Gide Monotherapy Subgroup", "endpoint": "PFS", **gide_mono_pfs},
        {"cohort": "TCGA-SKCM (Non-ICB)", "endpoint": "OS", **tcga_os},
    ]
    pd.DataFrame(indiv_rows).to_csv(surv_dir / "multivariable_cox_cohorts.csv", index=False)

    meta_rows = [
        {"analysis": "ICB_Primary_Pooled", "endpoint": "OS", "pooled_N": liu_os["n_adjusted"] + gide_os["n_adjusted"],
         **dersimonian_laird_and_hksj([liu_os["log_hr"], gide_os["log_hr"]], [liu_os["se"], gide_os["se"]])},
        {"analysis": "ICB_Primary_Pooled", "endpoint": "PFS", "pooled_N": liu_pfs["n_adjusted"] + gide_pfs["n_adjusted"],
         **dersimonian_laird_and_hksj([liu_pfs["log_hr"], gide_pfs["log_hr"]], [liu_pfs["se"], gide_pfs["se"]])},
        {"analysis": "ICB_AntiPD1_Monotherapy_Sensitivity", "endpoint": "OS", "pooled_N": liu_os["n_adjusted"] + gide_mono_os["n_adjusted"],
         **dersimonian_laird_and_hksj([liu_os["log_hr"], gide_mono_os["log_hr"]], [liu_os["se"], gide_mono_os["se"]])},
        {"analysis": "ICB_AntiPD1_Monotherapy_Sensitivity", "endpoint": "PFS", "pooled_N": liu_pfs["n_adjusted"] + gide_mono_pfs["n_adjusted"],
         **dersimonian_laird_and_hksj([liu_pfs["log_hr"], gide_mono_pfs["log_hr"]], [liu_pfs["se"], gide_mono_pfs["se"]])},
    ]
    pd.DataFrame(meta_rows).to_csv(surv_dir / "icb_meta_analysis_dl_hksj.csv", index=False)
    print(f"[Layer 3] Saved Cox models, biomarker benchmarks, and primary + monotherapy meta-analysis to {surv_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run external bulk validation, benchmarks, and meta-analysis.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    run_bulk_validation(args.data_dir, args.out_dir)
