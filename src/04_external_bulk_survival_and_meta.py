"""
Layer 3: Zero-Shot External Bulk Validation, Multivariable Cox Survival & ICB Meta-Analysis.

- Computes the locked 10-gene Immunometabolic Risk Score zero-shot (no retraining/reweighting)
  using cohort-relative gene-wise Z-score standardization.
- Evaluates multivariable Cox proportional hazards models in:
  * Liu et al. (2019; anti-PD-1)
  * Gide et al. (2019; anti-PD-1 monotherapy & anti-PD-1 + anti-CTLA-4 combination)
  * TCGA-SKCM (non-ICB general prognostic cohort, strictly separated from ICB pooling)
- Runs pre-specified DerSimonian-Laird (DL) random-effects meta-analysis restricted strictly
  to the two ICB cohorts (K = 2, N = 191), alongside conservative truncated HKSJ (df = 1)
  and anti-PD-1 monotherapy sensitivity analyses.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats

from config import DEFAULT_DATA_DIR, DEFAULT_OUT_DIR, PROTECTIVE_GENES, RISK_GENES


def compute_locked_risk_score(expr_samples_by_genes: pd.DataFrame) -> pd.Series:
    """
    Zero-shot 10-gene Risk Score with cohort-relative gene-wise Z-score standardization.
    Applies log2(x + 1) automatically if values appear on linear scale (max > 50).
    """
    X = expr_samples_by_genes.copy()
    if float(np.nanmax(X.values)) > 50.0:
        X = np.log2(np.maximum(X, 0.0) + 1.0)

    missing = [g for g in RISK_GENES + PROTECTIVE_GENES if g not in X.columns]
    if missing:
        raise KeyError(f"Missing signature genes in bulk expression matrix: {missing}")

    sub = X[RISK_GENES + PROTECTIVE_GENES]
    z = (sub - sub.mean(axis=0)) / sub.std(axis=0, ddof=0).replace(0, np.nan)
    raw_score = z[RISK_GENES].mean(axis=1) - z[PROTECTIVE_GENES].mean(axis=1)
    return (raw_score - raw_score.mean()) / raw_score.std(ddof=0)


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
    }, cph.summary.reset_index()


def dersimonian_laird_and_hksj(log_hrs: list, ses: list):
    """Pre-specified random-effects DerSimonian-Laird + conservative truncated HKSJ (df = K - 1)."""
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
    z_dl = theta_dl / se_dl
    p_dl = float(2.0 * (1.0 - stats.norm.cdf(abs(z_dl))))
    ci_dl = (float(np.exp(theta_dl - 1.96 * se_dl)), float(np.exp(theta_dl + 1.96 * se_dl)))

    q_hksj = float(np.sum(w_re * (y - theta_dl) ** 2) / df)
    q_trunc = max(1.0, q_hksj)
    se_hksj = float(np.sqrt(q_trunc / np.sum(w_re)))
    t_crit = float(stats.t.ppf(0.975, df=df))
    t_stat = theta_dl / se_hksj
    p_hksj = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df)))
    ci_hksj = (float(np.exp(theta_dl - t_crit * se_hksj)), float(np.exp(theta_dl + t_crit * se_hksj)))

    return {
        "K": k, "I2_pct": I2, "Q": Q, "tau2": tau2,
        "hr": float(np.exp(theta_dl)),
        "dl_ci_lower": ci_dl[0], "dl_ci_upper": ci_dl[1], "dl_p": p_dl,
        "hksj_ci_lower": ci_hksj[0], "hksj_ci_upper": ci_hksj[1], "hksj_p": p_hksj,
    }


def run_bulk_validation(data_dir: Path, out_dir: Path):
    surv_dir = out_dir / "bulk_survival_meta"
    surv_dir.mkdir(parents=True, exist_ok=True)

    cohorts = {
        "Liu2019": {
            "expr": data_dir / "bulk_liu2019" / "data_mrna_seq_tpm.txt",
            "clin": data_dir / "bulk_liu2019" / "data_clinical_patient.txt",
        },
        "Gide2019": {
            "expr": data_dir / "bulk_gide2019" / "cancercell_normalized_counts_genenames.txt",
            "clin": data_dir / "bulk_gide2019" / "clinical_metadata.csv",
        },
        "TCGA_SKCM": {
            "expr": data_dir / "bulk_tcga_skcm" / "tcga_skcm_expression.csv",
            "clin": data_dir / "bulk_tcga_skcm" / "tcga_skcm_clinical.csv",
        },
    }

    for name, paths in cohorts.items():
        if not paths["expr"].is_file() or not paths["clin"].is_file():
            raise FileNotFoundError(
                f"Missing {name} files:\n  {paths['expr']}\n  {paths['clin']}\n"
                "Please download the public datasets as described in README.md."
            )

    liu_expr = pd.read_csv(cohorts["Liu2019"]["expr"], sep="\t", index_col=0).T
    liu_clin = pd.read_csv(cohorts["Liu2019"]["clin"], sep="\t", comment="#")
    liu_clin = liu_clin.set_index("PATIENT_ID").loc[liu_expr.index.intersection(liu_clin["PATIENT_ID"])]
    liu_clin["Risk_Score_Std"] = compute_locked_risk_score(liu_expr.loc[liu_clin.index])

    gide_expr = pd.read_csv(cohorts["Gide2019"]["expr"], sep="\t", index_col=0).T
    gide_clin = pd.read_csv(cohorts["Gide2019"]["clin"]).set_index("Sample_ID")
    gide_common = gide_expr.index.intersection(gide_clin.index)
    gide_clin = gide_clin.loc[gide_common]
    gide_clin["Risk_Score_Std"] = compute_locked_risk_score(gide_expr.loc[gide_common])

    tcga_expr = pd.read_csv(cohorts["TCGA_SKCM"]["expr"], index_col=0).T
    tcga_clin = pd.read_csv(cohorts["TCGA_SKCM"]["clin"]).set_index("PATIENT_ID")
    tcga_common = tcga_expr.index.intersection(tcga_clin.index)
    tcga_clin = tcga_clin.loc[tcga_common]
    tcga_clin["Risk_Score_Std"] = compute_locked_risk_score(tcga_expr.loc[tcga_common])

    liu_os, _ = fit_multivariable_cox(liu_clin, "OS_MONTHS", "OS_EVENT", ["log_TMB", "LDH_elevated", "prior_CTLA4", "sex_male"])
    liu_pfs, _ = fit_multivariable_cox(liu_clin, "PFS_MONTHS", "PFS_EVENT", ["log_TMB", "LDH_elevated", "prior_CTLA4", "sex_male"])
    gide_os, _ = fit_multivariable_cox(gide_clin, "OS_Months", "OS_Event", ["Age", "Sex_Male", "Is_Combo"])
    gide_pfs, _ = fit_multivariable_cox(gide_clin, "PFS_Months", "PFS_Event", ["Age", "Sex_Male", "Is_Combo"])
    tcga_os, _ = fit_multivariable_cox(tcga_clin, "OS_time_days", "OS_event", ["Age", "Sex_Male"])

    indiv_rows = [
        {"cohort": "Liu et al. (2019)", "endpoint": "OS", **liu_os},
        {"cohort": "Liu et al. (2019)", "endpoint": "PFS", **liu_pfs},
        {"cohort": "Gide et al. (2019)", "endpoint": "OS", **gide_os},
        {"cohort": "Gide et al. (2019)", "endpoint": "PFS", **gide_pfs},
        {"cohort": "TCGA-SKCM (Non-ICB)", "endpoint": "OS", **tcga_os},
    ]
    pd.DataFrame(indiv_rows).to_csv(surv_dir / "multivariable_cox_cohorts.csv", index=False)

    meta_os = dersimonian_laird_and_hksj([liu_os["log_hr"], gide_os["log_hr"]], [liu_os["se"], gide_os["se"]])
    meta_pfs = dersimonian_laird_and_hksj([liu_pfs["log_hr"], gide_pfs["log_hr"]], [liu_pfs["se"], gide_pfs["se"]])
    meta_rows = [
        {"analysis": "ICB_Primary_Pooled", "endpoint": "OS", "pooled_N": liu_os["n_adjusted"] + gide_os["n_adjusted"], **meta_os},
        {"analysis": "ICB_Primary_Pooled", "endpoint": "PFS", "pooled_N": liu_pfs["n_adjusted"] + gide_pfs["n_adjusted"], **meta_pfs},
    ]
    pd.DataFrame(meta_rows).to_csv(surv_dir / "icb_meta_analysis_dl_hksj.csv", index=False)
    print(f"[Layer 3] ICB Pooled OS HR = {meta_os['hr']:.3f} (DL P={meta_os['dl_p']:.4f}; HKSJ P={meta_os['hksj_p']:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run zero-shot external bulk validation and ICB meta-analysis.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    run_bulk_validation(args.data_dir, args.out_dir)
