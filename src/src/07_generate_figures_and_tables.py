"""
Publication-Quality Figure Reproduction (Zero Synthetic Data).

Generates core analytical figures (Figures 2, 3, 4, 5, 6) strictly from real audited CSV
outputs produced by scripts 01-06.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DEFAULT_OUT_DIR

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
C_NAVY, C_RED, C_TEAL, C_AMBER, C_GRAY = "#1B3B6F", "#C0392B", "#0F766E", "#D68910", "#64748B"


def plot_pca_stability(out_dir: Path, fig_dir: Path):
    pw_path = out_dir / "pca_12fold_audit" / "pca_12fold_pairwise_cosines.csv"
    gene_path = out_dir / "pca_12fold_audit" / "pca_12fold_gene_stability.csv"
    if not pw_path.is_file() or not gene_path.is_file():
        return
    pw, gdf = pd.read_csv(pw_path), pd.read_csv(gene_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=300)
    yp = np.arange(len(gdf))[::-1]
    ax1.barh(yp, gdf["mean_loading"], xerr=gdf["sd_loading"], capsize=2.5,
             color=[C_RED if a == "Risk" else C_TEAL for a in gdf["arm"]], alpha=0.85, edgecolor="#222222", lw=0.6)
    ax1.axvline(0, color="#222222", lw=0.9)
    ax1.set_yticks(yp); ax1.set_yticklabels(gdf["gene"], fontweight="bold")
    ax1.set_title("Locked 10-Gene Immunometabolic Signature")

    pcs = ["PC1", "PC2", "PC8"]
    bp = ax2.boxplot([pw.loc[pw["pc"] == pc, "cosine_similarity"].values for pc in pcs],
                     positions=[1, 2, 3], widths=0.45, patch_artist=True, showfliers=False)
    for patch, col in zip(bp["boxes"], [C_NAVY, C_TEAL, C_AMBER]):
        patch.set_facecolor(col); patch.set_alpha(0.35); patch.set_edgecolor(col)
    ax2.set_xticks([1, 2, 3]); ax2.set_xticklabels(pcs, fontweight="bold")
    ax2.set_title("12-Fold Cross-Validation Axis Stability")
    fig.tight_layout()
    fig.savefig(fig_dir / "Figure3_PCA_12Fold_Stability.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "Figure3_PCA_12Fold_Stability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tme_part_whole(out_dir: Path, fig_dir: Path):
    p = out_dir / "sensitivity_controls" / "tme_part_whole_sensitivity.csv"
    if not p.is_file():
        return
    df = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0), dpi=300, sharey=True)
    for idx, (ax, cohort) in enumerate(zip(axes, ["Liu2019", "Gide2019"])):
        sub = df[df["cohort"] == cohort].reset_index(drop=True)
        y = np.arange(len(sub))[::-1]
        for yi, rf, rc in zip(y, sub["rho_full"], sub["rho_clean"]):
            ax.plot([rf, rc], [yi, yi], color="#444444", lw=1.5)
        ax.scatter(sub["rho_full"], y, color=C_NAVY, s=45, label="Full Pathway")
        ax.scatter(sub["rho_clean"], y, color=C_RED, s=45, marker="D", label="Non-Overlapping")
        ax.axvline(0, ls="--", lw=0.9, color=C_GRAY)
        ax.set_yticks(y)
        if idx == 0:
            ax.set_yticklabels(sub["pathway"], fontweight="bold")
        ax.set_title(cohort); ax.set_xlabel("Spearman rho")
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "Figure4_TME_PartWhole_Sensitivity.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "Figure4_TME_PartWhole_Sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_forest_meta(out_dir: Path, fig_dir: Path):
    cox_path = out_dir / "bulk_survival_meta" / "multivariable_cox_cohorts.csv"
    meta_path = out_dir / "bulk_survival_meta" / "icb_meta_analysis_dl_hksj.csv"
    if not cox_path.is_file() or not meta_path.is_file():
        return
    cdf, mdf = pd.read_csv(cox_path), pd.read_csv(meta_path)
    rows = []
    for _, r in cdf[cdf["cohort"].isin(["Liu et al. (2019)", "Gide et al. (2019)"])].iterrows():
        rows.append((f"{r['cohort']} - {r['endpoint']} (N={int(r['n_adjusted'])})", r["hr"], r["ci_lower"], r["ci_upper"], f"P={r['p_value']:.4f}", C_NAVY, False))
    for _, r in mdf[mdf["analysis"] == "ICB_Primary_Pooled"].iterrows():
        rows.append((f"ICB Pooled DL - {r['endpoint']} (K=2, N={int(r['pooled_N'])})", r["hr"], r["dl_ci_lower"], r["dl_ci_upper"], f"DL P={r['dl_p']:.4f} | HKSJ P={r['hksj_p']:.4f}", C_RED, True))
    for _, r in cdf[cdf["cohort"] == "TCGA-SKCM (Non-ICB)"].iterrows():
        rows.append((f"{r['cohort']} - {r['endpoint']} (N={int(r['n_adjusted'])})", r["hr"], r["ci_lower"], r["ci_upper"], f"P={r['p_value']:.4f} (Separate)", C_AMBER, False))

    fig, ax = plt.subplots(figsize=(10.2, 4.8), dpi=300)
    yf = np.arange(len(rows))[::-1]
    for y_i, (lbl, hr, lo, hi, ptxt, col, is_pool) in zip(yf, rows):
        ax.plot([lo, hi], [y_i, y_i], color=col, lw=2.3 if is_pool else 1.7)
        ax.scatter([hr], [y_i], color=col, s=85 if is_pool else 52, marker="D" if is_pool else "s", zorder=4)
        ax.text(2.62, y_i, f"{hr:.3f} ({lo:.3f}-{hi:.3f})  {ptxt}", va="center", ha="left", fontsize=8.2, fontweight="bold" if is_pool else "normal")
    ax.axvline(1.0, ls="--", lw=1.0, color="#222222")
    ax.axhline(0.5, ls=":", lw=1.0, color=C_GRAY)
    ax.set_yticks(yf); ax.set_yticklabels([r[0] for r in rows], fontweight="bold", fontsize=8.5)
    ax.set_xlim(0.75, 4.55)
    ax.set_title("Multivariable Cox Survival & Pre-Specified ICB Meta-Analysis")
    fig.tight_layout()
    fig.savefig(fig_dir / "Figure5_ForestPlot_ICB_MetaAnalysis.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "Figure5_ForestPlot_ICB_MetaAnalysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate real-data-only publication figures.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_pca_stability(args.out_dir, fig_dir)
    plot_tme_part_whole(args.out_dir, fig_dir)
    plot_forest_meta(args.out_dir, fig_dir)
