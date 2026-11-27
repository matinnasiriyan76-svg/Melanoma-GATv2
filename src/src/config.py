"""Centralized configuration, locked 10-gene signature, and 10 curated TME pathways."""

import os
from pathlib import Path

DEFAULT_DATA_DIR = Path(os.environ.get("MELANOMA_DATA_DIR", "./data"))
DEFAULT_OUT_DIR = Path(os.environ.get("MELANOMA_OUT_DIR", "./results"))

# Graph construction (verified against cell47_state_snapshot.json)
KNN_K = 15
N_HVGS = 2000
N_PCS = 32

# Locked 10-gene immunometabolic signature
RISK_GENES = ["GAPDH", "PGAM1", "FKBP1A"]
PROTECTIVE_GENES = ["IFI30", "IRF8", "NR4A1", "CD8A", "CCR7", "IL7R", "S1PR1"]
ALL_SIGNATURE_GENES = RISK_GENES + PROTECTIVE_GENES

# Exploratory 8-gene spatial subset (GAPDH & FKBP1A probes absent in Visium FFPE)
SPATIAL_RISK_GENES = ["PGAM1"]
SPATIAL_PROTECTIVE_GENES = PROTECTIVE_GENES

# Fixed seeds & iteration counts
GATV2_SEEDS = [42, 142, 242]
REWIRING_REPLICATES = 10
PERMUTATION_ITERATIONS = 1000
BOOTSTRAP_RESAMPLES = 2000

# 10 curated TME functional pathways (matches Figure 4 & Supplementary Table S6)
CURATED_TME_PATHWAYS = {
    "Glycolysis": ["GAPDH", "PGAM1", "LDHA", "PKM", "ENO1", "ALDOA", "TPI1", "PGK1", "HK2", "SLC2A1"],
    "Hypoxia": ["VEGFA", "LDHA", "SLC2A1", "BNIP3", "CA9", "ADM", "PDK1", "NDRG1"],
    "OXPHOS": ["ATP5F1A", "COX5A", "NDUFA9", "UQCRC2", "SDHB", "CYCS"],
    "Cell_Cycle": ["MKI67", "TOP2A", "CDK1", "CCNB1", "BIRC5", "PCNA", "MCM2"],
    "Curated_IFNG": ["IFNG", "STAT1", "CXCL9", "CXCL10", "IDO1", "HLA-DRA", "GBP1", "IRF1"],
    "Antigen_Presentation": ["HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2", "HLA-DRA", "HLA-DPA1"],
    "T_cell_Stemness_Memory": ["CCR7", "IL7R", "S1PR1", "TCF7", "LEF1", "SELL", "LTB"],
    "CD8_Cytotoxic": ["CD8A", "CD8B", "GZMA", "GZMB", "PRF1", "NKG7", "GNLY"],
    "T_cell_Exhaustion": ["PDCD1", "LAG3", "TIGIT", "HAVCR2", "CTLA4", "TOX", "ENTPD1"],
    "TLS_B_cell": ["MS4A1", "CD79A", "CD79B", "CD19", "BANK1", "BLK", "CXCL13"],
}

# Clinical ICB benchmark signatures
AYERS_IFNG_6 = ["IDO1", "CXCL10", "CXCL9", "HLA-DRA", "STAT1", "IFNG"]
CYT_GENES = ["GZMA", "PRF1"]
PDL1_GENE = ["CD274"]


def get_non_overlapping_pathways(signature_genes=None):
    """Return the 10 curated TME pathways with all signature-overlapping genes removed."""
    sig_set = set(signature_genes or ALL_SIGNATURE_GENES)
    return {
        name: [g for g in genes if g not in sig_set]
        for name, genes in CURATED_TME_PATHWAYS.items()
    }
