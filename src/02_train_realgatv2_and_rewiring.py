"""
Layer 1: Exploratory 3-Seed RealGATv2 LOPO Training & 10-Replicate Test-Time Topology Perturbation.

Architecture (locked):
- Layer 1: GATv2Conv(32 -> 32, heads=4, concat=True) + ELU + Dropout(0.3)
- Layer 2: GATv2Conv(128 -> 32, heads=1, concat=False) + ELU
- Readout: concat(global_mean_pool, global_max_pool) -> 64-d patient embedding
- Classifier: Linear(64 -> 32) + ReLU + Dropout(0.3) + Linear(32 -> 1) + Sigmoid
- Optimization: BCELoss, Adam(lr=1e-3, weight_decay=1e-4), batch_size=4, 80 epochs, seeds=[42, 142, 242].
- Test-time topology perturbation: 10 independent degree-preserving Maslov-Sneppen edge rewirings
  (10 * |E_k| accepted swaps per held-out test graph, max_tries = 100 * |E_k|).
"""

import argparse
import random
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool

from config import DEFAULT_OUT_DIR, GATV2_SEEDS, REWIRING_REPLICATES


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RealGATv2(nn.Module):
    def __init__(self, in_channels: int = 32, hidden_channels: int = 32, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x, edge_index, batch):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        h = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return torch.sigmoid(self.mlp(h)).squeeze(-1)


def load_graph_npz(path: Path) -> Data:
    d = np.load(path, allow_pickle=True)
    return Data(
        x=torch.tensor(d["x"], dtype=torch.float32),
        edge_index=torch.tensor(d["edge_index"], dtype=torch.long),
        y=torch.tensor([float(d["y"])], dtype=torch.float32),
        patient_id=str(d["patient_id"]),
    )


def maslov_sneppen_rewire(edge_index: torch.Tensor, n_nodes: int, seed: int):
    """Degree-preserving Maslov-Sneppen double-edge swap rewiring with clustering audit."""
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    edges = edge_index.cpu().numpy().T
    for u, v in edges:
        if u < v:
            G.add_edge(int(u), int(v))

    c_before = nx.average_clustering(G)
    deg_before = sorted(dict(G.degree()).values())
    n_edges = G.number_of_edges()

    G_rew = G.copy()
    nx.double_edge_swap(G_rew, nswap=10 * n_edges, max_tries=100 * n_edges, seed=seed)

    c_after = nx.average_clustering(G_rew)
    deg_after = sorted(dict(G_rew.degree()).values())
    deg_preserved = (deg_before == deg_after)

    src, dst = [], []
    for u, v in G_rew.edges():
        src.extend([u, v])
        dst.extend([v, u])
    new_edge_index = torch.tensor([src, dst], dtype=torch.long)
    return new_edge_index, c_before, c_after, deg_preserved


def run_training_and_rewiring(out_dir: Path, epochs: int = 80, batch_size: int = 4):
    graphs_root = out_dir / "lopo_patient_graphs"
    manifest_path = graphs_root / "lopo_graph_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run 01_sc_preprocessing_and_lopo_graphs.py first. Missing {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_dir = out_dir / "realgatv2_lopo"
    eval_dir.mkdir(parents=True, exist_ok=True)

    oof_rows = []
    rewire_rows = []
    topology_audit = []

    for fold_id, fdf in manifest.groupby("fold_id", sort=True):
        train_graphs = [load_graph_npz(Path(p)) for p in fdf.loc[~fdf["is_test"], "graph_path"]]
        test_row = fdf.loc[fdf["is_test"]].iloc[0]
        test_graph = load_graph_npz(Path(test_row["graph_path"]))
        test_pid = str(test_row["patient_id"])
        y_true = int(test_row["y"])

        for seed in GATV2_SEEDS:
            seed_everything(seed)
            model = RealGATv2().to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            loss_fn = nn.BCELoss()
            loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)

            model.train()
            for _ in range(epochs):
                for batch in loader:
                    batch = batch.to(device)
                    opt.zero_grad()
                    preds = model(batch.x, batch.edge_index, batch.batch)
                    loss = loss_fn(preds, batch.y.view(-1))
                    loss.backward()
                    opt.step()

            model.eval()
            with torch.no_grad():
                tg = test_graph.to(device)
                batch_vec = torch.zeros(tg.x.size(0), dtype=torch.long, device=device)
                prob = float(model(tg.x, tg.edge_index, batch_vec).item())

            oof_rows.append({"fold_id": fold_id, "patient_id": test_pid, "seed": seed, "y_true": y_true, "prob": prob})

            if seed == 42:
                for rep in range(1, REWIRING_REPLICATES + 1):
                    rew_edges, cb, ca, deg_ok = maslov_sneppen_rewire(
                        test_graph.edge_index, test_graph.x.size(0), seed=1000 * rep + sum(ord(c) for c in test_pid)
                    )
                    with torch.no_grad():
                        rew_prob = float(model(tg.x, rew_edges.to(device), batch_vec).item())
                    rewire_rows.append({
                        "replicate": rep, "patient_id": test_pid, "y_true": y_true, "rewired_prob": rew_prob
                    })
                    if rep == 1:
                        topology_audit.append({
                            "patient_id": test_pid, "y_true": y_true,
                            "clustering_intact": cb, "clustering_rewired": ca,
                            "pct_drop": (ca - cb) / cb * 100.0, "degree_preserved": deg_ok,
                        })

    oof_df = pd.DataFrame(oof_rows)
    oof_df.to_csv(eval_dir / "realgatv2_3seed_oof_predictions.csv", index=False)

    summary = []
    for seed, sdf in oof_df.groupby("seed"):
        summary.append({
            "setting": f"RealGATv2_seed_{seed}",
            "auroc": roc_auc_score(sdf["y_true"], sdf["prob"]),
            "pr_auc": average_precision_score(sdf["y_true"], sdf["prob"]),
            "brier": brier_score_loss(sdf["y_true"], sdf["prob"]),
            "bal_acc": balanced_accuracy_score(sdf["y_true"], (sdf["prob"] >= 0.5).astype(int)),
        })

    ens = oof_df.groupby(["patient_id", "y_true"])["prob"].agg(["mean", "std"]).reset_index()
    ens.to_csv(eval_dir / "realgatv2_ensemble_oof.csv", index=False)
    summary.append({
        "setting": "RealGATv2_3seed_ensemble",
        "auroc": roc_auc_score(ens["y_true"], ens["mean"]),
        "pr_auc": average_precision_score(ens["y_true"], ens["mean"]),
        "brier": brier_score_loss(ens["y_true"], ens["mean"]),
        "bal_acc": balanced_accuracy_score(ens["y_true"], (ens["mean"] >= 0.5).astype(int)),
    })

    rew_df = pd.DataFrame(rewire_rows)
    rew_aucs = [roc_auc_score(rdf["y_true"], rdf["rewired_prob"]) for _, rdf in rew_df.groupby("replicate")]
    pd.DataFrame({"replicate": range(1, len(rew_aucs) + 1), "rewired_auroc": rew_aucs}).to_csv(
        eval_dir / "rewiring_10rep_aurocs.csv", index=False
    )
    pd.DataFrame(topology_audit).to_csv(eval_dir / "rewiring_topology_audit.csv", index=False)
    pd.DataFrame(summary).to_csv(eval_dir / "realgatv2_summary_metrics.csv", index=False)

    print(
        f"[Layer 1] Ensemble OOF AUROC = {summary[-1]['auroc']:.4f} | "
        f"10x Rewired AUROC = {np.mean(rew_aucs):.4f} +/- {np.std(rew_aucs, ddof=1):.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3-seed RealGATv2 LOPO and run test-time rewiring.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    run_training_and_rewiring(args.out_dir)
