"""
Layer 1: Exploratory 3-Seed RealGATv2 LOPO, Baseline Comparisons (GCN, CellMLP_no_edges,
Logistic Regression with inner 3-fold CV), and 10-Replicate Test-Time Maslov-Sneppen Rewiring.

All clustering-coefficient and degree-preservation metrics are audited across all 10 rewiring
replicates and all 12 patient graphs.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, GCNConv, global_max_pool, global_mean_pool

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
            nn.Linear(hidden_channels * 2, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x, edge_index, batch):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        h = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return torch.sigmoid(self.mlp(h)).squeeze(-1)


class BaselineGCN(nn.Module):
    def __init__(self, in_channels: int = 32, hidden_channels: int = 32, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x, edge_index, batch):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        h = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return torch.sigmoid(self.mlp(h)).squeeze(-1)


class CellMLPNoEdges(nn.Module):
    """Graph-free DeepSets-style baseline ignoring all cell-cell edges."""
    def __init__(self, in_channels: int = 32, hidden_channels: int = 32, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.enc1 = nn.Linear(in_channels, hidden_channels)
        self.enc2 = nn.Linear(hidden_channels, hidden_channels)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x, edge_index, batch):
        x = F.elu(self.enc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.enc2(x))
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
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for u, v in edge_index.cpu().numpy().T:
        if u < v:
            G.add_edge(int(u), int(v))

    c_before = nx.average_clustering(G)
    deg_before = sorted(dict(G.degree()).values())
    n_edges = G.number_of_edges()

    G_rew = G.copy()
    nx.double_edge_swap(G_rew, nswap=10 * n_edges, max_tries=100 * n_edges, seed=seed)

    c_after = nx.average_clustering(G_rew)
    deg_preserved = (deg_before == sorted(dict(G_rew.degree()).values()))

    src, dst = [], []
    for u, v in G_rew.edges():
        src.extend([u, v])
        dst.extend([v, u])
    return torch.tensor([src, dst], dtype=torch.long), c_before, c_after, deg_preserved


def train_torch_model(model_cls, train_graphs, test_graph, seed, device, epochs=80, batch_size=4):
    seed_everything(seed)
    model = model_cls().to(device)
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
        bvec = torch.zeros(tg.x.size(0), dtype=torch.long, device=device)
        prob = float(model(tg.x, tg.edge_index, bvec).item())
    return model, prob


def eval_metrics(y_true, probs, name):
    return {
        "model": name,
        "auroc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "brier": float(brier_score_loss(y_true, probs)),
        "bal_acc": float(balanced_accuracy_score(y_true, (np.asarray(probs) >= 0.5).astype(int))),
    }


def run_training_and_rewiring(out_dir: Path, epochs: int = 80, batch_size: int = 4):
    manifest_path = out_dir / "lopo_patient_graphs" / "lopo_graph_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run 01_sc_preprocessing_and_lopo_graphs.py first. Missing {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_dir = out_dir / "realgatv2_lopo"
    eval_dir.mkdir(parents=True, exist_ok=True)

    oof_rows, base_rows, rewire_rows, topo_rows = [], [], [], []

    for fold_id, fdf in manifest.groupby("fold_id", sort=True):
        train_graphs = [load_graph_npz(Path(p)) for p in fdf.loc[~fdf["is_test"], "graph_path"]]
        test_row = fdf.loc[fdf["is_test"]].iloc[0]
        test_graph = load_graph_npz(Path(test_row["graph_path"]))
        test_pid = str(test_row["patient_id"])
        y_true = int(test_row["y"])

        # 1. Exploratory 3-seed RealGATv2 + 10-replicate rewiring on seed 42
        for seed in GATV2_SEEDS:
            model, prob = train_torch_model(RealGATv2, train_graphs, test_graph, seed, device, epochs, batch_size)
            oof_rows.append({"fold_id": fold_id, "patient_id": test_pid, "seed": seed, "y_true": y_true, "prob": prob})

            if seed == 42:
                tg = test_graph.to(device)
                bvec = torch.zeros(tg.x.size(0), dtype=torch.long, device=device)
                for rep in range(1, REWIRING_REPLICATES + 1):
                    rew_edges, cb, ca, deg_ok = maslov_sneppen_rewire(
                        test_graph.edge_index, test_graph.x.size(0), seed=1000 * rep + sum(ord(c) for c in test_pid)
                    )
                    with torch.no_grad():
                        rew_prob = float(model(tg.x, rew_edges.to(device), bvec).item())
                    rewire_rows.append({"replicate": rep, "patient_id": test_pid, "y_true": y_true, "rewired_prob": rew_prob})
                    topo_rows.append({
                        "replicate": rep, "patient_id": test_pid, "y_true": y_true,
                        "clustering_intact": cb, "clustering_rewired": ca,
                        "pct_drop": (ca - cb) / cb * 100.0, "degree_preserved": deg_ok,
                    })

        # 2. Baseline GCN & CellMLP_no_edges (seed 47)
        _, p_gcn = train_torch_model(BaselineGCN, train_graphs, test_graph, 47, device, epochs, batch_size)
        _, p_mlp = train_torch_model(CellMLPNoEdges, train_graphs, test_graph, 47, device, epochs, batch_size)

        # 3. Baseline Logistic Regression (patient mean PCA + inner 3-fold CV tuning)
        X_tr_lr = np.vstack([g.x.numpy().mean(axis=0) for g in train_graphs])
        y_tr_lr = np.array([int(g.y.item()) for g in train_graphs])
        X_te_lr = test_graph.x.numpy().mean(axis=0, keepdims=True)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=47)
        lr_gs = GridSearchCV(
            LogisticRegression(max_iter=1000, solver="liblinear"),
            param_grid={"C": [0.01, 0.1, 1.0, 10.0]},
            cv=cv, scoring="roc_auc",
        )
        lr_gs.fit(X_tr_lr, y_tr_lr)
        p_lr = float(lr_gs.predict_proba(X_te_lr)[0, 1])

        base_rows.append({
            "fold_id": fold_id, "patient_id": test_pid, "y_true": y_true,
            "GCN_prob": p_gcn, "CellMLP_prob": p_mlp, "LogReg_prob": p_lr,
        })

    oof_df = pd.DataFrame(oof_rows)
    base_df = pd.DataFrame(base_rows)
    oof_df.to_csv(eval_dir / "realgatv2_3seed_oof_predictions.csv", index=False)
    base_df.to_csv(eval_dir / "baseline_models_oof_predictions.csv", index=False)

    summary = [eval_metrics(sdf["y_true"], sdf["prob"], f"RealGATv2_seed_{s}") for s, sdf in oof_df.groupby("seed")]
    ens = oof_df.groupby(["patient_id", "y_true"])["prob"].agg(["mean", "std"]).reset_index()
    ens.to_csv(eval_dir / "realgatv2_ensemble_oof.csv", index=False)
    summary.append(eval_metrics(ens["y_true"], ens["mean"], "RealGATv2_3seed_ensemble"))
    summary.append(eval_metrics(base_df["y_true"], base_df["GCN_prob"], "Baseline_GCN"))
    summary.append(eval_metrics(base_df["y_true"], base_df["CellMLP_prob"], "Baseline_CellMLP_no_edges"))
    summary.append(eval_metrics(base_df["y_true"], base_df["LogReg_prob"], "Baseline_LogisticRegression_3foldCV"))

    rew_df = pd.DataFrame(rewire_rows)
    rew_aucs = [roc_auc_score(rdf["y_true"], rdf["rewired_prob"]) for _, rdf in rew_df.groupby("replicate")]
    pd.DataFrame({"replicate": range(1, len(rew_aucs) + 1), "rewired_auroc": rew_aucs}).to_csv(
        eval_dir / "rewiring_10rep_aurocs.csv", index=False
    )
    pd.DataFrame(topo_rows).to_csv(eval_dir / "rewiring_10rep_topology_audit.csv", index=False)
    pd.DataFrame(summary).to_csv(eval_dir / "model_benchmark_summary.csv", index=False)
    print(f"[Layer 1] Benchmark & 10-replicate rewiring audit saved to {eval_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RealGATv2, baselines, and 10x rewiring audit.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    run_training_and_rewiring(args.out_dir)
