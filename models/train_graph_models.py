#!/usr/bin/env python3
"""
Step 4-C: Graph Model Training (2 models) with Drug-Split 5-fold CV
Models: GraphSAGE, GAT
Graph: Bipartite (cell_line ↔ drug) with node features
Drug-split: folds split by unique drugs (generalization to unseen drugs)
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import json
import time
import os
import sys
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from scipy.stats import spearmanr, pearsonr

S3_BASE = os.getenv("S3_BASE", "s3://say2-4team/20260409_eseo")
FE_RUN_ID = os.getenv("FE_RUN_ID", "20260409_newfe_v8_eseo")
FEATURES_URI = f"{S3_BASE}/fe_output/{FE_RUN_ID}/features/features.parquet"
PAIR_FEATURES_URI = f"{S3_BASE}/fe_output/{FE_RUN_ID}/pair_features/pair_features_newfe_v2.parquet"
LABELS_URI = f"{S3_BASE}/fe_output/{FE_RUN_ID}/features/labels.parquet"
SEED = 42
N_FOLDS = 5
BENCH_SP = 0.713
BENCH_RMSE = 1.385
BENCH_P20 = 0.70
OUTPUT_DIR = Path(__file__).parent / "graph_results"
OUTPUT_DIR.mkdir(exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

# Device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")


def load_data():
    """Load and prepare data, returning features + identifiers."""
    print("Loading data from S3...")
    t0 = time.time()
    features = pd.read_parquet(FEATURES_URI)
    pair_features = pd.read_parquet(PAIR_FEATURES_URI)
    labels = pd.read_parquet(LABELS_URI)

    merged = features.merge(pair_features, on=["sample_id", "canonical_drug_id"], how="inner")
    labels = labels.set_index(["sample_id", "canonical_drug_id"])
    merged = merged.set_index(["sample_id", "canonical_drug_id"])
    labels = labels.loc[merged.index]

    sample_ids = merged.index.get_level_values("sample_id").values
    drug_ids = merged.index.get_level_values("canonical_drug_id").values
    X = merged.select_dtypes(include=[np.number]).fillna(0.0).values.astype(np.float32)
    y = labels["label_regression"].values.astype(np.float32)
    y_bin = labels["label_binary"].values.astype(np.float32) if "label_binary" in labels.columns else (y < np.median(y)).astype(np.float32)

    print(f"  Loaded: {X.shape[0]} x {X.shape[1]} features ({time.time()-t0:.1f}s)")
    print(f"  Unique cells: {len(np.unique(sample_ids))}, Unique drugs: {len(np.unique(drug_ids))}")
    return X, y, y_bin, sample_ids, drug_ids


def build_graph(X, sample_ids, drug_ids, scaler=None):
    """
    Build a bipartite graph: cell_line nodes + drug nodes + edges.
    Node features from X (split into cell and drug parts).
    """
    unique_cells = sorted(set(sample_ids))
    unique_drugs = sorted(set(drug_ids))
    cell_to_idx = {c: i for i, c in enumerate(unique_cells)}
    drug_to_idx = {d: i + len(unique_cells) for i, d in enumerate(unique_drugs)}

    n_cells = len(unique_cells)
    n_drugs = len(unique_drugs)
    n_nodes = n_cells + n_drugs

    # Feature dimensions: first 18311 are cell features, rest are drug+pair features
    cell_feat_dim = 18311
    drug_feat_dim = X.shape[1] - cell_feat_dim

    # Aggregate cell features (mean across all pairs for each cell)
    cell_feats = np.zeros((n_cells, cell_feat_dim), dtype=np.float32)
    cell_counts = np.zeros(n_cells, dtype=np.int32)
    drug_feats = np.zeros((n_drugs, drug_feat_dim), dtype=np.float32)
    drug_counts = np.zeros(n_drugs, dtype=np.int32)

    for i in range(len(sample_ids)):
        cidx = cell_to_idx[sample_ids[i]]
        didx = drug_to_idx[drug_ids[i]] - n_cells
        cell_feats[cidx] += X[i, :cell_feat_dim]
        cell_counts[cidx] += 1
        drug_feats[didx] += X[i, cell_feat_dim:]
        drug_counts[didx] += 1

    cell_counts[cell_counts == 0] = 1
    drug_counts[drug_counts == 0] = 1
    cell_feats /= cell_counts[:, None]
    drug_feats /= drug_counts[:, None]

    # Reduce dimensionality via PCA-like approach (top variance features)
    # For cell features: keep top 512 by variance
    cell_var = np.var(cell_feats, axis=0)
    top_cell_idx = np.argsort(cell_var)[-512:]
    cell_feats = cell_feats[:, top_cell_idx]

    # For drug features: keep top 128 by variance
    drug_var = np.var(drug_feats, axis=0)
    top_drug_idx = np.argsort(drug_var)[-min(128, drug_feat_dim):]
    drug_feats = drug_feats[:, top_drug_idx]

    # Pad to same dimension
    feat_dim = max(cell_feats.shape[1], drug_feats.shape[1])
    if cell_feats.shape[1] < feat_dim:
        cell_feats = np.pad(cell_feats, ((0, 0), (0, feat_dim - cell_feats.shape[1])))
    if drug_feats.shape[1] < feat_dim:
        drug_feats = np.pad(drug_feats, ((0, 0), (0, feat_dim - drug_feats.shape[1])))

    # Combine node features
    node_feats = np.vstack([cell_feats, drug_feats])

    # Standardize
    if scaler is None:
        scaler = StandardScaler()
        node_feats = scaler.fit_transform(node_feats).astype(np.float32)
    else:
        node_feats = scaler.transform(node_feats).astype(np.float32)

    # Build edges (cell_line ↔ drug)
    src, dst = [], []
    for i in range(len(sample_ids)):
        cidx = cell_to_idx[sample_ids[i]]
        didx = drug_to_idx[drug_ids[i]]
        src.append(cidx)
        dst.append(didx)
        src.append(didx)
        dst.append(cidx)  # bidirectional

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x = torch.tensor(node_feats, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index), cell_to_idx, drug_to_idx, scaler


def drug_split_cv(drug_ids, n_folds=5, seed=42):
    """Split by unique drugs for CV."""
    unique_drugs = sorted(set(drug_ids))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_drugs)

    fold_size = len(unique_drugs) // n_folds
    folds = []
    for i in range(n_folds):
        if i < n_folds - 1:
            val_drugs = set(unique_drugs[i * fold_size:(i + 1) * fold_size])
        else:
            val_drugs = set(unique_drugs[i * fold_size:])

        val_mask = np.array([d in val_drugs for d in drug_ids])
        train_mask = ~val_mask
        folds.append((np.where(train_mask)[0], np.where(val_mask)[0]))
    return folds


def precision_at_k(y_true, y_pred, k=20, threshold=None):
    """P@K: fraction of top-K predictions that are truly sensitive."""
    if threshold is None:
        threshold = np.median(y_true)
    # Lower IC50 = more sensitive, so we want smallest predicted values
    top_k_idx = np.argsort(y_pred)[:k]
    true_sensitive = (y_true[top_k_idx] < threshold).sum()
    return true_sensitive / k


# ── Model Definitions ──

class GraphSAGEModel(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=128, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return h


class GATModel(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=128, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden // heads, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden, out_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return h


class EdgePredictor(nn.Module):
    """Predicts edge (pair) values from node embeddings."""
    def __init__(self, gnn, emb_dim=128, hidden=64):
        super().__init__()
        self.gnn = gnn
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index, cell_indices, drug_indices):
        node_emb = self.gnn(x, edge_index)
        cell_emb = node_emb[cell_indices]
        drug_emb = node_emb[drug_indices]
        pair_emb = torch.cat([cell_emb, drug_emb], dim=1)
        return self.mlp(pair_emb).squeeze(-1)


def train_graph_model(model, graph_data, cell_indices_tr, drug_indices_tr, y_tr,
                      cell_indices_val, drug_indices_val, y_val,
                      epochs=150, lr=1e-3, patience=20):
    """Train graph model on edge prediction task."""
    if DEVICE.type == "mps":
        torch.mps.empty_cache()

    graph_data = graph_data.to(DEVICE)
    model = model.to(DEVICE)

    cell_idx_tr = torch.tensor(cell_indices_tr, dtype=torch.long, device=DEVICE)
    drug_idx_tr = torch.tensor(drug_indices_tr, dtype=torch.long, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)

    cell_idx_val = torch.tensor(cell_indices_val, dtype=torch.long, device=DEVICE)
    drug_idx_val = torch.tensor(drug_indices_val, dtype=torch.long, device=DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(graph_data.x, graph_data.edge_index, cell_idx_tr, drug_idx_tr)
        loss = criterion(pred, y_tr_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(graph_data.x, graph_data.edge_index, cell_idx_val, drug_idx_val).cpu().numpy()
            val_loss = mean_squared_error(y_val, val_pred)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred_val = model(graph_data.x, graph_data.edge_index, cell_idx_val, drug_idx_val).cpu().numpy()
        y_pred_tr = model(graph_data.x, graph_data.edge_index, cell_idx_tr, drug_idx_tr).cpu().numpy()
    return y_pred_val, y_pred_tr


def compute_metrics(y_true, y_pred, y_bin=None, y_tr_true=None, y_tr_pred=None):
    sp, _ = spearmanr(y_true, y_pred)
    pe, _ = pearsonr(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    p20 = precision_at_k(y_true, y_pred, k=20)

    m = {"spearman": sp, "pearson": pe, "rmse": rmse, "r2": r2, "p_at_20": p20}

    if y_bin is not None and len(np.unique(y_bin)) == 2:
        # Use negative predictions for AUROC (lower pred = more sensitive = positive)
        try:
            m["auroc"] = roc_auc_score(y_bin, -y_pred)
        except ValueError:
            m["auroc"] = float('nan')

    if y_tr_true is not None:
        tr_sp, _ = spearmanr(y_tr_true, y_tr_pred)
        m["train_spearman"] = tr_sp
        m["gap_spearman"] = tr_sp - sp
    return m


def run_graph_cv(name, gnn_cls, gnn_kwargs, X, y, y_bin, sample_ids, drug_ids):
    print(f"\n{'─'*60}")
    print(f"  [{name}] Training with 5-fold Drug-Split CV...")
    print(f"{'─'*60}")

    folds = drug_split_cv(drug_ids, N_FOLDS, SEED)
    fold_metrics = []
    total_t0 = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        t0 = time.time()
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

        # Build graph from ALL data (transductive setting)
        graph_data, cell_to_idx, drug_to_idx, scaler = build_graph(
            X, sample_ids, drug_ids
        )

        # Map pair indices to graph node indices
        cell_indices_tr = np.array([cell_to_idx[sample_ids[i]] for i in train_idx])
        drug_indices_tr = np.array([drug_to_idx[drug_ids[i]] for i in train_idx])
        cell_indices_val = np.array([cell_to_idx[sample_ids[i]] for i in val_idx])
        drug_indices_val = np.array([drug_to_idx[drug_ids[i]] for i in val_idx])

        y_tr = y[train_idx]
        y_val = y[val_idx]
        y_bin_val = y_bin[val_idx]

        torch.manual_seed(SEED + fold_idx)
        in_dim = graph_data.x.size(1)
        gnn = gnn_cls(in_dim=in_dim, **gnn_kwargs)
        emb_dim = gnn_kwargs.get("out_dim", 128)
        model = EdgePredictor(gnn, emb_dim=emb_dim)

        y_pred_val, y_pred_tr = train_graph_model(
            model, graph_data,
            cell_indices_tr, drug_indices_tr, y_tr,
            cell_indices_val, drug_indices_val, y_val,
            epochs=150, lr=1e-3, patience=20,
        )

        m = compute_metrics(y_val, y_pred_val, y_bin_val, y_tr, y_pred_tr)
        m["fold"] = fold_idx
        fold_metrics.append(m)
        dt = time.time() - t0

        auroc_str = f"  AUROC={m.get('auroc', float('nan')):.4f}" if 'auroc' in m else ""
        print(f"  Fold {fold_idx}: Sp={m['spearman']:.4f}  RMSE={m['rmse']:.4f}  "
              f"P@20={m['p_at_20']:.4f}  Gap(Sp)={m['gap_spearman']:.4f}"
              f"{auroc_str}  ({dt:.1f}s)")

    total_time = time.time() - total_t0
    df = pd.DataFrame(fold_metrics)
    summary = {
        "model": name,
        "spearman_mean": float(df["spearman"].mean()),
        "spearman_std": float(df["spearman"].std()),
        "rmse_mean": float(df["rmse"].mean()),
        "rmse_std": float(df["rmse"].std()),
        "pearson_mean": float(df["pearson"].mean()),
        "r2_mean": float(df["r2"].mean()),
        "r2_std": float(df["r2"].std()),
        "p_at_20_mean": float(df["p_at_20"].mean()),
        "p_at_20_std": float(df["p_at_20"].std()),
        "auroc_mean": float(df["auroc"].mean()) if "auroc" in df.columns else None,
        "train_spearman_mean": float(df["train_spearman"].mean()),
        "gap_spearman_mean": float(df["gap_spearman"].mean()),
        "elapsed_sec": total_time,
        "folds": fold_metrics,
    }

    sp_flag = "PASS" if summary["spearman_mean"] >= BENCH_SP else "FAIL"
    rm_flag = "PASS" if summary["rmse_mean"] <= BENCH_RMSE else "FAIL"
    p20_flag = "PASS" if summary["p_at_20_mean"] >= BENCH_P20 else "FAIL"

    print(f"\n  >>> {name} SUMMARY:")
    print(f"      Spearman: {summary['spearman_mean']:.4f} +/- {summary['spearman_std']:.4f}  [{sp_flag} vs {BENCH_SP}]")
    print(f"      RMSE:     {summary['rmse_mean']:.4f} +/- {summary['rmse_std']:.4f}  [{rm_flag} vs {BENCH_RMSE}]")
    print(f"      P@20:     {summary['p_at_20_mean']:.4f} +/- {summary['p_at_20_std']:.4f}  [{p20_flag} vs {BENCH_P20}]")
    if summary["auroc_mean"] is not None:
        print(f"      AUROC:    {summary['auroc_mean']:.4f}")
    print(f"      Pearson:  {summary['pearson_mean']:.4f}")
    print(f"      R2:       {summary['r2_mean']:.4f} +/- {summary['r2_std']:.4f}")
    print(f"      Train Sp: {summary['train_spearman_mean']:.4f}  Gap: {summary['gap_spearman_mean']:.4f}")
    print(f"      Time:     {total_time/60:.1f} min")

    return summary


def main():
    run_only = sys.argv[1] if len(sys.argv) > 1 else "all"

    X, y, y_bin, sample_ids, drug_ids = load_data()

    models_config = [
        ("14_GraphSAGE", GraphSAGEModel, {"hidden": 256, "out_dim": 128, "dropout": 0.3}),
        ("15_GAT", GATModel, {"hidden": 256, "out_dim": 128, "heads": 4, "dropout": 0.3}),
    ]

    if run_only != "all":
        models_config = [m for m in models_config if m[0] == run_only]
        if not models_config:
            print(f"Model '{run_only}' not found!")
            return

    all_results = []
    for name, cls, kwargs in models_config:
        result = run_graph_cv(name, cls, kwargs, X, y, y_bin, sample_ids, drug_ids)
        all_results.append(result)

    # Summary
    if len(all_results) > 1:
        print("\n" + "=" * 100)
        print(f"  GRAPH MODELS SUMMARY TABLE (Drug-Split CV)")
        print("=" * 100)
        print(f"{'Model':<18} {'Val Sp':>8} {'  std':>6} {'Train Sp':>9} {'Gap(Sp)':>8} "
              f"{'RMSE':>8} {'  std':>6} {'P@20':>6} {'AUROC':>7} {'R2':>8} {'Time':>6}")
        print("-" * 100)
        for r in all_results:
            sp_f = " *" if r["spearman_mean"] >= BENCH_SP else ""
            rm_f = " *" if r["rmse_mean"] <= BENCH_RMSE else ""
            p20_f = " *" if r["p_at_20_mean"] >= BENCH_P20 else ""
            auroc_s = f"{r['auroc_mean']:.4f}" if r["auroc_mean"] is not None else "  N/A"
            print(f"{r['model']:<18} {r['spearman_mean']:>8.4f}{sp_f:>2} "
                  f"{r['spearman_std']:>6.4f} {r['train_spearman_mean']:>9.4f} "
                  f"{r['gap_spearman_mean']:>8.4f} {r['rmse_mean']:>8.4f}{rm_f:>2} "
                  f"{r['rmse_std']:>6.4f} {r['p_at_20_mean']:>5.4f}{p20_f:>1} "
                  f"{auroc_s:>7} {r['r2_mean']:>8.4f} "
                  f"{r['elapsed_sec']/60:>5.1f}m")
        print("-" * 100)
        print(f"  Benchmarks: Spearman >= {BENCH_SP}, RMSE <= {BENCH_RMSE}, P@20 >= {BENCH_P20}")

    # Save
    def convert(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    out_path = OUTPUT_DIR / "graph_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
