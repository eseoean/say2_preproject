#!/usr/bin/env python3
"""
A/B test: exact slim + strong context vs exact slim + strong context + SMILES.

Starts from the current best exact-slim setup and adds a lightweight
character-level SMILES encoder branch to see whether structural string
information improves drug GroupCV.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_PATH = Path(__file__).resolve()
WORK_ROOT = SCRIPT_PATH.parents[1]
DATA_ROOT = WORK_ROOT / "v3_input_reproduction" / "exact_repo_match"
RESULT_ROOT = WORK_ROOT / "results"
PROGRESSIVE_RUNNER_PATH = SCRIPT_PATH.parent / "run_groupcv_dl_progressive.py"


def load_progressive_module():
    spec = importlib.util.spec_from_file_location("progressive_runner", PROGRESSIVE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load progressive runner from {PROGRESSIVE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encode_context(df: pd.DataFrame, cols: list[str], vocab_map: dict[str, dict[str, int]]) -> np.ndarray:
    if not cols:
        return np.zeros((len(df), 0), dtype=np.int64)
    out = np.zeros((len(df), len(cols)), dtype=np.int64)
    for i, col in enumerate(cols):
        series = df[col].astype(str).fillna("__MISSING__")
        mapping = vocab_map[col]
        default_id = mapping.get("__MISSING__", 0)
        out[:, i] = [mapping.get(val, default_id) for val in series]
    return out


def build_smiles_tensor(smiles_series: pd.Series, max_len: int = 256) -> tuple[np.ndarray, dict[str, int], int]:
    smiles = smiles_series.astype(str).fillna("").tolist()
    charset = sorted({ch for s in smiles for ch in s})
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for ch in charset:
        if ch not in vocab:
            vocab[ch] = len(vocab)
    token_ids = np.zeros((len(smiles), max_len), dtype=np.int64)
    for i, s in enumerate(smiles):
        for j, ch in enumerate(s[:max_len]):
            token_ids[i, j] = vocab.get(ch, 1)
    observed_max = max((len(s) for s in smiles), default=0)
    return token_ids, vocab, observed_max


class CatTokenEncoder(nn.Module):
    def __init__(self, vocab_sizes: list[int], d_token: int):
        super().__init__()
        self.d_token = d_token
        self.embeddings = nn.ModuleList([nn.Embedding(vs, d_token) for vs in vocab_sizes])

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        if len(self.embeddings) == 0:
            return torch.zeros((x_cat.size(0), 0, self.d_token), device=x_cat.device)
        toks = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.stack(toks, dim=1)


class SmilesCNNEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 32, out_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.conv3 = nn.Conv1d(emb_dim, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(emb_dim, 64, kernel_size=5, padding=2)
        self.proj = nn.Sequential(
            nn.Linear(128, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x_smiles: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x_smiles).transpose(1, 2)
        h3 = torch.amax(torch.relu(self.conv3(emb)), dim=2)
        h5 = torch.amax(torch.relu(self.conv5(emb)), dim=2)
        return self.proj(torch.cat([h3, h5], dim=1))


class FlatMLPSmilesModel(nn.Module):
    def __init__(self, num_dim: int, vocab_sizes: list[int], smiles_vocab_size: int, dropout: float = 0.3):
        super().__init__()
        self.cat_encoder = CatTokenEncoder(vocab_sizes, d_token=16)
        self.smiles_encoder = SmilesCNNEncoder(smiles_vocab_size, out_dim=64)
        in_dim = num_dim + len(vocab_sizes) * 16 + 64
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, x_smiles: torch.Tensor) -> torch.Tensor:
        cat_flat = self.cat_encoder(x_cat).reshape(x_num.size(0), -1)
        smiles = self.smiles_encoder(x_smiles)
        return self.net(torch.cat([x_num, cat_flat, smiles], dim=1)).squeeze(-1)


class WideDeepSmilesModel(nn.Module):
    def __init__(self, num_dim: int, vocab_sizes: list[int], smiles_vocab_size: int, hidden: int = 256):
        super().__init__()
        self.cat_encoder = CatTokenEncoder(vocab_sizes, d_token=16)
        self.smiles_encoder = SmilesCNNEncoder(smiles_vocab_size, out_dim=64)
        deep_dim = num_dim + len(vocab_sizes) * 16 + 64
        self.wide = nn.Linear(num_dim, 1)
        self.deep = nn.Sequential(
            nn.Linear(deep_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, x_smiles: torch.Tensor) -> torch.Tensor:
        cat_flat = self.cat_encoder(x_cat).reshape(x_num.size(0), -1)
        smiles = self.smiles_encoder(x_smiles)
        deep_x = torch.cat([x_num, cat_flat, smiles], dim=1)
        return (self.wide(x_num) + self.deep(deep_x)).squeeze(-1)


class CrossAttentionSmilesModel(nn.Module):
    def __init__(self, num_dim: int, sample_dim: int, vocab_sizes: list[int], smiles_vocab_size: int, d_model: int = 128):
        super().__init__()
        self.sample_dim = max(1, min(sample_dim, num_dim - 1))
        self.cat_encoder = CatTokenEncoder(vocab_sizes, d_token=16)
        self.smiles_encoder = SmilesCNNEncoder(smiles_vocab_size, out_dim=64)
        context_dim = (num_dim - self.sample_dim) + len(vocab_sizes) * 16 + 64
        self.sample_proj = nn.Sequential(nn.Linear(self.sample_dim, d_model), nn.GELU())
        self.context_proj = nn.Sequential(nn.Linear(context_dim, d_model), nn.GELU())
        self.cross_attn = nn.MultiheadAttention(d_model, 4, dropout=0.2, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, x_smiles: torch.Tensor) -> torch.Tensor:
        sample_x = x_num[:, : self.sample_dim]
        context_x = x_num[:, self.sample_dim :]
        cat_flat = self.cat_encoder(x_cat).reshape(x_num.size(0), -1)
        smiles = self.smiles_encoder(x_smiles)
        context = torch.cat([context_x, cat_flat, smiles], dim=1)
        sample_tok = self.sample_proj(sample_x).unsqueeze(1)
        context_tok = self.context_proj(context).unsqueeze(1)
        attn_out, _ = self.cross_attn(sample_tok, context_tok, context_tok)
        combined = torch.cat([attn_out.squeeze(1), sample_tok.squeeze(1)], dim=1)
        return self.ffn(combined).squeeze(-1)


def build_smiles_model(name: str, num_dim: int, sample_dim: int, vocab_sizes: list[int], smiles_vocab_size: int):
    key = name.lower().replace("-", "").replace("_", "")
    if key == "flatmlp":
        return FlatMLPSmilesModel(num_dim, vocab_sizes, smiles_vocab_size), {"epochs": 100, "lr": 1e-3, "batch_size": 256}
    if key == "widedeep":
        return WideDeepSmilesModel(num_dim, vocab_sizes, smiles_vocab_size), {"epochs": 100, "lr": 1e-3, "batch_size": 256}
    if key == "crossattention":
        return CrossAttentionSmilesModel(num_dim, sample_dim, vocab_sizes, smiles_vocab_size), {"epochs": 80, "lr": 5e-4, "batch_size": 256}
    raise ValueError(f"Unsupported SMILES model: {name}")


def train_smiles_model(
    mod,
    model: nn.Module,
    cfg: dict[str, float],
    x_num_tr: np.ndarray,
    x_cat_tr: np.ndarray,
    x_smiles_tr: np.ndarray,
    y_tr: np.ndarray,
    x_num_val: np.ndarray,
    x_cat_val: np.ndarray,
    x_smiles_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    device = mod.DEVICE
    model = model.to(device)
    train_ds = TensorDataset(
        torch.tensor(x_num_tr, dtype=torch.float32, device=device),
        torch.tensor(x_cat_tr, dtype=torch.long, device=device),
        torch.tensor(x_smiles_tr, dtype=torch.long, device=device),
        torch.tensor(y_tr, dtype=torch.float32, device=device),
    )
    drop_last = len(train_ds) >= int(cfg["batch_size"])
    train_dl = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=drop_last)

    x_num_val_t = torch.tensor(x_num_val, dtype=torch.float32, device=device)
    x_cat_val_t = torch.tensor(x_cat_val, dtype=torch.long, device=device)
    x_smiles_val_t = torch.tensor(x_smiles_val, dtype=torch.long, device=device)
    x_num_tr_t = torch.tensor(x_num_tr, dtype=torch.float32, device=device)
    x_cat_tr_t = torch.tensor(x_cat_tr, dtype=torch.long, device=device)
    x_smiles_tr_t = torch.tensor(x_smiles_tr, dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(cfg["epochs"]), 1))
    criterion = nn.MSELoss()
    patience = 15
    wait = 0
    best_val = float("inf")
    best_state = None

    for _ in range(int(cfg["epochs"])):
        model.train()
        for xb_num, xb_cat, xb_smiles, yb in train_dl:
            optimizer.zero_grad()
            pred = model(xb_num, xb_cat, xb_smiles)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_num_val_t, x_cat_val_t, x_smiles_val_t).detach().cpu().numpy()
        val_loss = float(np.mean((val_pred - y_val) ** 2))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_val = model(x_num_val_t, x_cat_val_t, x_smiles_val_t).detach().cpu().numpy()
        pred_tr = model(x_num_tr_t, x_cat_tr_t, x_smiles_tr_t).detach().cpu().numpy()
    return pred_val, pred_tr


def summarize_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    df = pd.DataFrame(rows)
    summary = {}
    for col in [
        "spearman",
        "rmse",
        "mae",
        "pearson",
        "r2",
        "ndcg@20",
        "train_spearman",
        "train_rmse",
        "train_mae",
        "gap_spearman",
        "gap_rmse",
        "gap_mae",
    ]:
        if col in df.columns:
            summary[f"{col}_mean"] = float(df[col].mean())
            summary[f"{col}_std"] = float(df[col].std())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="FlatMLP,WideDeep,CrossAttention")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--output-stem", default="exact_repo_slim_smiles_ab_top3_v1")
    args = parser.parse_args()

    mod = load_progressive_module()

    features = pd.read_parquet(DATA_ROOT / "features_slim_exact_repo.parquet")
    X = np.load(DATA_ROOT / "X_train_exact_repo_numeric.npy").astype(np.float32)
    y = np.load(DATA_ROOT / "y_train_exact_repo.npy").astype(np.float32)
    smiles_col = (
        features["drug__canonical_smiles_raw"]
        .astype("string")
        .fillna(features["drug__smiles"].astype("string"))
        .fillna("")
    )
    smiles_ids, smiles_vocab, observed_max_len = build_smiles_tensor(smiles_col, max_len=256)

    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    sample_dim = sum(col.startswith("sample__crispr") for col in numeric_cols)
    keys = features[["sample_id", "canonical_drug_id"]].copy()
    keys["sample_id"] = keys["sample_id"].astype(str)
    keys["canonical_drug_id"] = keys["canonical_drug_id"].astype(str)
    groups = keys["canonical_drug_id"].values

    ctx_df, _, vocab_map, context_summary = mod.build_reconstructed_context(keys)
    strong_cols = [col for col in mod.STRONG_CONTEXT_COLS if col in ctx_df.columns]
    strong_cat = encode_context(ctx_df, strong_cols, vocab_map)
    empty_aux = np.zeros((len(features), 0), dtype=np.int64)
    vocab_sizes = [len(vocab_map[col]) for col in strong_cols]

    variants = {
        "strong_context_only": {"with_smiles": False},
        "strong_context_plus_smiles": {"with_smiles": True},
    }

    splits = list(GroupKFold(n_splits=args.folds).split(X, y, groups=groups))
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    summary = {
        "input_features_path": str(DATA_ROOT / "features_slim_exact_repo.parquet"),
        "input_X_path": str(DATA_ROOT / "X_train_exact_repo_numeric.npy"),
        "input_y_path": str(DATA_ROOT / "y_train_exact_repo.npy"),
        "device": str(mod.DEVICE),
        "folds": int(args.folds),
        "n_rows": int(len(y)),
        "x_shape": list(X.shape),
        "sample_dim_detected": int(sample_dim),
        "strong_context_columns": strong_cols,
        "smiles_vocab_size": int(len(smiles_vocab)),
        "smiles_max_len_cap": 256,
        "smiles_observed_max_len": int(observed_max_len),
        "models": [],
        "context_summary": {
            "match_rate": float(context_summary.get("match_rate", 0.0)),
            "column_non_missing_rate": {
                col: float(context_summary["column_non_missing_rate"].get(col, 0.0))
                for col in strong_cols
            },
        },
    }

    for model_name in model_names:
        print("=" * 72)
        print(f"[{model_name}] strong context vs strong context + SMILES")
        print("=" * 72)
        model_result = {"model": model_name, "variants": {}}
        base_sp = None
        base_rmse = None
        base_mae = None
        base_ndcg = None

        for variant_name, variant in variants.items():
            fold_rows = []
            t0 = time.time()
            for fold_idx, (tr_idx, val_idx) in enumerate(splits):
                torch.manual_seed(mod.SEED + fold_idx)
                np.random.seed(mod.SEED + fold_idx)
                if variant["with_smiles"]:
                    model, cfg = build_smiles_model(model_name, X.shape[1], sample_dim, vocab_sizes, len(smiles_vocab))
                    pred_val, pred_tr = train_smiles_model(
                        mod,
                        model,
                        cfg,
                        X[tr_idx],
                        strong_cat[tr_idx],
                        smiles_ids[tr_idx],
                        y[tr_idx],
                        X[val_idx],
                        strong_cat[val_idx],
                        smiles_ids[val_idx],
                        y[val_idx],
                    )
                else:
                    model, cfg = mod.build_model(model_name, X.shape[1], vocab_sizes, [], sample_dim)
                    pred_val, pred_tr = mod.train_model(
                        model=model,
                        x_num_tr=X[tr_idx],
                        x_cat_tr=strong_cat[tr_idx],
                        x_aux_tr=empty_aux[tr_idx],
                        y_tr=y[tr_idx],
                        x_num_val=X[val_idx],
                        x_cat_val=strong_cat[val_idx],
                        x_aux_val=empty_aux[val_idx],
                        y_val=y[val_idx],
                        epochs=int(cfg["epochs"]),
                        lr=float(cfg["lr"]),
                        batch_size=int(cfg["batch_size"]),
                    )

                row = mod.compute_metrics(y[val_idx], pred_val, y[tr_idx], pred_tr)
                row["fold"] = int(fold_idx)
                fold_rows.append(row)
                print(
                    f"  {variant_name} fold {fold_idx}: "
                    f"Sp={row['spearman']:.4f} RMSE={row['rmse']:.4f} "
                    f"MAE={row['mae']:.4f} NDCG@20={row['ndcg@20']:.4f}"
                )

            summary_rows = summarize_rows(fold_rows)
            summary_rows["elapsed_sec"] = float(time.time() - t0)
            summary_rows["fold_metrics"] = fold_rows
            model_result["variants"][variant_name] = summary_rows
            print(
                f"  >>> {variant_name}: "
                f"Sp={summary_rows['spearman_mean']:.4f} "
                f"RMSE={summary_rows['rmse_mean']:.4f} "
                f"MAE={summary_rows['mae_mean']:.4f} "
                f"NDCG@20={summary_rows['ndcg@20_mean']:.4f}"
            )

            if variant_name == "strong_context_only":
                base_sp = summary_rows["spearman_mean"]
                base_rmse = summary_rows["rmse_mean"]
                base_mae = summary_rows["mae_mean"]
                base_ndcg = summary_rows["ndcg@20_mean"]

        if base_sp is not None:
            add = model_result["variants"]["strong_context_plus_smiles"]
            model_result["delta_vs_context_only"] = {
                "spearman": float(add["spearman_mean"] - base_sp),
                "rmse": float(add["rmse_mean"] - base_rmse),
                "mae": float(add["mae_mean"] - base_mae),
                "ndcg@20": float(add["ndcg@20_mean"] - base_ndcg),
            }

        summary["models"].append(model_result)

    out_path = RESULT_ROOT / f"{args.output_stem}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
