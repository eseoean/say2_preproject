#!/usr/bin/env python3
"""
Local GroupCV runner for the exact repo-matched slim input.

Uses the exact slim artifacts materialized in our workspace and reimplements
the simplified DL models found in the reference repo so we can compare them
under a consistent local evaluation flow.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
N_FOLDS = 3
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

WORK_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = WORK_ROOT / "v3_input_reproduction" / "exact_repo_match"
RESULT_ROOT = WORK_ROOT / "results"
RESULT_ROOT.mkdir(exist_ok=True)


def _safe_corr(x: np.ndarray, y: np.ndarray, fn) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(fn(x, y)[0])


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "spearman": _safe_corr(y_true, y_pred, spearmanr),
        "pearson": _safe_corr(y_true, y_pred, pearsonr),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


class FlatMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] | None = None):
        super().__init__()
        hidden_dims = hidden_dims or [512, 256, 128]
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc(x)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, n_blocks: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(n_blocks)])
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x).squeeze(-1)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class FTTransformerMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, 256)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embed(x).unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        return self.fc(x).squeeze(-1)


class NumericChunkTokenizer(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 128, n_tokens: int = 64):
        super().__init__()
        self.n_tokens = n_tokens
        self.chunk_size = math.ceil(in_dim / n_tokens)
        self.proj = nn.Linear(self.chunk_size, d_model)

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        batch = x_num.size(0)
        total = self.n_tokens * self.chunk_size
        pad_len = total - x_num.size(1)
        if pad_len > 0:
            x_num = F.pad(x_num, (0, pad_len))
        x_num = x_num.view(batch, self.n_tokens, self.chunk_size)
        return self.proj(x_num)


class TabTransformerMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        n_layers: int = 2,
        n_tokens: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_tokenizer = NumericChunkTokenizer(input_dim, d_model=d_model, n_tokens=n_tokens)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.num_tokenizer(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        h = self.transformer(torch.cat([cls, tokens], dim=1))
        return self.head(h[:, 0]).squeeze(-1)


class CrossAttentionMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.query = nn.Linear(input_dim, 256)
        self.key = nn.Linear(input_dim, 256)
        self.value = nn.Linear(input_dim, 256)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x).unsqueeze(1)
        k = self.key(x).unsqueeze(1)
        v = self.value(x).unsqueeze(1)
        attn_output, _ = self.multihead_attn(q, k, v)
        return self.mlp(attn_output.squeeze(1)).squeeze(-1)


class WideDeepMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.wide = nn.Linear(input_dim, 1)
        self.deep = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.wide(x) + self.deep(x)).squeeze(-1)


def build_model(name: str, input_dim: int):
    key = name.lower()
    if key == "flatmlp":
        return FlatMLP(input_dim), {"epochs": 50, "batch_size": 256, "lr": 1e-3}
    if key == "residualmlp":
        return ResidualMLP(input_dim), {"epochs": 50, "batch_size": 256, "lr": 1e-3}
    if key == "tabnet":
        return SimpleMLP(input_dim), {"epochs": 30, "batch_size": 256, "lr": 1e-3}
    if key == "fttransformer":
        return FTTransformerMLP(input_dim), {"epochs": 100, "batch_size": 64, "lr": 1e-3, "early_stopping": 10}
    if key == "tabtransformer":
        return TabTransformerMLP(input_dim), {"epochs": 80, "batch_size": 128, "lr": 5e-4, "early_stopping": 10}
    if key == "crossattention":
        return CrossAttentionMLP(input_dim), {"epochs": 100, "batch_size": 64, "lr": 1e-3, "early_stopping": 10}
    if key == "widedeep":
        return WideDeepMLP(input_dim), {"epochs": 80, "batch_size": 256, "lr": 1e-3, "early_stopping": 10}
    raise ValueError(f"Unsupported model: {name}")


def predict(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(preds)


def run_torch_model(
    model: nn.Module,
    cfg: Dict[str, float],
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> np.ndarray:
    train_dataset = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_dataset, batch_size=int(cfg["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    criterion = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    patience = 0
    early_stopping = int(cfg.get("early_stopping", 0))

    for _ in range(int(cfg["epochs"])):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        if early_stopping:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(DEVICE)
                    yb = yb.to(DEVICE)
                    val_losses.append(criterion(model(xb), yb).item())
            val_loss = float(np.mean(val_losses))
            if val_loss < best_val:
                best_val = val_loss
                patience = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= early_stopping:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return predict(model, val_loader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="FlatMLP,FTTransformer,CrossAttention")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--output-stem", default="exact_repo_slim_groupcv")
    args = parser.parse_args()

    features = pd.read_parquet(DATA_ROOT / "features_slim_exact_repo.parquet")
    X = np.load(DATA_ROOT / "X_train_exact_repo_numeric.npy").astype(np.float32)
    y = np.load(DATA_ROOT / "y_train_exact_repo.npy").astype(np.float32)
    groups = features["canonical_drug_id"].astype(str).values

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    splitter = GroupKFold(n_splits=args.folds)
    splits = list(splitter.split(X, y, groups=groups))

    summary = {
        "input_features_path": str(DATA_ROOT / "features_slim_exact_repo.parquet"),
        "input_X_path": str(DATA_ROOT / "X_train_exact_repo_numeric.npy"),
        "input_y_path": str(DATA_ROOT / "y_train_exact_repo.npy"),
        "device": str(DEVICE),
        "n_rows": int(len(y)),
        "x_shape": list(X.shape),
        "n_unique_drugs": int(features["canonical_drug_id"].astype(str).nunique()),
        "folds": int(args.folds),
        "models": [],
    }
    out_path = RESULT_ROOT / f"{args.output_stem}.json"
    oof_dir = RESULT_ROOT / f"{args.output_stem}_oof"
    oof_dir.mkdir(exist_ok=True)
    keys_path = oof_dir / "keys.parquet"
    features[["sample_id", "canonical_drug_id"]].to_parquet(keys_path, index=False)
    summary["oof_dir"] = str(oof_dir)
    summary["keys_path"] = str(keys_path)

    for model_name in models:
        print("=" * 72)
        print(f"[{model_name}] exact repo slim GroupCV")
        print("=" * 72)
        fold_rows = []
        t0 = time.time()
        oof = np.zeros_like(y, dtype=np.float32)
        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            torch.manual_seed(SEED + fold_idx)
            np.random.seed(SEED + fold_idx)
            model, cfg = build_model(model_name, X.shape[1])
            model = model.to(DEVICE)
            pred = run_torch_model(
                model=model,
                cfg=cfg,
                X_tr=X[tr_idx],
                y_tr=y[tr_idx],
                X_val=X[val_idx],
                y_val=y[val_idx],
            )
            oof[val_idx] = pred.astype(np.float32)
            row = metrics(y[val_idx], pred)
            row["fold"] = int(fold_idx)
            fold_rows.append(row)
            print(
                f"  Fold {fold_idx}: Sp={row['spearman']:.4f} "
                f"RMSE={row['rmse']:.4f} MAE={row['mae']:.4f}"
            )

        df = pd.DataFrame(fold_rows)
        result = {
            "model": model_name,
            "spearman_mean": float(df["spearman"].mean()),
            "spearman_std": float(df["spearman"].std()),
            "rmse_mean": float(df["rmse"].mean()),
            "rmse_std": float(df["rmse"].std()),
            "mae_mean": float(df["mae"].mean()),
            "mae_std": float(df["mae"].std()),
            "pearson_mean": float(df["pearson"].mean()),
            "r2_mean": float(df["r2"].mean()),
            "overall_metrics": metrics(y, oof),
            "elapsed_sec": float(time.time() - t0),
            "fold_metrics": fold_rows,
        }
        oof_path = oof_dir / f"{model_name}.npy"
        np.save(oof_path, oof)
        result["oof_path"] = str(oof_path)
        summary["models"].append(result)
        out_path.write_text(json.dumps(summary, indent=2))
        print(
            f"  >>> {model_name}: Sp={result['spearman_mean']:.4f} "
            f"RMSE={result['rmse_mean']:.4f} MAE={result['mae_mean']:.4f}"
        )

    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
