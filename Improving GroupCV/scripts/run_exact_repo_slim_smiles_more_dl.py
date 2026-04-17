#!/usr/bin/env python3
"""
Run additional DL models on exact slim + strong context + SMILES.

Targets:
- ResidualMLP
- TabNet
- FTTransformer
- TabTransformer
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold


SCRIPT_PATH = Path(__file__).resolve()
WORK_ROOT = SCRIPT_PATH.parents[1]
DATA_ROOT = WORK_ROOT / "v3_input_reproduction" / "exact_repo_match"
RESULT_ROOT = WORK_ROOT / "results"
PROGRESSIVE_RUNNER_PATH = SCRIPT_PATH.parent / "run_groupcv_dl_progressive.py"
SMILES_AB_PATH = SCRIPT_PATH.parent / "run_exact_repo_slim_smiles_ab.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


class ResidualMLPSmilesModel(nn.Module):
    def __init__(self, num_dim: int, vocab_sizes: list[int], smiles_vocab_size: int, hidden_dim: int = 256):
        super().__init__()
        self.cat_encoder = smiles_mod.CatTokenEncoder(vocab_sizes, d_token=16)
        self.smiles_encoder = smiles_mod.SmilesCNNEncoder(smiles_vocab_size, out_dim=64)
        in_dim = num_dim + len(vocab_sizes) * 16 + 64
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(3)])
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, x_smiles: torch.Tensor) -> torch.Tensor:
        cat_flat = self.cat_encoder(x_cat).reshape(x_num.size(0), -1)
        smiles = self.smiles_encoder(x_smiles)
        x = torch.cat([x_num, cat_flat, smiles], dim=1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x).squeeze(-1)


class TabNetLikeSmilesModel(nn.Module):
    def __init__(self, num_dim: int, vocab_sizes: list[int], smiles_vocab_size: int):
        super().__init__()
        self.cat_encoder = smiles_mod.CatTokenEncoder(vocab_sizes, d_token=16)
        self.smiles_encoder = smiles_mod.SmilesCNNEncoder(smiles_vocab_size, out_dim=64)
        in_dim = num_dim + len(vocab_sizes) * 16 + 64
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, x_smiles: torch.Tensor) -> torch.Tensor:
        cat_flat = self.cat_encoder(x_cat).reshape(x_num.size(0), -1)
        smiles = self.smiles_encoder(x_smiles)
        return self.net(torch.cat([x_num, cat_flat, smiles], dim=1)).squeeze(-1)


class FTTransformerSmilesModel(nn.Module):
    def __init__(self, num_dim: int, vocab_sizes: list[int], smiles_vocab_size: int):
        super().__init__()
        self.cat_encoder = smiles_mod.CatTokenEncoder(vocab_sizes, d_token=16)
        self.smiles_encoder = smiles_mod.SmilesCNNEncoder(smiles_vocab_size, out_dim=64)
        in_dim = num_dim + len(vocab_sizes) * 16 + 64
        self.token_embed = nn.Linear(in_dim, 256)
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

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, x_smiles: torch.Tensor) -> torch.Tensor:
        cat_flat = self.cat_encoder(x_cat).reshape(x_num.size(0), -1)
        smiles = self.smiles_encoder(x_smiles)
        x = torch.cat([x_num, cat_flat, smiles], dim=1)
        x = self.token_embed(x).unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        return self.fc(x).squeeze(-1)


class TabTransformerSmilesModel(nn.Module):
    def __init__(
        self,
        num_dim: int,
        vocab_sizes: list[int],
        smiles_vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        n_layers: int = 2,
        n_tokens: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_tokenizer = NumericChunkTokenizer(num_dim, d_model=d_model, n_tokens=n_tokens)
        self.cat_encoder = smiles_mod.CatTokenEncoder(vocab_sizes, d_token=d_model)
        self.smiles_encoder = smiles_mod.SmilesCNNEncoder(smiles_vocab_size, out_dim=d_model)
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

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, x_smiles: torch.Tensor) -> torch.Tensor:
        num_tokens = self.num_tokenizer(x_num)
        cat_tokens = self.cat_encoder(x_cat)
        smiles_token = self.smiles_encoder(x_smiles).unsqueeze(1)
        cls = self.cls_token.expand(x_num.size(0), -1, -1)
        h = self.transformer(torch.cat([cls, num_tokens, cat_tokens, smiles_token], dim=1))
        return self.head(h[:, 0]).squeeze(-1)


def build_smiles_model(name: str, num_dim: int, vocab_sizes: list[int], smiles_vocab_size: int):
    key = name.lower().replace("-", "").replace("_", "")
    if key == "residualmlp":
        return ResidualMLPSmilesModel(num_dim, vocab_sizes, smiles_vocab_size), {
            "epochs": 80,
            "lr": 1e-3,
            "batch_size": 256,
        }
    if key == "tabnet":
        return TabNetLikeSmilesModel(num_dim, vocab_sizes, smiles_vocab_size), {
            "epochs": 60,
            "lr": 1e-3,
            "batch_size": 256,
        }
    if key == "fttransformer":
        return FTTransformerSmilesModel(num_dim, vocab_sizes, smiles_vocab_size), {
            "epochs": 100,
            "lr": 1e-3,
            "batch_size": 64,
        }
    if key == "tabtransformer":
        return TabTransformerSmilesModel(num_dim, vocab_sizes, smiles_vocab_size), {
            "epochs": 80,
            "lr": 5e-4,
            "batch_size": 128,
        }
    raise ValueError(f"Unsupported model: {name}")


progressive_mod = load_module(PROGRESSIVE_RUNNER_PATH, "progressive_runner")
smiles_mod = load_module(SMILES_AB_PATH, "smiles_ab_runner")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="ResidualMLP,TabNet,FTTransformer,TabTransformer")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--output-stem", default="exact_repo_slim_smiles_more_dl_v1")
    parser.add_argument("--early-stop-model", default="")
    parser.add_argument("--early-stop-after-folds", type=int, default=2)
    parser.add_argument("--early-stop-spearman-threshold", type=float, default=0.57)
    args = parser.parse_args()

    features = pd.read_parquet(DATA_ROOT / "features_slim_exact_repo.parquet")
    X = np.load(DATA_ROOT / "X_train_exact_repo_numeric.npy").astype(np.float32)
    y = np.load(DATA_ROOT / "y_train_exact_repo.npy").astype(np.float32)

    keys = features[["sample_id", "canonical_drug_id"]].copy()
    keys["sample_id"] = keys["sample_id"].astype(str)
    keys["canonical_drug_id"] = keys["canonical_drug_id"].astype(str)
    groups = keys["canonical_drug_id"].values

    smiles_col = (
        features["drug__canonical_smiles_raw"]
        .astype("string")
        .fillna(features["drug__smiles"].astype("string"))
        .fillna("")
    )
    smiles_ids, smiles_vocab, observed_max_len = smiles_mod.build_smiles_tensor(smiles_col, max_len=256)

    ctx_df, _, vocab_map, context_summary = progressive_mod.build_reconstructed_context(keys)
    strong_cols = [col for col in progressive_mod.STRONG_CONTEXT_COLS if col in ctx_df.columns]
    strong_cat = smiles_mod.encode_context(ctx_df, strong_cols, vocab_map)
    vocab_sizes = [len(vocab_map[col]) for col in strong_cols]

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    splits = list(GroupKFold(n_splits=args.folds).split(X, y, groups=groups))

    summary = {
        "input_features_path": str(DATA_ROOT / "features_slim_exact_repo.parquet"),
        "input_X_path": str(DATA_ROOT / "X_train_exact_repo_numeric.npy"),
        "input_y_path": str(DATA_ROOT / "y_train_exact_repo.npy"),
        "device": str(progressive_mod.DEVICE),
        "folds": int(args.folds),
        "n_rows": int(len(y)),
        "x_shape": list(X.shape),
        "strong_context_columns": strong_cols,
        "smiles_vocab_size": int(len(smiles_vocab)),
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
    out_path = RESULT_ROOT / f"{args.output_stem}.json"

    for model_name in model_names:
        print("=" * 72)
        print(f"[{model_name}] exact slim + strong context + SMILES")
        print("=" * 72)
        fold_rows = []
        t0 = time.time()
        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            torch.manual_seed(progressive_mod.SEED + fold_idx)
            np.random.seed(progressive_mod.SEED + fold_idx)
            model, cfg = build_smiles_model(model_name, X.shape[1], vocab_sizes, len(smiles_vocab))
            pred_val, pred_tr = smiles_mod.train_smiles_model(
                progressive_mod,
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
            row = progressive_mod.compute_metrics(y[val_idx], pred_val, y[tr_idx], pred_tr)
            row["fold"] = int(fold_idx)
            fold_rows.append(row)
            print(
                f"  Fold {fold_idx}: Sp={row['spearman']:.4f} "
                f"RMSE={row['rmse']:.4f} MAE={row['mae']:.4f} "
                f"NDCG@20={row['ndcg@20']:.4f}"
            )
            if (
                args.early_stop_model
                and model_name.lower() == args.early_stop_model.lower()
                and len(fold_rows) >= args.early_stop_after_folds
            ):
                running_sp = float(pd.DataFrame(fold_rows)["spearman"].mean())
                if running_sp <= args.early_stop_spearman_threshold:
                    print(
                        f"  Early stop: first {len(fold_rows)} folds mean Spearman="
                        f"{running_sp:.4f} <= {args.early_stop_spearman_threshold:.4f}"
                    )
                    break

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
            "ndcg@20_mean": float(df["ndcg@20"].mean()),
            "elapsed_sec": float(time.time() - t0),
            "executed_folds": int(len(fold_rows)),
            "stopped_early": bool(len(fold_rows) < len(splits)),
            "fold_metrics": fold_rows,
        }
        summary["models"].append(result)
        out_path.write_text(json.dumps(summary, indent=2))
        print(
            f"  >>> {model_name}: Sp={result['spearman_mean']:.4f} "
            f"RMSE={result['rmse_mean']:.4f} MAE={result['mae_mean']:.4f} "
            f"NDCG@20={result['ndcg@20_mean']:.4f}"
        )

    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
