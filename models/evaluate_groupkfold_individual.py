#!/usr/bin/env python3
"""
Evaluate individual top models with GroupKFold.

Default grouping: canonical_drug_id (unseen-drug generalization)
Default models: CatBoost, LightGBM, XGBoost, FlatMLP, ResidualMLP, Cross-Attention
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from train_ensemble import (
    BENCH_RMSE,
    BENCH_SP,
    CrossAttentionNet,
    DEVICE,
    FlatMLP,
    ResidualMLP,
    SEED,
    load_data,
    train_catboost,
    train_dl_model,
    train_lightgbm,
    train_xgboost,
)


OUTPUT_DIR = Path(__file__).parent / os.getenv("GROUPKFOLD_OUTPUT_DIRNAME", "groupkfold_results_drug")
OUTPUT_DIR.mkdir(exist_ok=True)
N_FOLDS = int(os.getenv("GROUPKFOLD_N_FOLDS", "5"))
GROUP_BY = os.getenv("GROUPKFOLD_GROUP_BY", "drug").strip().lower()
SELECTED_MODELS = [m.strip() for m in os.getenv("GROUPKFOLD_MODELS", "").split(",") if m.strip()]

torch.manual_seed(SEED)
np.random.seed(SEED)


def compute_metrics(y_true, y_pred, y_train_true=None, y_train_pred=None):
    sp, _ = spearmanr(y_true, y_pred)
    pe, _ = pearsonr(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    out = {"spearman": sp, "pearson": pe, "rmse": rmse, "r2": r2}
    if y_train_true is not None and y_train_pred is not None:
        tr_sp, _ = spearmanr(y_train_true, y_train_pred)
        tr_rmse = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
        out["train_spearman"] = tr_sp
        out["train_rmse"] = tr_rmse
        out["gap_spearman"] = tr_sp - sp
        out["gap_rmse"] = rmse - tr_rmse
    return out


def save_json(path: Path, data) -> None:
    def _default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    path.write_text(json.dumps(data, indent=2, default=_default))


def get_groups(sample_ids, drug_ids):
    if GROUP_BY == "drug":
        return np.asarray(drug_ids)
    if GROUP_BY in {"sample", "cell", "cell_line"}:
        return np.asarray(sample_ids)
    raise ValueError(f"Unsupported GROUPKFOLD_GROUP_BY={GROUP_BY!r}; use 'drug' or 'sample'")


def main():
    X, y, sample_ids, drug_ids = load_data()
    groups = get_groups(sample_ids, drug_ids)
    splitter = GroupKFold(n_splits=N_FOLDS)
    in_dim = X.shape[1]
    sample_dim = 18311

    all_model_configs = [
        ("CatBoost", "ml", train_catboost),
        ("LightGBM", "ml", train_lightgbm),
        ("XGBoost", "ml", train_xgboost),
        ("FlatMLP", "dl", FlatMLP),
        ("ResidualMLP", "dl", ResidualMLP),
        ("Cross-Attention", "dl", CrossAttentionNet),
    ]
    if SELECTED_MODELS:
        wanted = set(SELECTED_MODELS)
        unknown = sorted(wanted - {name for name, _, _ in all_model_configs})
        if unknown:
            raise ValueError(f"Unknown GROUPKFOLD_MODELS: {unknown}")
        model_configs = [cfg for cfg in all_model_configs if cfg[0] in wanted]
    else:
        model_configs = all_model_configs

    dl_kwargs = {
        "FlatMLP": ({"in_dim": in_dim, "layers": [1024, 512, 256], "dropout": 0.3},
                    {"epochs": 100, "lr": 1e-3, "batch_size": 256}),
        "ResidualMLP": ({"in_dim": in_dim, "hidden": 512, "n_blocks": 3, "dropout": 0.3},
                        {"epochs": 100, "lr": 1e-3, "batch_size": 256}),
        "Cross-Attention": ({"in_dim": in_dim, "sample_dim": sample_dim, "d_model": 128, "nhead": 4, "dropout": 0.2},
                            {"epochs": 80, "lr": 5e-4, "batch_size": 256}),
    }

    print("=" * 72)
    print(f"GroupKFold individual evaluation")
    print(f"Grouping key: {GROUP_BY}")
    print(f"Unique groups: {len(set(groups))}")
    print(f"Models: {', '.join(name for name, _, _ in model_configs)}")
    print(f"Device: {DEVICE}")
    print("=" * 72)

    results = []
    partial_path = OUTPUT_DIR / f"groupkfold_{GROUP_BY}_results.partial.json"

    for name, mtype, trainer in model_configs:
        print(f"\n{'-' * 60}")
        print(f"[{name}] {N_FOLDS}-fold GroupKFold by {GROUP_BY}")
        print(f"{'-' * 60}")

        model_t0 = time.time()
        fold_metrics = []

        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            if mtype == "ml":
                pred_val, pred_tr = trainer(X_tr, y_tr, X_val, y_val)
            else:
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
                X_val_s = scaler.transform(X_val).astype(np.float32)

                torch.manual_seed(SEED + fold_idx)
                if DEVICE.type == "mps":
                    torch.mps.empty_cache()

                model_kw, train_kw = dl_kwargs[name]
                model = trainer(**model_kw)
                pred_val, pred_tr = train_dl_model(model, X_tr_s, y_tr, X_val_s, y_val, **train_kw)

                del model
                if DEVICE.type == "mps":
                    torch.mps.empty_cache()

            metrics = compute_metrics(y_val, pred_val, y_tr, pred_tr)
            metrics["fold"] = fold_idx
            metrics["n_train"] = int(len(train_idx))
            metrics["n_val"] = int(len(val_idx))
            metrics["n_train_groups"] = int(len(set(groups[train_idx])))
            metrics["n_val_groups"] = int(len(set(groups[val_idx])))
            fold_metrics.append(metrics)

            print(
                f"  Fold {fold_idx}: Sp={metrics['spearman']:.4f}  "
                f"RMSE={metrics['rmse']:.4f}  "
                f"Train Sp={metrics['train_spearman']:.4f}  "
                f"Gap={metrics['gap_spearman']:.4f}  "
                f"val_groups={metrics['n_val_groups']}"
            )

        df = pd.DataFrame(fold_metrics)
        summary = {
            "model": name,
            "group_by": GROUP_BY,
            "n_folds": N_FOLDS,
            "spearman_mean": df["spearman"].mean(),
            "spearman_std": df["spearman"].std(),
            "rmse_mean": df["rmse"].mean(),
            "rmse_std": df["rmse"].std(),
            "pearson_mean": df["pearson"].mean(),
            "r2_mean": df["r2"].mean(),
            "r2_std": df["r2"].std(),
            "train_spearman_mean": df["train_spearman"].mean(),
            "gap_spearman_mean": df["gap_spearman"].mean(),
            "gap_rmse_mean": df["gap_rmse"].mean(),
            "elapsed_sec": time.time() - model_t0,
            "folds": fold_metrics,
        }
        results.append(summary)
        save_json(partial_path, results)

        beat_sp = "PASS" if summary["spearman_mean"] >= BENCH_SP else "FAIL"
        beat_rm = "PASS" if summary["rmse_mean"] <= BENCH_RMSE else "FAIL"
        print(f"  >>> {name} SUMMARY")
        print(f"      Spearman: {summary['spearman_mean']:.4f} +/- {summary['spearman_std']:.4f}  [{beat_sp}]")
        print(f"      RMSE:     {summary['rmse_mean']:.4f} +/- {summary['rmse_std']:.4f}  [{beat_rm}]")
        print(f"      Pearson:  {summary['pearson_mean']:.4f}")
        print(f"      R2:       {summary['r2_mean']:.4f} +/- {summary['r2_std']:.4f}")
        print(f"      Train Sp: {summary['train_spearman_mean']:.4f}  Gap: {summary['gap_spearman_mean']:.4f}")
        print(f"      Time:     {summary['elapsed_sec'] / 60:.1f} min")

    out_path = OUTPUT_DIR / f"groupkfold_{GROUP_BY}_results.json"
    save_json(out_path, results)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
