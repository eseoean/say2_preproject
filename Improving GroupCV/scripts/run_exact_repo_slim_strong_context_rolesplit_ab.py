#!/usr/bin/env python3
"""
A/B test: exact slim + strong context concat vs role-split strong context.

Uses the exact slim numeric matrix and compares two ways of feeding the same
five strong context columns:

- concat: all 5 columns as categorical tokens
- role_split: 3 semantic categorical + 2 auxiliary context columns
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
from sklearn.model_selection import GroupKFold


SCRIPT_PATH = Path(__file__).resolve()
WORK_ROOT = SCRIPT_PATH.parents[1]
DATA_ROOT = WORK_ROOT / "v3_input_reproduction" / "exact_repo_match"
RESULT_ROOT = WORK_ROOT / "results"
PROGRESSIVE_RUNNER_PATH = SCRIPT_PATH.parent / "run_groupcv_dl_progressive.py"

PRIMARY_COLS = [
    "TCGA_DESC",
    "PATHWAY_NAME_NORMALIZED",
    "classification",
]
AUX_COLS = [
    "drug_bridge_strength",
    "stage3_resolution_status",
]


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
    parser.add_argument("--output-stem", default="exact_repo_slim_strong_context_rolesplit_ab_v1")
    args = parser.parse_args()

    mod = load_progressive_module()

    features = pd.read_parquet(DATA_ROOT / "features_slim_exact_repo.parquet")
    X = np.load(DATA_ROOT / "X_train_exact_repo_numeric.npy").astype(np.float32)
    y = np.load(DATA_ROOT / "y_train_exact_repo.npy").astype(np.float32)

    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    sample_dim = sum(col.startswith("sample__crispr") for col in numeric_cols)
    keys = features[["sample_id", "canonical_drug_id"]].copy()
    keys["sample_id"] = keys["sample_id"].astype(str)
    keys["canonical_drug_id"] = keys["canonical_drug_id"].astype(str)
    groups = keys["canonical_drug_id"].values

    ctx_df, _, vocab_map, context_summary = mod.build_reconstructed_context(keys)
    concat_cols = [col for col in mod.STRONG_CONTEXT_COLS if col in ctx_df.columns]
    primary_cols = [col for col in PRIMARY_COLS if col in ctx_df.columns]
    aux_cols = [col for col in AUX_COLS if col in ctx_df.columns]

    concat_cat = encode_context(ctx_df, concat_cols, vocab_map)
    rolesplit_cat = encode_context(ctx_df, primary_cols, vocab_map)
    rolesplit_aux = encode_context(ctx_df, aux_cols, vocab_map)
    empty_aux = np.zeros((len(features), 0), dtype=np.int64)

    variants = {
        "strong_context_concat": {
            "x_cat": concat_cat,
            "x_aux": empty_aux,
            "vocab_sizes": [len(vocab_map[col]) for col in concat_cols],
            "aux_vocab_sizes": [],
        },
        "strong_context_rolesplit": {
            "x_cat": rolesplit_cat,
            "x_aux": rolesplit_aux,
            "vocab_sizes": [len(vocab_map[col]) for col in primary_cols],
            "aux_vocab_sizes": [len(vocab_map[col]) for col in aux_cols],
        },
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
        "concat_columns": concat_cols,
        "primary_columns": primary_cols,
        "auxiliary_columns": aux_cols,
        "context_summary": {
            "match_rate": float(context_summary.get("match_rate", 0.0)),
            "column_non_missing_rate": {
                col: float(context_summary["column_non_missing_rate"].get(col, 0.0))
                for col in concat_cols
            },
        },
        "models": [],
    }

    for model_name in model_names:
        print("=" * 72)
        print(f"[{model_name}] strong context concat vs role split")
        print("=" * 72)
        model_result = {"model": model_name, "variants": {}}
        concat_sp = None
        concat_rmse = None
        concat_mae = None
        concat_ndcg = None

        for variant_name, variant in variants.items():
            fold_rows = []
            t0 = time.time()
            for fold_idx, (tr_idx, val_idx) in enumerate(splits):
                torch.manual_seed(mod.SEED + fold_idx)
                np.random.seed(mod.SEED + fold_idx)
                model, cfg = mod.build_model(
                    model_name,
                    X.shape[1],
                    variant["vocab_sizes"],
                    variant["aux_vocab_sizes"],
                    sample_dim,
                )
                pred_val, pred_tr = mod.train_model(
                    model=model,
                    x_num_tr=X[tr_idx],
                    x_cat_tr=variant["x_cat"][tr_idx],
                    x_aux_tr=variant["x_aux"][tr_idx],
                    y_tr=y[tr_idx],
                    x_num_val=X[val_idx],
                    x_cat_val=variant["x_cat"][val_idx],
                    x_aux_val=variant["x_aux"][val_idx],
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

            if variant_name == "strong_context_concat":
                concat_sp = summary_rows["spearman_mean"]
                concat_rmse = summary_rows["rmse_mean"]
                concat_mae = summary_rows["mae_mean"]
                concat_ndcg = summary_rows["ndcg@20_mean"]

        if concat_sp is not None:
            role = model_result["variants"]["strong_context_rolesplit"]
            model_result["delta_vs_concat"] = {
                "spearman": float(role["spearman_mean"] - concat_sp),
                "rmse": float(role["rmse_mean"] - concat_rmse),
                "mae": float(role["mae_mean"] - concat_mae),
                "ndcg@20": float(role["ndcg@20_mean"] - concat_ndcg),
            }

        summary["models"].append(model_result)

    out_path = RESULT_ROOT / f"{args.output_stem}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
