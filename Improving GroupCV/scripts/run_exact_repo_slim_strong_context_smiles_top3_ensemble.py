#!/usr/bin/env python3
"""
Ensemble for exact slim + strong context + SMILES top-3 models.

Starts from the best exact-slim strong-context stack and adds the lightweight
SMILES branch validated in the A/B script, then evaluates equal-weight and
Spearman-weighted ensembles on shared drug GroupKFold splits.
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
SMILES_AB_PATH = SCRIPT_PATH.parent / "run_exact_repo_slim_smiles_ab.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="FlatMLP,WideDeep,CrossAttention")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument(
        "--output-stem",
        default="exact_repo_slim_strong_context_smiles_top3_ensemble_v1",
    )
    args = parser.parse_args()

    mod = load_module(PROGRESSIVE_RUNNER_PATH, "progressive_runner")
    smiles_mod = load_module(SMILES_AB_PATH, "smiles_ab_runner")

    features = pd.read_parquet(DATA_ROOT / "features_slim_exact_repo.parquet")
    X = np.load(DATA_ROOT / "X_train_exact_repo_numeric.npy").astype(np.float32)
    y = np.load(DATA_ROOT / "y_train_exact_repo.npy").astype(np.float32)

    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    sample_dim = sum(col.startswith("sample__crispr") for col in numeric_cols)
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

    ctx_df, _, vocab_map, context_summary = mod.build_reconstructed_context(keys)
    strong_cols = [col for col in mod.STRONG_CONTEXT_COLS if col in ctx_df.columns]
    strong_codes = smiles_mod.encode_context(ctx_df, strong_cols, vocab_map)
    vocab_sizes = [len(vocab_map[col]) for col in strong_cols]

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    splits = list(GroupKFold(n_splits=args.folds).split(X, y, groups=groups))

    summary = {
        "input_features_path": str(DATA_ROOT / "features_slim_exact_repo.parquet"),
        "input_X_path": str(DATA_ROOT / "X_train_exact_repo_numeric.npy"),
        "input_y_path": str(DATA_ROOT / "y_train_exact_repo.npy"),
        "device": str(mod.DEVICE),
        "folds": int(args.folds),
        "models": model_names,
        "n_rows": int(len(y)),
        "x_shape": list(X.shape),
        "sample_dim_detected": int(sample_dim),
        "strong_context_columns": strong_cols,
        "smiles_vocab_size": int(len(smiles_vocab)),
        "smiles_max_len_cap": 256,
        "smiles_observed_max_len": int(observed_max_len),
        "context_summary": {
            "match_rate": float(context_summary.get("match_rate", 0.0)),
            "column_non_missing_rate": {
                col: float(context_summary["column_non_missing_rate"].get(col, 0.0))
                for col in strong_cols
            },
        },
        "ensemble_types": ["equal_weight_mean", "spearman_weighted"],
        "base_model_metrics": {},
        "equal_fold_metrics": [],
    }

    per_model_fold_metrics = {name: [] for name in model_names}
    oof_predictions = {name: np.zeros_like(y, dtype=np.float32) for name in model_names}
    t0 = time.time()

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        fold_preds = []
        for model_name in model_names:
            torch.manual_seed(mod.SEED + fold_idx)
            np.random.seed(mod.SEED + fold_idx)
            model, cfg = smiles_mod.build_smiles_model(
                model_name,
                X.shape[1],
                sample_dim,
                vocab_sizes,
                len(smiles_vocab),
            )
            pred_val, pred_tr = smiles_mod.train_smiles_model(
                mod,
                model,
                cfg,
                X[tr_idx],
                strong_codes[tr_idx],
                smiles_ids[tr_idx],
                y[tr_idx],
                X[val_idx],
                strong_codes[val_idx],
                smiles_ids[val_idx],
                y[val_idx],
            )
            oof_predictions[model_name][val_idx] = pred_val.astype(np.float32)
            fold_preds.append(pred_val)
            row = mod.compute_metrics(y[val_idx], pred_val, y[tr_idx], pred_tr)
            row["fold"] = int(fold_idx)
            per_model_fold_metrics[model_name].append(row)

        ensemble_pred = np.mean(np.stack(fold_preds, axis=0), axis=0)
        row = mod.compute_metrics(y[val_idx], ensemble_pred)
        row["fold"] = int(fold_idx)
        summary["equal_fold_metrics"].append(row)
        print(
            f"Equal fold {fold_idx}: "
            f"Sp={row['spearman']:.4f} RMSE={row['rmse']:.4f} "
            f"MAE={row['mae']:.4f} NDCG@20={row['ndcg@20']:.4f}"
        )

    for model_name in model_names:
        rows = pd.DataFrame(per_model_fold_metrics[model_name])
        summary["base_model_metrics"][model_name] = {
            "spearman_mean": float(rows["spearman"].mean()),
            "spearman_std": float(rows["spearman"].std()),
            "rmse_mean": float(rows["rmse"].mean()),
            "rmse_std": float(rows["rmse"].std()),
            "mae_mean": float(rows["mae"].mean()),
            "mae_std": float(rows["mae"].std()),
            "pearson_mean": float(rows["pearson"].mean()),
            "r2_mean": float(rows["r2"].mean()),
            "ndcg@20_mean": float(rows["ndcg@20"].mean()),
            "fold_metrics": per_model_fold_metrics[model_name],
        }

    equal_oof = np.mean(np.stack([oof_predictions[name] for name in model_names], axis=0), axis=0)
    model_spearman = np.array(
        [mod.compute_metrics(y, oof_predictions[name])["spearman"] for name in model_names],
        dtype=np.float64,
    )
    clipped = np.clip(model_spearman, a_min=0.0, a_max=None)
    if clipped.sum() == 0:
        weights = np.full(len(model_names), 1.0 / len(model_names), dtype=np.float64)
    else:
        weights = clipped / clipped.sum()
    weighted_oof = sum(oof_predictions[name] * weights[i] for i, name in enumerate(model_names))

    equal_metrics = mod.compute_metrics(y, equal_oof)
    weighted_metrics = mod.compute_metrics(y, weighted_oof)
    summary["weights"] = {name: float(weights[i]) for i, name in enumerate(model_names)}
    summary["equal_overall_metrics"] = {
        **equal_metrics,
        "elapsed_sec": float(time.time() - t0),
    }
    summary["weighted_overall_metrics"] = weighted_metrics

    out_path = RESULT_ROOT / f"{args.output_stem}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(
        "Equal overall: "
        f"Sp={summary['equal_overall_metrics']['spearman']:.4f} "
        f"RMSE={summary['equal_overall_metrics']['rmse']:.4f} "
        f"MAE={summary['equal_overall_metrics']['mae']:.4f} "
        f"NDCG@20={summary['equal_overall_metrics']['ndcg@20']:.4f}"
    )
    print("Spearman weights:", summary["weights"])
    print(
        "Weighted overall: "
        f"Sp={summary['weighted_overall_metrics']['spearman']:.4f} "
        f"RMSE={summary['weighted_overall_metrics']['rmse']:.4f} "
        f"MAE={summary['weighted_overall_metrics']['mae']:.4f} "
        f"NDCG@20={summary['weighted_overall_metrics']['ndcg@20']:.4f}"
    )
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
