#!/usr/bin/env python3
"""
Equal/weighted ensemble for the top exact-slim GroupCV models.

Runs the selected models on the same GroupKFold splits, stores OOF
predictions, and reports both equal-weight and Spearman-weighted ensembles.
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
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupKFold


SCRIPT_PATH = Path(__file__).resolve()
WORK_ROOT = SCRIPT_PATH.parents[1]
RESULT_ROOT = WORK_ROOT / "results"
RUNNER_PATH = SCRIPT_PATH.parent / "run_exact_repo_slim_groupcv.py"


def load_runner_module():
    spec = importlib.util.spec_from_file_location("exact_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load runner module from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _safe_corr(x, y, fn):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(fn(x, y)[0])


def compute_diversity(oof_predictions: dict[str, np.ndarray], y_true: np.ndarray):
    names = list(oof_predictions.keys())
    pair_rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = oof_predictions[names[i]]
            b = oof_predictions[names[j]]
            ra = y_true - a
            rb = y_true - b
            pair_rows.append(
                {
                    "pair": f"{names[i]} vs {names[j]}",
                    "prediction_pearson": _safe_corr(a, b, pearsonr),
                    "prediction_spearman": _safe_corr(a, b, spearmanr),
                    "residual_pearson": _safe_corr(ra, rb, pearsonr),
                    "residual_spearman": _safe_corr(ra, rb, spearmanr),
                    "mean_abs_prediction_gap": float(np.mean(np.abs(a - b))),
                }
            )
    df = pd.DataFrame(pair_rows)
    return {
        "pairwise": pair_rows,
        "summary": {
            "avg_prediction_pearson": float(df["prediction_pearson"].mean()),
            "avg_prediction_spearman": float(df["prediction_spearman"].mean()),
            "avg_residual_pearson": float(df["residual_pearson"].mean()),
            "avg_residual_spearman": float(df["residual_spearman"].mean()),
            "avg_mean_abs_prediction_gap": float(df["mean_abs_prediction_gap"].mean()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="WideDeep,CrossAttention,FlatMLP")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--output-stem", default="exact_repo_slim_top3_ensemble_weighted_v1")
    args = parser.parse_args()

    runner = load_runner_module()
    features = pd.read_parquet(runner.DATA_ROOT / "features_slim_exact_repo.parquet")
    X = np.load(runner.DATA_ROOT / "X_train_exact_repo_numeric.npy").astype(np.float32)
    y = np.load(runner.DATA_ROOT / "y_train_exact_repo.npy").astype(np.float32)
    groups = features["canonical_drug_id"].astype(str).values
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    splits = list(GroupKFold(n_splits=args.folds).split(X, y, groups=groups))

    summary = {
        "input_features_path": str(runner.DATA_ROOT / "features_slim_exact_repo.parquet"),
        "input_X_path": str(runner.DATA_ROOT / "X_train_exact_repo_numeric.npy"),
        "input_y_path": str(runner.DATA_ROOT / "y_train_exact_repo.npy"),
        "device": str(runner.DEVICE),
        "models": model_names,
        "folds": int(args.folds),
        "n_rows": int(len(y)),
        "x_shape": list(X.shape),
        "n_unique_drugs": int(features["canonical_drug_id"].astype(str).nunique()),
        "ensemble_types": ["equal_weight_mean", "spearman_weighted"],
        "equal_fold_metrics": [],
        "base_model_metrics": {},
    }
    oof_dir = RESULT_ROOT / f"{args.output_stem}_oof"
    oof_dir.mkdir(exist_ok=True)
    keys_path = oof_dir / "keys.parquet"
    features[["sample_id", "canonical_drug_id"]].to_parquet(keys_path, index=False)
    summary["oof_dir"] = str(oof_dir)
    summary["keys_path"] = str(keys_path)

    per_model_fold_metrics = {name: [] for name in model_names}
    oof_predictions = {name: np.zeros_like(y, dtype=np.float32) for name in model_names}
    t0 = time.time()

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        fold_preds = []
        for model_name in model_names:
            torch.manual_seed(runner.SEED + fold_idx)
            np.random.seed(runner.SEED + fold_idx)
            model, cfg = runner.build_model(model_name, X.shape[1])
            model = model.to(runner.DEVICE)
            pred = runner.run_torch_model(
                model=model,
                cfg=cfg,
                X_tr=X[tr_idx],
                y_tr=y[tr_idx],
                X_val=X[val_idx],
                y_val=y[val_idx],
            )
            fold_preds.append(pred)
            oof_predictions[model_name][val_idx] = pred.astype(np.float32)
            metric_row = runner.metrics(y[val_idx], pred)
            metric_row["fold"] = int(fold_idx)
            per_model_fold_metrics[model_name].append(metric_row)

        ensemble_pred = np.mean(np.stack(fold_preds, axis=0), axis=0)
        row = runner.metrics(y[val_idx], ensemble_pred)
        row["fold"] = int(fold_idx)
        summary["equal_fold_metrics"].append(row)
        print(
            f"Equal fold {fold_idx}: "
            f"Sp={row['spearman']:.4f} RMSE={row['rmse']:.4f} MAE={row['mae']:.4f}"
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
            "fold_metrics": per_model_fold_metrics[model_name],
        }
        model_oof_path = oof_dir / f"{model_name}.npy"
        np.save(model_oof_path, oof_predictions[model_name])
        summary["base_model_metrics"][model_name]["oof_path"] = str(model_oof_path)

    equal_oof = np.mean(np.stack([oof_predictions[name] for name in model_names], axis=0), axis=0)
    model_spearman = np.array(
        [runner.metrics(y, oof_predictions[name])["spearman"] for name in model_names],
        dtype=np.float64,
    )
    if np.all(model_spearman <= 0):
        weights = np.full(len(model_names), 1.0 / len(model_names), dtype=np.float64)
    else:
        clipped = np.clip(model_spearman, a_min=0.0, a_max=None)
        if clipped.sum() == 0:
            weights = np.full(len(model_names), 1.0 / len(model_names), dtype=np.float64)
        else:
            weights = clipped / clipped.sum()
    weighted_oof = sum(oof_predictions[name] * weights[i] for i, name in enumerate(model_names))

    equal_metrics = runner.metrics(y, equal_oof)
    weighted_metrics = runner.metrics(y, weighted_oof)
    np.save(oof_dir / "equal_ensemble.npy", equal_oof)
    np.save(oof_dir / "weighted_ensemble.npy", weighted_oof)
    np.save(oof_dir / "y_true.npy", y)
    summary["weights"] = {name: float(weights[i]) for i, name in enumerate(model_names)}
    summary["diversity"] = compute_diversity(oof_predictions, y)
    best_base_spearman = max(summary["base_model_metrics"][name]["spearman_mean"] for name in model_names)
    best_base_rmse = min(summary["base_model_metrics"][name]["rmse_mean"] for name in model_names)
    summary["ensemble_gain_vs_best_base"] = {
        "weighted_spearman_gain": float(weighted_metrics["spearman"] - best_base_spearman),
        "weighted_rmse_gain": float(best_base_rmse - weighted_metrics["rmse"]),
    }
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
        f"MAE={summary['equal_overall_metrics']['mae']:.4f}"
    )
    print("Spearman weights:", summary["weights"])
    print(
        "Weighted overall: "
        f"Sp={summary['weighted_overall_metrics']['spearman']:.4f} "
        f"RMSE={summary['weighted_overall_metrics']['rmse']:.4f} "
        f"MAE={summary['weighted_overall_metrics']['mae']:.4f}"
    )
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
