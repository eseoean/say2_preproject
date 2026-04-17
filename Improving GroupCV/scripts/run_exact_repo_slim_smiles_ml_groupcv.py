#!/usr/bin/env python3
"""
Run GroupCV for ML models on the shared exact-slim + SMILES matrix.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_exact_repo_slim_strong_context_smiles_ml_groupcv import (
    N_FOLDS,
    build_model,
    compute_metrics,
    fit_predict,
    summarize_rows,
)
from sklearn.model_selection import GroupKFold


SCRIPT_PATH = Path(__file__).resolve()
WORK_ROOT = SCRIPT_PATH.parents[1]
DATA_ROOT = WORK_ROOT / "v3_input_reproduction" / "exact_repo_match_ml_smiles"
RESULT_ROOT = WORK_ROOT / "results"
RESULT_ROOT.mkdir(exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="LightGBM,LightGBM_DART,XGBoost,CatBoost,RandomForest,ExtraTrees",
    )
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--output-stem", default="exact_repo_slim_smiles_ml_groupcv_v1")
    args = parser.parse_args()

    X = np.load(DATA_ROOT / "X_ml_exact_slim_smiles.npy").astype(np.float32)
    y = np.load(DATA_ROOT / "y_ml_exact_slim_smiles.npy").astype(np.float32)
    keys = pd.read_parquet(DATA_ROOT / "keys_ml_exact_slim_smiles.parquet")
    feature_names = json.loads((DATA_ROOT / "feature_names_ml_exact_slim_smiles.json").read_text())
    groups = keys["canonical_drug_id"].astype(str).values

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    splits = list(GroupKFold(n_splits=args.folds).split(X, y, groups=groups))

    results = {
        "input_X_path": str(DATA_ROOT / "X_ml_exact_slim_smiles.npy"),
        "input_y_path": str(DATA_ROOT / "y_ml_exact_slim_smiles.npy"),
        "input_keys_path": str(DATA_ROOT / "keys_ml_exact_slim_smiles.parquet"),
        "folds": int(args.folds),
        "rows": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "models": [],
    }
    out_path = RESULT_ROOT / f"{args.output_stem}.json"
    oof_dir = RESULT_ROOT / f"{args.output_stem}_oof"
    oof_dir.mkdir(exist_ok=True)
    keys_path = oof_dir / "keys.parquet"
    keys.to_parquet(keys_path, index=False)
    results["oof_dir"] = str(oof_dir)
    results["keys_path"] = str(keys_path)

    for model_name in model_names:
        print("=" * 72)
        print(f"[{model_name}] exact slim + SMILES")
        print("=" * 72)
        fold_rows = []
        t0 = time.time()
        oof = np.zeros_like(y, dtype=np.float32)
        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            model = build_model(model_name, feature_names, fold_idx)
            pred_val, pred_tr = fit_predict(
                model_name,
                model,
                X[tr_idx],
                y[tr_idx],
                X[val_idx],
                y[val_idx],
                feature_names,
            )
            oof[val_idx] = pred_val.astype(np.float32)
            row = compute_metrics(y[val_idx], pred_val, y[tr_idx], pred_tr)
            row["fold"] = int(fold_idx)
            fold_rows.append(row)
            print(
                f"  Fold {fold_idx}: Sp={row['spearman']:.4f} RMSE={row['rmse']:.4f} "
                f"MAE={row['mae']:.4f} NDCG@20={row['ndcg@20']:.4f}"
            )

        result = {
            "model": model_name,
            "fold_metrics": fold_rows,
            "summary": summarize_rows(fold_rows),
            "overall_metrics": compute_metrics(y, oof),
            "elapsed_sec": float(time.time() - t0),
        }
        oof_path = oof_dir / f"{model_name}.npy"
        np.save(oof_path, oof)
        result["oof_path"] = str(oof_path)
        results["models"].append(result)
        out_path.write_text(json.dumps(results, indent=2))
        print(
            f"  >>> overall: Sp={result['overall_metrics']['spearman']:.4f} "
            f"RMSE={result['overall_metrics']['rmse']:.4f} "
            f"MAE={result['overall_metrics']['mae']:.4f} "
            f"NDCG@20={result['overall_metrics']['ndcg@20']:.4f}"
        )

    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
