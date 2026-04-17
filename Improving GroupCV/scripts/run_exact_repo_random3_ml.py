#!/usr/bin/env python3
"""
Run random sample 3-fold CV for ML models across three input datasets:
- exact slim (numeric-only)
- exact slim + SMILES
- exact slim + strong context + SMILES
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


SCRIPT_PATH = Path(__file__).resolve()
WORK_ROOT = SCRIPT_PATH.parents[1]
RESULT_ROOT = WORK_ROOT / "results"
RESULT_ROOT.mkdir(exist_ok=True)

EXACT_ROOT = WORK_ROOT / "v3_input_reproduction" / "exact_repo_match"
BUNDLE_ROOT = WORK_ROOT / "v3_input_reproduction" / "exact_repo_match_context_smiles_bundle"
STRONG_ML_PATH = SCRIPT_PATH.parent / "run_exact_repo_slim_strong_context_smiles_ml_groupcv.py"

SEED = 42
FOLDS = 3


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_input_bundle(input_mode: str):
    if input_mode == "numeric":
        features = pd.read_parquet(EXACT_ROOT / "features_slim_exact_repo.parquet")
        X = np.load(EXACT_ROOT / "X_train_exact_repo_numeric.npy").astype(np.float32)
        y = np.load(EXACT_ROOT / "y_train_exact_repo.npy").astype(np.float32)
        keys = features[["sample_id", "canonical_drug_id"]].copy()
        feature_names = features.select_dtypes(include=[np.number]).columns.tolist()
        return {
            "X": X,
            "y": y,
            "keys": keys,
            "feature_names": feature_names,
            "input_X_path": str(EXACT_ROOT / "X_train_exact_repo_numeric.npy"),
            "input_y_path": str(EXACT_ROOT / "y_train_exact_repo.npy"),
            "input_keys_path": str(EXACT_ROOT / "features_slim_exact_repo.parquet"),
        }
    if input_mode == "smiles":
        X = np.load(BUNDLE_ROOT / "X_ml_exact_slim_smiles.npy").astype(np.float32)
        y = np.load(BUNDLE_ROOT / "y_exact_slim.npy").astype(np.float32)
        keys = pd.read_parquet(BUNDLE_ROOT / "keys_exact_slim.parquet")
        feature_names = json.loads((BUNDLE_ROOT / "feature_names_ml_exact_slim_smiles.json").read_text())
        return {
            "X": X,
            "y": y,
            "keys": keys,
            "feature_names": feature_names,
            "input_X_path": str(BUNDLE_ROOT / "X_ml_exact_slim_smiles.npy"),
            "input_y_path": str(BUNDLE_ROOT / "y_exact_slim.npy"),
            "input_keys_path": str(BUNDLE_ROOT / "keys_exact_slim.parquet"),
        }
    if input_mode == "strong_context_smiles":
        X = np.load(BUNDLE_ROOT / "X_ml_exact_slim_strong_context_smiles.npy").astype(np.float32)
        y = np.load(BUNDLE_ROOT / "y_exact_slim.npy").astype(np.float32)
        keys = pd.read_parquet(BUNDLE_ROOT / "keys_exact_slim.parquet")
        feature_names = json.loads((BUNDLE_ROOT / "feature_names_ml_exact_slim_strong_context_smiles.json").read_text())
        return {
            "X": X,
            "y": y,
            "keys": keys,
            "feature_names": feature_names,
            "input_X_path": str(BUNDLE_ROOT / "X_ml_exact_slim_strong_context_smiles.npy"),
            "input_y_path": str(BUNDLE_ROOT / "y_exact_slim.npy"),
            "input_keys_path": str(BUNDLE_ROOT / "keys_exact_slim.parquet"),
        }
    raise ValueError(f"Unsupported input_mode: {input_mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-mode", choices=["numeric", "smiles", "strong_context_smiles"], required=True)
    parser.add_argument(
        "--models",
        default="LightGBM,LightGBM_DART,XGBoost,CatBoost,RandomForest,ExtraTrees",
    )
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--output-stem", default="")
    args = parser.parse_args()

    ml_mod = load_module(STRONG_ML_PATH, "strong_ml_random_runner")
    bundle = load_input_bundle(args.input_mode)
    X = bundle["X"]
    y = bundle["y"]
    keys = bundle["keys"].copy()
    keys["sample_id"] = keys["sample_id"].astype(str)
    keys["canonical_drug_id"] = keys["canonical_drug_id"].astype(str)
    feature_names = bundle["feature_names"]
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    output_stem = args.output_stem or f"exact_repo_random3_{args.input_mode}_ml_v1"
    out_path = RESULT_ROOT / f"{output_stem}.json"
    oof_dir = RESULT_ROOT / f"{output_stem}_oof"
    oof_dir.mkdir(exist_ok=True)
    keys_path = oof_dir / "keys.parquet"
    keys.to_parquet(keys_path, index=False)

    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    splits = list(splitter.split(X, y))

    results = {
        "split_mode": "random_sample_kfold",
        "seed": SEED,
        "input_mode": args.input_mode,
        "input_X_path": bundle["input_X_path"],
        "input_y_path": bundle["input_y_path"],
        "input_keys_path": bundle["input_keys_path"],
        "folds": int(args.folds),
        "rows": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "models": [],
        "oof_dir": str(oof_dir),
        "keys_path": str(keys_path),
    }

    for model_name in model_names:
        print("=" * 72)
        print(f"[{model_name}] random sample 3-fold / {args.input_mode}")
        print("=" * 72)
        t0 = time.time()
        fold_rows = []
        oof = np.zeros_like(y, dtype=np.float32)
        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            model = ml_mod.build_model(model_name, feature_names, fold_idx)
            pred_val, pred_tr = ml_mod.fit_predict(
                model_name,
                model,
                X[tr_idx],
                y[tr_idx],
                X[val_idx],
                y[val_idx],
                feature_names,
            )
            oof[val_idx] = pred_val.astype(np.float32)
            row = ml_mod.compute_metrics(y[val_idx], pred_val, y[tr_idx], pred_tr)
            row["fold"] = int(fold_idx)
            fold_rows.append(row)
            print(
                f"  Fold {fold_idx}: Sp={row['spearman']:.4f} RMSE={row['rmse']:.4f} "
                f"MAE={row['mae']:.4f} NDCG@20={row['ndcg@20']:.4f}"
            )

        result = {
            "model": model_name,
            "fold_metrics": fold_rows,
            "summary": ml_mod.summarize_rows(fold_rows),
            "overall_metrics": ml_mod.compute_metrics(y, oof),
            "elapsed_sec": float(time.time() - t0),
        }
        oof_path = oof_dir / f"{model_name}.npy"
        np.save(oof_path, oof)
        result["oof_path"] = str(oof_path)
        results["models"].append(result)
        out_path.write_text(json.dumps(results, indent=2))

    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
