#!/usr/bin/env python3
"""
Run GroupCV for ML models on the shared exact-slim + strong-context + SMILES matrix.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score, r2_score
from sklearn.model_selection import GroupKFold


SCRIPT_PATH = Path(__file__).resolve()
WORK_ROOT = SCRIPT_PATH.parents[1]
DATA_ROOT = WORK_ROOT / "v3_input_reproduction" / "exact_repo_match_ml_strong_context_smiles"
RESULT_ROOT = WORK_ROOT / "results"
RESULT_ROOT.mkdir(exist_ok=True)

SEED = 42
N_FOLDS = 3
CPU_THREADS = int(os.getenv("MODEL_CPU_THREADS", "4"))


def _safe_pearson(x, y) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(pearsonr(x, y)[0])


def _safe_spearman(x, y) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(spearmanr(x, y)[0])


def compute_ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 20) -> float:
    k = min(k, len(y_true))
    true_gain = -np.asarray(y_true, dtype=np.float64)
    pred_gain = -np.asarray(y_pred, dtype=np.float64)
    true_gain = true_gain - true_gain.min() + 1e-6
    pred_gain = pred_gain - pred_gain.min() + 1e-6
    return float(ndcg_score(true_gain.reshape(1, -1), pred_gain.reshape(1, -1), k=k))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_tr_true=None, y_tr_pred=None):
    out = {
        "spearman": _safe_spearman(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "pearson": _safe_pearson(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "ndcg@20": compute_ndcg_at_k(y_true, y_pred, k=20),
    }
    if y_tr_true is not None and y_tr_pred is not None:
        tr_sp = _safe_spearman(y_tr_true, y_tr_pred)
        tr_rmse = float(np.sqrt(mean_squared_error(y_tr_true, y_tr_pred)))
        tr_mae = float(mean_absolute_error(y_tr_true, y_tr_pred))
        out["train_spearman"] = tr_sp
        out["train_rmse"] = tr_rmse
        out["train_mae"] = tr_mae
        out["gap_spearman"] = tr_sp - out["spearman"]
        out["gap_rmse"] = out["rmse"] - tr_rmse
        out["gap_mae"] = out["mae"] - tr_mae
    return out


def build_model(name: str, feature_names: list[str], fold_idx: int):
    key = name.lower().replace("-", "").replace("_", "")
    if key == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            objective="regression",
            n_estimators=1200,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=7,
            min_child_samples=20,
            colsample_bytree=0.7,
            subsample=0.8,
            subsample_freq=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=SEED + fold_idx,
            n_jobs=CPU_THREADS,
            verbosity=-1,
        )
    if key == "lightgbmdart":
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            objective="regression",
            boosting_type="dart",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=7,
            min_child_samples=20,
            colsample_bytree=0.7,
            subsample=0.8,
            subsample_freq=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            drop_rate=0.1,
            skip_drop=0.5,
            random_state=SEED + fold_idx,
            n_jobs=CPU_THREADS,
            verbosity=-1,
        )
    if key == "xgboost":
        import xgboost as xgb

        return xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=1200,
            max_depth=7,
            learning_rate=0.05,
            colsample_bytree=0.7,
            subsample=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5,
            tree_method="hist",
            random_state=SEED + fold_idx,
            n_jobs=CPU_THREADS,
            verbosity=0,
        )
    if key == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=1200,
            learning_rate=0.05,
            depth=7,
            l2_leaf_reg=3.0,
            rsm=0.7,
            subsample=0.8,
            random_seed=SEED + fold_idx,
            verbose=False,
            thread_count=1,
        )
    if key == "randomforest":
        return RandomForestRegressor(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=5,
            n_jobs=CPU_THREADS,
            random_state=SEED + fold_idx,
        )
    if key == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=3,
            n_jobs=CPU_THREADS,
            random_state=SEED + fold_idx,
        )
    raise ValueError(f"Unsupported model: {name}")


def fit_predict(name: str, model, X_tr, y_tr, X_val, y_val, feature_names):
    key = name.lower().replace("-", "").replace("_", "")
    if key.startswith("lightgbm"):
        callbacks = []
        if key == "lightgbm":
            callbacks = []
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        return model.predict(X_val), model.predict(X_tr)
    if key == "xgboost":
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return model.predict(X_val), model.predict(X_tr)
    if key == "catboost":
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        return model.predict(X_val), model.predict(X_tr)
    model.fit(X_tr, y_tr)
    return model.predict(X_val), model.predict(X_tr)


def summarize_rows(rows):
    df = pd.DataFrame(rows)
    out = {}
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
            out[f"{col}_mean"] = float(df[col].mean())
            out[f"{col}_std"] = float(df[col].std())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="LightGBM,LightGBM_DART,XGBoost,CatBoost,RandomForest,ExtraTrees",
    )
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument(
        "--output-stem",
        default="exact_repo_slim_strong_context_smiles_ml_groupcv_v1",
    )
    args = parser.parse_args()

    X = np.load(DATA_ROOT / "X_ml_exact_slim_strong_context_smiles.npy").astype(np.float32)
    y = np.load(DATA_ROOT / "y_ml_exact_slim_strong_context_smiles.npy").astype(np.float32)
    keys = pd.read_parquet(DATA_ROOT / "keys_ml_exact_slim_strong_context_smiles.parquet")
    feature_names = json.loads(
        (DATA_ROOT / "feature_names_ml_exact_slim_strong_context_smiles.json").read_text()
    )
    groups = keys["canonical_drug_id"].astype(str).values

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    splits = list(GroupKFold(n_splits=args.folds).split(X, y, groups=groups))

    results = {
        "input_X_path": str(DATA_ROOT / "X_ml_exact_slim_strong_context_smiles.npy"),
        "input_y_path": str(DATA_ROOT / "y_ml_exact_slim_strong_context_smiles.npy"),
        "input_keys_path": str(DATA_ROOT / "keys_ml_exact_slim_strong_context_smiles.parquet"),
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
        print(f"[{model_name}] exact slim + strong context + SMILES")
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

        summary = summarize_rows(fold_rows)
        overall = compute_metrics(y, oof)
        result = {
            "model": model_name,
            "fold_metrics": fold_rows,
            "summary": summary,
            "overall_metrics": overall,
            "elapsed_sec": float(time.time() - t0),
        }
        oof_path = oof_dir / f"{model_name}.npy"
        np.save(oof_path, oof)
        result["oof_path"] = str(oof_path)
        results["models"].append(result)
        out_path.write_text(json.dumps(results, indent=2))
        print(
            f"  >>> overall: Sp={overall['spearman']:.4f} RMSE={overall['rmse']:.4f} "
            f"MAE={overall['mae']:.4f} NDCG@20={overall['ndcg@20']:.4f}"
        )

    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
