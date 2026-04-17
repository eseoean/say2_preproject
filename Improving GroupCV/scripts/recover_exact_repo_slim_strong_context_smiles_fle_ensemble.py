#!/usr/bin/env python3
"""
Recover the mixed FlatMLP + LightGBM_DART + ExtraTrees ensemble by:
- reusing the existing FlatMLP OOF from the FRC ensemble run
- training only LightGBM_DART and ExtraTrees
- rebuilding equal/weighted ensemble metrics and diversity
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupKFold


SCRIPT_PATH = Path(__file__).resolve()
WORK_ROOT = SCRIPT_PATH.parents[1]
RESULT_ROOT = WORK_ROOT / "results"
EXACT_ROOT = WORK_ROOT / "v3_input_reproduction" / "exact_repo_match"
ML_ROOT = WORK_ROOT / "v3_input_reproduction" / "exact_repo_match_ml_strong_context_smiles"

FRC_JSON_PATH = RESULT_ROOT / "exact_repo_slim_strong_context_smiles_frc_ensemble_oof_v2.json"
ML_RUNNER_PATH = SCRIPT_PATH.parent / "run_exact_repo_slim_strong_context_smiles_ml_groupcv.py"

FOLDS = 3
MODEL_NAMES = ["FlatMLP", "LightGBM_DART", "ExtraTrees"]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _safe_corr(x: np.ndarray, y: np.ndarray, fn) -> float:
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
    if pair_rows:
        df = pd.DataFrame(pair_rows)
        summary = {
            "avg_prediction_pearson": float(df["prediction_pearson"].mean()),
            "avg_prediction_spearman": float(df["prediction_spearman"].mean()),
            "avg_residual_pearson": float(df["residual_pearson"].mean()),
            "avg_residual_spearman": float(df["residual_spearman"].mean()),
            "avg_mean_abs_prediction_gap": float(df["mean_abs_prediction_gap"].mean()),
        }
    else:
        summary = {}
    return {"pairwise": pair_rows, "summary": summary}


def main() -> None:
    out_path = RESULT_ROOT / "exact_repo_slim_strong_context_smiles_fle_ensemble_v1.json"
    oof_dir = RESULT_ROOT / "exact_repo_slim_strong_context_smiles_fle_ensemble_v1_oof"
    oof_dir.mkdir(exist_ok=True)

    ml_mod = load_module(ML_RUNNER_PATH, "strong_ml_runner_fle_recover")

    features = pd.read_parquet(EXACT_ROOT / "features_slim_exact_repo.parquet")
    X_ml = np.load(ML_ROOT / "X_ml_exact_slim_strong_context_smiles.npy").astype(np.float32)
    y = np.load(EXACT_ROOT / "y_train_exact_repo.npy").astype(np.float32)
    keys = features[["sample_id", "canonical_drug_id"]].copy()
    keys["sample_id"] = keys["sample_id"].astype(str)
    keys["canonical_drug_id"] = keys["canonical_drug_id"].astype(str)
    groups = keys["canonical_drug_id"].values
    feature_names_ml = json.loads((ML_ROOT / "feature_names_ml_exact_slim_strong_context_smiles.json").read_text())
    splits = list(GroupKFold(n_splits=FOLDS).split(X_ml, y, groups=groups))

    keys_path = oof_dir / "keys.parquet"
    y_path = oof_dir / "y_true.npy"
    keys.to_parquet(keys_path, index=False)
    np.save(y_path, y)

    frc_obj = json.loads(FRC_JSON_PATH.read_text())
    flat_meta = frc_obj["base_model_metrics"]["FlatMLP"]
    flat_oof_src = Path(flat_meta["oof_path"])
    flat_oof = np.load(flat_oof_src).astype(np.float32)
    flat_oof_dst = oof_dir / "FlatMLP.npy"
    if flat_oof_src != flat_oof_dst:
        shutil.copy2(flat_oof_src, flat_oof_dst)

    summary = {
        "input_features_path": str(EXACT_ROOT / "features_slim_exact_repo.parquet"),
        "input_ml_X_path": str(ML_ROOT / "X_ml_exact_slim_strong_context_smiles.npy"),
        "input_y_path": str(EXACT_ROOT / "y_train_exact_repo.npy"),
        "folds": FOLDS,
        "n_rows": int(len(y)),
        "ml_x_shape": list(X_ml.shape),
        "models": MODEL_NAMES,
        "oof_dir": str(oof_dir),
        "keys_path": str(keys_path),
        "base_model_metrics": {
            "FlatMLP": {
                "summary": ml_mod.summarize_rows(flat_meta["fold_metrics"]),
                "overall_metrics": ml_mod.compute_metrics(y, flat_oof),
                "elapsed_sec": None,
                "fold_metrics": flat_meta["fold_metrics"],
                "oof_path": str(flat_oof_dst),
            }
        },
    }

    oof_predictions: dict[str, np.ndarray] = {"FlatMLP": flat_oof}

    for model_name in ["LightGBM_DART", "ExtraTrees"]:
        print("=" * 72)
        print(f"[{model_name}] recover mixed ensemble")
        print("=" * 72)
        fold_rows = []
        t0 = time.time()
        oof = np.zeros_like(y, dtype=np.float32)
        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            model = ml_mod.build_model(model_name, feature_names_ml, fold_idx)
            pred_val, pred_tr = ml_mod.fit_predict(
                model_name,
                model,
                X_ml[tr_idx],
                y[tr_idx],
                X_ml[val_idx],
                y[val_idx],
                feature_names_ml,
            )
            oof[val_idx] = pred_val.astype(np.float32)
            row = ml_mod.compute_metrics(y[val_idx], pred_val, y[tr_idx], pred_tr)
            row["fold"] = int(fold_idx)
            fold_rows.append(row)
            print(
                f"  Fold {fold_idx}: Sp={row['spearman']:.4f} RMSE={row['rmse']:.4f} "
                f"MAE={row['mae']:.4f} NDCG@20={row['ndcg@20']:.4f}"
            )
        oof_path = oof_dir / f"{model_name}.npy"
        np.save(oof_path, oof)
        oof_predictions[model_name] = oof
        summary["base_model_metrics"][model_name] = {
            "summary": ml_mod.summarize_rows(fold_rows),
            "overall_metrics": ml_mod.compute_metrics(y, oof),
            "elapsed_sec": float(time.time() - t0),
            "fold_metrics": fold_rows,
            "oof_path": str(oof_path),
        }
        out_path.write_text(json.dumps(summary, indent=2))

    equal_oof = np.mean(np.stack([oof_predictions[name] for name in MODEL_NAMES], axis=0), axis=0)
    model_spearman = np.array(
        [ml_mod.compute_metrics(y, oof_predictions[name])["spearman"] for name in MODEL_NAMES],
        dtype=np.float64,
    )
    clipped = np.clip(model_spearman, a_min=0.0, a_max=None)
    weights = clipped / clipped.sum() if clipped.sum() > 0 else np.ones_like(clipped) / len(clipped)
    weighted_oof = np.zeros_like(y, dtype=np.float32)
    for w, name in zip(weights, MODEL_NAMES):
        weighted_oof += float(w) * oof_predictions[name]

    np.save(oof_dir / "equal_ensemble.npy", equal_oof)
    np.save(oof_dir / "weighted_ensemble.npy", weighted_oof)

    summary["equal_overall_metrics"] = ml_mod.compute_metrics(y, equal_oof)
    summary["weighted_overall_metrics"] = ml_mod.compute_metrics(y, weighted_oof)
    summary["weights"] = {name: float(w) for name, w in zip(MODEL_NAMES, weights)}
    summary["diversity"] = compute_diversity(oof_predictions, y)

    best_base_sp = max(summary["base_model_metrics"][name]["overall_metrics"]["spearman"] for name in MODEL_NAMES)
    best_base_rmse = min(summary["base_model_metrics"][name]["overall_metrics"]["rmse"] for name in MODEL_NAMES)
    summary["ensemble_gain_vs_best_base"] = {
        "weighted_spearman_gain": float(summary["weighted_overall_metrics"]["spearman"] - best_base_sp),
        "weighted_rmse_gain": float(summary["weighted_overall_metrics"]["rmse"] - best_base_rmse),
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
