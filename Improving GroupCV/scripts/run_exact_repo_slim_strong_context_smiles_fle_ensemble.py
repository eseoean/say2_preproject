#!/usr/bin/env python3
"""
Run a mixed OOF ensemble on exact slim + strong context + SMILES:
- FlatMLP (DL)
- LightGBM_DART (ML)
- ExtraTrees (ML)
"""

from __future__ import annotations

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
RESULT_ROOT.mkdir(exist_ok=True)

EXACT_ROOT = WORK_ROOT / "v3_input_reproduction" / "exact_repo_match"
ML_ROOT = WORK_ROOT / "v3_input_reproduction" / "exact_repo_match_ml_strong_context_smiles"

SMILES_AB_PATH = SCRIPT_PATH.parent / "run_exact_repo_slim_smiles_ab.py"
STRONG_ML_PATH = SCRIPT_PATH.parent / "run_exact_repo_slim_strong_context_smiles_ml_groupcv.py"
PROGRESSIVE_RUNNER_PATH = SCRIPT_PATH.parent / "run_groupcv_dl_progressive.py"

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
    summary = {}
    if pair_rows:
        df = pd.DataFrame(pair_rows)
        summary = {
            "avg_prediction_pearson": float(df["prediction_pearson"].mean()),
            "avg_prediction_spearman": float(df["prediction_spearman"].mean()),
            "avg_residual_pearson": float(df["residual_pearson"].mean()),
            "avg_residual_spearman": float(df["residual_spearman"].mean()),
            "avg_mean_abs_prediction_gap": float(df["mean_abs_prediction_gap"].mean()),
        }
    return {"pairwise": pair_rows, "summary": summary}


def main() -> None:
    output_stem = "exact_repo_slim_strong_context_smiles_fle_ensemble_v1"
    out_path = RESULT_ROOT / f"{output_stem}.json"
    oof_dir = RESULT_ROOT / f"{output_stem}_oof"
    oof_dir.mkdir(exist_ok=True)

    smiles_ab_mod = load_module(SMILES_AB_PATH, "smiles_ab_runner_fle")
    ml_mod = load_module(STRONG_ML_PATH, "strong_ml_runner_fle")
    progressive_mod = load_module(PROGRESSIVE_RUNNER_PATH, "progressive_runner_fle")

    features = pd.read_parquet(EXACT_ROOT / "features_slim_exact_repo.parquet")
    X_dl = np.load(EXACT_ROOT / "X_train_exact_repo_numeric.npy").astype(np.float32)
    y = np.load(EXACT_ROOT / "y_train_exact_repo.npy").astype(np.float32)
    X_ml = np.load(ML_ROOT / "X_ml_exact_slim_strong_context_smiles.npy").astype(np.float32)
    keys = features[["sample_id", "canonical_drug_id"]].copy()
    keys["sample_id"] = keys["sample_id"].astype(str)
    keys["canonical_drug_id"] = keys["canonical_drug_id"].astype(str)
    groups = keys["canonical_drug_id"].values
    keys_path = oof_dir / "keys.parquet"
    y_path = oof_dir / "y_true.npy"
    keys.to_parquet(keys_path, index=False)
    np.save(y_path, y)

    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    sample_dim = sum(col.startswith("sample__crispr") for col in numeric_cols)
    smiles_col = (
        features["drug__canonical_smiles_raw"]
        .astype("string")
        .fillna(features["drug__smiles"].astype("string"))
        .fillna("")
    )
    smiles_ids, smiles_vocab, observed_max_len = smiles_ab_mod.build_smiles_tensor(smiles_col, max_len=256)
    ctx_df, _, vocab_map, context_summary = progressive_mod.build_reconstructed_context(keys)
    strong_cols = [col for col in progressive_mod.STRONG_CONTEXT_COLS if col in ctx_df.columns]
    strong_codes = smiles_ab_mod.encode_context(ctx_df, strong_cols, vocab_map)
    vocab_sizes = [len(vocab_map[col]) for col in strong_cols]

    feature_names_ml = json.loads((ML_ROOT / "feature_names_ml_exact_slim_strong_context_smiles.json").read_text())
    splits = list(GroupKFold(n_splits=FOLDS).split(X_dl, y, groups=groups))

    summary = {
        "input_features_path": str(EXACT_ROOT / "features_slim_exact_repo.parquet"),
        "input_dl_X_path": str(EXACT_ROOT / "X_train_exact_repo_numeric.npy"),
        "input_ml_X_path": str(ML_ROOT / "X_ml_exact_slim_strong_context_smiles.npy"),
        "input_y_path": str(EXACT_ROOT / "y_train_exact_repo.npy"),
        "folds": FOLDS,
        "n_rows": int(len(y)),
        "dl_x_shape": list(X_dl.shape),
        "ml_x_shape": list(X_ml.shape),
        "sample_dim_detected": int(sample_dim),
        "strong_context_columns": strong_cols,
        "smiles_vocab_size": int(len(smiles_vocab)),
        "smiles_observed_max_len": int(observed_max_len),
        "context_summary": {
            "match_rate": float(context_summary.get("match_rate", 0.0)),
            "column_non_missing_rate": {
                col: float(context_summary["column_non_missing_rate"].get(col, 0.0))
                for col in strong_cols
            },
        },
        "models": MODEL_NAMES,
        "oof_dir": str(oof_dir),
        "keys_path": str(keys_path),
        "base_model_metrics": {},
    }

    oof_predictions: dict[str, np.ndarray] = {}
    per_model_fold_metrics: dict[str, list[dict[str, float]]] = {}

    for model_name in MODEL_NAMES:
        print("=" * 72)
        print(f"[{model_name}] exact slim + strong context + SMILES (mixed ensemble)")
        print("=" * 72)
        fold_rows = []
        t0 = time.time()
        oof = np.zeros_like(y, dtype=np.float32)
        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            np.random.seed(progressive_mod.SEED + fold_idx)
            torch.manual_seed(progressive_mod.SEED + fold_idx)
            if model_name == "FlatMLP":
                model, cfg = smiles_ab_mod.build_smiles_model(
                    model_name,
                    X_dl.shape[1],
                    sample_dim,
                    vocab_sizes,
                    len(smiles_vocab),
                )
                pred_val, pred_tr = smiles_ab_mod.train_smiles_model(
                    progressive_mod,
                    model,
                    cfg,
                    X_dl[tr_idx],
                    strong_codes[tr_idx],
                    smiles_ids[tr_idx],
                    y[tr_idx],
                    X_dl[val_idx],
                    strong_codes[val_idx],
                    smiles_ids[val_idx],
                    y[val_idx],
                )
            else:
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
                f"  Fold {fold_idx}: Sp={row['spearman']:.4f} "
                f"RMSE={row['rmse']:.4f} MAE={row['mae']:.4f} "
                f"NDCG@20={row['ndcg@20']:.4f}"
            )

        per_model_fold_metrics[model_name] = fold_rows
        oof_predictions[model_name] = oof
        result = {
            "summary": ml_mod.summarize_rows(fold_rows),
            "overall_metrics": ml_mod.compute_metrics(y, oof),
            "elapsed_sec": float(time.time() - t0),
            "fold_metrics": fold_rows,
        }
        oof_path = oof_dir / f"{model_name}.npy"
        np.save(oof_path, oof)
        result["oof_path"] = str(oof_path)
        summary["base_model_metrics"][model_name] = result
        out_path.write_text(json.dumps(summary, indent=2))

    equal_oof = np.mean(np.stack([oof_predictions[name] for name in MODEL_NAMES], axis=0), axis=0)
    model_spearman = np.array(
        [ml_mod.compute_metrics(y, oof_predictions[name])["spearman"] for name in MODEL_NAMES],
        dtype=np.float64,
    )
    clipped = np.clip(model_spearman, a_min=0.0, a_max=None)
    if clipped.sum() == 0:
        weights = np.ones_like(clipped) / len(clipped)
    else:
        weights = clipped / clipped.sum()
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
