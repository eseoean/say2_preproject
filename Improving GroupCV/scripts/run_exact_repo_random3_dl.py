#!/usr/bin/env python3
"""
Run random sample 3-fold CV for DL models across three input datasets:
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
import torch
from sklearn.model_selection import KFold


SCRIPT_PATH = Path(__file__).resolve()
WORK_ROOT = SCRIPT_PATH.parents[1]
RESULT_ROOT = WORK_ROOT / "results"
RESULT_ROOT.mkdir(exist_ok=True)

EXACT_ROOT = WORK_ROOT / "v3_input_reproduction" / "exact_repo_match"
BUNDLE_ROOT = WORK_ROOT / "v3_input_reproduction" / "exact_repo_match_context_smiles_bundle"

NUMERIC_DL_PATH = SCRIPT_PATH.parent / "run_exact_repo_slim_groupcv.py"
SMILES_AB_PATH = SCRIPT_PATH.parent / "run_exact_repo_slim_smiles_ab.py"
SMILES_MORE_PATH = SCRIPT_PATH.parent / "run_exact_repo_slim_smiles_more_dl.py"
SMILES_ALL_PATH = SCRIPT_PATH.parent / "run_exact_repo_slim_smiles_all_dl.py"
PROGRESSIVE_PATH = SCRIPT_PATH.parent / "run_groupcv_dl_progressive.py"

SEED = 42
FOLDS = 3


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_base():
    features = pd.read_parquet(EXACT_ROOT / "features_slim_exact_repo.parquet")
    X = np.load(EXACT_ROOT / "X_train_exact_repo_numeric.npy").astype(np.float32)
    y = np.load(EXACT_ROOT / "y_train_exact_repo.npy").astype(np.float32)
    keys = features[["sample_id", "canonical_drug_id"]].copy()
    keys["sample_id"] = keys["sample_id"].astype(str)
    keys["canonical_drug_id"] = keys["canonical_drug_id"].astype(str)
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    sample_dim = sum(col.startswith("sample__crispr") for col in numeric_cols)
    smiles_col = (
        features["drug__canonical_smiles_raw"]
        .astype("string")
        .fillna(features["drug__smiles"].astype("string"))
        .fillna("")
    )
    return features, X, y, keys, sample_dim, smiles_col


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-mode", choices=["numeric", "smiles", "strong_context_smiles"], required=True)
    parser.add_argument(
        "--models",
        default="FlatMLP,WideDeep,CrossAttention,ResidualMLP,TabNet,FTTransformer,TabTransformer",
    )
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--output-stem", default="")
    parser.add_argument("--early-stop-model", default="")
    parser.add_argument("--early-stop-after-folds", type=int, default=0)
    parser.add_argument("--early-stop-spearman-threshold", type=float, default=0.0)
    args = parser.parse_args()

    numeric_mod = load_module(NUMERIC_DL_PATH, "numeric_dl_random_runner")
    smiles_ab_mod = load_module(SMILES_AB_PATH, "smiles_ab_random_runner")
    smiles_more_mod = load_module(SMILES_MORE_PATH, "smiles_more_random_runner")
    smiles_all_mod = load_module(SMILES_ALL_PATH, "smiles_all_random_runner")
    progressive_mod = load_module(PROGRESSIVE_PATH, "progressive_random_runner")

    features, X, y, keys, sample_dim, smiles_col = load_base()
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    output_stem = args.output_stem or f"exact_repo_random3_{args.input_mode}_dl_v1"
    out_path = RESULT_ROOT / f"{output_stem}.json"
    oof_dir = RESULT_ROOT / f"{output_stem}_oof"
    oof_dir.mkdir(exist_ok=True)
    keys_path = oof_dir / "keys.parquet"
    keys.to_parquet(keys_path, index=False)

    if args.input_mode == "numeric":
        strong_cols = []
        strong_codes = np.zeros((len(features), 0), dtype=np.int64)
        smiles_ids = np.zeros((len(features), 1), dtype=np.int64)
        smiles_vocab = {"<PAD>": 0}
        vocab_sizes = []
        context_summary = {"match_rate": 0.0, "column_non_missing_rate": {}}
    else:
        smiles_ids, smiles_vocab, observed_max_len = smiles_ab_mod.build_smiles_tensor(smiles_col, max_len=256)
        if args.input_mode == "strong_context_smiles":
            strong_codes = np.load(BUNDLE_ROOT / "strong_context_codes.npy").astype(np.int64)
            strong_vocab = json.loads((BUNDLE_ROOT / "strong_context_vocab.json").read_text())
            strong_cols = list(strong_vocab.keys())
            vocab_sizes = [len(strong_vocab[col]) for col in strong_cols]
            context_summary = json.loads((BUNDLE_ROOT / "context_smiles_bundle_summary.json").read_text())["strong_context"]["context_summary"]
        else:
            strong_cols = []
            strong_codes = np.zeros((len(features), 0), dtype=np.int64)
            vocab_sizes = []
            context_summary = {"match_rate": 0.0, "column_non_missing_rate": {}}

    if args.input_mode == "numeric":
        observed_max_len = 0
    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    splits = list(splitter.split(X, y))

    summary = {
        "split_mode": "random_sample_kfold",
        "seed": SEED,
        "input_mode": args.input_mode,
        "input_features_path": str(EXACT_ROOT / "features_slim_exact_repo.parquet"),
        "input_X_path": str(EXACT_ROOT / "X_train_exact_repo_numeric.npy"),
        "input_y_path": str(EXACT_ROOT / "y_train_exact_repo.npy"),
        "device": str(progressive_mod.DEVICE),
        "folds": int(args.folds),
        "n_rows": int(len(y)),
        "x_shape": list(X.shape),
        "sample_dim_detected": int(sample_dim),
        "strong_context_columns": strong_cols,
        "smiles_vocab_size": int(len(smiles_vocab)),
        "smiles_observed_max_len": int(observed_max_len),
        "models": [],
        "context_summary": {
            "match_rate": float(context_summary.get("match_rate", 0.0)),
            "column_non_missing_rate": {
                col: float(context_summary.get("column_non_missing_rate", {}).get(col, 0.0))
                for col in strong_cols
            },
        },
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
            torch.manual_seed(progressive_mod.SEED + fold_idx)
            np.random.seed(progressive_mod.SEED + fold_idx)
            if args.input_mode == "numeric":
                model, cfg = numeric_mod.build_model(model_name, X.shape[1])
                pred_val = numeric_mod.run_torch_model(
                    model,
                    cfg,
                    X[tr_idx],
                    y[tr_idx],
                    X[val_idx],
                    y[val_idx],
                )
                train_loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(torch.from_numpy(X[tr_idx]), torch.from_numpy(y[tr_idx])),
                    batch_size=256,
                    shuffle=False,
                )
                pred_tr = numeric_mod.predict(model.to(numeric_mod.DEVICE), train_loader)
            else:
                model, cfg = smiles_all_mod.build_any_smiles_model(
                    smiles_ab_mod,
                    smiles_more_mod,
                    model_name,
                    X.shape[1],
                    sample_dim,
                    vocab_sizes,
                    len(smiles_vocab),
                )
                pred_val, pred_tr = smiles_ab_mod.train_smiles_model(
                    progressive_mod,
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

            oof[val_idx] = pred_val.astype(np.float32)
            row = progressive_mod.compute_metrics(y[val_idx], pred_val, y[tr_idx], pred_tr)
            row["fold"] = int(fold_idx)
            fold_rows.append(row)
            print(
                f"  Fold {fold_idx}: Sp={row['spearman']:.4f} RMSE={row['rmse']:.4f} "
                f"MAE={row['mae']:.4f} NDCG@20={row['ndcg@20']:.4f}"
            )
            if (
                args.early_stop_model
                and model_name.lower() == args.early_stop_model.lower()
                and args.early_stop_after_folds > 0
                and len(fold_rows) >= args.early_stop_after_folds
            ):
                running_sp = float(pd.DataFrame(fold_rows)["spearman"].mean())
                if running_sp <= float(args.early_stop_spearman_threshold):
                    print(
                        f"  Early stop triggered for {model_name}: "
                        f"mean Spearman {running_sp:.4f} <= {args.early_stop_spearman_threshold:.4f}"
                    )
                    break

        result = {
            "model": model_name,
            "spearman_mean": float(pd.DataFrame(fold_rows)["spearman"].mean()),
            "spearman_std": float(pd.DataFrame(fold_rows)["spearman"].std()),
            "rmse_mean": float(pd.DataFrame(fold_rows)["rmse"].mean()),
            "rmse_std": float(pd.DataFrame(fold_rows)["rmse"].std()),
            "mae_mean": float(pd.DataFrame(fold_rows)["mae"].mean()),
            "mae_std": float(pd.DataFrame(fold_rows)["mae"].std()),
            "pearson_mean": float(pd.DataFrame(fold_rows)["pearson"].mean()),
            "r2_mean": float(pd.DataFrame(fold_rows)["r2"].mean()),
            "ndcg@20_mean": float(pd.DataFrame(fold_rows)["ndcg@20"].mean()),
            "overall_metrics": progressive_mod.compute_metrics(y, oof),
            "elapsed_sec": float(time.time() - t0),
            "executed_folds": int(len(fold_rows)),
            "stopped_early": bool(len(fold_rows) < len(splits)),
            "fold_metrics": fold_rows,
        }
        oof_path = oof_dir / f"{model_name}.npy"
        np.save(oof_path, oof)
        result["oof_path"] = str(oof_path)
        summary["models"].append(result)
        out_path.write_text(json.dumps(summary, indent=2))

    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
