#!/usr/bin/env python3
"""
Run DL GroupCV on exact slim numeric + SMILES, optionally with strong context.

Supports:
- FlatMLP
- WideDeep
- CrossAttention
- ResidualMLP
- TabNet
- FTTransformer
- TabTransformer
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
MORE_DL_PATH = SCRIPT_PATH.parent / "run_exact_repo_slim_smiles_more_dl.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_any_smiles_model(smiles_ab_mod, more_dl_mod, name: str, num_dim: int, sample_dim: int, vocab_sizes: list[int], smiles_vocab_size: int):
    key = name.lower().replace("-", "").replace("_", "")
    if key in {"flatmlp", "widedeep", "crossattention"}:
        return smiles_ab_mod.build_smiles_model(name, num_dim, sample_dim, vocab_sizes, smiles_vocab_size)
    return more_dl_mod.build_smiles_model(name, num_dim, vocab_sizes, smiles_vocab_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="FlatMLP,WideDeep,CrossAttention,ResidualMLP,TabNet,FTTransformer,TabTransformer",
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--context-mode", choices=["none", "strong"], default="none")
    parser.add_argument("--output-stem", default="exact_repo_slim_smiles_all_dl_v1")
    parser.add_argument("--early-stop-model", default="")
    parser.add_argument("--early-stop-after-folds", type=int, default=0)
    parser.add_argument("--early-stop-spearman-threshold", type=float, default=0.0)
    args = parser.parse_args()

    mod = load_module(PROGRESSIVE_RUNNER_PATH, "progressive_runner")
    smiles_ab_mod = load_module(SMILES_AB_PATH, "smiles_ab_runner")
    more_dl_mod = load_module(MORE_DL_PATH, "smiles_more_runner")

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
    smiles_ids, smiles_vocab, observed_max_len = smiles_ab_mod.build_smiles_tensor(smiles_col, max_len=256)

    if args.context_mode == "strong":
        ctx_df, _, vocab_map, context_summary = mod.build_reconstructed_context(keys)
        strong_cols = [col for col in mod.STRONG_CONTEXT_COLS if col in ctx_df.columns]
        strong_codes = smiles_ab_mod.encode_context(ctx_df, strong_cols, vocab_map)
        vocab_sizes = [len(vocab_map[col]) for col in strong_cols]
    else:
        strong_cols = []
        strong_codes = np.zeros((len(features), 0), dtype=np.int64)
        vocab_sizes = []
        context_summary = {
            "match_rate": 0.0,
            "column_non_missing_rate": {},
        }

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    splits = list(GroupKFold(n_splits=args.folds).split(X, y, groups=groups))
    out_path = RESULT_ROOT / f"{args.output_stem}.json"
    oof_dir = RESULT_ROOT / f"{args.output_stem}_oof"
    oof_dir.mkdir(exist_ok=True)
    keys_path = oof_dir / "keys.parquet"
    keys.to_parquet(keys_path, index=False)

    summary = {
        "input_features_path": str(DATA_ROOT / "features_slim_exact_repo.parquet"),
        "input_X_path": str(DATA_ROOT / "X_train_exact_repo_numeric.npy"),
        "input_y_path": str(DATA_ROOT / "y_train_exact_repo.npy"),
        "device": str(mod.DEVICE),
        "folds": int(args.folds),
        "context_mode": args.context_mode,
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
                col: float(context_summary["column_non_missing_rate"].get(col, 0.0))
                for col in strong_cols
            },
        },
        "oof_dir": str(oof_dir),
        "keys_path": str(keys_path),
    }

    for model_name in model_names:
        print("=" * 72)
        print(f"[{model_name}] exact slim + SMILES ({args.context_mode} context)")
        print("=" * 72)
        fold_rows = []
        t0 = time.time()
        oof = np.zeros_like(y, dtype=np.float32)
        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            torch.manual_seed(mod.SEED + fold_idx)
            np.random.seed(mod.SEED + fold_idx)
            model, cfg = build_any_smiles_model(
                smiles_ab_mod,
                more_dl_mod,
                model_name,
                X.shape[1],
                sample_dim,
                vocab_sizes,
                len(smiles_vocab),
            )
            pred_val, pred_tr = smiles_ab_mod.train_smiles_model(
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
            oof[val_idx] = pred_val.astype(np.float32)
            row = mod.compute_metrics(y[val_idx], pred_val, y[tr_idx], pred_tr)
            row["fold"] = int(fold_idx)
            fold_rows.append(row)
            print(
                f"  Fold {fold_idx}: Sp={row['spearman']:.4f} "
                f"RMSE={row['rmse']:.4f} MAE={row['mae']:.4f} "
                f"NDCG@20={row['ndcg@20']:.4f}"
            )
            if (
                args.early_stop_model
                and model_name.lower() == args.early_stop_model.lower()
                and args.early_stop_after_folds > 0
                and len(fold_rows) >= args.early_stop_after_folds
            ):
                running_sp = float(pd.DataFrame(fold_rows)["spearman"].mean())
                if running_sp <= args.early_stop_spearman_threshold:
                    print(
                        f"  Early stop: first {len(fold_rows)} folds mean Spearman="
                        f"{running_sp:.4f} <= {args.early_stop_spearman_threshold:.4f}"
                    )
                    break

        df = pd.DataFrame(fold_rows)
        result = {
            "model": model_name,
            "spearman_mean": float(df["spearman"].mean()),
            "spearman_std": float(df["spearman"].std()),
            "rmse_mean": float(df["rmse"].mean()),
            "rmse_std": float(df["rmse"].std()),
            "mae_mean": float(df["mae"].mean()),
            "mae_std": float(df["mae"].std()),
            "pearson_mean": float(df["pearson"].mean()),
            "r2_mean": float(df["r2"].mean()),
            "ndcg@20_mean": float(df["ndcg@20"].mean()),
            "overall_metrics": mod.compute_metrics(y[: len(oof)], oof) if len(oof) else {},
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
        print(
            f"  >>> {model_name}: Sp={result['spearman_mean']:.4f} "
            f"RMSE={result['rmse_mean']:.4f} MAE={result['mae_mean']:.4f} "
            f"NDCG@20={result['ndcg@20_mean']:.4f}"
        )

    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
