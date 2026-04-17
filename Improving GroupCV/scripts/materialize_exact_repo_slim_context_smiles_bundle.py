#!/usr/bin/env python3
"""
Materialize reproducible context/SMILES artifacts starting from the local exact slim table.

Starting point:
- v3_input_reproduction/exact_repo_match/features_slim_exact_repo.parquet
- v3_input_reproduction/exact_repo_match/y_train_exact_repo.npy

Outputs:
- reconstructed full context table
- strong-context-only table
- DL-ready strong context codes + vocab
- DL-ready SMILES token ids + vocab
- ML-ready exact slim + SMILES matrix
- ML-ready exact slim + strong context + SMILES matrix
- a summary JSON describing every output and dimension
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
WORK_ROOT = SCRIPT_PATH.parents[1]
EXACT_ROOT = WORK_ROOT / "v3_input_reproduction" / "exact_repo_match"
OUT_ROOT = WORK_ROOT / "v3_input_reproduction" / "exact_repo_match_context_smiles_bundle"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

PROGRESSIVE_RUNNER_PATH = SCRIPT_PATH.parent / "run_groupcv_dl_progressive.py"
SMILES_AB_PATH = SCRIPT_PATH.parent / "run_exact_repo_slim_smiles_ab.py"
ML_STRONG_SMILES_BUILDER_PATH = SCRIPT_PATH.parent / "build_exact_repo_slim_strong_context_smiles_ml_matrix.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    progressive = load_module(PROGRESSIVE_RUNNER_PATH, "progressive_runner")
    smiles_mod = load_module(SMILES_AB_PATH, "smiles_ab_runner")
    ml_builder = load_module(ML_STRONG_SMILES_BUILDER_PATH, "ml_strong_smiles_builder")

    features_path = EXACT_ROOT / "features_slim_exact_repo.parquet"
    y_path = EXACT_ROOT / "y_train_exact_repo.npy"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing exact slim features: {features_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing exact slim labels: {y_path}")

    features = pd.read_parquet(features_path)
    y = np.load(y_path).astype(np.float32)

    keys = features[["sample_id", "canonical_drug_id"]].copy()
    keys["sample_id"] = keys["sample_id"].astype(str)
    keys["canonical_drug_id"] = keys["canonical_drug_id"].astype(str)

    numeric_df = features.select_dtypes(include=[np.number]).copy().fillna(0.0)
    numeric_names = numeric_df.columns.tolist()

    ctx_df, cat_cols, vocab_map, context_summary = progressive.build_reconstructed_context(keys)
    strong_cols = [col for col in progressive.STRONG_CONTEXT_COLS if col in ctx_df.columns]
    strong_df = pd.concat([keys.reset_index(drop=True), ctx_df[strong_cols].reset_index(drop=True)], axis=1)

    strong_codes = smiles_mod.encode_context(ctx_df, strong_cols, vocab_map)
    strong_vocab = {col: vocab_map[col] for col in strong_cols}

    smiles_col = (
        features["drug__canonical_smiles_raw"]
        .astype("string")
        .fillna(features["drug__smiles"].astype("string"))
        .fillna("")
    )
    smiles_ids, smiles_vocab, observed_max_len = smiles_mod.build_smiles_tensor(smiles_col, max_len=256)

    drug_smiles = (
        features[["canonical_drug_id", "drug__canonical_smiles_raw", "drug__smiles"]]
        .drop_duplicates(subset=["canonical_drug_id"])
        .reset_index(drop=True)
    )
    smiles_svd_df, smiles_svd_names, smiles_svd_summary = ml_builder.build_smiles_svd(drug_smiles, dim=64)
    smiles_row_df = keys[["canonical_drug_id"]].merge(smiles_svd_df, on="canonical_drug_id", how="left")
    smiles_svd_arr = smiles_row_df[smiles_svd_names].values.astype(np.float32)

    context_onehot_arr, context_onehot_names, _ = ml_builder.build_onehot_context(ctx_df, strong_cols)

    x_ml_smiles = np.concatenate(
        [numeric_df.values.astype(np.float32), smiles_svd_arr],
        axis=1,
    ).astype(np.float32)
    x_ml_strong_context_smiles = np.concatenate(
        [
            numeric_df.values.astype(np.float32),
            context_onehot_arr.astype(np.float32),
            smiles_svd_arr,
        ],
        axis=1,
    ).astype(np.float32)

    outputs = {
        "keys": OUT_ROOT / "keys_exact_slim.parquet",
        "y": OUT_ROOT / "y_exact_slim.npy",
        "numeric_feature_names": OUT_ROOT / "numeric_feature_names.json",
        "reconstructed_context_full": OUT_ROOT / "reconstructed_context_full.parquet",
        "strong_context_only": OUT_ROOT / "strong_context_only.parquet",
        "strong_context_vocab": OUT_ROOT / "strong_context_vocab.json",
        "strong_context_codes": OUT_ROOT / "strong_context_codes.npy",
        "smiles_token_ids": OUT_ROOT / "smiles_token_ids.npy",
        "smiles_vocab": OUT_ROOT / "smiles_vocab.json",
        "ml_smiles_X": OUT_ROOT / "X_ml_exact_slim_smiles.npy",
        "ml_smiles_feature_names": OUT_ROOT / "feature_names_ml_exact_slim_smiles.json",
        "ml_strong_smiles_X": OUT_ROOT / "X_ml_exact_slim_strong_context_smiles.npy",
        "ml_strong_smiles_feature_names": OUT_ROOT / "feature_names_ml_exact_slim_strong_context_smiles.json",
        "summary": OUT_ROOT / "context_smiles_bundle_summary.json",
    }

    keys.to_parquet(outputs["keys"], index=False)
    np.save(outputs["y"], y)
    outputs["numeric_feature_names"].write_text(json.dumps(numeric_names, indent=2))
    ctx_df.to_parquet(outputs["reconstructed_context_full"], index=False)
    strong_df.to_parquet(outputs["strong_context_only"], index=False)
    outputs["strong_context_vocab"].write_text(json.dumps(strong_vocab, indent=2, ensure_ascii=False))
    np.save(outputs["strong_context_codes"], strong_codes)
    np.save(outputs["smiles_token_ids"], smiles_ids)
    outputs["smiles_vocab"].write_text(json.dumps(smiles_vocab, indent=2, ensure_ascii=False))
    np.save(outputs["ml_smiles_X"], x_ml_smiles)
    outputs["ml_smiles_feature_names"].write_text(json.dumps(numeric_names + smiles_svd_names, indent=2))
    np.save(outputs["ml_strong_smiles_X"], x_ml_strong_context_smiles)
    outputs["ml_strong_smiles_feature_names"].write_text(
        json.dumps(numeric_names + context_onehot_names + smiles_svd_names, indent=2)
    )

    summary = {
        "starting_point": {
            "features_path": str(features_path),
            "y_path": str(y_path),
            "rows": int(len(features)),
            "numeric_feature_dim": int(len(numeric_names)),
        },
        "strong_context": {
            "columns": strong_cols,
            "full_categorical_columns": cat_cols,
            "context_summary": context_summary,
            "code_shape": list(strong_codes.shape),
            "onehot_dim_for_ml": int(len(context_onehot_names)),
        },
        "smiles": {
            "token_ids_shape": list(smiles_ids.shape),
            "smiles_vocab_size": int(len(smiles_vocab)),
            "observed_max_len": int(observed_max_len),
            "svd_dim": int(len(smiles_svd_names)),
            "svd_summary": smiles_svd_summary,
        },
        "ml_matrices": {
            "exact_slim_plus_smiles_shape": list(x_ml_smiles.shape),
            "exact_slim_plus_strong_context_plus_smiles_shape": list(x_ml_strong_context_smiles.shape),
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
        "notes": [
            "DL runs consume strong_context_codes.npy and smiles_token_ids.npy together with the exact slim numeric matrix.",
            "ML runs consume the prebuilt shared numeric matrices with either SMILES-only or strong-context-plus-SMILES features concatenated.",
            "This script reproduces dataset materialization only. Training/evaluation is handled by the run_exact_repo_slim_* scripts.",
        ],
    }
    outputs["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
