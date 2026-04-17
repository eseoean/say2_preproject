#!/usr/bin/env python3
"""
Materialize a shared numeric matrix for ML experiments:

- exact slim numeric features
- drug-level SMILES TF-IDF + SVD embedding
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_exact_repo_slim_strong_context_smiles_ml_matrix import build_smiles_svd


SCRIPT_PATH = Path(__file__).resolve()
WORK_ROOT = SCRIPT_PATH.parents[1]
DATA_ROOT = WORK_ROOT / "v3_input_reproduction" / "exact_repo_match"
OUTPUT_ROOT = WORK_ROOT / "v3_input_reproduction" / "exact_repo_match_ml_smiles"
OUTPUT_ROOT.mkdir(exist_ok=True)


def main() -> None:
    features = pd.read_parquet(DATA_ROOT / "features_slim_exact_repo.parquet")
    y = np.load(DATA_ROOT / "y_train_exact_repo.npy").astype(np.float32)

    keys = features[["sample_id", "canonical_drug_id"]].copy()
    keys["sample_id"] = keys["sample_id"].astype(str)
    keys["canonical_drug_id"] = keys["canonical_drug_id"].astype(str)

    numeric_df = features.select_dtypes(include=[np.number]).copy().fillna(0.0)
    numeric_names = numeric_df.columns.tolist()

    drug_smiles = (
        features[["canonical_drug_id", "drug__canonical_smiles_raw", "drug__smiles"]]
        .drop_duplicates(subset=["canonical_drug_id"])
        .reset_index(drop=True)
    )
    smiles_df, smiles_names, smiles_summary = build_smiles_svd(drug_smiles, dim=64)
    smiles_row_df = keys[["canonical_drug_id"]].merge(smiles_df, on="canonical_drug_id", how="left")
    smiles_arr = smiles_row_df[smiles_names].values.astype(np.float32)

    X = np.concatenate([numeric_df.values.astype(np.float32), smiles_arr], axis=1).astype(np.float32)
    feature_names = numeric_names + smiles_names

    np.save(OUTPUT_ROOT / "X_ml_exact_slim_smiles.npy", X)
    np.save(OUTPUT_ROOT / "y_ml_exact_slim_smiles.npy", y)
    keys.to_parquet(OUTPUT_ROOT / "keys_ml_exact_slim_smiles.parquet", index=False)
    (OUTPUT_ROOT / "feature_names_ml_exact_slim_smiles.json").write_text(json.dumps(feature_names, indent=2))

    summary = {
        "input_features_path": str(DATA_ROOT / "features_slim_exact_repo.parquet"),
        "input_y_path": str(DATA_ROOT / "y_train_exact_repo.npy"),
        "output_X_path": str(OUTPUT_ROOT / "X_ml_exact_slim_smiles.npy"),
        "output_y_path": str(OUTPUT_ROOT / "y_ml_exact_slim_smiles.npy"),
        "output_keys_path": str(OUTPUT_ROOT / "keys_ml_exact_slim_smiles.parquet"),
        "rows": int(X.shape[0]),
        "feature_dim_total": int(X.shape[1]),
        "feature_dim_numeric_slim": int(len(numeric_names)),
        "feature_dim_smiles_svd": int(len(smiles_names)),
        "smiles_summary": smiles_summary,
    }
    (OUTPUT_ROOT / "matrix_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
