#!/usr/bin/env python3
"""
Materialize a shared numeric matrix for ML experiments:

- exact slim numeric features
- strong context one-hot features
- drug-level SMILES TF-IDF + SVD embedding
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder


SCRIPT_PATH = Path(__file__).resolve()
WORK_ROOT = SCRIPT_PATH.parents[1]
DATA_ROOT = WORK_ROOT / "v3_input_reproduction" / "exact_repo_match"
OUTPUT_ROOT = WORK_ROOT / "v3_input_reproduction" / "exact_repo_match_ml_strong_context_smiles"
OUTPUT_ROOT.mkdir(exist_ok=True)
PROGRESSIVE_RUNNER_PATH = SCRIPT_PATH.parent / "run_groupcv_dl_progressive.py"

SMILES_SVD_DIM = 64


def load_progressive_module():
    spec = importlib.util.spec_from_file_location("progressive_runner", PROGRESSIVE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load progressive runner from {PROGRESSIVE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_onehot_context(ctx_df: pd.DataFrame, cols: list[str]):
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    arr = encoder.fit_transform(ctx_df[cols].astype(str))
    names = encoder.get_feature_names_out(cols).tolist()
    return arr.astype(np.float32), names, encoder


def build_smiles_svd(drug_df: pd.DataFrame, dim: int = SMILES_SVD_DIM):
    smiles = (
        drug_df["drug__canonical_smiles_raw"]
        .astype("string")
        .fillna(drug_df["drug__smiles"].astype("string"))
        .fillna("")
        .astype(str)
        .tolist()
    )
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=1,
        lowercase=False,
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(smiles)
    max_dim = min(dim, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
    max_dim = max(1, max_dim)
    svd = TruncatedSVD(n_components=max_dim, random_state=42)
    emb = svd.fit_transform(tfidf).astype(np.float32)
    names = [f"smiles_svd_{i:03d}" for i in range(emb.shape[1])]
    out = drug_df[["canonical_drug_id"]].copy()
    for i, name in enumerate(names):
        out[name] = emb[:, i]
    summary = {
        "tfidf_shape": [int(tfidf.shape[0]), int(tfidf.shape[1])],
        "svd_dim": int(emb.shape[1]),
        "explained_variance_sum": float(svd.explained_variance_ratio_.sum()),
        "vectorizer_vocab_size": int(len(vectorizer.vocabulary_)),
    }
    return out, names, summary


def main() -> None:
    mod = load_progressive_module()

    features = pd.read_parquet(DATA_ROOT / "features_slim_exact_repo.parquet")
    y = np.load(DATA_ROOT / "y_train_exact_repo.npy").astype(np.float32)

    keys = features[["sample_id", "canonical_drug_id"]].copy()
    keys["sample_id"] = keys["sample_id"].astype(str)
    keys["canonical_drug_id"] = keys["canonical_drug_id"].astype(str)

    numeric_df = features.select_dtypes(include=[np.number]).copy().fillna(0.0)
    numeric_names = numeric_df.columns.tolist()

    ctx_df, _, _, context_summary = mod.build_reconstructed_context(keys)
    strong_cols = [col for col in mod.STRONG_CONTEXT_COLS if col in ctx_df.columns]
    context_arr, context_names, _ = build_onehot_context(ctx_df, strong_cols)

    drug_smiles = (
        features[
            ["canonical_drug_id", "drug__canonical_smiles_raw", "drug__smiles"]
        ]
        .drop_duplicates(subset=["canonical_drug_id"])
        .reset_index(drop=True)
    )
    smiles_df, smiles_names, smiles_summary = build_smiles_svd(drug_smiles, dim=SMILES_SVD_DIM)
    smiles_row_df = keys[["canonical_drug_id"]].merge(smiles_df, on="canonical_drug_id", how="left")
    smiles_arr = smiles_row_df[smiles_names].values.astype(np.float32)

    X = np.concatenate(
        [
            numeric_df.values.astype(np.float32),
            context_arr,
            smiles_arr,
        ],
        axis=1,
    ).astype(np.float32)
    feature_names = numeric_names + context_names + smiles_names

    np.save(OUTPUT_ROOT / "X_ml_exact_slim_strong_context_smiles.npy", X)
    np.save(OUTPUT_ROOT / "y_ml_exact_slim_strong_context_smiles.npy", y)
    keys.to_parquet(OUTPUT_ROOT / "keys_ml_exact_slim_strong_context_smiles.parquet", index=False)
    (OUTPUT_ROOT / "feature_names_ml_exact_slim_strong_context_smiles.json").write_text(
        json.dumps(feature_names, indent=2)
    )

    summary = {
        "input_features_path": str(DATA_ROOT / "features_slim_exact_repo.parquet"),
        "input_y_path": str(DATA_ROOT / "y_train_exact_repo.npy"),
        "output_X_path": str(OUTPUT_ROOT / "X_ml_exact_slim_strong_context_smiles.npy"),
        "output_y_path": str(OUTPUT_ROOT / "y_ml_exact_slim_strong_context_smiles.npy"),
        "output_keys_path": str(OUTPUT_ROOT / "keys_ml_exact_slim_strong_context_smiles.parquet"),
        "rows": int(X.shape[0]),
        "feature_dim_total": int(X.shape[1]),
        "feature_dim_numeric_slim": int(len(numeric_names)),
        "feature_dim_context_onehot": int(len(context_names)),
        "feature_dim_smiles_svd": int(len(smiles_names)),
        "strong_context_columns": strong_cols,
        "context_summary": context_summary,
        "smiles_summary": smiles_summary,
    }
    (OUTPUT_ROOT / "matrix_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
