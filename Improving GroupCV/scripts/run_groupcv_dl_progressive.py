#!/usr/bin/env python3
"""
Progressive GroupCV DL experiments.

Goal:
- Start from the current common numeric-only input.
- Add guideline-inspired categorical/context features step by step.
- Evaluate whether drug GroupCV generalization improves.

Current implemented variants:
- baseline_numeric
- context_categorical
- reconstructed_context_full
- role_split_context_full
- x_repacked_blocksvd
- x_repacked_reconstructed_context_full
- x_repacked_role_split_context_full
- strong_context_only
- x_repacked_strong_context_only

Models:
- ResidualMLP
- FlatMLP
- TabNet
- FTTransformer
- CrossAttention
- TabTransformer
- WideDeep
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
N_FOLDS = 3
BENCH_SP = 0.713
BENCH_RMSE = 1.385
CPU_THREADS = int(os.getenv("MODEL_CPU_THREADS", "4"))

S3_BASE = os.getenv("S3_BASE", "s3://say2-4team/20260409_eseo")
FE_RUN_ID = os.getenv("FE_RUN_ID", "20260409_newfe_v8_eseo")
FEATURES_URI = f"{S3_BASE}/fe_output/{FE_RUN_ID}/features/features.parquet"
PAIR_FEATURES_URI = f"{S3_BASE}/fe_output/{FE_RUN_ID}/pair_features/pair_features_newfe_v2.parquet"
LABELS_URI = f"{S3_BASE}/fe_output/{FE_RUN_ID}/features/labels.parquet"

SCRIPT_PATH = Path(__file__).resolve()
WORK_ROOT = SCRIPT_PATH.parents[1]
OUTPUT_ROOT = WORK_ROOT / "results"
CACHE_ROOT = WORK_ROOT / "tmp_schema"
OUTPUT_ROOT.mkdir(exist_ok=True)

ROLE_TABLE_CSV = CACHE_ROOT / "feature_role_table.csv"
VOCAB_CSV = CACHE_ROOT / "categorical_vocab_candidates.csv"
TRAIN_READY = CACHE_ROOT / "train_ready.parquet"
VALID_READY = CACHE_ROOT / "valid_ready.parquet"
TRAIN_META = CACHE_ROOT / "train_metadata.parquet"
VALID_META = CACHE_ROOT / "valid_metadata.parquet"
DRUG_FEATURE_CATALOG = CACHE_ROOT / "drug_features_catalog.parquet"
DRUG_TARGET_MAPPING = CACHE_ROOT / "drug_target_mapping.parquet"
GDSC_DRUG_ANN = CACHE_ROOT / "gdsc2_drug_annotation_master.parquet"
GDSC_CELLLINE_ANN = CACHE_ROOT / "gdsc2_cellline_annotation_table.parquet"
PAIR_FEATURE_CACHE = CACHE_ROOT / "pair_features_newfe_v2.parquet"
COMMON_NUMERIC_CACHE = CACHE_ROOT / "common_numeric_input.parquet"

PRIMARY_CONTEXT_COLS = [
    "TCGA_DESC",
    "PATHWAY_NAME_NORMALIZED",
    "classification",
]
AUXILIARY_CONTEXT_COLS = [
    "drug_bridge_strength",
    "stage3_resolution_status",
    "WEBRELEASE",
    "drugbank_match_rule",
    "chembl_match_rule",
    "lincs_match_rule",
    "admet_match_rule",
    "cell_bridge_match_rule",
]
DEFAULT_SAMPLE_SVD_COMPONENTS = 256
DEFAULT_DRUG_SVD_COMPONENTS = 96
STRONG_CONTEXT_COLS = [
    "TCGA_DESC",
    "PATHWAY_NAME_NORMALIZED",
    "classification",
    "drug_bridge_strength",
    "stage3_resolution_status",
]

CROSS_SAMPLE_DIM_LEGACY = 18311

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, str(CPU_THREADS))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(CPU_THREADS)

requested_device = os.getenv("DL_DEVICE", "").strip().lower()
if requested_device:
    DEVICE = torch.device(requested_device)
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


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


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_tr_true: np.ndarray | None = None,
    y_tr_pred: np.ndarray | None = None,
) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    sp = _safe_spearman(y_true, y_pred)
    pe = _safe_pearson(y_true, y_pred)
    r2 = float(r2_score(y_true, y_pred))
    ndcg20 = compute_ndcg_at_k(y_true, y_pred, k=20)
    out = {
        "rmse": rmse,
        "mae": mae,
        "spearman": sp,
        "pearson": pe,
        "r2": r2,
        "ndcg@20": ndcg20,
    }
    if y_tr_true is not None and y_tr_pred is not None:
        tr_sp = _safe_spearman(y_tr_true, y_tr_pred)
        tr_rmse = float(np.sqrt(mean_squared_error(y_tr_true, y_tr_pred)))
        tr_mae = float(mean_absolute_error(y_tr_true, y_tr_pred))
        out["train_spearman"] = tr_sp
        out["train_rmse"] = tr_rmse
        out["train_mae"] = tr_mae
        out["gap_spearman"] = tr_sp - sp
        out["gap_rmse"] = rmse - tr_rmse
        out["gap_mae"] = mae - tr_mae
    return out


def ensure_context_cache_exists() -> None:
    required = [ROLE_TABLE_CSV, VOCAB_CSV, TRAIN_READY, VALID_READY, TRAIN_META, VALID_META]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Context cache is incomplete. Missing files:\n- " + "\n- ".join(missing)
        )


def ensure_reconstructed_cache_exists() -> None:
    required = [
        ROLE_TABLE_CSV,
        VOCAB_CSV,
        DRUG_FEATURE_CATALOG,
        DRUG_TARGET_MAPPING,
        GDSC_DRUG_ANN,
        GDSC_CELLLINE_ANN,
        PAIR_FEATURE_CACHE,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Reconstructed context cache is incomplete. Missing files:\n- " + "\n- ".join(missing)
        )


def load_common_numeric_input() -> Dict[str, object]:
    t0 = time.time()
    if COMMON_NUMERIC_CACHE.exists():
        print("Loading current common input from local cache...")
        cached = pd.read_parquet(COMMON_NUMERIC_CACHE)
        keys = cached[["sample_id", "canonical_drug_id"]].copy()
        keys["canonical_drug_id"] = keys["canonical_drug_id"].astype(str)
        y = cached["label_regression"].values.astype(np.float32)
        numeric_df = cached.drop(columns=["sample_id", "canonical_drug_id", "label_regression"])
    else:
        print("Loading current common input from S3...")
        features = pd.read_parquet(FEATURES_URI)
        pair_features = pd.read_parquet(PAIR_FEATURES_URI)
        labels = pd.read_parquet(LABELS_URI)

        merged = features.merge(pair_features, on=["sample_id", "canonical_drug_id"], how="inner")
        labels = labels.set_index(["sample_id", "canonical_drug_id"])
        merged = merged.set_index(["sample_id", "canonical_drug_id"])
        labels = labels.loc[merged.index]

        numeric_df = merged.select_dtypes(include=[np.number]).fillna(0.0).copy()
        keys = merged.reset_index()[["sample_id", "canonical_drug_id"]].copy()
        keys["canonical_drug_id"] = keys["canonical_drug_id"].astype(str)
        y = labels["label_regression"].values.astype(np.float32)

        cache_df = pd.concat(
            [
                keys.reset_index(drop=True),
                numeric_df.reset_index(drop=True),
                pd.DataFrame({"label_regression": y}),
            ],
            axis=1,
        )
        cache_df.to_parquet(COMMON_NUMERIC_CACHE, index=False)
        print(f"  Saved local cache to {COMMON_NUMERIC_CACHE}")

    numeric_cols = list(numeric_df.columns)
    sample_dim_detected = sum(col.startswith("sample__crispr__") for col in numeric_cols)

    print(
        f"  Loaded: {numeric_df.shape[0]} rows x {numeric_df.shape[1]} numeric features "
        f"({time.time() - t0:.1f}s)"
    )
    print(f"  Detected sample feature width: {sample_dim_detected}")
    return {
        "keys": keys,
        "numeric_df": numeric_df,
        "numeric_cols": numeric_cols,
        "y": y,
        "sample_dim_detected": sample_dim_detected,
    }


def _dedupe_context_rows(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    for col in cat_cols:
        df[col] = df[col].astype("string").fillna("__MISSING__")
    df["DRUG_ID"] = df["DRUG_ID"].astype(str)

    if df.duplicated(["CELL_LINE_NAME", "DRUG_ID"]).any():
        agg = {col: "first" for col in cat_cols}
        df = (
            df.groupby(["CELL_LINE_NAME", "DRUG_ID"], dropna=False, as_index=False)
            .agg(agg)
            .reset_index(drop=True)
        )
    else:
        df = df.drop_duplicates(["CELL_LINE_NAME", "DRUG_ID"]).reset_index(drop=True)
    return df


def load_context_lookup() -> Tuple[pd.DataFrame, List[str], Dict[str, Dict[str, int]]]:
    ensure_context_cache_exists()

    role = pd.read_csv(ROLE_TABLE_CSV)
    vocab_df = pd.read_csv(VOCAB_CSV)
    cat_cols = role.loc[role["role"] == "categorical", "column_name"].tolist()

    train_ready = pd.read_parquet(TRAIN_READY, columns=cat_cols)
    valid_ready = pd.read_parquet(VALID_READY, columns=cat_cols)
    train_meta = pd.read_parquet(TRAIN_META, columns=["CELL_LINE_NAME", "DRUG_ID"])
    valid_meta = pd.read_parquet(VALID_META, columns=["CELL_LINE_NAME", "DRUG_ID"])

    ctx_train = pd.concat([train_meta.reset_index(drop=True), train_ready.reset_index(drop=True)], axis=1)
    ctx_valid = pd.concat([valid_meta.reset_index(drop=True), valid_ready.reset_index(drop=True)], axis=1)
    context_df = pd.concat([ctx_train, ctx_valid], ignore_index=True)
    context_df = _dedupe_context_rows(context_df, cat_cols)

    vocab_map: Dict[str, Dict[str, int]] = {}
    for col in cat_cols:
        values = (
            vocab_df.loc[vocab_df["column_name"] == col, "category_value"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        ordered = ["__MISSING__"]
        for val in values:
            if val != "__MISSING__":
                ordered.append(val)
        vocab_map[col] = {val: idx for idx, val in enumerate(ordered)}
    return context_df, cat_cols, vocab_map


def _is_ambiguous_target_token(token: str) -> bool:
    t = str(token).strip().upper()
    return t in {"", "<NA>", "NA", "NAN", "NONE", "__MISSING__"}


def _is_gene_like_target_token(token: str) -> bool:
    t = str(token).strip().upper()
    if _is_ambiguous_target_token(t):
        return False
    if " " in t or any(ch in t for ch in ["(", ")", "/", ",", "."]):
        return False
    if t in {"IIA", "IIB", "IIC", "IID", "III", "IV", "V", "VI"}:
        return False
    return bool(re.fullmatch(r"[A-Z0-9\-]+", t))


def _classification_from_targets(tokens: List[str]) -> str:
    cleaned = [str(t).strip() for t in tokens if str(t).strip()]
    if not cleaned:
        return "mixed_gene_and_ambiguous"
    if any(_is_ambiguous_target_token(t) for t in cleaned):
        return "mixed_gene_and_ambiguous"
    if all(_is_gene_like_target_token(t) for t in cleaned):
        return "all_tokens_gene_matched"
    return "mixed_gene_and_non_gene"


def _non_missing_rate(series: pd.Series) -> float:
    return float((series.astype(str) != "__MISSING__").mean())


def build_reconstructed_context(
    keys: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Dict[str, int]], Dict[str, object]]:
    ensure_reconstructed_cache_exists()

    role = pd.read_csv(ROLE_TABLE_CSV)
    vocab_df = pd.read_csv(VOCAB_CSV)
    cat_cols = role.loc[role["role"] == "categorical", "column_name"].tolist()

    cell_ann = pd.read_parquet(GDSC_CELLLINE_ANN)[["CELL_LINE_NAME", "TCGA_DESC"]].drop_duplicates("CELL_LINE_NAME")
    cell_ann = cell_ann.rename(columns={"CELL_LINE_NAME": "sample_id"})
    cell_ann["sample_id"] = cell_ann["sample_id"].astype(str)
    cell_ann["TCGA_DESC"] = cell_ann["TCGA_DESC"].astype("string").fillna("__MISSING__")
    cell_ann["cell_bridge_match_rule"] = "exact_cell_line_name"

    drug_ann = pd.read_parquet(GDSC_DRUG_ANN)[
        ["DRUG_ID", "PATHWAY_NAME_NORMALIZED", "PUTATIVE_TARGET_NORMALIZED"]
    ].copy()
    drug_ann["canonical_drug_id"] = drug_ann["DRUG_ID"].astype(str)
    drug_ann = drug_ann.drop(columns=["DRUG_ID"]).drop_duplicates("canonical_drug_id")

    catalog = pd.read_parquet(DRUG_FEATURE_CATALOG)[["DRUG_ID", "match_source", "has_smiles"]].copy()
    catalog["canonical_drug_id"] = catalog["DRUG_ID"].astype(str)
    catalog = catalog.drop(columns=["DRUG_ID"]).drop_duplicates("canonical_drug_id")

    target_map = pd.read_parquet(DRUG_TARGET_MAPPING).copy()
    target_map["canonical_drug_id"] = target_map["canonical_drug_id"].astype(str)
    target_class = (
        target_map.groupby("canonical_drug_id")["target_gene_symbol"]
        .apply(lambda s: _classification_from_targets(s.astype(str).tolist()))
        .rename("classification")
        .reset_index()
    )

    pair = pd.read_parquet(
        PAIR_FEATURE_CACHE,
        columns=[
            "canonical_drug_id",
            "drug_has_valid_smiles",
            "lincs_cosine",
            "lincs_pearson",
            "lincs_spearman",
            "lincs_reverse_score_top50",
            "lincs_reverse_score_top100",
            "target_gene_count",
        ],
    ).copy()
    pair["canonical_drug_id"] = pair["canonical_drug_id"].astype(str)
    pair_agg = pair.groupby("canonical_drug_id").agg(
        drug_has_valid_smiles=("drug_has_valid_smiles", "max"),
        lincs_cosine=("lincs_cosine", lambda s: float((s.abs() > 1e-12).any())),
        lincs_pearson=("lincs_pearson", lambda s: float((s.abs() > 1e-12).any())),
        lincs_spearman=("lincs_spearman", lambda s: float((s.abs() > 1e-12).any())),
        lincs_reverse_score_top50=("lincs_reverse_score_top50", lambda s: float((s.abs() > 1e-12).any())),
        lincs_reverse_score_top100=("lincs_reverse_score_top100", lambda s: float((s.abs() > 1e-12).any())),
        target_gene_count=("target_gene_count", "max"),
    ).reset_index()
    pair_agg["has_lincs_any"] = pair_agg[
        [
            "lincs_cosine",
            "lincs_pearson",
            "lincs_spearman",
            "lincs_reverse_score_top50",
            "lincs_reverse_score_top100",
        ]
    ].max(axis=1)
    pair_agg["has_target_any"] = (pair_agg["target_gene_count"] > 0).astype(int)

    drug_ctx = (
        drug_ann.merge(catalog, on="canonical_drug_id", how="left")
        .merge(target_class, on="canonical_drug_id", how="left")
        .merge(
            pair_agg[
                ["canonical_drug_id", "drug_has_valid_smiles", "has_lincs_any", "has_target_any"]
            ],
            on="canonical_drug_id",
            how="left",
        )
    )

    drug_ctx["PATHWAY_NAME_NORMALIZED"] = drug_ctx["PATHWAY_NAME_NORMALIZED"].astype("string").fillna("__MISSING__")
    drug_ctx["classification"] = drug_ctx["classification"].astype("string").fillna("mixed_gene_and_ambiguous")
    drug_ctx["match_source"] = drug_ctx["match_source"].astype("string").fillna("unmatched")
    drug_ctx["has_smiles"] = drug_ctx["has_smiles"].fillna(0).astype(int)
    drug_ctx["has_lincs_any"] = drug_ctx["has_lincs_any"].fillna(0).astype(int)
    drug_ctx["has_target_any"] = drug_ctx["has_target_any"].fillna(0).astype(int)

    source_count = (
        (drug_ctx["has_smiles"] > 0).astype(int)
        + drug_ctx["has_lincs_any"]
        + drug_ctx["has_target_any"]
    )
    drug_ctx["drug_bridge_strength"] = np.where(source_count >= 2, "multi_source", "single_source")
    drug_ctx["stage3_resolution_status"] = np.where(
        drug_ctx["classification"] == "all_tokens_gene_matched",
        "resolved_or_cleaned",
        "partial_gene_resolution_with_family_remaining",
    )
    drug_ctx["WEBRELEASE"] = "Y"

    drug_ctx["drugbank_match_rule"] = "__MISSING__"
    drug_ctx.loc[drug_ctx["match_source"] == "drugbank_name", "drugbank_match_rule"] = "exact_name_key"
    drug_ctx.loc[
        drug_ctx["match_source"].isin(["drugbank_synonym", "drugbank_fuzzy"]),
        "drugbank_match_rule",
    ] = "compact_name_key"

    drug_ctx["chembl_match_rule"] = "__MISSING__"
    drug_ctx.loc[drug_ctx["match_source"] == "chembl_norm", "chembl_match_rule"] = "exact_name_key"
    drug_ctx["lincs_match_rule"] = np.where(drug_ctx["has_lincs_any"] > 0, "exact_name_key", "__MISSING__")
    drug_ctx["admet_match_rule"] = "__MISSING__"

    drug_ctx = drug_ctx[
        [
            "canonical_drug_id",
            "PATHWAY_NAME_NORMALIZED",
            "classification",
            "drug_bridge_strength",
            "stage3_resolution_status",
            "WEBRELEASE",
            "drugbank_match_rule",
            "chembl_match_rule",
            "lincs_match_rule",
            "admet_match_rule",
        ]
    ].drop_duplicates("canonical_drug_id")

    merged = (
        keys.copy()
        .merge(cell_ann[["sample_id", "TCGA_DESC", "cell_bridge_match_rule"]], on="sample_id", how="left")
        .merge(drug_ctx, on="canonical_drug_id", how="left")
    )

    for col in cat_cols:
        merged[col] = merged[col].astype("string").fillna("__MISSING__")

    vocab_map: Dict[str, Dict[str, int]] = {}
    for col in cat_cols:
        values = (
            vocab_df.loc[vocab_df["column_name"] == col, "category_value"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        derived_values = merged[col].astype(str).drop_duplicates().tolist()
        ordered = ["__MISSING__"]
        for val in values + derived_values:
            if val != "__MISSING__" and val not in ordered:
                ordered.append(val)
        vocab_map[col] = {val: idx for idx, val in enumerate(ordered)}

    summary = {
        "variant": "reconstructed_context_full",
        "matched_rows": int(len(merged)),
        "unmatched_rows": 0,
        "match_rate": 1.0,
        "categorical_columns": cat_cols,
        "column_non_missing_rate": {col: _non_missing_rate(merged[col]) for col in cat_cols},
        "coverage_note": "Context reconstructed from current pipeline sources: cellline annotation, GDSC drug annotation, drug catalog, target mapping, and current pair feature availability.",
    }
    return merged, cat_cols, vocab_map, summary


def build_variant_dataset(variant: str, cross_sample_dim_mode: str) -> Dict[str, object]:
    base = load_common_numeric_input()
    keys = base["keys"]
    numeric_df = base["numeric_df"]
    y = base["y"]
    sample_dim_detected = int(base["sample_dim_detected"])
    sample_dim_used = (
        CROSS_SAMPLE_DIM_LEGACY if cross_sample_dim_mode == "legacy" else sample_dim_detected
    )

    if variant in {"baseline_numeric", "x_repacked_blocksvd"}:
        cat_cols: List[str] = []
        cat_codes = np.zeros((len(keys), 0), dtype=np.int64)
        aux_cols: List[str] = []
        aux_codes = np.zeros((len(keys), 0), dtype=np.int64)
        context_summary = {
            "variant": variant,
            "matched_rows": 0,
            "unmatched_rows": len(keys),
            "match_rate": 0.0,
            "categorical_columns": [],
            "auxiliary_columns": [],
        }
        if variant == "x_repacked_blocksvd":
            context_summary["numeric_transform_note"] = (
                "Current numeric X is re-packed by block: sample CRISPR and drug Morgan "
                "are compressed with train-fold SVD and row summary statistics; smaller "
                "drug/target/LINCS blocks pass through."
            )
        vocab_sizes: List[int] = []
        aux_vocab_sizes: List[int] = []
    elif variant == "context_categorical":
        context_df, cat_cols, vocab_map = load_context_lookup()
        merged = keys.merge(
            context_df,
            left_on=["sample_id", "canonical_drug_id"],
            right_on=["CELL_LINE_NAME", "DRUG_ID"],
            how="left",
            indicator=True,
        )

        for col in cat_cols:
            merged[col] = merged[col].astype("string").fillna("__MISSING__")

        cat_arrays = []
        vocab_sizes = []
        for col in cat_cols:
            col_map = vocab_map[col]
            encoded = merged[col].map(lambda x: col_map.get(str(x), 0)).astype(np.int64).values
            cat_arrays.append(encoded)
            vocab_sizes.append(len(col_map))
        cat_codes = np.stack(cat_arrays, axis=1) if cat_arrays else np.zeros((len(keys), 0), dtype=np.int64)
        aux_cols = []
        aux_codes = np.zeros((len(keys), 0), dtype=np.int64)
        aux_vocab_sizes = []

        context_summary = {
            "variant": variant,
            "matched_rows": int((merged["_merge"] == "both").sum()),
            "unmatched_rows": int((merged["_merge"] != "both").sum()),
            "match_rate": float((merged["_merge"] == "both").mean()),
            "categorical_columns": cat_cols,
            "auxiliary_columns": [],
            "coverage_note": "Context rows are matched from hybrid sample-aware cache to current common input by (CELL_LINE_NAME/sample_id, DRUG_ID/canonical_drug_id).",
        }
    elif variant in {
        "reconstructed_context_full",
        "x_repacked_reconstructed_context_full",
        "strong_context_only",
        "x_repacked_strong_context_only",
    }:
        merged, cat_cols, vocab_map, context_summary = build_reconstructed_context(keys)
        if variant in {"strong_context_only", "x_repacked_strong_context_only"}:
            cat_cols = [col for col in STRONG_CONTEXT_COLS if col in cat_cols]
            context_summary = dict(context_summary)
            context_summary["variant"] = variant
            context_summary["categorical_columns"] = cat_cols
            context_summary["auxiliary_columns"] = []
            context_summary["strong_context_note"] = (
                "Only higher-signal reconstructed context columns are retained. "
                "Low-coverage or near-constant matching-rule columns are excluded."
            )
        cat_arrays = []
        vocab_sizes = []
        for col in cat_cols:
            col_map = vocab_map[col]
            encoded = merged[col].map(lambda x: col_map.get(str(x), 0)).astype(np.int64).values
            cat_arrays.append(encoded)
            vocab_sizes.append(len(col_map))
        cat_codes = np.stack(cat_arrays, axis=1) if cat_arrays else np.zeros((len(keys), 0), dtype=np.int64)
        aux_cols = []
        aux_codes = np.zeros((len(keys), 0), dtype=np.int64)
        aux_vocab_sizes = []
        if variant in {
            "x_repacked_reconstructed_context_full",
            "x_repacked_strong_context_only",
        }:
            context_summary = dict(context_summary)
            context_summary["variant"] = variant
            context_summary["numeric_transform_note"] = (
                "Current numeric X is re-packed by block: sample CRISPR and drug Morgan "
                "are compressed with train-fold SVD and row summary statistics; smaller "
                "drug/target/LINCS blocks pass through."
            )
    elif variant in {"role_split_context_full", "x_repacked_role_split_context_full"}:
        merged, cat_cols_all, vocab_map, context_summary = build_reconstructed_context(keys)
        cat_cols = [col for col in PRIMARY_CONTEXT_COLS if col in cat_cols_all]
        aux_cols = [col for col in AUXILIARY_CONTEXT_COLS if col in cat_cols_all]

        cat_arrays = []
        vocab_sizes = []
        for col in cat_cols:
            col_map = vocab_map[col]
            encoded = merged[col].map(lambda x: col_map.get(str(x), 0)).astype(np.int64).values
            cat_arrays.append(encoded)
            vocab_sizes.append(len(col_map))
        cat_codes = np.stack(cat_arrays, axis=1) if cat_arrays else np.zeros((len(keys), 0), dtype=np.int64)

        aux_arrays = []
        aux_vocab_sizes = []
        for col in aux_cols:
            col_map = vocab_map[col]
            encoded = merged[col].map(lambda x: col_map.get(str(x), 0)).astype(np.int64).values
            aux_arrays.append(encoded)
            aux_vocab_sizes.append(len(col_map))
        aux_codes = np.stack(aux_arrays, axis=1) if aux_arrays else np.zeros((len(keys), 0), dtype=np.int64)

        context_summary = dict(context_summary)
        context_summary["variant"] = variant
        context_summary["categorical_columns"] = cat_cols
        context_summary["auxiliary_columns"] = aux_cols
        context_summary["role_split_note"] = "Primary semantic categorical columns are modeled separately from auxiliary context and matching-quality columns."
        if variant == "x_repacked_role_split_context_full":
            context_summary["numeric_transform_note"] = (
                "Current numeric X is re-packed by block: sample CRISPR and drug Morgan "
                "are compressed with train-fold SVD and row summary statistics; smaller "
                "drug/target/LINCS blocks pass through."
            )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return {
        "variant": variant,
        "keys": keys,
        "y": y,
        "numeric": numeric_df.values.astype(np.float32),
        "numeric_dim": numeric_df.shape[1],
        "numeric_cols": list(numeric_df.columns),
        "sample_dim_detected": sample_dim_detected,
        "sample_dim_used": sample_dim_used,
        "cat_codes": cat_codes,
        "cat_cols": cat_cols,
        "cat_vocab_sizes": vocab_sizes,
        "aux_codes": aux_codes,
        "aux_cols": aux_cols,
        "aux_vocab_sizes": aux_vocab_sizes,
        "context_summary": context_summary,
    }


def _select_idx(cols: List[str], predicate) -> np.ndarray:
    return np.array([i for i, col in enumerate(cols) if predicate(col)], dtype=np.int64)


def _row_summary(block: np.ndarray) -> np.ndarray:
    if block.shape[1] == 0:
        return np.zeros((block.shape[0], 0), dtype=np.float32)
    nonzero_ratio = (np.abs(block) > 1e-12).mean(axis=1, keepdims=True)
    return np.concatenate(
        [
            block.mean(axis=1, keepdims=True),
            block.std(axis=1, keepdims=True),
            np.abs(block).mean(axis=1, keepdims=True),
            np.linalg.norm(block, axis=1, keepdims=True),
            nonzero_ratio,
        ],
        axis=1,
    ).astype(np.float32)


def _fit_transform_svd(
    x_tr: np.ndarray,
    x_val: np.ndarray,
    n_components: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    max_comp = min(n_components, x_tr.shape[0] - 1, x_tr.shape[1] - 1)
    if max_comp < 2:
        return x_tr.astype(np.float32), x_val.astype(np.float32)
    svd = TruncatedSVD(n_components=max_comp, random_state=seed)
    tr = svd.fit_transform(x_tr).astype(np.float32)
    val = svd.transform(x_val).astype(np.float32)
    return tr, val


def transform_numeric_blocks(
    variant: str,
    x_num_tr: np.ndarray,
    x_num_val: np.ndarray,
    numeric_cols: List[str],
    seed: int,
    sample_svd_components: int,
    drug_svd_components: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    if variant not in {
        "x_repacked_blocksvd",
        "x_repacked_reconstructed_context_full",
        "x_repacked_role_split_context_full",
        "x_repacked_strong_context_only",
    }:
        return x_num_tr, x_num_val, {"numeric_transform_dim": int(x_num_tr.shape[1])}

    sample_idx = _select_idx(numeric_cols, lambda c: c.startswith("sample__crispr__"))
    drug_morgan_idx = _select_idx(numeric_cols, lambda c: c.startswith("drug_morgan_"))
    small_idx = _select_idx(
        numeric_cols,
        lambda c: (
            c.startswith("drug_desc_")
            or c.startswith("drug__")
            or c == "drug_has_valid_smiles"
            or c.startswith("target_")
            or c.startswith("lincs_")
        ),
    )
    used = set(sample_idx.tolist()) | set(drug_morgan_idx.tolist()) | set(small_idx.tolist())
    other_idx = np.array([i for i in range(len(numeric_cols)) if i not in used], dtype=np.int64)

    pieces_tr: List[np.ndarray] = []
    pieces_val: List[np.ndarray] = []
    dims: Dict[str, int] = {}

    if len(sample_idx) > 0:
        tr_sample = x_num_tr[:, sample_idx]
        val_sample = x_num_val[:, sample_idx]
        tr_sample_svd, val_sample_svd = _fit_transform_svd(
            tr_sample,
            val_sample,
            n_components=sample_svd_components,
            seed=seed,
        )
        tr_sample_sum = _row_summary(tr_sample)
        val_sample_sum = _row_summary(val_sample)
        pieces_tr.extend([tr_sample_svd, tr_sample_sum])
        pieces_val.extend([val_sample_svd, val_sample_sum])
        dims["sample_svd_dim"] = int(tr_sample_svd.shape[1])
        dims["sample_summary_dim"] = int(tr_sample_sum.shape[1])

    if len(drug_morgan_idx) > 0:
        tr_drug = x_num_tr[:, drug_morgan_idx]
        val_drug = x_num_val[:, drug_morgan_idx]
        tr_drug_svd, val_drug_svd = _fit_transform_svd(
            tr_drug,
            val_drug,
            n_components=drug_svd_components,
            seed=seed + 97,
        )
        tr_drug_sum = _row_summary(tr_drug)
        val_drug_sum = _row_summary(val_drug)
        pieces_tr.extend([tr_drug_svd, tr_drug_sum])
        pieces_val.extend([val_drug_svd, val_drug_sum])
        dims["drug_svd_dim"] = int(tr_drug_svd.shape[1])
        dims["drug_summary_dim"] = int(tr_drug_sum.shape[1])

    passthrough_idx = np.concatenate([small_idx, other_idx]) if (len(small_idx) + len(other_idx)) > 0 else np.array([], dtype=np.int64)
    if len(passthrough_idx) > 0:
        tr_pass = x_num_tr[:, passthrough_idx].astype(np.float32)
        val_pass = x_num_val[:, passthrough_idx].astype(np.float32)
        pieces_tr.append(tr_pass)
        pieces_val.append(val_pass)
        dims["passthrough_dim"] = int(tr_pass.shape[1])

    x_tr_out = np.concatenate(pieces_tr, axis=1).astype(np.float32)
    x_val_out = np.concatenate(pieces_val, axis=1).astype(np.float32)
    dims["numeric_transform_dim"] = int(x_tr_out.shape[1])
    dims["sample_svd_requested"] = int(sample_svd_components)
    dims["drug_svd_requested"] = int(drug_svd_components)
    return x_tr_out, x_val_out, dims


class CatTokenEncoder(nn.Module):
    def __init__(self, vocab_sizes: List[int], d_token: int):
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.d_token = d_token
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, d_token) for vocab_size in vocab_sizes]
        )

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        if len(self.embeddings) == 0:
            return torch.zeros((x_cat.size(0), 0, self.d_token), device=x_cat.device)
        tokens = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.stack(tokens, dim=1)


class AuxContextEncoder(nn.Module):
    def __init__(self, vocab_sizes: List[int], d_token: int, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        self.token_encoder = CatTokenEncoder(vocab_sizes, d_token=d_token)
        in_dim = len(vocab_sizes) * d_token
        if in_dim > 0:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
            )
        else:
            self.proj = None

    def forward(self, x_aux: torch.Tensor) -> torch.Tensor:
        batch = x_aux.size(0)
        if self.proj is None:
            return torch.zeros((batch, self.out_dim), device=x_aux.device)
        aux_flat = self.token_encoder(x_aux).reshape(batch, -1)
        return self.proj(aux_flat)


class NumericChunkTokenizer(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 128, n_tokens: int = 64):
        super().__init__()
        self.n_tokens = n_tokens
        self.chunk_size = math.ceil(in_dim / n_tokens)
        self.proj = nn.Linear(self.chunk_size, d_model)

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        batch = x_num.size(0)
        total = self.n_tokens * self.chunk_size
        pad_len = total - x_num.size(1)
        if pad_len > 0:
            x_num = F.pad(x_num, (0, pad_len))
        x_num = x_num.view(batch, self.n_tokens, self.chunk_size)
        return self.proj(x_num)


class ResidualMLPModel(nn.Module):
    def __init__(
        self,
        num_dim: int,
        vocab_sizes: List[int],
        aux_vocab_sizes: List[int],
        hidden: int = 512,
        n_blocks: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.cat_encoder = CatTokenEncoder(vocab_sizes, d_token=16)
        self.aux_encoder = AuxContextEncoder(aux_vocab_sizes, d_token=8, out_dim=hidden)
        in_dim = num_dim + len(vocab_sizes) * 16
        self.input_proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(
                nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden),
                    nn.Dropout(dropout),
                )
            )
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, x_aux: torch.Tensor) -> torch.Tensor:
        cat_flat = self.cat_encoder(x_cat).reshape(x_num.size(0), -1)
        x = torch.cat([x_num, cat_flat], dim=1)
        h = self.input_proj(x) + self.aux_encoder(x_aux)
        for block in self.blocks:
            h = h + block(h)
        return self.head(h).squeeze(-1)


class FlatMLPModel(nn.Module):
    def __init__(
        self,
        num_dim: int,
        vocab_sizes: List[int],
        aux_vocab_sizes: List[int],
        layers: List[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = layers or [1024, 512, 256]
        self.cat_encoder = CatTokenEncoder(vocab_sizes, d_token=16)
        self.aux_encoder = AuxContextEncoder(aux_vocab_sizes, d_token=8, out_dim=64)
        in_dim = num_dim + len(vocab_sizes) * 16 + 64
        mods: List[nn.Module] = []
        prev = in_dim
        for hidden in layers:
            mods += [nn.Linear(prev, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout)]
            prev = hidden
        mods.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*mods)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, x_aux: torch.Tensor) -> torch.Tensor:
        cat_flat = self.cat_encoder(x_cat).reshape(x_num.size(0), -1)
        aux_state = self.aux_encoder(x_aux)
        x = torch.cat([x_num, cat_flat, aux_state], dim=1)
        return self.net(x).squeeze(-1)


class TabNetModel(nn.Module):
    def __init__(
        self,
        num_dim: int,
        vocab_sizes: List[int],
        aux_vocab_sizes: List[int],
        n_steps: int = 3,
        hidden: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.cat_encoder = CatTokenEncoder(vocab_sizes, d_token=16)
        self.aux_encoder = AuxContextEncoder(aux_vocab_sizes, d_token=8, out_dim=num_dim + len(vocab_sizes) * 16)
        in_dim = num_dim + len(vocab_sizes) * 16
        self.bn = nn.BatchNorm1d(in_dim)
        self.steps = nn.ModuleList()
        for _ in range(n_steps):
            self.steps.append(
                nn.ModuleDict(
                    {
                        "attn": nn.Sequential(
                            nn.Linear(in_dim, hidden),
                            nn.GELU(),
                            nn.Linear(hidden, in_dim),
                            nn.Sigmoid(),
                        ),
                        "fc": nn.Sequential(
                            nn.Linear(in_dim, hidden),
                            nn.BatchNorm1d(hidden),
                            nn.GELU(),
                            nn.Dropout(dropout),
                        ),
                    }
                )
            )
        self.head = nn.Linear(hidden * n_steps, 1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, x_aux: torch.Tensor) -> torch.Tensor:
        cat_flat = self.cat_encoder(x_cat).reshape(x_num.size(0), -1)
        x = torch.cat([x_num, cat_flat], dim=1)
        aux_gate = torch.sigmoid(self.aux_encoder(x_aux))
        x = self.bn(x * (1.0 + aux_gate))
        outs = []
        for step in self.steps:
            mask = step["attn"](x)
            outs.append(step["fc"](x * mask))
        return self.head(torch.cat(outs, dim=1)).squeeze(-1)


class FTTransformerModel(nn.Module):
    def __init__(
        self,
        num_dim: int,
        vocab_sizes: List[int],
        aux_vocab_sizes: List[int],
        d_model: int = 128,
        nhead: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_tokenizer = NumericChunkTokenizer(num_dim, d_model=d_model, n_tokens=64)
        self.cat_encoder = CatTokenEncoder(vocab_sizes, d_token=d_model)
        self.aux_encoder = AuxContextEncoder(aux_vocab_sizes, d_token=16, out_dim=d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, 1))

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, x_aux: torch.Tensor) -> torch.Tensor:
        num_tokens = self.num_tokenizer(x_num)
        cat_tokens = self.cat_encoder(x_cat)
        tokens = torch.cat([num_tokens, cat_tokens], dim=1)
        cls = self.cls_token.expand(x_num.size(0), -1, -1)
        h = self.transformer(torch.cat([cls, tokens], dim=1))
        aux_state = self.aux_encoder(x_aux)
        fused = torch.cat([h[:, 0], aux_state], dim=1)
        return self.head(fused).squeeze(-1)


class CrossAttentionModel(nn.Module):
    def __init__(
        self,
        num_dim: int,
        vocab_sizes: List[int],
        aux_vocab_sizes: List[int],
        sample_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.sample_dim = max(1, min(sample_dim, num_dim - 1))
        self.cat_encoder = CatTokenEncoder(vocab_sizes, d_token=16)
        self.aux_encoder = AuxContextEncoder(aux_vocab_sizes, d_token=8, out_dim=d_model)
        context_dim = (num_dim - self.sample_dim) + len(vocab_sizes) * 16
        self.sample_proj = nn.Sequential(nn.Linear(self.sample_dim, d_model), nn.GELU())
        self.context_proj = nn.Sequential(nn.Linear(context_dim, d_model), nn.GELU())
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, x_aux: torch.Tensor) -> torch.Tensor:
        sample_x = x_num[:, : self.sample_dim]
        context_x = x_num[:, self.sample_dim :]
        cat_flat = self.cat_encoder(x_cat).reshape(x_num.size(0), -1)
        context = torch.cat([context_x, cat_flat], dim=1)
        sample_tok = self.sample_proj(sample_x).unsqueeze(1)
        context_tok = self.context_proj(context).unsqueeze(1)
        attn_out, _ = self.cross_attn(sample_tok, context_tok, context_tok)
        aux_state = self.aux_encoder(x_aux)
        combined = torch.cat([attn_out.squeeze(1), sample_tok.squeeze(1), aux_state], dim=1)
        return self.ffn(combined).squeeze(-1)


class TabTransformerModel(nn.Module):
    def __init__(
        self,
        num_dim: int,
        vocab_sizes: List[int],
        aux_vocab_sizes: List[int],
        d_model: int = 128,
        nhead: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_proj = nn.Linear(num_dim, d_model)
        self.cat_encoder = CatTokenEncoder(vocab_sizes, d_token=d_model)
        self.aux_encoder = AuxContextEncoder(aux_vocab_sizes, d_token=16, out_dim=d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, 1))

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, x_aux: torch.Tensor) -> torch.Tensor:
        num_token = self.num_proj(x_num).unsqueeze(1)
        cat_tokens = self.cat_encoder(x_cat)
        tokens = torch.cat([num_token, cat_tokens], dim=1)
        cls = self.cls_token.expand(x_num.size(0), -1, -1)
        h = self.transformer(torch.cat([cls, tokens], dim=1))
        aux_state = self.aux_encoder(x_aux)
        fused = torch.cat([h[:, 0], aux_state], dim=1)
        return self.head(fused).squeeze(-1)


class WideDeepModel(nn.Module):
    def __init__(
        self,
        num_dim: int,
        vocab_sizes: List[int],
        aux_vocab_sizes: List[int],
        hidden: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.cat_encoder = CatTokenEncoder(vocab_sizes, d_token=16)
        self.aux_encoder = AuxContextEncoder(aux_vocab_sizes, d_token=8, out_dim=64)
        deep_dim = num_dim + len(vocab_sizes) * 16 + 64
        self.wide = nn.Linear(num_dim, 1)
        self.deep = nn.Sequential(
            nn.Linear(deep_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, x_aux: torch.Tensor) -> torch.Tensor:
        cat_flat = self.cat_encoder(x_cat).reshape(x_num.size(0), -1)
        aux_state = self.aux_encoder(x_aux)
        deep_x = torch.cat([x_num, cat_flat, aux_state], dim=1)
        return (self.wide(x_num) + self.deep(deep_x)).squeeze(-1)


def train_model(
    model: nn.Module,
    x_num_tr: np.ndarray,
    x_cat_tr: np.ndarray,
    x_aux_tr: np.ndarray,
    y_tr: np.ndarray,
    x_num_val: np.ndarray,
    x_cat_val: np.ndarray,
    x_aux_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    if DEVICE.type == "mps":
        torch.mps.empty_cache()

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = nn.MSELoss()

    x_num_tr_t = torch.tensor(x_num_tr, dtype=torch.float32, device=DEVICE)
    x_cat_tr_t = torch.tensor(x_cat_tr, dtype=torch.long, device=DEVICE)
    x_aux_tr_t = torch.tensor(x_aux_tr, dtype=torch.long, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
    x_num_val_t = torch.tensor(x_num_val, dtype=torch.float32, device=DEVICE)
    x_cat_val_t = torch.tensor(x_cat_val, dtype=torch.long, device=DEVICE)
    x_aux_val_t = torch.tensor(x_aux_val, dtype=torch.long, device=DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=DEVICE)

    train_ds = TensorDataset(x_num_tr_t, x_cat_tr_t, x_aux_tr_t, y_tr_t)
    drop_last = len(train_ds) >= batch_size
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)

    best_val = float("inf")
    best_state = None
    wait = 0

    for _ in range(epochs):
        model.train()
        for xb_num, xb_cat, xb_aux, yb in train_dl:
            optimizer.zero_grad()
            pred = model(xb_num, xb_cat, xb_aux)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_num_val_t, x_cat_val_t, x_aux_val_t).detach().cpu().numpy()
            val_loss = mean_squared_error(y_val, val_pred)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_val = model(x_num_val_t, x_cat_val_t, x_aux_val_t).detach().cpu().numpy()
        pred_tr = model(x_num_tr_t, x_cat_tr_t, x_aux_tr_t).detach().cpu().numpy()
    return pred_val, pred_tr


def build_model(
    name: str,
    num_dim: int,
    vocab_sizes: List[int],
    aux_vocab_sizes: List[int],
    sample_dim_used: int,
) -> Tuple[nn.Module, Dict[str, float]]:
    key = name.lower().replace("-", "").replace("_", "").replace("&", "")
    if key == "residualmlp":
        return ResidualMLPModel(num_dim, vocab_sizes, aux_vocab_sizes), {"epochs": 100, "lr": 1e-3, "batch_size": 256}
    if key == "flatmlp":
        return FlatMLPModel(num_dim, vocab_sizes, aux_vocab_sizes), {"epochs": 100, "lr": 1e-3, "batch_size": 256}
    if key == "tabnet":
        return TabNetModel(num_dim, vocab_sizes, aux_vocab_sizes), {"epochs": 100, "lr": 1e-3, "batch_size": 256}
    if key == "fttransformer":
        return FTTransformerModel(num_dim, vocab_sizes, aux_vocab_sizes), {"epochs": 80, "lr": 5e-4, "batch_size": 128}
    if key == "crossattention":
        return CrossAttentionModel(num_dim, vocab_sizes, aux_vocab_sizes, sample_dim=sample_dim_used), {"epochs": 80, "lr": 5e-4, "batch_size": 256}
    if key == "tabtransformer":
        return TabTransformerModel(num_dim, vocab_sizes, aux_vocab_sizes), {"epochs": 80, "lr": 5e-4, "batch_size": 128}
    if key == "widedeep":
        return WideDeepModel(num_dim, vocab_sizes, aux_vocab_sizes), {"epochs": 100, "lr": 1e-3, "batch_size": 256}
    raise ValueError(f"Unsupported model: {name}")


def run_groupcv(
    variant: str,
    model_names: List[str],
    max_folds: int,
    prepare_only: bool,
    output_stem: str,
    cross_sample_dim_mode: str,
    sample_svd_components: int,
    drug_svd_components: int,
) -> Path:
    bundle = build_variant_dataset(variant=variant, cross_sample_dim_mode=cross_sample_dim_mode)
    keys = bundle["keys"]
    x_num = bundle["numeric"]
    numeric_cols = bundle["numeric_cols"]
    x_cat = bundle["cat_codes"]
    x_aux = bundle["aux_codes"]
    y = bundle["y"]
    groups = keys["canonical_drug_id"].astype(str).values

    summary = {
        "variant": variant,
        "device": str(DEVICE),
        "n_rows": int(len(y)),
        "numeric_dim": int(bundle["numeric_dim"]),
        "categorical_dim": int(x_cat.shape[1]),
        "auxiliary_dim": int(x_aux.shape[1]),
        "sample_dim_detected": int(bundle["sample_dim_detected"]),
        "sample_dim_used": int(bundle["sample_dim_used"]),
        "sample_svd_components": int(sample_svd_components),
        "drug_svd_components": int(drug_svd_components),
        "context_summary": bundle["context_summary"],
        "models": [],
    }

    out_path = OUTPUT_ROOT / f"{output_stem}.json"
    partial_path = OUTPUT_ROOT / f"{output_stem}.partial.json"

    if prepare_only:
        out_path.write_text(json.dumps(summary, indent=2, default=_json_default))
        print(f"Prepared dataset summary saved to {out_path}")
        return out_path

    splitter = GroupKFold(n_splits=N_FOLDS)
    split_iter = list(splitter.split(x_num, y, groups))
    if max_folds < len(split_iter):
        split_iter = split_iter[:max_folds]

    print("=" * 72)
    print(f"Variant: {variant}")
    print(
        f"Rows: {len(y)} | Numeric dim: {x_num.shape[1]} | "
        f"Cat dim: {x_cat.shape[1]} | Aux dim: {x_aux.shape[1]}"
    )
    print(f"Context match rate: {summary['context_summary'].get('match_rate', 0.0):.4f}")
    print(f"GroupCV folds: {len(split_iter)}")
    print("=" * 72)

    for model_name in model_names:
        model_t0 = time.time()
        fold_metrics = []
        print(f"\n{'-' * 60}")
        print(f"[{model_name}] {variant} drug GroupCV")
        print(f"{'-' * 60}")

        for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
            x_num_tr = x_num[train_idx]
            x_num_val = x_num[val_idx]
            x_cat_tr = x_cat[train_idx]
            x_cat_val = x_cat[val_idx]
            x_aux_tr = x_aux[train_idx]
            x_aux_val = x_aux[val_idx]
            y_tr = y[train_idx]
            y_val = y[val_idx]

            x_num_tr, x_num_val, numeric_transform_info = transform_numeric_blocks(
                variant=variant,
                x_num_tr=x_num_tr,
                x_num_val=x_num_val,
                numeric_cols=numeric_cols,
                seed=SEED + fold_idx,
                sample_svd_components=sample_svd_components,
                drug_svd_components=drug_svd_components,
            )

            scaler = StandardScaler()
            x_num_tr_s = scaler.fit_transform(x_num_tr).astype(np.float32)
            x_num_val_s = scaler.transform(x_num_val).astype(np.float32)

            torch.manual_seed(SEED + fold_idx)
            model, train_cfg = build_model(
                name=model_name,
                num_dim=x_num_tr_s.shape[1],
                vocab_sizes=bundle["cat_vocab_sizes"],
                aux_vocab_sizes=bundle["aux_vocab_sizes"],
                sample_dim_used=int(bundle["sample_dim_used"]),
            )
            pred_val, pred_tr = train_model(
                model=model,
                x_num_tr=x_num_tr_s,
                x_cat_tr=x_cat_tr,
                x_aux_tr=x_aux_tr,
                y_tr=y_tr,
                x_num_val=x_num_val_s,
                x_cat_val=x_cat_val,
                x_aux_val=x_aux_val,
                y_val=y_val,
                **train_cfg,
            )

            metrics = compute_metrics(y_val, pred_val, y_tr, pred_tr)
            metrics["fold"] = fold_idx
            metrics["n_train"] = int(len(train_idx))
            metrics["n_val"] = int(len(val_idx))
            metrics["n_train_groups"] = int(len(set(groups[train_idx])))
            metrics["n_val_groups"] = int(len(set(groups[val_idx])))
            metrics.update(numeric_transform_info)
            fold_metrics.append(metrics)
            print(
                f"  Fold {fold_idx}: Sp={metrics['spearman']:.4f}  "
                f"RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  "
                f"NDCG@20={metrics['ndcg@20']:.4f}  Gap(Sp)={metrics['gap_spearman']:.4f}"
            )

        df = pd.DataFrame(fold_metrics)
        result = {
            "model": model_name,
            "variant": variant,
            "spearman_mean": float(df["spearman"].mean()),
            "spearman_std": float(df["spearman"].std()),
            "rmse_mean": float(df["rmse"].mean()),
            "rmse_std": float(df["rmse"].std()),
            "mae_mean": float(df["mae"].mean()),
            "mae_std": float(df["mae"].std()),
            "ndcg@20_mean": float(df["ndcg@20"].mean()),
            "ndcg@20_std": float(df["ndcg@20"].std()),
            "pearson_mean": float(df["pearson"].mean()),
            "r2_mean": float(df["r2"].mean()),
            "r2_std": float(df["r2"].std()),
            "train_spearman_mean": float(df["train_spearman"].mean()),
            "gap_spearman_mean": float(df["gap_spearman"].mean()),
            "elapsed_sec": float(time.time() - model_t0),
            "folds": fold_metrics,
        }
        summary["models"].append(result)
        partial_path.write_text(json.dumps(summary, indent=2, default=_json_default))

        sp_flag = "PASS" if result["spearman_mean"] >= BENCH_SP else "FAIL"
        rm_flag = "PASS" if result["rmse_mean"] <= BENCH_RMSE else "FAIL"
        print(f"  >>> {model_name} SUMMARY")
        print(
            f"      Spearman: {result['spearman_mean']:.4f} +/- {result['spearman_std']:.4f} "
            f"[{sp_flag}]"
        )
        print(
            f"      RMSE:     {result['rmse_mean']:.4f} +/- {result['rmse_std']:.4f} "
            f"[{rm_flag}]"
        )
        print(f"      MAE:      {result['mae_mean']:.4f} +/- {result['mae_std']:.4f}")
        print(
            f"      NDCG@20:  {result['ndcg@20_mean']:.4f} +/- {result['ndcg@20_std']:.4f}"
        )
        print(f"      Train Sp: {result['train_spearman_mean']:.4f}")
        print(f"      Gap Sp:   {result['gap_spearman_mean']:.4f}")
        print(f"      Time:     {result['elapsed_sec'] / 60:.1f} min")

    out_path.write_text(json.dumps(summary, indent=2, default=_json_default))
    print(f"\nSaved results to {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        default="baseline_numeric",
        choices=[
            "baseline_numeric",
            "context_categorical",
            "reconstructed_context_full",
            "role_split_context_full",
            "x_repacked_blocksvd",
            "x_repacked_reconstructed_context_full",
            "x_repacked_role_split_context_full",
            "strong_context_only",
            "x_repacked_strong_context_only",
        ],
    )
    parser.add_argument(
        "--models",
        default="ResidualMLP,FlatMLP,TabNet,FTTransformer,CrossAttention,TabTransformer,WideDeep",
    )
    parser.add_argument("--max-folds", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output-stem", default=None)
    parser.add_argument(
        "--cross-sample-dim-mode",
        default="legacy",
        choices=["legacy", "detected"],
    )
    parser.add_argument("--sample-svd-components", type=int, default=DEFAULT_SAMPLE_SVD_COMPONENTS)
    parser.add_argument("--drug-svd-components", type=int, default=DEFAULT_DRUG_SVD_COMPONENTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    output_stem = args.output_stem or f"groupcv_progressive_{args.variant}"
    print(f"Using device: {DEVICE}")
    print(f"Models: {', '.join(model_names)}")
    run_groupcv(
        variant=args.variant,
        model_names=model_names,
        max_folds=args.max_folds,
        prepare_only=args.prepare_only,
        output_stem=output_stem,
        cross_sample_dim_mode=args.cross_sample_dim_mode,
        sample_svd_components=args.sample_svd_components,
        drug_svd_components=args.drug_svd_components,
    )


if __name__ == "__main__":
    main()
