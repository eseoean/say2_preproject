#!/usr/bin/env python3
"""
Build a v3-like slim input table from the current common X cache.

Purpose
-------
- Start from the current merged common input that our models actually use.
- Reconstruct the teammate's "features_slim" idea in a local, reproducible way.
- Save a cleaned table, a slim table, labels, and a summary report.

Notes
-----
- This script follows the document spec as closely as possible from the local cache.
- Some published counts in the document are internally inconsistent, so this script
  reports the *actual* counts obtained from the current local data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COMMON_INPUT = ROOT / "tmp_schema" / "common_numeric_input.parquet"
OUT_ROOT = ROOT / "v3_input_reproduction"

MORGAN_VAR_THRESHOLD = 0.01
GENE_VAR_REMOVE_COUNT = 13719
CORR_THRESHOLD = 0.95

PATHWAY_COLS = [
    "target_pathway_score_mean",
    "target_pathway_hit_count",
]
TARGET_COLS = [
    "target_overlap_count",
    "target_overlap_ratio",
    "target_overlap_down_count",
    "target_overlap_down_ratio",
    "target_expr_mean",
    "target_expr_std",
    "target_gene_coverage_ratio",
    "target_gene_count",
]


def correlation_prune(frame: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, List[str]]:
    if frame.shape[1] <= 1:
        return frame.copy(), []
    corr = frame.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if (upper[col] > threshold).any()]
    kept = frame.drop(columns=drop_cols)
    return kept, drop_cols


def split_feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    gene_cols = [c for c in df.columns if c.startswith("sample__crispr__")]
    morgan_cols = [c for c in df.columns if c.startswith("drug_morgan_")]
    lincs_cols = [c for c in df.columns if c.startswith("lincs_")]
    descriptor_cols = [c for c in df.columns if c.startswith("drug_desc_")]
    misc_flag_cols = [c for c in ["drug__has_smiles", "drug_has_valid_smiles"] if c in df.columns]
    target_cols = [c for c in TARGET_COLS if c in df.columns]
    pathway_cols = [c for c in PATHWAY_COLS if c in df.columns]
    return {
        "gene": gene_cols,
        "morgan": morgan_cols,
        "lincs": lincs_cols,
        "descriptors": descriptor_cols,
        "misc_flags": misc_flag_cols,
        "target": target_cols,
        "pathway": pathway_cols,
    }


def build_doclike_tables(
    df: pd.DataFrame,
    corr_threshold: float,
    gene_remove_count: int,
    morgan_var_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    groups = split_feature_groups(df)
    keys = df[["sample_id", "canonical_drug_id"]].copy()

    gene = df[groups["gene"]].copy()
    morgan = df[groups["morgan"]].copy()
    lincs = df[groups["lincs"]].copy()
    target = df[groups["target"]].copy()
    pathway = df[groups["pathway"]].copy()
    descriptors = df[groups["descriptors"]].copy()
    misc_flags = df[groups["misc_flags"]].copy()

    cleaned = pd.concat(
        [keys, gene, morgan, lincs, target, pathway, descriptors, misc_flags],
        axis=1,
    )

    gene_var = gene.var(axis=0)
    gene_remove_count = min(gene_remove_count, gene.shape[1] - 1)
    gene_lowvar_drop = gene_var.sort_values(kind="mergesort").index[:gene_remove_count].tolist()
    gene_after_lowvar = gene.drop(columns=gene_lowvar_drop)
    gene_after_corr, gene_corr_drop = correlation_prune(gene_after_lowvar, corr_threshold)

    morgan_var = morgan.var(axis=0)
    morgan_lowvar_drop = morgan_var.index[morgan_var < morgan_var_threshold].tolist()
    morgan_after_lowvar = morgan.drop(columns=morgan_lowvar_drop)
    morgan_after_corr, morgan_corr_drop = correlation_prune(morgan_after_lowvar, corr_threshold)

    slim = pd.concat(
        [keys, gene_after_corr, morgan_after_corr, lincs, target, pathway, descriptors],
        axis=1,
    )

    summary = {
        "rows": int(len(df)),
        "unique_drugs": int(df["canonical_drug_id"].astype(str).nunique()),
        "cleaned_shape": list(cleaned.shape),
        "slim_shape": list(slim.shape),
        "group_counts_cleaned": {
            "gene": int(gene.shape[1]),
            "morgan": int(morgan.shape[1]),
            "lincs": int(lincs.shape[1]),
            "target": int(target.shape[1]),
            "pathway": int(pathway.shape[1]),
            "drug_descriptors": int(descriptors.shape[1]),
            "misc_flags": int(misc_flags.shape[1]),
            "id_cols": 2,
        },
        "group_counts_slim": {
            "gene": int(gene_after_corr.shape[1]),
            "morgan": int(morgan_after_corr.shape[1]),
            "lincs": int(lincs.shape[1]),
            "target": int(target.shape[1]),
            "pathway": int(pathway.shape[1]),
            "drug_descriptors": int(descriptors.shape[1]),
            "id_cols": 2,
        },
        "feature_selection": {
            "gene_low_variance_removed": int(len(gene_lowvar_drop)),
            "gene_corr_removed": int(len(gene_corr_drop)),
            "morgan_low_variance_removed": int(len(morgan_lowvar_drop)),
            "morgan_corr_removed": int(len(morgan_corr_drop)),
            "gene_low_variance_mode": "remove_lowest_variance_by_count",
            "gene_low_variance_remove_count": int(gene_remove_count),
            "morgan_low_variance_threshold": float(morgan_var_threshold),
            "corr_threshold": float(corr_threshold),
        },
        "doc_alignment_notes": [
            "Rows and unique drugs after removing invalid/all-zero Morgan drugs should match the document if the same source cache is used.",
            "Morgan low-variance stage is expected to align closely with the document at threshold 0.01.",
            "Gene correlation removal may differ from the document because the exact teammate script and hidden preprocessing order were not shared.",
        ],
    }
    return cleaned, slim, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(COMMON_INPUT))
    parser.add_argument("--outdir", default=str(OUT_ROOT))
    parser.add_argument("--corr-threshold", type=float, default=CORR_THRESHOLD)
    parser.add_argument("--gene-remove-count", type=int, default=GENE_VAR_REMOVE_COUNT)
    parser.add_argument("--morgan-var-threshold", type=float, default=MORGAN_VAR_THRESHOLD)
    args = parser.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)
    morgan_cols = [c for c in df.columns if c.startswith("drug_morgan_")]
    invalid_mask = (df["drug_has_valid_smiles"] <= 0) | (df[morgan_cols].abs().sum(axis=1) <= 0)
    filtered = df.loc[~invalid_mask].reset_index(drop=True)

    cleaned, slim, summary = build_doclike_tables(
        df=filtered,
        corr_threshold=args.corr_threshold,
        gene_remove_count=args.gene_remove_count,
        morgan_var_threshold=args.morgan_var_threshold,
    )

    cleaned_path = outdir / "features_cleaned_doclike.parquet"
    slim_path = outdir / "features_slim_doclike.parquet"
    y_path = outdir / "y_train.npy"
    summary_path = outdir / "build_summary.json"

    cleaned.to_parquet(cleaned_path, index=False)
    slim.to_parquet(slim_path, index=False)
    np.save(y_path, filtered["label_regression"].to_numpy(dtype=np.float32))
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"Saved cleaned table: {cleaned_path}")
    print(f"Saved slim table:    {slim_path}")
    print(f"Saved labels:        {y_path}")
    print(f"Saved summary:       {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
