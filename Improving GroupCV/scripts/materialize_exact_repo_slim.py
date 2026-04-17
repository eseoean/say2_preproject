#!/usr/bin/env python3
"""
Materialize the exact repo-provided v3 slim inputs into our workspace.

This does not try to reverse-engineer the teammate's hidden selection script.
Instead, it uses the actual artifacts committed in the reference repo and
stores verified copies plus a local validation summary.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


WORK_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = WORK_ROOT / "v3_input_reproduction" / "exact_repo_match"
REPO_ROOT = Path("/tmp/biso_myprotocol_repo/20260414_re_pre_project_v3")

REPO_SLIM = REPO_ROOT / "features_slim.parquet"
REPO_CLEANED = REPO_ROOT / "features_cleaned.parquet"
REPO_LOG = REPO_ROOT / "feature_selection_log.json"
REPO_Y = REPO_ROOT / "step4_results" / "y_train.npy"

CURRENT_COMMON = WORK_ROOT / "tmp_schema" / "common_numeric_input.parquet"


def group_counts(df: pd.DataFrame) -> dict[str, int]:
    return {
        "sample__crispr": int(sum(c.startswith("sample__crispr__") for c in df.columns)),
        "drug_morgan": int(sum(c.startswith("drug_morgan_") for c in df.columns)),
        "lincs": int(sum(c.startswith("lincs_") for c in df.columns)),
        "target": int(sum(c.startswith("target_") for c in df.columns)),
        "drug_desc": int(sum(c.startswith("drug_desc_") for c in df.columns)),
        "drug_meta_like": int(sum(c.startswith("drug__") or c == "drug_has_valid_smiles" for c in df.columns)),
        "id": int(sum(c in {"sample_id", "canonical_drug_id"} for c in df.columns)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not REPO_SLIM.exists():
        raise FileNotFoundError(f"Missing repo slim file: {REPO_SLIM}")
    if not REPO_CLEANED.exists():
        raise FileNotFoundError(f"Missing repo cleaned file: {REPO_CLEANED}")
    if not REPO_Y.exists():
        raise FileNotFoundError(f"Missing repo label file: {REPO_Y}")
    if not CURRENT_COMMON.exists():
        raise FileNotFoundError(f"Missing current common cache: {CURRENT_COMMON}")

    repo_slim = pd.read_parquet(REPO_SLIM)
    repo_cleaned = pd.read_parquet(REPO_CLEANED)
    current = pd.read_parquet(CURRENT_COMMON)

    morgan_cols = [c for c in current.columns if c.startswith("drug_morgan_")]
    valid_mask = (current["drug_has_valid_smiles"] > 0) & (current[morgan_cols].abs().sum(axis=1) > 0)
    current_filtered = current.loc[valid_mask].reset_index(drop=True)

    repo_keys = repo_slim[["sample_id", "canonical_drug_id"]].astype(str)
    current_keys = current_filtered[["sample_id", "canonical_drug_id"]].astype(str)

    repo_key_set = set(map(tuple, repo_keys.to_numpy()))
    current_key_set = set(map(tuple, current_keys.to_numpy()))

    repo_slim_numeric = repo_slim.select_dtypes(include="number")
    current_filtered_numeric = current_filtered.select_dtypes(include="number")

    slim_out = OUT_DIR / "features_slim_exact_repo.parquet"
    cleaned_out = OUT_DIR / "features_cleaned_reference_repo.parquet"
    y_out = OUT_DIR / "y_train_exact_repo.npy"
    x_out = OUT_DIR / "X_train_exact_repo_numeric.npy"
    summary_out = OUT_DIR / "exact_repo_match_summary.json"
    slim_cols_out = OUT_DIR / "features_slim_columns.txt"

    shutil.copy2(REPO_SLIM, slim_out)
    shutil.copy2(REPO_CLEANED, cleaned_out)
    shutil.copy2(REPO_Y, y_out)
    np.save(x_out, repo_slim_numeric.to_numpy(dtype=np.float32))
    slim_cols_out.write_text("\n".join(repo_slim.columns.tolist()) + "\n")

    feature_log = json.loads(REPO_LOG.read_text()) if REPO_LOG.exists() else {}

    summary = {
        "reference_repo_root": str(REPO_ROOT),
        "outputs": {
            "features_slim_exact_repo": str(slim_out),
            "features_cleaned_reference_repo": str(cleaned_out),
            "y_train_exact_repo": str(y_out),
            "X_train_exact_repo_numeric": str(x_out),
            "features_slim_columns_txt": str(slim_cols_out),
        },
        "repo_feature_selection_log": feature_log,
        "repo_slim_shape": list(repo_slim.shape),
        "repo_slim_numeric_shape": list(repo_slim_numeric.shape),
        "repo_cleaned_shape": list(repo_cleaned.shape),
        "repo_group_counts": group_counts(repo_slim),
        "current_filtered_shape": list(current_filtered.shape),
        "current_filtered_numeric_shape": list(current_filtered_numeric.shape),
        "key_overlap": {
            "repo_rows": int(len(repo_keys)),
            "current_filtered_rows": int(len(current_keys)),
            "overlap_exact": int(len(repo_key_set & current_key_set)),
            "repo_only_keys": int(len(repo_key_set - current_key_set)),
            "current_only_keys": int(len(current_key_set - repo_key_set)),
        },
        "schema_overlap_vs_current_filtered": {
            "repo_slim_cols": int(len(repo_slim.columns)),
            "current_filtered_cols": int(len(current_filtered.columns)),
            "common_cols": int(len(set(repo_slim.columns) & set(current_filtered.columns))),
            "repo_only_cols": int(len(set(repo_slim.columns) - set(current_filtered.columns))),
            "current_only_cols": int(len(set(current_filtered.columns) - set(repo_slim.columns))),
            "repo_only_cols_sample": sorted(list(set(repo_slim.columns) - set(current_filtered.columns)))[:50],
            "current_only_cols_sample": sorted(list(set(current_filtered.columns) - set(repo_slim.columns)))[:50],
        },
        "note": (
            "The exact repo slim is materialized locally for faithful reproduction. "
            "This is more accurate than inferring hidden selection steps from the document alone."
        ),
    }

    summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"Saved exact slim: {slim_out}")
    print(f"Saved reference cleaned: {cleaned_out}")
    print(f"Saved numeric X: {x_out}")
    print(f"Saved y: {y_out}")
    print(f"Saved summary: {summary_out}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
