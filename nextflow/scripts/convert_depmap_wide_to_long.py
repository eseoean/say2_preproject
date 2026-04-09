"""Convert DepMap CRISPR wide-format to long-format for FE pipeline.

Input:  depmap_crispr_gene_dependency_basic_clean (wide: ModelID + gene columns)
        depmap_model_basic_clean (ModelID -> CellLineName mapping)
Output: long-format parquet with columns [cell_line_name, gene_name, dependency]
"""
from __future__ import annotations

import argparse
import re

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert DepMap CRISPR wide→long format.")
    p.add_argument("--crispr-uri", required=True, help="DepMap CRISPR gene dependency wide parquet.")
    p.add_argument("--model-uri", required=True, help="DepMap model basic clean parquet (ModelID→CellLineName).")
    p.add_argument("--output-uri", required=True, help="Output long-format parquet.")
    return p.parse_args()


def parse_gene_col(col: str) -> str | None:
    """Parse gene name from column like 'A1BG (1)' → 'A1BG'."""
    m = re.match(r"^(.+?)\s*\(\d+\)$", col)
    if m:
        return m.group(1).strip()
    return None


def main() -> None:
    args = parse_args()

    print("Loading CRISPR data...")
    crispr = pd.read_parquet(args.crispr_uri)
    print(f"  Shape: {crispr.shape}")

    print("Loading model data...")
    model = pd.read_parquet(args.model_uri)
    print(f"  Model rows: {len(model)}")

    # Build ModelID → CellLineName map
    model_map = model[["ModelID", "CellLineName"]].dropna().drop_duplicates("ModelID")
    id_to_name = dict(zip(model_map["ModelID"], model_map["CellLineName"]))
    print(f"  ModelID→CellLineName mappings: {len(id_to_name)}")

    # Parse gene columns
    id_col = crispr.columns[0]  # "ModelID"
    gene_cols = [c for c in crispr.columns[1:] if parse_gene_col(c) is not None]
    print(f"  Gene columns: {len(gene_cols)}")

    # Melt wide → long
    print("Melting wide → long...")
    long = crispr[[id_col] + gene_cols].melt(
        id_vars=[id_col],
        var_name="gene_col",
        value_name="dependency",
    )
    print(f"  Long rows: {len(long):,}")

    # Parse gene names
    long["gene_name"] = long["gene_col"].apply(parse_gene_col)

    # Map ModelID → cell_line_name
    long["cell_line_name"] = long[id_col].map(id_to_name)
    n_mapped = long["cell_line_name"].notna().sum()
    n_unmapped = long["cell_line_name"].isna().sum()
    print(f"  Mapped: {n_mapped:,}, Unmapped: {n_unmapped:,}")

    # Drop unmapped and select final columns
    long = long[long["cell_line_name"].notna()].copy()
    long = long[["cell_line_name", "gene_name", "dependency"]]

    # Drop rows with NaN dependency
    n_before = len(long)
    long = long.dropna(subset=["dependency"])
    n_dropped = n_before - len(long)
    print(f"  Dropped NaN dependency: {n_dropped:,}")
    print(f"  Final rows: {len(long):,}")

    # Stats
    print(f"\n  Unique cell lines: {long['cell_line_name'].nunique()}")
    print(f"  Unique genes: {long['gene_name'].nunique()}")

    # Save
    long.to_parquet(args.output_uri, index=False)
    print(f"\nSaved: {args.output_uri}")


if __name__ == "__main__":
    main()
