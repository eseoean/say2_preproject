#!/usr/bin/env python3
"""Prepare gdsc_ic50.parquet for the FE pipeline.

Default behavior matches the reference protocol:
  - Use curated GDSC2 input
  - Keep BRCA rows
  - Optionally keep additional cell lines such as HCC1806
  - Rename columns to the schema expected by prepare_fe_inputs.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build gdsc_ic50.parquet for the FE pipeline.")
    p.add_argument("--input-uri", required=True, help="Curated GDSC parquet input.")
    p.add_argument("--output-uri", required=True, help="Output parquet path/URI.")
    p.add_argument("--out-report", default="", help="Optional QC report JSON path.")
    p.add_argument("--cancer-code", default="BRCA", help="Cancer code to keep from TCGA_DESC.")
    p.add_argument(
        "--extra-cell-lines",
        default="HCC1806",
        help="Comma-separated extra cell lines to keep even if TCGA_DESC differs.",
    )
    p.add_argument(
        "--gdsc-version",
        default="GDSC2",
        help="Value written into the gdsc_version column.",
    )
    return p.parse_args()


def _require_column(df: pd.DataFrame, names: list[str], label: str) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise ValueError(f"Missing required {label} column. Tried: {names}")


def _mkdir_parent(uri: str) -> None:
    if uri.startswith("s3://"):
        return
    Path(uri).parent.mkdir(parents=True, exist_ok=True)


def _write_parquet(df: pd.DataFrame, uri: str) -> None:
    if uri.startswith("s3://"):
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = tmp.name
        df.to_parquet(tmp_path, index=False)
        subprocess.run(["aws", "s3", "cp", tmp_path, uri], check=True)
        return
    df.to_parquet(uri, index=False)


def _write_json(obj: dict[str, object], uri: str) -> None:
    content = json.dumps(obj, ensure_ascii=False, indent=2)
    if uri.startswith("s3://"):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        subprocess.run(["aws", "s3", "cp", tmp_path, uri], check=True)
        return
    Path(uri).write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()

    print("Loading curated GDSC data...")
    df = pd.read_parquet(args.input_uri)
    print(f"  Shape: {df.shape}")

    tcga_col = _require_column(df, ["TCGA_DESC", "tcga_desc"], "TCGA_DESC")
    cell_col = _require_column(df, ["CELL_LINE_NAME", "cell_line_name"], "cell line")
    drug_id_col = _require_column(df, ["DRUG_ID", "drug_id"], "drug id")
    drug_name_col = _require_column(df, ["DRUG_NAME", "drug_name"], "drug name")
    ic50_col = _require_column(df, ["LN_IC50", "ln_IC50"], "ln_IC50")

    extra_cells = [c.strip() for c in str(args.extra_cell_lines).split(",") if c.strip()]

    keep_mask = df[tcga_col].astype(str).eq(args.cancer_code)
    if extra_cells:
        keep_mask |= df[cell_col].astype(str).isin(extra_cells)

    filtered = df.loc[keep_mask].copy()
    print(f"  Filtered rows: {len(filtered):,}")

    out = pd.DataFrame(
        {
            "gdsc_version": args.gdsc_version,
            "cell_line_name": filtered[cell_col].astype(str).str.strip(),
            "DRUG_ID": pd.to_numeric(filtered[drug_id_col], errors="coerce").astype("Int64"),
            "drug_name": filtered[drug_name_col].astype(str).str.strip(),
            "ln_IC50": pd.to_numeric(filtered[ic50_col], errors="coerce"),
            "TCGA_DESC": filtered[tcga_col].astype(str).str.strip(),
        }
    )
    out = out.dropna(subset=["DRUG_ID", "ln_IC50"]).copy()
    out["DRUG_ID"] = out["DRUG_ID"].astype(int)

    _mkdir_parent(args.output_uri)
    _write_parquet(out, args.output_uri)
    print(f"Saved: {args.output_uri}")

    report = {
        "input_uri": args.input_uri,
        "output_uri": args.output_uri,
        "cancer_code": args.cancer_code,
        "extra_cell_lines": extra_cells,
        "gdsc_version": args.gdsc_version,
        "rows": int(out.shape[0]),
        "unique_cell_lines": int(out["cell_line_name"].nunique()),
        "unique_drugs": int(out["DRUG_ID"].nunique()),
        "ic50_missing_rows": int(out["ln_IC50"].isna().sum()),
    }

    if args.out_report:
        _mkdir_parent(args.out_report)
        _write_json(report, args.out_report)
        print(f"Saved report: {args.out_report}")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
