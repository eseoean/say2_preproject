"""Normalize LINCS MCF7 signatures for the FE pipeline.

Steps:
  1. Rename Entrez Gene ID columns → crispr__GENE_SYMBOL using LINCS gene_info
  2. Extract BRD IDs from sig_id, map → pert_iname via pert_info
  3. Match pert_iname → canonical_drug_id via GDSC drug catalog
  4. Average signatures per canonical_drug_id
  5. Output: parquet keyed by canonical_drug_id with crispr__SYMBOL columns
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BRD_RE = re.compile(r"(BRD-[A-Z0-9]+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Normalize LINCS MCF7 for pipeline.")
    p.add_argument("--lincs-uri", required=True, help="LINCS MCF7 signature parquet (sig_id + Entrez Gene ID columns)")
    p.add_argument("--gene-info-uri", required=True, help="LINCS gene_info parquet (pr_gene_id, pr_gene_symbol)")
    p.add_argument("--pert-info-uri", required=True, help="LINCS pert_info parquet (pert_id, pert_iname)")
    p.add_argument("--drug-catalog-uri", required=True, help="Drug catalog parquet (DRUG_NAME, canonical_smiles, ...)")
    p.add_argument("--gdsc-ic50-uri", default="", help="Optional GDSC IC50 parquet for DRUG_ID → drug_name mapping")
    p.add_argument("--out-parquet", required=True, help="Output normalized LINCS signature parquet")
    p.add_argument("--out-report", required=True, help="Output mapping report JSON")
    return p.parse_args()


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower().strip())


def main() -> None:
    args = parse_args()
    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)

    # 1. Load gene_info for Entrez → symbol mapping
    gene_info = pd.read_parquet(args.gene_info_uri)
    entrez_to_symbol = dict(
        zip(gene_info["pr_gene_id"].astype(str), gene_info["pr_gene_symbol"].astype(str))
    )
    print(f"Gene info: {len(entrez_to_symbol)} Entrez→Symbol mappings")

    # 2. Load LINCS signatures
    lincs = pd.read_parquet(args.lincs_uri)
    print(f"LINCS raw: {lincs.shape}")

    # Rename Entrez ID columns → crispr__SYMBOL
    rename_map = {}
    unmapped_cols = []
    for col in lincs.columns:
        if col == "sig_id":
            continue
        symbol = entrez_to_symbol.get(str(col))
        if symbol:
            new_name = f"crispr__{symbol}"
            rename_map[col] = new_name
        else:
            unmapped_cols.append(col)

    lincs = lincs.rename(columns=rename_map)
    # Drop unmapped columns (Entrez IDs not in gene_info)
    lincs = lincs.drop(columns=unmapped_cols, errors="ignore")
    print(f"Mapped {len(rename_map)} columns to crispr__SYMBOL, dropped {len(unmapped_cols)} unmapped")

    # 3. Extract BRD IDs from sig_id
    lincs["brd_id"] = lincs["sig_id"].apply(
        lambda x: m.group(1) if (m := BRD_RE.search(str(x).upper())) else ""
    )

    # 4. Load pert_info for BRD → pert_iname mapping
    pert_info = pd.read_parquet(args.pert_info_uri)
    # Filter to compounds (trt_cp)
    pert_cp = pert_info[pert_info["pert_type"] == "trt_cp"].copy()
    pert_cp["brd_id"] = pert_cp["pert_id"].apply(
        lambda x: m.group(1) if (m := BRD_RE.search(str(x).upper())) else str(x).upper().strip()
    )
    brd_to_name = dict(
        zip(pert_cp["brd_id"].astype(str), pert_cp["pert_iname"].astype(str))
    )
    print(f"Pert info: {len(brd_to_name)} BRD→pert_iname mappings")

    # 5. Build drug name → canonical_drug_id mapping from GDSC
    drug_catalog = pd.read_parquet(args.drug_catalog_uri)

    # Build lookup: normalized drug name → DRUG_ID
    name_to_drug_id = {}
    if "DRUG_NAME" in drug_catalog.columns and "DRUG_ID" in drug_catalog.columns:
        for _, row in drug_catalog[["DRUG_ID", "DRUG_NAME"]].dropna().iterrows():
            name_to_drug_id[_norm_name(row["DRUG_NAME"])] = str(int(row["DRUG_ID"]))
    elif "drug_name_norm" in drug_catalog.columns and "DRUG_ID" in drug_catalog.columns:
        for _, row in drug_catalog[["DRUG_ID", "drug_name_norm"]].dropna().iterrows():
            name_to_drug_id[_norm_name(row["drug_name_norm"])] = str(int(row["DRUG_ID"]))

    # Also build from GDSC IC50 if available (has DRUG_ID + drug_name)
    if args.gdsc_ic50_uri:
        gdsc = pd.read_parquet(args.gdsc_ic50_uri, columns=["DRUG_ID", "drug_name"])
        for _, row in gdsc[["DRUG_ID", "drug_name"]].dropna().drop_duplicates().iterrows():
            key = _norm_name(row["drug_name"])
            if key not in name_to_drug_id:
                name_to_drug_id[key] = str(int(row["DRUG_ID"]))
    print(f"Drug name→DRUG_ID: {len(name_to_drug_id)} mappings")

    # 6. Map BRD → pert_iname → canonical_drug_id
    lincs["pert_iname"] = lincs["brd_id"].map(brd_to_name).fillna("")
    lincs["canonical_drug_id"] = lincs["pert_iname"].apply(
        lambda x: name_to_drug_id.get(_norm_name(x), "")
    )

    # Also try SMILES matching for unmatched
    pert_smiles = dict(
        zip(pert_cp["brd_id"].astype(str), pert_cp["canonical_smiles"].astype(str))
    )
    catalog_smiles_to_id = {}
    smiles_col = "canonical_smiles" if "canonical_smiles" in drug_catalog.columns else None
    if smiles_col:
        for _, row in drug_catalog.dropna(subset=[smiles_col]).iterrows():
            smi = str(row[smiles_col]).strip()
            if smi and smi.lower() not in ("nan", "none", ""):
                did = str(int(row["DRUG_ID"])) if "DRUG_ID" in drug_catalog.columns else ""
                if did:
                    catalog_smiles_to_id[smi] = did

    smiles_matched = 0
    for idx in lincs.index:
        if lincs.at[idx, "canonical_drug_id"]:
            continue
        brd = lincs.at[idx, "brd_id"]
        smi = pert_smiles.get(brd, "")
        if smi and smi.lower() not in ("nan", "none", "<na>", ""):
            did = catalog_smiles_to_id.get(smi, "")
            if did:
                lincs.at[idx, "canonical_drug_id"] = did
                smiles_matched += 1

    matched = lincs[lincs["canonical_drug_id"] != ""].copy()
    print(f"Matched signatures: {len(matched)}/{len(lincs)} ({100*len(matched)/len(lincs):.1f}%)")
    print(f"  by name: {len(matched) - smiles_matched}, by SMILES: {smiles_matched}")
    print(f"Unique drugs matched: {matched['canonical_drug_id'].nunique()}")

    # 7. Aggregate: average signatures per canonical_drug_id
    num_cols = [c for c in matched.columns if c.startswith("crispr__")]
    grouped = matched.groupby("canonical_drug_id", as_index=False)[num_cols].mean()
    print(f"Final LINCS drug signature: {grouped.shape}")

    grouped.to_parquet(args.out_parquet, index=False)

    # 8. Report
    report = {
        "lincs_total_sigs": int(lincs.shape[0]),
        "gene_columns_mapped": len(rename_map),
        "gene_columns_unmapped": len(unmapped_cols),
        "unique_brds": int(lincs["brd_id"].nunique()),
        "matched_sigs": int(len(matched)),
        "matched_ratio": float(len(matched) / max(len(lincs), 1)),
        "matched_drugs": int(grouped.shape[0]),
        "by_name": int(len(matched) - smiles_matched),
        "by_smiles": int(smiles_matched),
        "output_shape": list(grouped.shape),
    }
    Path(args.out_report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
