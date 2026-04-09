"""Build drug_features catalog by multi-source SMILES matching.

Matching priority:
  1. ChEMBL compound_master (canonical_smiles, highest quality)
  2. DrugBank drug_master + synonym_table (only for ChEMBL-unmatched)
  3. Fuzzy matching: name cleaning + suffix/prefix variants
  4. PubChem PUG REST API (external lookup for remaining)
  5. Unmatched drugs proceed with SMILES=NA

Output: drug_features.parquet with columns
  [DRUG_ID, DRUG_NAME, drug_name_norm, canonical_smiles, match_source, has_smiles]
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:
    requests = None


def _norm(s: str) -> str:
    """Lowercase, strip non-alnum."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower().strip())


def _clean_drug_name(name: str) -> str:
    """Clean drug name for PubChem/fuzzy lookup: remove concentrations, parenthetical info."""
    s = str(name).strip()
    # Remove concentration info like "(50 uM)", "(10 uM)"
    s = re.sub(r"\s*\(\d+\s*[un]?[Mm]\)", "", s)
    # Remove trailing parenthetical like "(-)", "(+)"
    s = re.sub(r"\s*\([+-]\)\s*$", "", s)
    # Remove trailing salt forms like "-2HCl"
    s = re.sub(r"-\d*HCl\s*$", "", s, flags=re.IGNORECASE)
    # Remove "vitamin C" style parentheticals
    s = re.sub(r"\s*\(vitamin\s+\w+\)\s*$", "", s, flags=re.IGNORECASE)
    # Remove trailing letters like "A", "B", "C" that are variant suffixes
    # (but not if the whole name is short)
    return s.strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build drug features catalog via multi-source SMILES matching.")
    p.add_argument("--gdsc-annotation-uri", required=True)
    p.add_argument("--gdsc-ic50-uri", required=True)
    p.add_argument("--chembl-uri", required=True)
    p.add_argument("--drugbank-uri", required=True)
    p.add_argument("--drugbank-synonym-uri", required=True)
    p.add_argument("--output-uri", required=True)
    p.add_argument("--qc-output-uri", default=None)
    p.add_argument("--skip-pubchem", action="store_true")
    return p.parse_args()


def _load(uri: str) -> pd.DataFrame:
    return pd.read_parquet(uri)


def _build_chembl_lookup(chembl: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Build normalized-name -> (smiles, chembl_id) lookup from ChEMBL."""
    c = chembl[chembl["pref_name"].notna() & chembl["canonical_smiles"].notna()].copy()
    c["name_norm"] = c["pref_name"].apply(_norm)
    c = c.drop_duplicates("name_norm", keep="first")
    return {row["name_norm"]: (row["canonical_smiles"], row.get("chembl_id", ""))
            for _, row in c.iterrows()}


def _build_chembl_lower_lookup(chembl: pd.DataFrame) -> dict[str, str]:
    """Build lowercase name -> smiles lookup (preserving hyphens/spaces)."""
    c = chembl[chembl["pref_name"].notna() & chembl["canonical_smiles"].notna()].copy()
    c["name_lower"] = c["pref_name"].str.lower().str.strip()
    c = c.drop_duplicates("name_lower", keep="first")
    return dict(zip(c["name_lower"], c["canonical_smiles"]))


def _build_drugbank_lookups(db_master: pd.DataFrame, db_syn: pd.DataFrame):
    """Build DrugBank name and synonym lookups."""
    db = db_master[db_master["smiles"].notna()].copy()
    db["name_norm"] = db["name"].apply(_norm)
    db["name_lower"] = db["name"].str.lower().str.strip()

    norm_lookup = {}
    lower_lookup = {}
    for _, row in db.drop_duplicates("name_norm", keep="first").iterrows():
        norm_lookup[row["name_norm"]] = row["smiles"]
    for _, row in db.drop_duplicates("name_lower", keep="first").iterrows():
        lower_lookup[row["name_lower"]] = row["smiles"]

    # Synonym lookup
    syn_with_smiles = db_syn.merge(
        db[["drugbank_id", "smiles"]].drop_duplicates("drugbank_id"),
        on="drugbank_id", how="inner"
    )
    syn_norm_lookup = {}
    for _, row in syn_with_smiles.iterrows():
        key = _norm(str(row["synonym"]))
        if key not in syn_norm_lookup:
            syn_norm_lookup[key] = row["smiles"]

    return norm_lookup, lower_lookup, syn_norm_lookup


def match_pubchem_batch(names_and_ids: list[tuple[str, int]]) -> dict[int, str]:
    """PubChem PUG REST: lookup SMILES by drug name."""
    if requests is None:
        print("  [WARN] requests not available, skipping PubChem")
        return {}

    results = {}
    for raw_name, drug_id in names_and_ids:
        # Try original name first, then cleaned name
        candidates = [raw_name]
        cleaned = _clean_drug_name(raw_name)
        if cleaned != raw_name:
            candidates.append(cleaned)
        # Also try without hyphens for compound codes
        no_hyphen = cleaned.replace("-", "")
        if no_hyphen != cleaned:
            candidates.append(no_hyphen)

        matched = False
        for name in candidates:
            try:
                encoded = urllib.parse.quote(name, safe="")
                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/CanonicalSMILES/JSON"
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    props = data.get("PropertyTable", {}).get("Properties", [])
                    if props:
                        smiles = props[0].get("CanonicalSMILES") or props[0].get("ConnectivitySMILES")
                        if smiles:
                            results[drug_id] = smiles
                        print(f"  PubChem OK: {raw_name} -> matched via '{name}'")
                        matched = True
                        break
                time.sleep(0.22)
            except Exception as e:
                print(f"  PubChem err: {name}: {e}")
                time.sleep(0.5)
        if not matched and len(candidates) > 0:
            pass  # silently skip
    return results


def main() -> None:
    args = parse_args()

    print("Loading data...")
    gdsc_ann = _load(args.gdsc_annotation_uri)
    gdsc_ic50 = _load(args.gdsc_ic50_uri)
    chembl = _load(args.chembl_uri)
    db_master = _load(args.drugbank_uri)
    db_syn = _load(args.drugbank_synonym_uri)

    # Build GDSC drug list
    gdsc_drugs = gdsc_ann[["DRUG_ID", "DRUG_NAME"]].drop_duplicates("DRUG_ID").copy()
    gdsc_drugs["name_norm"] = gdsc_drugs["DRUG_NAME"].apply(_norm)
    total = len(gdsc_drugs)
    print(f"GDSC drugs: {total}")

    # Initialize result columns
    gdsc_drugs["canonical_smiles"] = pd.NA
    gdsc_drugs["match_source"] = pd.NA
    gdsc_drugs["chembl_id"] = pd.NA

    # Build all lookups
    print("Building lookups...")
    chembl_norm = _build_chembl_lookup(chembl)
    chembl_lower = _build_chembl_lower_lookup(chembl)
    db_norm, db_lower, db_syn_norm = _build_drugbank_lookups(db_master, db_syn)

    # ── Priority 1: ChEMBL normalized match ──
    print("\n[1/4] ChEMBL normalized match...")
    for idx, row in gdsc_drugs.iterrows():
        if row["name_norm"] in chembl_norm:
            smiles, cid = chembl_norm[row["name_norm"]]
            gdsc_drugs.loc[idx, "canonical_smiles"] = smiles
            gdsc_drugs.loc[idx, "match_source"] = "chembl_norm"
            gdsc_drugs.loc[idx, "chembl_id"] = cid
    n1 = int(gdsc_drugs["canonical_smiles"].notna().sum())
    print(f"  Matched: {n1}/{total}")

    # ── Priority 2: DrugBank normalized + synonym (only unmatched) ──
    print("\n[2/4] DrugBank name + synonym match...")
    unmatched_mask = gdsc_drugs["canonical_smiles"].isna()
    n2 = 0
    for idx in gdsc_drugs[unmatched_mask].index:
        nn = gdsc_drugs.loc[idx, "name_norm"]
        if nn in db_norm:
            gdsc_drugs.loc[idx, "canonical_smiles"] = db_norm[nn]
            gdsc_drugs.loc[idx, "match_source"] = "drugbank_name"
            n2 += 1
        elif nn in db_syn_norm:
            gdsc_drugs.loc[idx, "canonical_smiles"] = db_syn_norm[nn]
            gdsc_drugs.loc[idx, "match_source"] = "drugbank_synonym"
            n2 += 1
    print(f"  Matched: {n2} (cumulative: {n1+n2}/{total})")

    # ── Priority 3: Fuzzy matching (cleaned names, case-insensitive, suffix variants) ──
    print("\n[3/4] Fuzzy matching...")
    unmatched_mask = gdsc_drugs["canonical_smiles"].isna()
    n3 = 0
    salt_suffixes = [
        " hydrochloride", " hcl", " sodium", " potassium", " mesylate",
        " maleate", " fumarate", " succinate", " tartrate", " acetate",
        " citrate", " sulfate", " phosphate", " tosylate", " besylate",
        " dihydrochloride", " monohydrate", " hydrate", " bromide",
    ]
    for idx in gdsc_drugs[unmatched_mask].index:
        drug_raw = str(gdsc_drugs.loc[idx, "DRUG_NAME"])
        drug_lower = drug_raw.lower().strip()
        drug_cleaned = _clean_drug_name(drug_raw).lower().strip()

        # Try case-insensitive exact against ChEMBL lower lookup
        for candidate in [drug_lower, drug_cleaned]:
            if candidate in chembl_lower:
                gdsc_drugs.loc[idx, "canonical_smiles"] = chembl_lower[candidate]
                gdsc_drugs.loc[idx, "match_source"] = "chembl_fuzzy"
                n3 += 1
                break
            if candidate in db_lower:
                gdsc_drugs.loc[idx, "canonical_smiles"] = db_lower[candidate]
                gdsc_drugs.loc[idx, "match_source"] = "drugbank_fuzzy"
                n3 += 1
                break
        else:
            # Try adding/stripping salt suffixes
            found = False
            for suffix in salt_suffixes:
                for base in [drug_lower, drug_cleaned]:
                    added = base + suffix
                    stripped = base.replace(suffix, "").strip()
                    for c in [added, stripped]:
                        if c and c in chembl_lower:
                            gdsc_drugs.loc[idx, "canonical_smiles"] = chembl_lower[c]
                            gdsc_drugs.loc[idx, "match_source"] = "chembl_salt_variant"
                            n3 += 1
                            found = True
                            break
                        if c and c in db_lower:
                            gdsc_drugs.loc[idx, "canonical_smiles"] = db_lower[c]
                            gdsc_drugs.loc[idx, "match_source"] = "drugbank_salt_variant"
                            n3 += 1
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

    cum3 = int(gdsc_drugs["canonical_smiles"].notna().sum())
    print(f"  Matched: {n3} (cumulative: {cum3}/{total})")

    # ── Priority 4: PubChem API ──
    unmatched_mask = gdsc_drugs["canonical_smiles"].isna()
    unmatched_drugs = gdsc_drugs[unmatched_mask]
    n_before_pc = cum3

    if not args.skip_pubchem and len(unmatched_drugs) > 0:
        # Filter out pure numeric IDs (internal compound codes)
        pubchem_candidates = [
            (row["DRUG_NAME"], row["DRUG_ID"])
            for _, row in unmatched_drugs.iterrows()
            if not str(row["DRUG_NAME"]).strip().isdigit()
        ]
        print(f"\n[4/4] PubChem API for {len(pubchem_candidates)} drugs (skipping {len(unmatched_drugs)-len(pubchem_candidates)} numeric IDs)...")
        pc_results = match_pubchem_batch(pubchem_candidates)
        for did, smiles in pc_results.items():
            mask = gdsc_drugs["DRUG_ID"] == did
            gdsc_drugs.loc[mask, "canonical_smiles"] = smiles
            gdsc_drugs.loc[mask, "match_source"] = "pubchem_api"
        n4 = len(pc_results)
        print(f"  PubChem matched: {n4}")
    else:
        n4 = 0
        if len(unmatched_drugs) > 0:
            print(f"\n[4/4] PubChem skipped. {len(unmatched_drugs)} remain.")

    # ── Final summary ──
    n_final = int(gdsc_drugs["canonical_smiles"].notna().sum())
    n_missing = total - n_final
    print(f"\n{'='*60}")
    print(f"FINAL: {n_final}/{total} matched ({n_final/total*100:.1f}%), {n_missing} missing")
    print(f"{'='*60}")

    # Output
    output = gdsc_drugs[["DRUG_ID", "DRUG_NAME", "name_norm", "canonical_smiles", "match_source"]].copy()
    output = output.rename(columns={"name_norm": "drug_name_norm"})
    output["has_smiles"] = output["canonical_smiles"].notna().astype(int)
    output["match_source"] = output["match_source"].fillna("unmatched")

    output.to_parquet(args.output_uri, index=False)
    print(f"\nSaved: {args.output_uri} ({output.shape})")

    breakdown = output["match_source"].value_counts().to_dict()
    print("\nBreakdown:")
    for src, cnt in sorted(breakdown.items(), key=lambda x: -x[1]):
        print(f"  {src}: {cnt}")

    unmatched_list = sorted(output[output["has_smiles"] == 0]["DRUG_NAME"].tolist())
    if unmatched_list:
        print(f"\nUnmatched ({len(unmatched_list)}):")
        for d in unmatched_list:
            print(f"  - {d}")

    if args.qc_output_uri:
        qc = {
            "total_drugs": total,
            "matched": n_final,
            "match_rate": round(n_final / total, 4),
            "missing": n_missing,
            "breakdown": {k: int(v) for k, v in breakdown.items()},
            "unmatched_drugs": unmatched_list,
        }
        content = json.dumps(qc, ensure_ascii=False, indent=2)
        if args.qc_output_uri.startswith("s3://"):
            import tempfile, subprocess
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                f.write(content)
                tmp = f.name
            subprocess.run(["aws", "s3", "cp", tmp, args.qc_output_uri], check=True)
        else:
            Path(args.qc_output_uri).parent.mkdir(parents=True, exist_ok=True)
            with open(args.qc_output_uri, "w") as f:
                f.write(content)
        print(f"QC saved: {args.qc_output_uri}")


if __name__ == "__main__":
    main()
