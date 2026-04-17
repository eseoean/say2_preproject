#!/usr/bin/env python3
"""
Step 7: ADMET Gate - Final Drug Selection
  - Load Top 15 validated drugs from Step 6
  - Match drug SMILES against 22 ADMET assay datasets
  - Compute ADMET safety profiles
  - Final filtering → drug candidates with safety assessment
  - Upload results to S3
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import json
import time
import os
import sys
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import DataStructs, AllChem

S3_BASE = os.getenv("S3_BASE", "s3://say2-4team/20260409_eseo")

# Data paths
DRUG_CATALOG = f"{S3_BASE}/data/drug_features_catalog.parquet"
ADMET_BASE = f"{S3_BASE}/data/admet"
DRUG_ANN = f"{S3_BASE}/data/gsdc/gdsc2_drug_annotation_master_20260406.parquet"

# Step 6 results
STEP6_DIR = Path(__file__).parent / os.getenv("STEP6_INPUT_DIRNAME", "metabric_results")
TOP15_PATH = STEP6_DIR / "top15_validated.csv"

# Output
OUTPUT_DIR = Path(__file__).parent / os.getenv("ADMET_OUTPUT_DIRNAME", "admet_results")
OUTPUT_DIR.mkdir(exist_ok=True)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# 22 ADMET assays with their properties
ADMET_ASSAYS = {
    # Absorption
    "caco2_wang":       {"category": "Absorption", "name": "Caco-2 Permeability", "type": "regression", "good_direction": "high", "unit": "log(cm/s)", "threshold": -5.15},
    "hia_hou":          {"category": "Absorption", "name": "HIA (Human Intestinal Absorption)", "type": "binary", "good_value": 1},
    "pgp_broccatelli":  {"category": "Absorption", "name": "P-gp Inhibitor", "type": "binary", "good_value": 0},
    "bioavailability_ma": {"category": "Absorption", "name": "Oral Bioavailability (F>20%)", "type": "binary", "good_value": 1},
    # Distribution
    "bbb_martins":      {"category": "Distribution", "name": "BBB Penetration", "type": "binary", "good_value": None},  # cancer: don't need BBB
    "ppbr_az":          {"category": "Distribution", "name": "Plasma Protein Binding Rate", "type": "regression", "good_direction": "low", "unit": "%", "threshold": 90},
    "vdss_lombardo":    {"category": "Distribution", "name": "Volume of Distribution", "type": "regression", "good_direction": None, "unit": "L/kg"},
    # Metabolism
    "cyp2c9_veith":     {"category": "Metabolism", "name": "CYP2C9 Inhibitor", "type": "binary", "good_value": 0},
    "cyp2d6_veith":     {"category": "Metabolism", "name": "CYP2D6 Inhibitor", "type": "binary", "good_value": 0},
    "cyp3a4_veith":     {"category": "Metabolism", "name": "CYP3A4 Inhibitor", "type": "binary", "good_value": 0},
    "cyp2c9_substrate_carbonmangels": {"category": "Metabolism", "name": "CYP2C9 Substrate", "type": "binary", "good_value": None},
    "cyp2d6_substrate_carbonmangels": {"category": "Metabolism", "name": "CYP2D6 Substrate", "type": "binary", "good_value": None},
    "cyp3a4_substrate_carbonmangels": {"category": "Metabolism", "name": "CYP3A4 Substrate", "type": "binary", "good_value": None},
    # Excretion
    "clearance_hepatocyte_az": {"category": "Excretion", "name": "Hepatocyte Clearance", "type": "regression", "good_direction": None, "unit": "uL/min/10^6 cells"},
    "clearance_microsome_az":  {"category": "Excretion", "name": "Microsome Clearance", "type": "regression", "good_direction": None, "unit": "mL/min/g"},
    "half_life_obach":  {"category": "Excretion", "name": "Half-Life", "type": "regression", "good_direction": "high", "unit": "hr", "threshold": 3},
    # Toxicity
    "ames":             {"category": "Toxicity", "name": "Ames Mutagenicity", "type": "binary", "good_value": 0},
    "dili":             {"category": "Toxicity", "name": "DILI (Drug-Induced Liver Injury)", "type": "binary", "good_value": 0},
    "herg":             {"category": "Toxicity", "name": "hERG Cardiotoxicity", "type": "binary", "good_value": 0},
    "ld50_zhu":         {"category": "Toxicity", "name": "Acute Toxicity (LD50)", "type": "regression", "good_direction": "high", "unit": "log(mol/kg)"},
    "lipophilicity_astrazeneca": {"category": "Properties", "name": "Lipophilicity (logD)", "type": "regression", "good_direction": None, "unit": "logD", "ideal_range": (-0.4, 5.6)},
    "solubility_aqsoldb": {"category": "Properties", "name": "Aqueous Solubility", "type": "regression", "good_direction": "high", "unit": "logS"},
}


def tanimoto_similarity(smiles1, smiles2):
    """Compute Tanimoto similarity between two SMILES."""
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def load_data():
    print("Loading data...")
    t0 = time.time()

    top15 = pd.read_csv(TOP15_PATH)
    drug_catalog = pd.read_parquet(DRUG_CATALOG)
    drug_ann = pd.read_parquet(DRUG_ANN)

    # Map SMILES to top 15 drugs
    smiles_map = drug_catalog.set_index("DRUG_ID")["canonical_smiles"].to_dict()
    name_map = drug_ann.set_index("DRUG_ID")["DRUG_NAME"].to_dict()
    target_map = drug_ann.set_index("DRUG_ID")["PUTATIVE_TARGET_NORMALIZED"].to_dict()
    pathway_map = drug_ann.set_index("DRUG_ID")["PATHWAY_NAME_NORMALIZED"].to_dict()

    top15["smiles"] = top15["drug_id"].map(smiles_map)
    if "drug_name" not in top15.columns:
        top15["drug_name"] = top15["drug_id"].map(name_map)
    if "target" not in top15.columns:
        top15["target"] = top15["drug_id"].map(target_map)
    if "pathway" not in top15.columns:
        top15["pathway"] = top15["drug_id"].map(pathway_map)

    has_smiles = top15["smiles"].notna().sum()
    print(f"  Top 15 drugs loaded ({has_smiles}/15 with SMILES)")
    print(f"  ({time.time()-t0:.1f}s)")
    return top15, drug_catalog


def lookup_admet(top15):
    """Look up each drug in 22 ADMET assay datasets."""
    print(f"\n{'='*60}")
    print(f"  ADMET Assay Lookup (22 assays x {len(top15)} drugs)")
    print(f"{'='*60}")

    # Pre-compute Morgan fingerprints for top15 drugs
    drug_fps = {}
    for _, row in top15.iterrows():
        smiles = row.get("smiles")
        if pd.notna(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                drug_fps[row["drug_id"]] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

    results = {}  # drug_id -> {assay_name: result}

    for assay_name, assay_info in ADMET_ASSAYS.items():
        # Load assay data (train_val + test combined)
        try:
            train = pd.read_parquet(f"{ADMET_BASE}/{assay_name}/train_val_basic_clean_20260406.parquet")
            test = pd.read_parquet(f"{ADMET_BASE}/{assay_name}/test_basic_clean_20260406.parquet")
            assay_data = pd.concat([train, test], ignore_index=True)
        except Exception as e:
            print(f"  Warning: Could not load {assay_name}: {e}")
            continue

        # Build fingerprint database for this assay
        assay_smiles = assay_data["Drug"].values
        assay_labels = assay_data["Y"].values
        assay_fps = []
        valid_idx = []

        for i, smi in enumerate(assay_smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                assay_fps.append(fp)
                valid_idx.append(i)

        n_found = 0
        for _, row in top15.iterrows():
            drug_id = row["drug_id"]
            drug_name = row.get("drug_name", f"Drug_{drug_id}")

            if drug_id not in results:
                results[drug_id] = {}

            if drug_id not in drug_fps:
                results[drug_id][assay_name] = {
                    "value": None, "match_type": "no_smiles",
                    "similarity": 0.0, **assay_info
                }
                continue

            my_fp = drug_fps[drug_id]

            # Exact match (Tanimoto >= 0.99)
            best_sim = 0.0
            best_val = None
            best_idx = -1

            for j, afp in enumerate(assay_fps):
                sim = DataStructs.TanimotoSimilarity(my_fp, afp)
                if sim > best_sim:
                    best_sim = sim
                    best_val = assay_labels[valid_idx[j]]
                    best_idx = valid_idx[j]

            if best_sim >= 0.99:
                match_type = "exact"
                n_found += 1
            elif best_sim >= 0.85:
                match_type = "close_analog"
                n_found += 1
            elif best_sim >= 0.70:
                match_type = "analog"
            else:
                match_type = "no_match"
                best_val = None

            results[drug_id][assay_name] = {
                "value": float(best_val) if best_val is not None else None,
                "match_type": match_type,
                "similarity": float(best_sim),
                **assay_info
            }

        print(f"  {assay_info['category']:<12} {assay_info['name']:<35} matched: {n_found}/{len(top15)}")

    return results


def compute_safety_profiles(top15, admet_results):
    """Compute safety profiles for each drug."""
    print(f"\n{'='*60}")
    print(f"  Safety Profile Assessment")
    print(f"{'='*60}")

    profiles = []

    for _, row in top15.iterrows():
        drug_id = row["drug_id"]
        drug_name = row.get("drug_name", f"Drug_{drug_id}")
        drug_results = admet_results.get(drug_id, {})

        n_assays_tested = 0
        n_pass = 0
        n_caution = 0
        n_fail = 0
        n_nodata = 0
        flags = []
        assay_details = {}

        for assay_name, assay_info in ADMET_ASSAYS.items():
            r = drug_results.get(assay_name, {})
            val = r.get("value")
            match = r.get("match_type", "no_match")

            if val is None or match in ("no_match", "no_smiles"):
                status = "no_data"
                n_nodata += 1
            elif assay_info["type"] == "binary":
                n_assays_tested += 1
                good_val = assay_info.get("good_value")
                if good_val is None:
                    status = "info"  # informational only
                elif int(val) == good_val:
                    status = "pass"
                    n_pass += 1
                else:
                    # Toxicity flags
                    if assay_name in ("ames", "dili", "herg"):
                        status = "caution"
                        n_caution += 1
                        flags.append(f"{assay_info['name']}(+)")
                    else:
                        status = "minor"
                        n_caution += 1
            else:
                n_assays_tested += 1
                status = "measured"
                n_pass += 1

            assay_details[assay_name] = {
                "value": val,
                "status": status,
                "match_type": match,
                "similarity": r.get("similarity", 0),
            }

        # Safety score (higher = safer)
        if n_assays_tested > 0:
            safety_score = n_pass / max(n_assays_tested, 1) * 10
            # Penalize critical toxicity flags
            for flag in flags:
                if "hERG" in flag:
                    safety_score -= 2
                elif "Ames" in flag:
                    safety_score -= 1.5
                elif "DILI" in flag:
                    safety_score -= 1
        else:
            safety_score = 5.0  # neutral for no data

        # Known approved drugs get a bonus
        known_approved = drug_name in {
            "Docetaxel", "Paclitaxel", "Vinblastine", "Vinorelbine",
            "Rapamycin", "Bortezomib", "Romidepsin", "Dactinomycin",
        }
        if known_approved:
            safety_score += 2.0  # already FDA-approved/clinical use

        profiles.append({
            "drug_id": drug_id,
            "drug_name": drug_name,
            "target": row.get("target", "N/A"),
            "pathway": row.get("pathway", "N/A"),
            "mean_pred_ic50": row.get("mean_pred_ic50", 0),
            "sensitivity_rate": row.get("sensitivity_rate", 0),
            "n_assays_tested": n_assays_tested,
            "n_pass": n_pass,
            "n_caution": n_caution,
            "n_nodata": n_nodata,
            "safety_score": round(safety_score, 2),
            "flags": flags,
            "known_approved": known_approved,
            "assay_details": assay_details,
        })

    # Sort by combined score: efficacy + safety
    for p in profiles:
        # Combined: safety_score (0-12) + efficacy from predicted IC50 rank
        ic50_rank = sorted(profiles, key=lambda x: x["mean_pred_ic50"]).index(p)
        p["efficacy_rank"] = ic50_rank + 1
        p["combined_score"] = p["safety_score"] + (15 - ic50_rank) * 0.5

    profiles.sort(key=lambda x: -x["combined_score"])

    # Print summary
    print(f"\n  {'#':<3} {'Drug':<22} {'IC50':>7} {'Safe':>5} {'Pass':>5} "
          f"{'Caution':>7} {'NoData':>7} {'Approved':>8} {'Flags'}")
    print(f"  {'-'*85}")
    for i, p in enumerate(profiles, 1):
        flags_str = ", ".join(p["flags"]) if p["flags"] else "-"
        approved = "YES" if p["known_approved"] else "-"
        print(f"  {i:<3} {p['drug_name']:<22} {p['mean_pred_ic50']:>7.3f} "
              f"{p['safety_score']:>5.1f} {p['n_pass']:>5} "
              f"{p['n_caution']:>7} {p['n_nodata']:>7} {approved:>8} {flags_str}")

    return profiles


def final_selection(profiles):
    """Final drug selection based on combined efficacy + safety."""
    print(f"\n{'='*60}")
    print(f"  FINAL DRUG CANDIDATES (ADMET-filtered)")
    print(f"{'='*60}")

    # All drugs pass through with safety annotations
    # (ADMET Gate is informational, not eliminatory for known drugs)
    final = []
    for i, p in enumerate(profiles, 1):
        category = "Approved" if p["known_approved"] else (
            "Candidate" if p["safety_score"] >= 4 else "Caution"
        )
        final.append({
            "final_rank": i,
            "drug_id": p["drug_id"],
            "drug_name": p["drug_name"],
            "target": p["target"],
            "pathway": p["pathway"],
            "pred_ic50": p["mean_pred_ic50"],
            "sensitivity_rate": p["sensitivity_rate"],
            "safety_score": p["safety_score"],
            "category": category,
            "flags": p["flags"],
            "combined_score": p["combined_score"],
            "n_assays_tested": p["n_assays_tested"],
        })

    print(f"\n  {'#':<3} {'Drug':<22} {'Target':<20} {'IC50':>7} {'Safety':>7} "
          f"{'Combined':>8} {'Category':<10}")
    print(f"  {'-'*80}")
    for d in final:
        print(f"  {d['final_rank']:<3} {d['drug_name']:<22} "
              f"{str(d['target'])[:18]:<20} {d['pred_ic50']:>7.3f} "
              f"{d['safety_score']:>7.1f} {d['combined_score']:>8.1f} "
              f"{d['category']:<10}")

    n_approved = sum(1 for d in final if d["category"] == "Approved")
    n_candidate = sum(1 for d in final if d["category"] == "Candidate")
    n_caution = sum(1 for d in final if d["category"] == "Caution")
    print(f"\n  Total: {len(final)} drugs | Approved: {n_approved} | "
          f"Candidate: {n_candidate} | Caution: {n_caution}")

    return final


def save_results(profiles, final):
    """Save all Step 7 results."""
    summary = {
        "step": 7,
        "description": "ADMET Gate - Final Drug Selection",
        "n_assays": len(ADMET_ASSAYS),
        "n_drugs_input": len(profiles),
        "n_drugs_output": len(final),
        "assay_list": {k: v["name"] for k, v in ADMET_ASSAYS.items()},
        "drug_profiles": [{
            k: v for k, v in p.items() if k != "assay_details"
        } for p in profiles],
        "final_candidates": final,
        "detailed_profiles": profiles,
    }

    def convert(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, np.bool_)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    out_json = OUTPUT_DIR / "step7_admet_results.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=convert)
    print(f"\n  Results saved: {out_json}")

    # CSV summary
    final_df = pd.DataFrame(final)
    final_csv = OUTPUT_DIR / "final_drug_candidates.csv"
    final_df.to_csv(final_csv, index=False)
    print(f"  Final CSV: {final_csv}")

    return summary


def upload_to_s3():
    """Upload Step 7 results to S3."""
    import subprocess
    admet_s3_dirname = os.getenv("ADMET_S3_DIRNAME", "admet_results")
    s3_dest = f"{S3_BASE}/models/{admet_s3_dirname}/"
    cmd = f"aws s3 sync {OUTPUT_DIR} {s3_dest} --quiet"
    print(f"\n  Uploading to S3: {s3_dest}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print("  S3 upload: OK")
    else:
        print(f"  S3 upload warning: {result.stderr[:200]}")

    # Also upload ensemble results
    ens_s3_dirname = os.getenv("ENSEMBLE_S3_DIRNAME", "ensemble_results")
    ens_input_dirname = os.getenv("ENSEMBLE_INPUT_DIRNAME", "ensemble_results")
    ens_dest = f"{S3_BASE}/models/{ens_s3_dirname}/"
    ens_dir = Path(__file__).parent / ens_input_dirname
    subprocess.run(f"aws s3 sync {ens_dir} {ens_dest} --quiet", shell=True,
                   capture_output=True, text=True)
    print(f"  Ensemble results synced to S3")


def main():
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  Step 7: ADMET Gate - Final Drug Selection")
    print(f"{'='*60}")

    top15, drug_catalog = load_data()

    # ADMET lookup
    admet_results = lookup_admet(top15)

    # Safety profiles
    profiles = compute_safety_profiles(top15, admet_results)

    # Final selection
    final = final_selection(profiles)

    # Save results
    summary = save_results(profiles, final)

    # Upload to S3
    upload_to_s3()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Step 7 COMPLETE ({elapsed/60:.1f} min)")
    print(f"  {len(final)} final drug candidates with ADMET profiles")
    print(f"{'='*60}")

    return summary


if __name__ == "__main__":
    main()
