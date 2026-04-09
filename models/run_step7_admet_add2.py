#!/usr/bin/env python3
"""
Step 7 추가: 중복 제거 후 15개 채우기 위해 Romidepsin, Sepantronium bromide ADMET 분석
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import json
import time
import os
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import DataStructs, AllChem

S3_BASE = os.getenv("S3_BASE", "s3://say2-4team/20260409_eseo")
ADMET_BASE = f"{S3_BASE}/data/admet"
DRUG_CATALOG = f"{S3_BASE}/data/drug_features_catalog.parquet"
DRUG_ANN = f"{S3_BASE}/data/gsdc/gdsc2_drug_annotation_master_20260406.parquet"

OUTPUT_DIR = Path(__file__).parent / "admet_results"

# Target drug IDs (from top30, next-ranked after dedup)
NEW_DRUG_IDS = [1817, 1941]  # Romidepsin, Sepantronium bromide

# Existing top15_validated info for these drugs
NEW_DRUGS_INFO = {
    1817: {"mean_pred_ic50": -4.7611876, "sensitivity_rate": 1.0, "n_samples": 26},
    1941: {"mean_pred_ic50": -3.73555,   "sensitivity_rate": 1.0, "n_samples": 28},
}

ADMET_ASSAYS = {
    "caco2_wang":       {"category": "Absorption", "name": "Caco-2 Permeability", "type": "regression", "good_direction": "high", "unit": "log(cm/s)", "threshold": -5.15},
    "hia_hou":          {"category": "Absorption", "name": "HIA (Human Intestinal Absorption)", "type": "binary", "good_value": 1},
    "pgp_broccatelli":  {"category": "Absorption", "name": "P-gp Inhibitor", "type": "binary", "good_value": 0},
    "bioavailability_ma": {"category": "Absorption", "name": "Oral Bioavailability (F>20%)", "type": "binary", "good_value": 1},
    "bbb_martins":      {"category": "Distribution", "name": "BBB Penetration", "type": "binary", "good_value": None},
    "ppbr_az":          {"category": "Distribution", "name": "Plasma Protein Binding Rate", "type": "regression", "good_direction": "low", "unit": "%", "threshold": 90},
    "vdss_lombardo":    {"category": "Distribution", "name": "Volume of Distribution", "type": "regression", "good_direction": None, "unit": "L/kg"},
    "cyp2c9_veith":     {"category": "Metabolism", "name": "CYP2C9 Inhibitor", "type": "binary", "good_value": 0},
    "cyp2d6_veith":     {"category": "Metabolism", "name": "CYP2D6 Inhibitor", "type": "binary", "good_value": 0},
    "cyp3a4_veith":     {"category": "Metabolism", "name": "CYP3A4 Inhibitor", "type": "binary", "good_value": 0},
    "cyp2c9_substrate_carbonmangels": {"category": "Metabolism", "name": "CYP2C9 Substrate", "type": "binary", "good_value": None},
    "cyp2d6_substrate_carbonmangels": {"category": "Metabolism", "name": "CYP2D6 Substrate", "type": "binary", "good_value": None},
    "cyp3a4_substrate_carbonmangels": {"category": "Metabolism", "name": "CYP3A4 Substrate", "type": "binary", "good_value": None},
    "clearance_hepatocyte_az": {"category": "Excretion", "name": "Hepatocyte Clearance", "type": "regression", "good_direction": None, "unit": "uL/min/10^6 cells"},
    "clearance_microsome_az":  {"category": "Excretion", "name": "Microsome Clearance", "type": "regression", "good_direction": None, "unit": "mL/min/g"},
    "half_life_obach":  {"category": "Excretion", "name": "Half-Life", "type": "regression", "good_direction": "high", "unit": "hr", "threshold": 3},
    "ames":             {"category": "Toxicity", "name": "Ames Mutagenicity", "type": "binary", "good_value": 0},
    "dili":             {"category": "Toxicity", "name": "DILI (Drug-Induced Liver Injury)", "type": "binary", "good_value": 0},
    "herg":             {"category": "Toxicity", "name": "hERG Cardiotoxicity", "type": "binary", "good_value": 0},
    "ld50_zhu":         {"category": "Toxicity", "name": "Acute Toxicity (LD50)", "type": "regression", "good_direction": "high", "unit": "log(mol/kg)"},
    "lipophilicity_astrazeneca": {"category": "Properties", "name": "Lipophilicity (logD)", "type": "regression", "good_direction": None, "unit": "logD", "ideal_range": (-0.4, 5.6)},
    "solubility_aqsoldb": {"category": "Properties", "name": "Aqueous Solubility", "type": "regression", "good_direction": "high", "unit": "logS"},
}


def main():
    t0 = time.time()
    print("Loading drug catalog and annotations...")
    drug_catalog = pd.read_parquet(DRUG_CATALOG)
    drug_ann = pd.read_parquet(DRUG_ANN)

    smiles_map = drug_catalog.set_index("DRUG_ID")["canonical_smiles"].to_dict()
    name_map = drug_ann.set_index("DRUG_ID")["DRUG_NAME"].to_dict()
    target_map = drug_ann.set_index("DRUG_ID")["PUTATIVE_TARGET_NORMALIZED"].to_dict()
    pathway_map = drug_ann.set_index("DRUG_ID")["PATHWAY_NAME_NORMALIZED"].to_dict()

    # Build drug info
    drugs = []
    for did in NEW_DRUG_IDS:
        drugs.append({
            "drug_id": did,
            "drug_name": name_map.get(did, f"Drug_{did}"),
            "target": target_map.get(did, "N/A"),
            "pathway": pathway_map.get(did, "N/A"),
            "smiles": smiles_map.get(did),
            **NEW_DRUGS_INFO[did],
        })
    drugs_df = pd.DataFrame(drugs)

    for d in drugs:
        print(f"  {d['drug_id']}: {d['drug_name']} ({d['target']}) SMILES={'YES' if d['smiles'] else 'NO'}")

    # Pre-compute fingerprints
    drug_fps = {}
    for d in drugs:
        if d["smiles"]:
            mol = Chem.MolFromSmiles(d["smiles"])
            if mol:
                drug_fps[d["drug_id"]] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

    # ADMET lookup
    print(f"\nRunning 22 ADMET assays for {len(drugs)} drugs...")
    results = {d["drug_id"]: {} for d in drugs}

    for assay_name, assay_info in ADMET_ASSAYS.items():
        try:
            train = pd.read_parquet(f"{ADMET_BASE}/{assay_name}/train_val_basic_clean_20260406.parquet")
            test = pd.read_parquet(f"{ADMET_BASE}/{assay_name}/test_basic_clean_20260406.parquet")
            assay_data = pd.concat([train, test], ignore_index=True)
        except Exception as e:
            print(f"  Warning: {assay_name}: {e}")
            for d in drugs:
                results[d["drug_id"]][assay_name] = {"value": None, "status": "no_data", "match_type": "no_match", "similarity": 0.0}
            continue

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

        for d in drugs:
            did = d["drug_id"]
            if did not in drug_fps:
                results[did][assay_name] = {"value": None, "status": "no_data", "match_type": "no_smiles", "similarity": 0.0}
                continue

            my_fp = drug_fps[did]
            best_sim = 0.0
            best_val = None
            for j, afp in enumerate(assay_fps):
                sim = DataStructs.TanimotoSimilarity(my_fp, afp)
                if sim > best_sim:
                    best_sim = sim
                    best_val = assay_labels[valid_idx[j]]

            if best_sim >= 0.99:
                match_type = "exact"
            elif best_sim >= 0.85:
                match_type = "close_analog"
            elif best_sim >= 0.70:
                match_type = "analog"
            else:
                match_type = "no_match"
                best_val = None

            results[did][assay_name] = {
                "value": float(best_val) if best_val is not None else None,
                "match_type": match_type,
                "similarity": float(best_sim),
            }

        matched = sum(1 for d in drugs if results[d["drug_id"]][assay_name]["value"] is not None)
        print(f"  {assay_info['category']:<12} {assay_info['name']:<35} matched: {matched}/{len(drugs)}")

    # Compute safety profiles
    print("\nComputing safety profiles...")
    profiles = []
    for d in drugs:
        did = d["drug_id"]
        drug_results = results[did]
        n_assays_tested = 0
        n_pass = 0
        n_caution = 0
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
                    status = "info"
                elif int(val) == good_val:
                    status = "pass"
                    n_pass += 1
                else:
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

        if n_assays_tested > 0:
            safety_score = n_pass / max(n_assays_tested, 1) * 10
            for flag in flags:
                if "hERG" in flag:
                    safety_score -= 2
                elif "Ames" in flag:
                    safety_score -= 1.5
                elif "DILI" in flag:
                    safety_score -= 1
        else:
            safety_score = 5.0

        known_approved = d["drug_name"] in {
            "Docetaxel", "Paclitaxel", "Vinblastine", "Vinorelbine",
            "Rapamycin", "Bortezomib", "Romidepsin", "Dactinomycin",
        }
        if known_approved:
            safety_score += 2.0

        profiles.append({
            "drug_id": did,
            "drug_name": d["drug_name"],
            "target": d["target"],
            "pathway": d["pathway"],
            "mean_pred_ic50": d["mean_pred_ic50"],
            "sensitivity_rate": d["sensitivity_rate"],
            "n_assays_tested": n_assays_tested,
            "n_pass": n_pass,
            "n_caution": n_caution,
            "n_nodata": n_nodata,
            "safety_score": round(safety_score, 2),
            "flags": flags,
            "known_approved": known_approved,
            "assay_details": assay_details,
        })

    # Print results
    for p in profiles:
        print(f"\n  {p['drug_name']} (ID={p['drug_id']}):")
        print(f"    IC50={p['mean_pred_ic50']:.3f}, Safety={p['safety_score']:.2f}")
        print(f"    Tested={p['n_assays_tested']}, Pass={p['n_pass']}, Caution={p['n_caution']}")
        print(f"    Flags={p['flags']}")
        print(f"    Known approved={p['known_approved']}")

    # Save as JSON for integration
    out = OUTPUT_DIR / "new_2_drugs_admet.json"
    def convert(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, np.bool_)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out, "w") as f:
        json.dump(profiles, f, indent=2, default=convert)
    print(f"\nSaved: {out}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
