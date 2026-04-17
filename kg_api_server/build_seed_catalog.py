#!/usr/bin/env python3
"""
Build a local seed catalog for the KG/API MVP server from
both the FRC route and the random3 route post-ADMET outputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
FRC_POST_DIR = MODELS_DIR / "post_admet_summary_frc_strong_context_smiles"
FRC_STEP7_DIR = MODELS_DIR / "admet_results_frc_strong_context_smiles"
R3_POST_DIR = MODELS_DIR / "post_admet_summary_random3_strong_context_smiles"
R3_STEP7_DIR = MODELS_DIR / "admet_results_random3_strong_context_smiles"
OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(exist_ok=True)


def make_aliases(name: str) -> list[str]:
    raw = str(name)
    normalized = raw.lower().replace(" ", "").replace("-", "").replace("_", "")
    aliases = {raw, raw.lower(), normalized}
    if raw.lower() == "sepantroniumbromide":
        aliases.add("sepantronium bromide")
    return sorted(aliases)


def build_records(post_dir: Path, step7_dir: Path, route_name: str) -> list[dict]:
    top15 = pd.read_csv(post_dir / "top15_comprehensive_table.csv")
    step7 = json.loads((step7_dir / "step7_admet_results.json").read_text())

    detailed_map = {int(x["drug_id"]): x for x in step7.get("detailed_profiles", [])}
    final_map = {int(x["drug_id"]): x for x in step7.get("final_candidates", [])}

    records = []
    for _, row in top15.iterrows():
        drug_id = int(row["drug_id"])
        detailed = detailed_map.get(drug_id, {})
        final = final_map.get(drug_id, {})

        flags = row["flags"]
        if pd.isna(flags):
            flags = "[]"

        record = {
            "route_name": route_name,
            "drug_id": drug_id,
            "drug_name": row["drug_name"],
            "aliases": make_aliases(row["drug_name"]),
            "target": row["target"],
            "pathway": row["pathway"],
            "predicted_ic50": float(row["predicted_ic50"]),
            "sensitivity_rate": float(row["sensitivity_rate"]),
            "validation_score": float(row["validation_score"]),
            "safety_score": float(row["safety_score"]),
            "combined_score": float(row["combined_score"]),
            "final_rank": int(row["final_rank"]),
            "category": row["category"],
            "clinical_bucket": row["clinical_bucket"],
            "known_brca": bool(row["known_brca"]),
            "target_expressed": bool(row["target_expressed"]),
            "brca_pathway": bool(row["brca_pathway"]),
            "survival_sig": bool(row["survival_sig"]),
            "survival_p": None if pd.isna(row["survival_p"]) else float(row["survival_p"]),
            "flags": flags,
            "recommendation_note": row["recommendation_note"],
            "n_assays_tested": int(row["n_assays_tested"]),
            "known_approved": bool(detailed.get("known_approved", False)),
            "assay_details": detailed.get("assay_details", {}),
            "final_candidate_payload": final,
        }
        records.append(record)
    return records


def dedupe_records(records: list[dict]) -> list[dict]:
    by_name: dict[str, dict] = {}
    for rec in records:
        key = str(rec["drug_name"]).lower()
        prev = by_name.get(key)
        if prev is None:
            by_name[key] = rec
            continue
        prev_score = float(prev.get("combined_score", 0.0))
        new_score = float(rec.get("combined_score", 0.0))
        if new_score > prev_score:
            by_name[key] = rec
    return sorted(by_name.values(), key=lambda x: (-float(x["combined_score"]), int(x["final_rank"])))


def main() -> None:
    frc_records = build_records(FRC_POST_DIR, FRC_STEP7_DIR, "FRC")
    r3_records = build_records(R3_POST_DIR, R3_STEP7_DIR, "random3")
    records = dedupe_records(frc_records + r3_records)

    out = {
        "ensemble_name": "Combined local seed catalog (FRC + random3)",
        "input_bundle": "exact slim + strong context + SMILES",
        "routes": ["FRC", "random3"],
        "n_drugs": len(records),
        "records": records,
    }

    out_path = OUT_DIR / "frc_seed_catalog.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved seed catalog: {out_path}")


if __name__ == "__main__":
    main()
