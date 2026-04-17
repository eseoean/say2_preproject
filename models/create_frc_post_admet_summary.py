#!/usr/bin/env python3
"""
Create teammate-flow style post-ADMET summary tables for the
FRC ensemble (FlatMLP + ResidualMLP + CrossAttention).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).parent
STEP6_DIR = BASE_DIR / "metabric_results_frc_strong_context_smiles"
STEP7_DIR = BASE_DIR / "admet_results_frc_strong_context_smiles"
OUT_DIR = BASE_DIR / "post_admet_summary_frc_strong_context_smiles"
OUT_DIR.mkdir(exist_ok=True)


def build_recommendation_note(row: pd.Series) -> str:
    category = str(row.get("category", ""))
    known_brca = int(row.get("known_brca", 0) or 0)
    target_expressed = int(row.get("target_expressed", 0) or 0)
    brca_pathway = int(row.get("brca_pathway", 0) or 0)
    survival_sig = int(row.get("survival_sig", 0) or 0)
    flags = str(row.get("flags", "[]"))

    if category == "Approved" and known_brca:
        return "Breast-cancer relevant and ADMET-screened. Suitable as a primary shortlist candidate."
    if category == "Approved":
        return "ADMET-screened and externally validated. Worth considering as an expansion candidate."
    if category == "Candidate" and target_expressed and survival_sig:
        return "Biologically plausible candidate with supportive validation, but additional safety or translational review is needed."
    if category == "Candidate":
        return "Promising candidate, but requires follow-up validation before prioritizing."
    if category == "Caution":
        if "Ames" in flags or "DILI" in flags:
            return "Efficacy signal exists, but safety flags were detected. Keep as a low-priority caution item."
        return "Not a first-line recommendation due to safety concerns."
    return "Needs additional review."


def build_clinical_bucket(row: pd.Series) -> str:
    known_brca = int(row.get("known_brca", 0) or 0)
    category = str(row.get("category", ""))

    if known_brca:
        return "Breast cancer current/relevant use"
    if category == "Approved":
        return "Breast cancer expansion candidate"
    if category == "Candidate":
        return "Breast cancer exploratory candidate"
    return "Breast cancer caution / low priority"


def main() -> None:
    step6 = pd.read_csv(STEP6_DIR / "top15_validated.csv")
    step7 = pd.read_csv(STEP7_DIR / "final_drug_candidates.csv")
    step7_json = json.loads((STEP7_DIR / "step7_admet_results.json").read_text())

    merged = step7.merge(
        step6[
            [
                "drug_id",
                "validation_score",
                "final_rank",
                "known_brca",
                "target_expressed",
                "brca_pathway",
                "survival_sig",
                "survival_p",
            ]
        ],
        on="drug_id",
        how="left",
        suffixes=("_step7", "_step6"),
    )

    merged = merged.rename(
        columns={
            "final_rank_step7": "final_rank",
            "pred_ic50": "predicted_ic50",
            "category": "category",
        }
    )
    if "final_rank" not in merged.columns and "final_rank_step7" in merged.columns:
        merged["final_rank"] = merged["final_rank_step7"]

    merged["clinical_bucket"] = merged.apply(build_clinical_bucket, axis=1)
    merged["recommendation_note"] = merged.apply(build_recommendation_note, axis=1)

    cols = [
        "final_rank",
        "drug_id",
        "drug_name",
        "target",
        "pathway",
        "predicted_ic50",
        "sensitivity_rate",
        "validation_score",
        "safety_score",
        "combined_score",
        "category",
        "clinical_bucket",
        "known_brca",
        "target_expressed",
        "brca_pathway",
        "survival_sig",
        "survival_p",
        "flags",
        "n_assays_tested",
        "recommendation_note",
    ]
    merged = merged[cols].sort_values(["final_rank", "combined_score"], ascending=[True, False])

    approved = merged[merged["category"] == "Approved"].copy()
    candidate = merged[merged["category"] == "Candidate"].copy()
    caution = merged[merged["category"] == "Caution"].copy()

    summary = {
        "ensemble_name": "FRC (FlatMLP + ResidualMLP + CrossAttention)",
        "input_bundle": "exact slim + strong context + SMILES",
        "n_total": int(len(merged)),
        "n_approved": int(len(approved)),
        "n_candidate": int(len(candidate)),
        "n_caution": int(len(caution)),
        "top5_final": merged.head(5)["drug_name"].tolist(),
        "assay_count": int(step7_json.get("n_assays", 0)),
    }

    merged.to_csv(OUT_DIR / "top15_comprehensive_table.csv", index=False)
    approved.to_csv(OUT_DIR / "approved_shortlist.csv", index=False)
    candidate.to_csv(OUT_DIR / "candidate_shortlist.csv", index=False)
    caution.to_csv(OUT_DIR / "caution_shortlist.csv", index=False)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    html = f"""<html>
<head>
  <meta charset="utf-8">
  <title>FRC Post-ADMET Summary</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; padding: 24px; line-height: 1.5; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .meta {{ background: #f6f8fa; border: 1px solid #d0d7de; padding: 12px 16px; border-radius: 8px; margin-bottom: 20px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 14px 0 28px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f6f8fa; }}
    tr:nth-child(even) {{ background: #fcfcfd; }}
  </style>
</head>
<body>
  <h1>FRC Ensemble Post-ADMET Summary</h1>
  <div class="meta">
    <div><strong>Ensemble</strong>: {summary["ensemble_name"]}</div>
    <div><strong>Input</strong>: {summary["input_bundle"]}</div>
    <div><strong>Approved / Candidate / Caution</strong>: {summary["n_approved"]} / {summary["n_candidate"]} / {summary["n_caution"]}</div>
    <div><strong>Top 5</strong>: {", ".join(summary["top5_final"])}</div>
  </div>

  <h2>Top 15 Comprehensive Table</h2>
  {merged.to_html(index=False)}

  <h2>Approved Shortlist</h2>
  {approved.to_html(index=False)}

  <h2>Candidate Shortlist</h2>
  {candidate.to_html(index=False)}

  <h2>Caution Shortlist</h2>
  {caution.to_html(index=False)}
</body>
</html>"""

    (OUT_DIR / "top15_comprehensive_table.html").write_text(html, encoding="utf-8")

    print(f"Saved comprehensive summary to: {OUT_DIR}")


if __name__ == "__main__":
    main()
