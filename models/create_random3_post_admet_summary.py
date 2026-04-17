#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).parent
STEP6_DIR = BASE_DIR / "metabric_results_random3_strong_context_smiles"
STEP7_DIR = BASE_DIR / "admet_results_random3_strong_context_smiles"
OUT_DIR = BASE_DIR / "post_admet_summary_random3_strong_context_smiles"
OUT_DIR.mkdir(exist_ok=True)


CURRENT_USE = {
    "Docetaxel": "FDA 유방암 적응증 승인 (보조/전이성). NCCN 표준요법",
    "Paclitaxel": "FDA 유방암 적응증 승인 (보조/전이성). NCCN 표준요법",
    "Vinorelbine": "전이성 유방암 표준요법. NCCN 가이드라인 포함",
    "Vinblastine": "FDA 유방암 적응증 포함 (호르몬 불응성 유방암)",
    "Epirubicin": "FDA 유방암 적응증 승인 (보조요법 FEC 레지멘)",
}

EXPANSION = {
    "Romidepsin": "FDA 승인: CTCL/PTCL. 유방암 TNBC 임상시험 (NCT02393794, NCT01638533)",
    "SN-38": "Sacituzumab govitecan (SN-38 접합체) FDA TNBC 승인. SN-38 자체는 미승인",
    "Bortezomib": "FDA 승인: 다발성 골수종. 유방암 임상시험 진행 (NCT00025246 등)",
    "Dinaciclib": "CDK 억제제. 유방암 1/2상 임상시험 진행 (NCT01624441 등)",
    "Rapamycin": "유도체 Everolimus FDA HR+/HER2- 유방암 승인. Rapamycin 자체는 면역억제제 승인",
    "Luminespib": "HSP90 억제제. 유방암 2상 임상시험 (HER2+, TNBC 대상)",
}

NOVEL = {
    "Sepantronium bromide": "연구용 화합물. Survivin 억제제. 유방암 적응증 없음",
    "Dactinomycin": "FDA 승인: 윌름스 종양, 융모성 질환. 유방암 적응증 없음",
    "Staurosporine": "연구용 천연 화합물. FDA 미승인. 임상시험 없음",
    "Camptothecin": "연구용 천연 화합물. Irinotecan/Topotecan의 모 화합물. 직접 임상 사용 없음",
}


def load_admet_assays() -> dict:
    path = BASE_DIR / "run_step7_admet.py"
    spec = importlib.util.spec_from_file_location("run_step7_admet_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.ADMET_ASSAYS


def build_bucket(drug_name: str) -> str:
    if drug_name in CURRENT_USE:
        return "유방암 현재 사용"
    if drug_name in EXPANSION:
        return "적응증 확장/연구 중"
    if drug_name in NOVEL:
        return "유방암 미사용"
    return "추가 검토 필요"


def build_rationale(drug_name: str) -> str:
    if drug_name in CURRENT_USE:
        return CURRENT_USE[drug_name]
    if drug_name in EXPANSION:
        return EXPANSION[drug_name]
    if drug_name in NOVEL:
        return NOVEL[drug_name]
    return "추가 임상 적응증 확인 필요"


def build_recommendation_note(row: pd.Series) -> str:
    bucket = row["clinical_bucket"]
    flags = str(row.get("flags", "[]"))
    if bucket == "유방암 현재 사용":
        return "임상 현실과 일치하는 재발견 후보로, 우선 shortlist에 두기 좋습니다."
    if bucket == "적응증 확장/연구 중":
        return "유방암으로 확장 가능성이 있고, 추가 translational 검토 가치가 있습니다."
    if bucket == "유방암 미사용":
        return "재창출 관점의 exploratory candidate로 적절합니다."
    if "Ames" in flags or "DILI" in flags:
        return "효능 신호는 있지만 안전성 플래그가 있어 low-priority caution으로 유지합니다."
    return "추가 검토가 필요합니다."


def compute_admet_coverage(detailed_profiles: list[dict], assay_list: dict, assay_meta: dict) -> pd.DataFrame:
    rows = []
    total = len(detailed_profiles)
    for assay_key, assay_name in assay_list.items():
        meta = assay_meta[assay_key]
        matched = 0
        for profile in detailed_profiles:
            det = profile["assay_details"].get(assay_key, {})
            if det.get("match_type") in {"exact", "close_analog"}:
                matched += 1
        rows.append(
            {
                "category": meta["category"],
                "name": assay_name,
                "type": "이진" if meta["type"] == "binary" else "회귀",
                "matched": f"{matched}/{total}",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    step6 = pd.read_csv(STEP6_DIR / "top15_validated.csv")
    step7 = pd.read_csv(STEP7_DIR / "final_drug_candidates.csv")
    step7_json = json.loads((STEP7_DIR / "step7_admet_results.json").read_text())
    assay_meta = load_admet_assays()

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
        }
    )
    if "final_rank" not in merged.columns and "final_rank_step7" in merged.columns:
        merged["final_rank"] = merged["final_rank_step7"]

    merged["clinical_bucket"] = merged["drug_name"].map(build_bucket)
    merged["clinical_rationale"] = merged["drug_name"].map(build_rationale)
    merged["recommendation_note"] = merged.apply(build_recommendation_note, axis=1)
    merged = merged.sort_values(["final_rank", "combined_score"], ascending=[True, False])

    coverage_df = compute_admet_coverage(
        step7_json["detailed_profiles"],
        step7_json["assay_list"],
        assay_meta,
    )

    summary = {
        "ensemble_name": "random3 weighted ensemble (CatBoost + XGBoost + LightGBM + TabNet + ResidualMLP + WideDeep)",
        "input_bundle": "exact slim + strong context + SMILES",
        "split_mode": "random sample 3-fold",
        "n_total": int(len(merged)),
        "n_current_use": int((merged["clinical_bucket"] == "유방암 현재 사용").sum()),
        "n_expansion": int((merged["clinical_bucket"] == "적응증 확장/연구 중").sum()),
        "n_novel": int((merged["clinical_bucket"] == "유방암 미사용").sum()),
        "n_assays": int(step7_json.get("n_assays", 0)),
        "n_approved_raw": int((merged["category"] == "Approved").sum()),
        "n_candidate_raw": int((merged["category"] == "Candidate").sum()),
        "n_caution_raw": int((merged["category"] == "Caution").sum()),
        "top5_final": merged.head(5)["drug_name"].tolist(),
    }

    merged.to_csv(OUT_DIR / "top15_comprehensive_table.csv", index=False)
    coverage_df.to_csv(OUT_DIR / "admet_coverage_table.csv", index=False)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"Saved random3 post-ADMET summary to: {OUT_DIR}")


if __name__ == "__main__":
    main()
