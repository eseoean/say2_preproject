from __future__ import annotations

import csv
import html
import json
from pathlib import Path


ROOT = Path("/Users/skku_aws2_18/pre_project/say2_preproject")
MODELS = ROOT / "models"
OUT = ROOT / "step7plus_detail.html"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fmt_num(value: object, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def normalize_clinical_bucket(value: str) -> str:
    mapping = {
        "Breast cancer current/relevant use": "유방암 현재 사용",
        "Breast cancer exploratory candidate": "적응증 확장/연구 중",
    }
    return mapping.get(value, value)


def badge_class(category: str) -> str:
    return {
        "유방암 현재 사용": "cat-approved",
        "적응증 확장/연구 중": "cat-candidate",
        "유방암 미사용": "cat-caution",
    }.get(category, "cat-neutral")


def yes_no_class(value: str) -> str:
    return "val-good" if str(value) == "1" else "val-dim"


def build_top15_rows(rows: list[dict]) -> str:
    parts: list[str] = []
    for row in rows:
        category = normalize_clinical_bucket(row.get("clinical_bucket", row["category"]))
        flags = row["flags"].strip()
        flags_text = flags if flags not in {"[]", ""} else "-"
        parts.append(
            "<tr>"
            f"<td>{html.escape(row['final_rank'])}</td>"
            f"<td><b>{html.escape(row['drug_name'])}</b></td>"
            f"<td><span class='{badge_class(category)}'>{html.escape(category)}</span></td>"
            f"<td>{html.escape(row['target'])}</td>"
            f"<td>{html.escape(row['pathway'])}</td>"
            f"<td class='val-good'>{fmt_num(row['predicted_ic50'], 3)}</td>"
            f"<td>{fmt_num(row['validation_score'], 2)}</td>"
            f"<td>{fmt_num(row['safety_score'], 2)}</td>"
            f"<td class='val-good'>{fmt_num(row['combined_score'], 2)}</td>"
            f"<td class='{yes_no_class(row['target_expressed'])}'>{html.escape(row['target_expressed'])}</td>"
            f"<td class='{yes_no_class(row['brca_pathway'])}'>{html.escape(row['brca_pathway'])}</td>"
            f"<td class='{yes_no_class(row['survival_sig'])}'>{html.escape(row['survival_sig'])}</td>"
            f"<td>{html.escape(flags_text)}</td>"
            f"<td>{html.escape(row['recommendation_note'])}</td>"
            "</tr>"
        )
    return "\n".join(parts)


def build_kg_rows(rows: list[dict], top15_rows: list[dict]) -> str:
    category_map = {
        row["drug_name"]: normalize_clinical_bucket(row.get("clinical_bucket", row.get("category", "-")))
        for row in top15_rows
    }
    parts: list[str] = []
    for row in rows:
        category = category_map.get(row["drug_name"], normalize_clinical_bucket(row.get("category", "-")))
        parts.append(
            "<tr>"
            f"<td><b>{html.escape(row['drug_name'])}</b></td>"
            f"<td><span class='{badge_class(category)}'>{html.escape(category)}</span></td>"
            f"<td>{fmt_num(row['combined_score'], 2)}</td>"
            f"<td class='val-good'>{html.escape(row['api_success_count'])}</td>"
            f"<td>{html.escape(row['faers_count'])}</td>"
            f"<td>{html.escape(row['trial_count'])}</td>"
            f"<td>{html.escape(row['target_count'])}</td>"
            f"<td>{html.escape(row['pathway_count'])}</td>"
            f"<td>{html.escape(row['pubmed_general_count'])}</td>"
            f"<td>{html.escape(row['pubmed_breast_cancer_count'])}</td>"
            "</tr>"
        )
    return "\n".join(parts)


def main() -> None:
    post_dir = MODELS / "post_admet_summary_frc_strong_context_smiles"
    kg_dir = MODELS / "kg_api_results_frc_strong_context_smiles"
    metabric_dir = MODELS / "metabric_results_frc_strong_context_smiles"
    admet_dir = MODELS / "admet_results_frc_strong_context_smiles"

    summary = load_json(post_dir / "summary.json")
    step6 = load_json(metabric_dir / "step6_metabric_results.json")
    step7 = load_json(admet_dir / "step7_admet_results.json")
    top15 = load_csv(post_dir / "top15_comprehensive_table.csv")
    kg_rows = load_csv(kg_dir / "kg_api_summary.csv")

    metrics = step6["ensemble_metrics"]
    top5 = ", ".join(summary["top5_final"])
    approved = load_csv(post_dir / "approved_shortlist.csv")
    candidates = load_csv(post_dir / "candidate_shortlist.csv")
    caution = load_csv(post_dir / "caution_shortlist.csv")

    approved_names = ", ".join(r["drug_name"] for r in approved)
    candidate_names = ", ".join(r["drug_name"] for r in candidates)
    caution_names = ", ".join(r["drug_name"] for r in caution)

    html_text = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>단계 7+: KG/API 검증 - 최종 후보 근거 요약</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }}
  .header {{ background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 20px 32px; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; gap: 16px; }}
  .header h1 {{ font-size: 1.3rem; color: #f8fafc; }}
  .header a {{ color: #38bdf8; text-decoration: none; font-size: 0.85rem; }}
  .container {{ max-width: 1380px; margin: 0 auto; padding: 24px; }}
  .section {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 20px; margin-bottom: 20px; }}
  .section-title {{ font-size: 1.08rem; font-weight: 700; margin-bottom: 4px; display: flex; align-items: center; gap: 10px; }}
  .section-sub {{ font-size: 0.78rem; color: #64748b; margin-bottom: 16px; line-height: 1.5; }}
  .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
  .metric-card {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px; text-align: center; }}
  .metric-card .value {{ font-size: 1.4rem; font-weight: 800; }}
  .metric-card .label {{ font-size: 0.72rem; color: #64748b; margin-top: 4px; }}
  .highlight {{ background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.30); border-radius: 10px; padding: 16px; margin-bottom: 16px; }}
  .stack {{ display: grid; grid-template-columns: 1.3fr 1fr; gap: 16px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.78rem; }}
  th {{ text-align: left; padding: 8px 10px; background: #0f172a; color: #94a3b8; border-bottom: 2px solid #334155; font-weight: 600; white-space: nowrap; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid rgba(51,65,85,0.6); vertical-align: top; }}
  tr:hover {{ background: rgba(59,130,246,0.05); }}
  .val-good {{ color: #4ade80; font-weight: 700; }}
  .val-dim {{ color: #94a3b8; }}
  .cat-approved, .cat-candidate, .cat-caution, .cat-neutral {{ padding: 2px 8px; border-radius: 999px; font-size: 0.7rem; font-weight: 700; display: inline-block; }}
  .cat-approved {{ background: #22c55e20; color: #4ade80; }}
  .cat-candidate {{ background: #3b82f620; color: #60a5fa; }}
  .cat-caution {{ background: #ef444420; color: #f87171; }}
  .cat-neutral {{ background: #64748b20; color: #cbd5e1; }}
  .list-card {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px; }}
  .list-card h3 {{ font-size: 0.92rem; margin-bottom: 8px; }}
  .list-card p {{ color: #cbd5e1; line-height: 1.6; font-size: 0.8rem; }}
  .toolbar {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }}
  .pill {{ padding: 6px 10px; border-radius: 999px; font-size: 0.72rem; font-weight: 700; }}
  .pill-blue {{ background: rgba(56,189,248,0.12); color: #7dd3fc; }}
  .pill-green {{ background: rgba(34,197,94,0.12); color: #86efac; }}
  .pill-amber {{ background: rgba(245,158,11,0.12); color: #fcd34d; }}
  @media (max-width: 980px) {{
    .stack {{ grid-template-columns: 1fr; }}
  }}
  @media (max-width: 640px) {{
    .container {{ padding: 14px; }}
    .section {{ padding: 16px; }}
    table {{ display: block; overflow-x: auto; white-space: nowrap; }}
  }}
</style>
</head>
<body>
<div class="header">
  <h1>단계 7+: KG/API 검증 - 최종 후보 근거 요약</h1>
  <a href="dashboard.html">&larr; 파이프라인 대시보드로 돌아가기</a>
</div>
<div class="container">

  <div class="highlight">
    <div style="font-size:1rem; font-weight:700; color:#7dd3fc; margin-bottom:8px;">FRC 최종 라우트 요약</div>
    <div style="line-height:1.7; color:#cbd5e1;">
      <b>{html.escape(summary['ensemble_name'])}</b> 조합과 <b>{html.escape(summary['input_bundle'])}</b> 입력을 사용해
      METABRIC 외부 검증, ADMET 게이트, KG/API 근거 수집까지 팀원 레포 흐름 기준으로 이어서 정리했습니다.
    </div>
    <div class="toolbar">
      <span class="pill pill-blue">Step 6: METABRIC</span>
      <span class="pill pill-green">Step 7: ADMET Gate</span>
      <span class="pill pill-amber">Step 7+: KG/API Evidence</span>
    </div>
  </div>

  <div class="section">
    <div class="section-title">핵심 성능 및 최종 분류</div>
    <div class="section-sub">GroupCV 기준 best ensemble 성능과 Step 6/7 최종 후보 분포를 함께 요약했습니다.</div>
    <div class="metric-grid">
      <div class="metric-card"><div class="value">{fmt_num(metrics['spearman'], 3)}</div><div class="label">Ensemble Spearman</div></div>
      <div class="metric-card"><div class="value">{fmt_num(metrics['rmse'], 3)}</div><div class="label">Ensemble RMSE</div></div>
      <div class="metric-card"><div class="value">{fmt_num(metrics['mae'], 3)}</div><div class="label">Ensemble MAE</div></div>
      <div class="metric-card"><div class="value">{summary['n_approved']}</div><div class="label">Approved</div></div>
      <div class="metric-card"><div class="value">{summary['n_candidate']}</div><div class="label">Candidate</div></div>
      <div class="metric-card"><div class="value">{summary['n_caution']}</div><div class="label">Caution</div></div>
      <div class="metric-card"><div class="value">{step7['n_assays']}</div><div class="label">ADMET Assays</div></div>
      <div class="metric-card"><div class="value">{len(kg_rows)}</div><div class="label">KG/API Drug Count</div></div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">추천 요약</div>
    <div class="section-sub">Top 5와 category별 shortlist를 빠르게 확인할 수 있게 묶었습니다.</div>
    <div class="stack">
      <div class="list-card">
        <h3>Top 5 Final Candidates</h3>
        <p>{html.escape(top5)}</p>
      </div>
      <div class="list-card">
        <h3>Approved / Candidate / Caution</h3>
        <p><b>Approved:</b> {html.escape(approved_names)}<br><br>
        <b>Candidate:</b> {html.escape(candidate_names)}<br><br>
        <b>Caution:</b> {html.escape(caution_names)}</p>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Top 15 종합 후보 표</div>
    <div class="section-sub">Step 6 검증 지표, Step 7 이후 임상 분류, 최종 recommendation note를 한 번에 확인할 수 있는 통합 표입니다.</div>
    <table>
      <thead>
        <tr>
          <th>#</th><th>Drug</th><th>임상 분류</th><th>Target</th><th>Pathway</th>
          <th>Pred IC50</th><th>Validation</th><th>Safety</th><th>Combined</th>
          <th>Target Expr</th><th>BRCA Pathway</th><th>Survival</th><th>Flags</th><th>Recommendation</th>
        </tr>
      </thead>
      <tbody>
        {build_top15_rows(top15)}
      </tbody>
    </table>
  </div>

  <div class="section">
    <div class="section-title">KG/API 근거 수집 요약</div>
    <div class="section-sub">ClinicalTrials, PubMed, FAERS, target/pathway 근거를 최종 15개 후보별로 집계했고, 표의 분류는 post-ADMET 임상 분류를 따릅니다.</div>
    <table>
      <thead>
        <tr>
          <th>Drug</th><th>임상 분류</th><th>Combined</th><th>API Success</th><th>FAERS</th>
          <th>Trials</th><th>Targets</th><th>Pathways</th><th>PubMed</th><th>PubMed BRCA</th>
        </tr>
      </thead>
      <tbody>
        {build_kg_rows(kg_rows, top15)}
      </tbody>
    </table>
  </div>

</div>
</body>
</html>
"""

    OUT.write_text(html_text)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
