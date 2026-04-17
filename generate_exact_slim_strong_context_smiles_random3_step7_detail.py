from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("/Users/skku_aws2_18/pre_project/say2_preproject")
MODELS = ROOT / "models"
POST = MODELS / "post_admet_summary_random3_strong_context_smiles"
OUT = ROOT / "exact_slim_strong_context_smiles_random3_step7_detail.html"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt_num(value: object, digits: int = 2) -> str:
    if value in {None, "", "-"}:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def fmt_intish(value: object) -> str:
    if value in {None, "", "-"}:
        return "-"
    try:
        return str(int(float(value)))
    except Exception:
        return str(value)


def bucket_class(bucket: str) -> str:
    return {
        "유방암 현재 사용": "cat-current",
        "적응증 확장/연구 중": "cat-trial",
        "유방암 미사용": "cat-novel",
    }.get(bucket, "cat-novel")


def short_flag(flag: str) -> str:
    flag = flag.strip()
    replacements = {
        "Ames Mutagenicity(+)": "Ames",
        "DILI (Drug-Induced Liver Injury)(+)": "DILI",
        "hERG Cardiotoxicity(+)": "hERG",
    }
    return replacements.get(flag, flag)


def parse_flags(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text or text == "[]":
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = [part.strip().strip("'").strip('"') for part in text.split(",")]
    return [short_flag(part) for part in parts if part]


def coverage_type_class(category: str) -> str:
    return " style='color:#f87171'" if category == "Toxicity" else ""


def render_coverage_rows(rows: list[dict]) -> str:
    out = []
    for row in rows:
        out.append(
            "<tr>"
            f"<td{coverage_type_class(row['category'])}>{row['category']}</td>"
            f"<td>{row['name']}</td>"
            f"<td>{row['type']}</td>"
            f"<td>{row['matched']}</td>"
            "</tr>"
        )
    return "\n".join(out)


def render_candidate_rows(rows: list[dict]) -> str:
    out = []
    for row in rows:
        flags = parse_flags(row.get("flags", ""))
        flags_html = (
            "".join(f"<span class='flag'>{flag}</span>" for flag in flags)
            if flags
            else "-"
        )
        out.append(
            "<tr>"
            f"<td>{fmt_intish(row['final_rank'])}</td>"
            f"<td><b>{row['drug_name']}</b></td>"
            f"<td>{row['target']}</td>"
            f"<td class='val-good'>{fmt_num(row['predicted_ic50'], 3)}</td>"
            f"<td class='{'val-bad' if float(row['safety_score']) < 4 else 'val-good' if float(row['safety_score']) >= 7 else 'val-warn'}'>{fmt_num(row['safety_score'], 2)}</td>"
            f"<td class='val-good'>{fmt_num(row['combined_score'], 2)}</td>"
            f"<td>{fmt_intish(row['n_assays_tested'])}</td>"
            f"<td>{len(flags)}</td>"
            f"<td><span class='{bucket_class(row['clinical_bucket'])}'>{row['clinical_bucket']}</span></td>"
            f"<td><div class='rationale'>{row['clinical_rationale']}</div></td>"
            f"<td>{flags_html}</td>"
            "</tr>"
        )
    return "\n".join(out)


def main() -> None:
    summary = load_json(POST / "summary.json")
    top15 = load_csv(POST / "top15_comprehensive_table.csv")
    coverage = load_csv(POST / "admet_coverage_table.csv")

    top_names = ", ".join(summary["top5_final"])
    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>exact slim + strong context + SMILES - random3 Step 7 ADMET 게이트</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }}
  .header {{ background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 20px 32px; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; }}
  .header h1 {{ font-size: 1.3rem; color: #f8fafc; }}
  .header a {{ color: #38bdf8; text-decoration: none; font-size: 0.85rem; }}
  .flow-nav {{ display:flex; gap:10px; flex-wrap:wrap; margin: 0 0 16px; }}
  .flow-btn {{ text-decoration:none; padding:8px 12px; border-radius:999px; border:1px solid #334155; background:#0f172a; color:#cbd5e1; font-size:0.78rem; font-weight:700; }}
  .flow-btn.active {{ background:#1d4ed8; color:#eff6ff; border-color:#3b82f6; }}
  .container {{ max-width: 1300px; margin: 0 auto; padding: 24px; }}
  .section {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
  .section-title {{ font-size: 1.1rem; font-weight: 700; margin-bottom: 4px; display: flex; align-items: center; gap: 10px; }}
  .section-sub {{ font-size: 0.75rem; color: #64748b; margin-bottom: 16px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.78rem; }}
  th {{ text-align: left; padding: 8px 10px; background: #0f172a; color: #94a3b8; border-bottom: 2px solid #334155; font-weight: 600; white-space: nowrap; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #1e293b80; vertical-align: top; }}
  tr:hover {{ background: rgba(59,130,246,0.05); }}
  .val-good {{ color: #4ade80; font-weight: 700; }}
  .val-warn {{ color: #fbbf24; font-weight: 700; }}
  .val-bad {{ color: #f87171; font-weight: 700; }}
  .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
  .metric-card {{ background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 12px; text-align: center; }}
  .metric-card .value {{ font-size: 1.4rem; font-weight: 800; }}
  .metric-card .label {{ font-size: 0.72rem; color: #64748b; margin-top: 4px; }}
  .cat-current {{ background: #22c55e20; color: #22c55e; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 600; }}
  .cat-trial {{ background: #3b82f620; color: #3b82f6; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 600; }}
  .cat-novel {{ background: #a855f720; color: #a855f7; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 600; }}
  .flag {{ background: #f8717120; color: #f87171; padding: 1px 6px; border-radius: 3px; font-size: 0.68rem; margin: 1px; display: inline-block; }}
  .highlight {{ background: rgba(34,197,94,0.08); border: 1px solid #22c55e40; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
  .rationale {{ font-size: 0.68rem; color: #94a3b8; margin-top: 2px; line-height: 1.5; }}
  .dedup-note {{ background: #06b6d415; border: 1px solid #06b6d440; border-radius: 6px; padding: 10px 14px; margin-bottom: 16px; font-size: 0.8rem; color: #06b6d4; }}
  ul {{ font-size: 0.82rem; line-height: 1.8; padding-left: 20px; }}
</style>
</head>
<body>
<div class="header">
  <h1>exact slim + strong context + SMILES - random sample 3-fold Step 7 ADMET 게이트</h1>
  <a href="exact_slim_strong_context_smiles_random3_detail.html">&larr; random3 상세 페이지로 돌아가기</a>
</div>
<div class="container">

  <div class="flow-nav">
    <a class="flow-btn" href="exact_slim_strong_context_smiles_random3_detail.html#step4">Step 4 모델</a>
    <a class="flow-btn" href="exact_slim_strong_context_smiles_random3_detail.html#step5">Step 5 앙상블</a>
    <a class="flow-btn" href="exact_slim_strong_context_smiles_random3_step6_detail.html">Step 6 METABRIC</a>
    <a class="flow-btn active" href="exact_slim_strong_context_smiles_random3_step7_detail.html">Step 7 ADMET</a>
    <a class="flow-btn" href="exact_slim_strong_context_smiles_random3_step7plus_detail.html">Step 7+ KG/API</a>
  </div>

  <div class="dedup-note">
    <b>random3 post-ADMET summary 적용:</b> 이번 route는 <b>top15가 이미 고유 약물</b>로 구성되어 있어 별도 중복 제거는 없었고, raw ADMET 분류(Approved/Candidate/Caution) 위에 <b>유방암 현재 사용 / 적응증 확장·연구 중 / 유방암 미사용</b> 임상 분류를 추가해 팀원 레포의 제출 형식에 맞게 정리했습니다. Top 5는 <b>{top_names}</b> 입니다.
  </div>

  <div class="highlight">
    <div style="color:#4ade80; font-size:1rem; font-weight:700; margin-bottom:12px;">ADMET 안전성 평가 및 유방암 적응증 분류 요약</div>
    <div class="metric-grid">
      <div class="metric-card"><div class="value" style="color:#4ade80;">{summary['n_current_use']}</div><div class="label">유방암 현재 사용 (FDA 승인)</div></div>
      <div class="metric-card"><div class="value" style="color:#3b82f6;">{summary['n_expansion']}</div><div class="label">유방암 적응증 확장/연구 중</div></div>
      <div class="metric-card"><div class="value" style="color:#a855f7;">{summary['n_novel']}</div><div class="label">유방암 미사용 (신약 후보물질)</div></div>
      <div class="metric-card"><div class="value" style="color:#94a3b8;">{summary['n_assays']}</div><div class="label">ADMET 분석 항목</div></div>
      <div class="metric-card"><div class="value" style="color:#fbbf24;">{summary['n_total']}</div><div class="label">고유 약물 후보</div></div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">유방암 임상 적응증 분류 기준</div>
    <div class="section-sub">DrugBank · ClinicalTrials.gov · 팀원 레포 Step 7 해석 규칙 기반</div>
    <table>
      <thead><tr><th>분류</th><th>정의</th><th>약물 수</th></tr></thead>
      <tbody>
        <tr><td><span class="cat-current">유방암 현재 사용</span></td><td>FDA 유방암 적응증 승인 또는 NCCN 가이드라인 포함 표준요법</td><td class="val-good">{summary['n_current_use']}</td></tr>
        <tr><td><span class="cat-trial">적응증 확장/연구 중</span></td><td>유방암 대상 임상시험 진행 중이거나 유도체가 유방암 승인됨</td><td style="color:#3b82f6; font-weight:700;">{summary['n_expansion']}</td></tr>
        <tr><td><span class="cat-novel">유방암 미사용</span></td><td>유방암 적응증 없음, 신약 후보물질 또는 재창출 대상</td><td style="color:#a855f7; font-weight:700;">{summary['n_novel']}</td></tr>
      </tbody>
    </table>
  </div>

  <div class="section">
    <div class="section-title">ADMET 분석 커버리지 ({summary['n_assays']}개 항목)</div>
    <div class="section-sub">TDC ADMET benchmark | random3 Top15 후보 기준 | assay별 실제 매칭 수</div>
    <table>
      <thead><tr><th>범주</th><th>분석 항목</th><th>유형</th><th>매칭</th></tr></thead>
      <tbody>
        {render_coverage_rows(coverage)}
      </tbody>
    </table>
  </div>

  <div class="section">
    <div class="section-title">최종 약물 후보 15개</div>
    <div class="section-sub">random3 weighted ensemble → Step 6 validated top15 → Step 7 ADMET → 임상 분류 후 종합 정리</div>
    <table>
      <thead>
        <tr><th>#</th><th>약물</th><th>타겟</th><th>예측 IC50</th><th>안전성</th><th>종합</th><th>통과</th><th>주의</th><th>유방암 분류</th><th>분류 근거</th><th>우려사항</th></tr>
      </thead>
      <tbody>
        {render_candidate_rows(top15)}
      </tbody>
    </table>
  </div>

  <div class="section">
    <div class="section-title">주요 ADMET 결과</div>
    <div class="section-sub">random3 route의 Step 7 raw 결과와 post-ADMET 임상 분류를 함께 해석한 메모입니다.</div>
    <ul>
      <li><b>Raw ADMET 분류</b>: Approved {summary['n_approved_raw']} / Candidate {summary['n_candidate_raw']} / Caution {summary['n_caution_raw']}</li>
      <li><b>임상 분류 후 요약</b>: 현재 사용 {summary['n_current_use']} / 확장·연구 중 {summary['n_expansion']} / 미사용 {summary['n_novel']}</li>
      <li><b>가장 강한 최종 후보</b>: Romidepsin, Sepantronium bromide, Staurosporine, SN-38, Docetaxel</li>
      <li><b>DILI 플래그</b>: Docetaxel, Bortezomib, Dactinomycin, Paclitaxel, Epirubicin 등 세포독성 계열에서 주로 관찰됐습니다.</li>
      <li><b>가장 보수적으로 봐야 할 후보</b>: Epirubicin은 임상 현실과 일치하는 재발견이지만 Ames/DILI 동시 플래그 때문에 Caution으로 남습니다.</li>
      <li><b>random split 해석</b>: GroupCV보다 성능이 낙관적이므로, 임상 의미 해석은 현재 화면처럼 Step 6/7 보조 검증과 함께 보는 게 안전합니다.</li>
    </ul>
  </div>

</div>
</body>
</html>
"""
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
