from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/skku_aws2_18/pre_project/say2_preproject")
MODELS = ROOT / "models"
OUT = ROOT / "exact_slim_strong_context_smiles_random3_step6_detail.html"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def fmt_num(value: object, digits: int = 4) -> str:
    if value in {None, "", "-"}:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def fmt_pct(value: object, digits: int = 1) -> str:
    if value in {None, "", "-"}:
        return "-"
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except Exception:
        return str(value)


def yes_no(v: object) -> str:
    return "YES" if bool(v) else "-"


def render_method_a(rows: list[dict]) -> str:
    parts = []
    for i, row in enumerate(rows, start=1):
        pct = float(row.get("pct_patients_expressing", 0.0))
        if pct > 0:
            expr = fmt_pct(pct, 0)
        else:
            # Match the teammate dashboard convention:
            # show YES for pathway/mechanism-level targets that are treated as expressed,
            # and reserve N/A only for truly unresolved cases.
            expr = "YES" if row.get("target_expressed") else "N/A"
        brca = yes_no(row.get("brca_pathway_relevant"))
        parts.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{row['drug_name']}</td>"
            f"<td>{row['target']}</td>"
            f"<td>{row['pathway']}</td>"
            f"<td>{expr}</td>"
            f"<td>{fmt_num(row.get('expr_rank_pct'), 1)}</td>"
            f"<td>{brca}</td>"
            "</tr>"
        )
    return "\n".join(parts)


def render_method_b(rows: list[dict]) -> str:
    parts = []
    for i, row in enumerate(rows, start=1):
        pval = float(row.get("log_rank_p", 1.0))
        sig = "YES" if row.get("survival_significant") else "-"
        parts.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{row['drug_name']}</td>"
            f"<td>{fmt_num(pval, 4)}</td>"
            f"<td>{fmt_num(row.get('median_os_high'), 1)}</td>"
            f"<td>{fmt_num(row.get('median_os_low'), 1)}</td>"
            f"<td>{row.get('hr_direction', '-')}</td>"
            f"<td>{sig}</td>"
            "</tr>"
        )
    return "\n".join(parts)


def render_top30(rows: list[dict], top15_names: set[str]) -> str:
    parts = []
    for i, row in enumerate(rows, start=1):
        selected = row["drug_name"] in top15_names
        klass = "selected" if selected else ""
        status = "상위 15" if selected else "탈락"
        parts.append(
            f"<tr class='{klass}'>"
            f"<td>{i}</td>"
            f"<td><b>{row['drug_name']}</b></td>"
            f"<td>{row['target']} / {row['pathway']}</td>"
            f"<td>{fmt_num(row['mean_pred_ic50'], 3)}</td>"
            f"<td>{fmt_pct(row['sensitivity_rate'], 0)}</td>"
            f"<td>{yes_no(row['target_expressed'])}</td>"
            f"<td>{yes_no(row['survival_sig'])}</td>"
            f"<td>{yes_no(row['known_brca'])}</td>"
            f"<td>{fmt_num(row['validation_score'], 2)}</td>"
            f"<td>{status}</td>"
            "</tr>"
        )
    return "\n".join(parts)


def main() -> None:
    step6 = load_json(MODELS / "metabric_results_random3_strong_context_smiles" / "step6_metabric_results.json")
    top15 = pd.read_csv(MODELS / "metabric_results_random3_strong_context_smiles" / "top15_validated.csv")
    top15_names = set(top15["drug_name"].astype(str))
    pk = step6["method_c"]["precision_at_k"]

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>exact slim + strong context + SMILES - random3 METABRIC 외부 검증</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }}
  .header {{ background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 20px 32px; border-bottom: 1px solid #334155; display:flex; justify-content:space-between; align-items:center; }}
  .header h1 {{ font-size: 1.3rem; color:#f8fafc; }}
  .header a {{ color:#38bdf8; text-decoration:none; font-size:0.85rem; }}
  .flow-nav {{ display:flex; gap:10px; flex-wrap:wrap; margin: 0 0 16px; }}
  .flow-btn {{ text-decoration:none; padding:8px 12px; border-radius:999px; border:1px solid #334155; background:#0f172a; color:#cbd5e1; font-size:0.78rem; font-weight:700; }}
  .flow-btn.active {{ background:#1d4ed8; color:#eff6ff; border-color:#3b82f6; }}
  .container {{ max-width:1300px; margin:0 auto; padding:24px; }}
  .section {{ background:#1e293b; border:1px solid #334155; border-radius:8px; padding:20px; margin-bottom:20px; }}
  .section-title {{ font-size:1.1rem; font-weight:700; margin-bottom:4px; display:flex; gap:10px; align-items:center; }}
  .section-sub {{ font-size:0.75rem; color:#64748b; margin-bottom:16px; }}
  .metric-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; }}
  .metric-card {{ background:#0f172a; border:1px solid #334155; border-radius:6px; padding:12px; text-align:center; }}
  .metric-card .value {{ font-size:1.4rem; font-weight:800; color:#4ade80; }}
  .metric-card .label {{ font-size:0.72rem; color:#64748b; margin-top:4px; }}
  .method-header {{ color:#38bdf8; font-size:1rem; font-weight:700; margin-bottom:8px; padding:8px 0; border-bottom:1px solid #334155; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.78rem; }}
  th {{ text-align:left; padding:8px 10px; background:#0f172a; color:#94a3b8; border-bottom:2px solid #334155; font-weight:600; white-space:nowrap; }}
  td {{ padding:7px 10px; border-bottom:1px solid #1e293b80; }}
  tr:hover {{ background:rgba(59,130,246,0.05); }}
  tr.selected {{ border-left:3px solid #4ade80; background:rgba(34,197,94,0.05); }}
  .highlight {{ background:#0f172a; border:1px solid #334155; border-radius:6px; padding:12px 16px; margin-bottom:16px; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:4px; font-size:0.7rem; font-weight:600; background:#22c55e20; color:#22c55e; border:1px solid #22c55e40; }}
</style>
</head>
<body>
<div class="header">
  <h1>exact slim + strong context + SMILES - random sample 3-fold METABRIC 외부 검증</h1>
  <a href="exact_slim_strong_context_smiles_random3_detail.html">&larr; random3 상세 페이지로 돌아가기</a>
</div>
<div class="container">

  <div class="flow-nav">
    <a class="flow-btn" href="exact_slim_strong_context_smiles_random3_detail.html#step4">Step 4 모델</a>
    <a class="flow-btn" href="exact_slim_strong_context_smiles_random3_detail.html#step5">Step 5 앙상블</a>
    <a class="flow-btn active" href="exact_slim_strong_context_smiles_random3_step6_detail.html">Step 6 METABRIC</a>
    <a class="flow-btn" href="exact_slim_strong_context_smiles_random3_step7_detail.html">Step 7 ADMET</a>
    <a class="flow-btn" href="exact_slim_strong_context_smiles_random3_step7plus_detail.html">Step 7+ KG/API</a>
  </div>

  <div class="section" style="border-color:#22c55e40;">
    <div class="section-title" style="color:#4ade80;">검증 요약 <span class="badge">완료</span></div>
    <div class="section-sub">random3 weighted ensemble top30을 기준으로 METABRIC BRCA 외부 검증을 다시 계산한 결과입니다.</div>
    <div class="metric-grid">
      <div class="metric-card"><div class="value">{step6['method_a']['n_targets_expressed']}/{step6['method_a']['n_total']}</div><div class="label">BRCA 발현 타겟</div></div>
      <div class="metric-card"><div class="value">{step6['method_a']['n_brca_pathway']}/{step6['method_a']['n_total']}</div><div class="label">BRCA 관련 경로</div></div>
      <div class="metric-card"><div class="value">{step6['method_b']['n_significant']}/{step6['method_a']['n_total']}</div><div class="label">생존 유의</div></div>
      <div class="metric-card"><div class="value">{fmt_pct(pk['P@15']['precision'], 1)}</div><div class="label">P@15 기존 BRCA 약물 정밀도</div></div>
      <div class="metric-card"><div class="value">{fmt_num(step6['method_b']['rsf_c_index'], 3)}</div><div class="label">RSF C-index</div></div>
      <div class="metric-card"><div class="value">{fmt_num(step6['method_c']['graphsage_p20'], 2)}</div><div class="label">GraphSAGE P@20</div></div>
    </div>
  </div>

  <div class="section">
    <div class="method-header">방법 A: 타겟 유전자 발현 검증</div>
    <div class="section-sub">METABRIC BRCA 발현 데이터 대비 약물 타겟/경로가 실제로 잡히는지 확인했습니다.</div>
    <table>
      <thead><tr><th>#</th><th>약물</th><th>타겟</th><th>경로</th><th>발현%</th><th>Rank%</th><th>BRCA 경로</th></tr></thead>
      <tbody>{render_method_a(step6['method_a']['details'])}</tbody>
    </table>
  </div>

  <div class="section">
    <div class="method-header">방법 B: 생존 계층화</div>
    <div class="section-sub">타겟 발현 기준 환자를 나눴을 때 생존 차이가 유의한지 확인했습니다.</div>
    <div class="highlight"><b>RSF 참고값:</b> C-index {fmt_num(step6['method_b']['rsf_c_index'], 4)} / AUROC {fmt_num(step6['method_b']['rsf_auroc'], 4)}</div>
    <table>
      <thead><tr><th>#</th><th>약물</th><th>P-value</th><th>OS high</th><th>OS low</th><th>Direction</th><th>Sig</th></tr></thead>
      <tbody>{render_method_b(step6['method_b']['details'])}</tbody>
    </table>
  </div>

  <div class="section">
    <div class="method-header">방법 C: 기존 약물 정밀도 (P@K)</div>
    <div class="section-sub">Top K 안에 기존 BRCA 관련 약물이 얼마나 포함되는지 봤습니다.</div>
    <div class="metric-grid">
      <div class="metric-card"><div class="value">{fmt_pct(pk['P@5']['precision'], 0)}</div><div class="label">P@5</div></div>
      <div class="metric-card"><div class="value">{fmt_pct(pk['P@10']['precision'], 0)}</div><div class="label">P@10</div></div>
      <div class="metric-card"><div class="value">{fmt_pct(pk['P@15']['precision'], 0)}</div><div class="label">P@15</div></div>
      <div class="metric-card"><div class="value">{fmt_pct(pk['P@20']['precision'], 0)}</div><div class="label">P@20</div></div>
      <div class="metric-card"><div class="value">{fmt_pct(pk['P@25']['precision'], 0)}</div><div class="label">P@25</div></div>
      <div class="metric-card"><div class="value">{fmt_pct(pk['P@30']['precision'], 0)}</div><div class="label">P@30</div></div>
    </div>
  </div>

  <div class="section" style="border-color:#fbbf2440;">
    <div class="section-title" style="color:#fbbf24;">상위 30개 약물 - 검증 결과 및 선별</div>
    <div class="section-sub">녹색 행은 최종 상위 15개입니다. 검증 점수는 발현 + 생존 + 기존 BRCA 약물 정보를 합쳐 계산했습니다.</div>
    <table>
      <thead>
        <tr><th>#</th><th>약물</th><th>타겟 / 경로</th><th>예측 IC50</th><th>민감도</th><th>발현</th><th>생존</th><th>기존 BRCA</th><th>검증 점수</th><th>상태</th></tr>
      </thead>
      <tbody>{render_top30(step6['all_30_scores'], top15_names)}</tbody>
    </table>
  </div>

</div>
</body>
</html>
"""
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
