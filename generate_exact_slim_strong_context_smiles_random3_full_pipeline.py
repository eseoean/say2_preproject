from __future__ import annotations

import csv
import html
import json
from pathlib import Path


ROOT = Path("/Users/skku_aws2_18/pre_project/say2_preproject")
RESULTS = ROOT / "Improving GroupCV" / "results"
MODELS = ROOT / "models"
OUT = ROOT / "exact_slim_strong_context_smiles_random3_full_pipeline.html"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def fmt_elapsed(seconds: object) -> str:
    if seconds in {None, "", "-"}:
        return "-"
    try:
        sec = float(seconds)
    except Exception:
        return str(seconds)
    if sec >= 3600:
        return f"{sec / 3600:.1f}h"
    if sec >= 60:
        return f"{sec / 60:.1f}m"
    return f"{sec:.0f}s"


def normalize_clinical_bucket(value: str) -> str:
    mapping = {
        "Breast cancer current/relevant use": "유방암 현재 사용",
        "Breast cancer exploratory candidate": "적응증 확장/연구 중",
    }
    return mapping.get(value, value)


def bucket_class(bucket: str) -> str:
    return {
        "유방암 현재 사용": "bucket-current",
        "적응증 확장/연구 중": "bucket-trial",
        "유방암 미사용": "bucket-novel",
    }.get(bucket, "bucket-neutral")


def build_model_rows(models: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for model in models:
        overall = model["overall_metrics"]
        rows.append(
            {
                "model": model["model"],
                "spearman": overall["spearman"],
                "rmse": overall["rmse"],
                "mae": overall["mae"],
                "pearson": overall["pearson"],
                "r2": overall["r2"],
                "ndcg": overall["ndcg@20"],
                "time": fmt_elapsed(model.get("elapsed_sec")),
                "family": model.get("family", "-"),
            }
        )
    return sorted(rows, key=lambda row: row["spearman"], reverse=True)


def render_model_table(rows: list[dict]) -> str:
    out: list[str] = []
    for idx, row in enumerate(rows, start=1):
        out.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td><b>{html.escape(row['model'])}</b></td>"
            f"<td>{fmt_num(row['spearman'])}</td>"
            f"<td>{fmt_num(row['rmse'])}</td>"
            f"<td>{fmt_num(row['mae'])}</td>"
            f"<td>{fmt_num(row['pearson'])}</td>"
            f"<td>{fmt_num(row['r2'])}</td>"
            f"<td>{fmt_num(row['ndcg'])}</td>"
            f"<td>{row['time']}</td>"
            "</tr>"
        )
    return "\n".join(out)


def render_weights_table(weights: dict, meta: dict) -> str:
    rows = []
    for model, weight in sorted(weights.items(), key=lambda kv: kv[1], reverse=True):
        info = meta[model]
        rows.append(
            "<tr>"
            f"<td><b>{html.escape(model)}</b></td>"
            f"<td>{html.escape(info['family'])}</td>"
            f"<td>{fmt_num(weight, 4)}</td>"
            f"<td>{fmt_num(info['overall_metrics']['spearman'])}</td>"
            f"<td>{fmt_num(info['overall_metrics']['rmse'])}</td>"
            f"<td>{fmt_elapsed(info.get('elapsed_sec'))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def render_step6_table(rows: list[dict]) -> str:
    out: list[str] = []
    for row in rows:
        out.append(
            "<tr>"
            f"<td>{row['final_rank']}</td>"
            f"<td><b>{html.escape(row['drug_name'])}</b></td>"
            f"<td>{html.escape(row['target'])}</td>"
            f"<td>{html.escape(row['pathway'])}</td>"
            f"<td>{fmt_num(row['mean_pred_ic50'], 3)}</td>"
            f"<td>{fmt_pct(row['sensitivity_rate'], 0)}</td>"
            f"<td>{'YES' if row['target_expressed'] == '1' else '-'}</td>"
            f"<td>{'YES' if row['brca_pathway'] == '1' else '-'}</td>"
            f"<td>{'YES' if row['survival_sig'] == '1' else '-'}</td>"
            f"<td>{fmt_num(row['validation_score'], 2)}</td>"
            "</tr>"
        )
    return "\n".join(out)


def render_step7_table(rows: list[dict]) -> str:
    out: list[str] = []
    for row in rows:
        bucket = normalize_clinical_bucket(row.get("clinical_bucket", "-"))
        flags = row.get("flags", "").strip()
        flags_text = flags if flags not in {"", "[]"} else "-"
        out.append(
            "<tr>"
            f"<td>{row['final_rank']}</td>"
            f"<td><b>{html.escape(row['drug_name'])}</b></td>"
            f"<td><span class='{bucket_class(bucket)}'>{html.escape(bucket)}</span></td>"
            f"<td>{html.escape(row['target'])}</td>"
            f"<td>{html.escape(row['pathway'])}</td>"
            f"<td>{fmt_num(row['predicted_ic50'], 3)}</td>"
            f"<td>{fmt_num(row['validation_score'], 2)}</td>"
            f"<td>{fmt_num(row['safety_score'], 2)}</td>"
            f"<td>{fmt_num(row['combined_score'], 2)}</td>"
            f"<td>{html.escape(flags_text)}</td>"
            f"<td>{html.escape(row['recommendation_note'])}</td>"
            "</tr>"
        )
    return "\n".join(out)


def render_step7plus_table(rows: list[dict], top15_rows: list[dict]) -> str:
    bucket_map = {
        row["drug_name"]: normalize_clinical_bucket(row.get("clinical_bucket", row.get("category", "-")))
        for row in top15_rows
    }
    out: list[str] = []
    for row in rows:
        bucket = bucket_map.get(row["drug_name"], normalize_clinical_bucket(row.get("category", "-")))
        out.append(
            "<tr>"
            f"<td><b>{html.escape(row['drug_name'])}</b></td>"
            f"<td><span class='{bucket_class(bucket)}'>{html.escape(bucket)}</span></td>"
            f"<td>{fmt_num(row['combined_score'], 2)}</td>"
            f"<td>{row['api_success_count']}</td>"
            f"<td>{row['faers_count']}</td>"
            f"<td>{row['trial_count']}</td>"
            f"<td>{row['target_count']}</td>"
            f"<td>{row['pathway_count']}</td>"
            f"<td>{row['pubmed_general_count']}</td>"
            f"<td>{row['pubmed_breast_cancer_count']}</td>"
            "</tr>"
        )
    return "\n".join(out)


def main() -> None:
    ml = load_json(RESULTS / "exact_repo_random3_strong_context_smiles_ml_v1.json")
    dl = load_json(RESULTS / "exact_repo_random3_strong_context_smiles_dl_v1.json")
    ensemble = load_json(RESULTS / "exact_repo_random3_strong_context_smiles_ensemble_v1.json")
    step6 = load_json(MODELS / "metabric_results_random3_strong_context_smiles" / "step6_metabric_results.json")
    step6_top15 = load_csv(MODELS / "metabric_results_random3_strong_context_smiles" / "top15_validated.csv")
    step7_summary = load_json(MODELS / "post_admet_summary_random3_strong_context_smiles" / "summary.json")
    step7_top15 = load_csv(MODELS / "post_admet_summary_random3_strong_context_smiles" / "top15_comprehensive_table.csv")
    step7plus_rows = load_csv(MODELS / "kg_api_results_random3_strong_context_smiles" / "kg_api_summary.csv")

    ml_rows = build_model_rows(ml["models"])
    dl_rows = build_model_rows(dl["models"])
    best_ml = ml_rows[0]
    best_dl = dl_rows[0]
    weighted = ensemble["weighted_overall_metrics"]
    diversity = ensemble["diversity"]["summary"]

    html_text = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>sample 3-fold 통합 문서 - Step 4 to Step 7+ KG/API</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }}
  .header {{ background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 24px 32px; border-bottom: 1px solid #334155; }}
  .header h1 {{ font-size: 1.45rem; color: #f8fafc; margin-bottom: 8px; }}
  .header p {{ color: #cbd5e1; line-height: 1.7; max-width: 980px; }}
  .header a {{ color: #38bdf8; text-decoration: none; font-size: 0.9rem; display:inline-block; margin-top:10px; }}
  .sticky-nav {{ position: sticky; top: 0; z-index: 20; background: rgba(15,23,42,0.92); backdrop-filter: blur(10px); border-bottom: 1px solid #334155; }}
  .sticky-nav .inner {{ max-width: 1380px; margin: 0 auto; padding: 12px 24px; display: flex; gap: 10px; flex-wrap: wrap; }}
  .sticky-nav a {{ text-decoration: none; padding: 8px 12px; border-radius: 999px; border: 1px solid #334155; background: #111827; color: #cbd5e1; font-size: 0.78rem; font-weight: 700; }}
  .container {{ max-width: 1380px; margin: 0 auto; padding: 24px; }}
  .section {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 22px; margin-bottom: 20px; }}
  .section h2 {{ font-size: 1.18rem; margin-bottom: 6px; }}
  .section .sub {{ color: #94a3b8; font-size: 0.78rem; line-height: 1.6; margin-bottom: 16px; }}
  .hero-grid, .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
  .hero-card, .metric-card {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px; }}
  .hero-card .value, .metric-card .value {{ font-size: 1.35rem; font-weight: 800; }}
  .hero-card .label, .metric-card .label {{ font-size: 0.72rem; color: #64748b; margin-top: 5px; }}
  .banner {{ background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.30); border-radius: 10px; padding: 14px 16px; line-height: 1.7; margin-bottom: 18px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.78rem; }}
  th {{ text-align: left; padding: 8px 10px; background: #0f172a; color: #94a3b8; border-bottom: 2px solid #334155; font-weight: 600; white-space: nowrap; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid rgba(51,65,85,0.6); vertical-align: top; }}
  tr:hover {{ background: rgba(59,130,246,0.05); }}
  .split {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .callout {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px 16px; line-height: 1.7; color: #cbd5e1; }}
  .bucket-current, .bucket-trial, .bucket-novel, .bucket-neutral {{ padding: 2px 8px; border-radius: 999px; font-size: 0.7rem; font-weight: 700; display: inline-block; }}
  .bucket-current {{ background: #22c55e20; color: #4ade80; }}
  .bucket-trial {{ background: #3b82f620; color: #60a5fa; }}
  .bucket-novel {{ background: #a855f720; color: #c084fc; }}
  .bucket-neutral {{ background: #64748b20; color: #cbd5e1; }}
  .footnote {{ margin-top: 12px; color: #94a3b8; font-size: 0.76rem; line-height: 1.6; }}
  @media (max-width: 960px) {{ .split {{ grid-template-columns: 1fr; }} }}
  @media (max-width: 640px) {{
    .container {{ padding: 14px; }}
    .section {{ padding: 16px; }}
    table {{ display: block; overflow-x: auto; white-space: nowrap; }}
  }}
</style>
</head>
<body>
<div class="header">
  <h1>sample 3-fold 통합 문서 - Step 4 → Step 7+ KG/API</h1>
  <p><b>exact slim + strong context + SMILES</b> 입력과 <b>random sample 3-fold (seed=42)</b> 기준으로, 모델 비교부터 앙상블, METABRIC, ADMET, KG/API까지 한 흐름으로 이어서 보는 통합 문서입니다.</p>
  <a href="dashboard.html">&larr; 대시보드로 돌아가기</a>
</div>

<div class="sticky-nav">
  <div class="inner">
    <a href="#overview">Overview</a>
    <a href="#step4">Step 4 모델</a>
    <a href="#step5">Step 5 앙상블</a>
    <a href="#step6">Step 6 METABRIC</a>
    <a href="#step7">Step 7 ADMET</a>
    <a href="#step7plus">Step 7+ KG/API</a>
  </div>
</div>

<div class="container">
  <section class="section" id="overview">
    <h2>Overview</h2>
    <div class="sub">현재 문서의 모든 수치는 sample 3-fold 기준입니다. 같은 약물이 train/valid에 동시에 등장할 수 있으므로, 이 문서는 ceiling과 interpolation 관점의 요약으로 읽는 것이 맞습니다.</div>
    <div class="banner">
      <b>실험 요약:</b> Step 4에서는 <b>CatBoost</b>가 최고 ML, <b>TabNet</b>이 최고 DL이었고, Step 5에서는 <b>CatBoost + XGBoost + LightGBM + TabNet + ResidualMLP + WideDeep</b> 혼합 weighted ensemble이 최고 성능을 냈습니다. 이후 Step 6/7/7+에서 이 경로를 기준으로 외부 검증, ADMET, KG/API 해석을 붙였습니다.
    </div>
    <div class="hero-grid">
      <div class="hero-card"><div class="value">6,366</div><div class="label">Rows</div></div>
      <div class="hero-card"><div class="value">5,625</div><div class="label">ML Feature Dim</div></div>
      <div class="hero-card"><div class="value">{best_ml['model']}</div><div class="label">Best ML</div></div>
      <div class="hero-card"><div class="value">{fmt_num(best_ml['spearman'], 4)}</div><div class="label">Best ML Spearman</div></div>
      <div class="hero-card"><div class="value">{best_dl['model']}</div><div class="label">Best DL</div></div>
      <div class="hero-card"><div class="value">{fmt_num(best_dl['spearman'], 4)}</div><div class="label">Best DL Spearman</div></div>
      <div class="hero-card"><div class="value">{fmt_num(weighted['spearman'], 4)}</div><div class="label">Best Ensemble Spearman</div></div>
      <div class="hero-card"><div class="value">{fmt_num(weighted['rmse'], 4)}</div><div class="label">Best Ensemble RMSE</div></div>
    </div>
  </section>

  <section class="section" id="step4">
    <h2>Step 4. 단일 모델 비교</h2>
    <div class="sub">ML 6개, DL 6개를 같은 sample 3-fold 분할에서 비교했습니다.</div>
    <div class="split">
      <div>
        <div class="callout" style="margin-bottom:12px;"><b>ML 요약:</b> CatBoost, XGBoost, LightGBM이 상위권을 형성했고, CatBoost가 최고 ML이었습니다.</div>
        <table>
          <thead><tr><th>#</th><th>Model</th><th>Sp</th><th>RMSE</th><th>MAE</th><th>Pearson</th><th>R²</th><th>NDCG@20</th><th>Time</th></tr></thead>
          <tbody>{render_model_table(ml_rows)}</tbody>
        </table>
      </div>
      <div>
        <div class="callout" style="margin-bottom:12px;"><b>DL 요약:</b> TabNet, ResidualMLP, WideDeep이 상위권을 형성했고, TabNet이 최고 DL이었습니다.</div>
        <table>
          <thead><tr><th>#</th><th>Model</th><th>Sp</th><th>RMSE</th><th>MAE</th><th>Pearson</th><th>R²</th><th>NDCG@20</th><th>Time</th></tr></thead>
          <tbody>{render_model_table(dl_rows)}</tbody>
        </table>
      </div>
    </div>
  </section>

  <section class="section" id="step5">
    <h2>Step 5. 혼합 앙상블</h2>
    <div class="sub">상위 ML/DL을 섞은 6-model weighted ensemble을 최종 sample 3-fold 기준 경로로 사용했습니다.</div>
    <div class="metric-grid">
      <div class="metric-card"><div class="value">{fmt_num(weighted['spearman'], 4)}</div><div class="label">Weighted Spearman</div></div>
      <div class="metric-card"><div class="value">{fmt_num(weighted['rmse'], 4)}</div><div class="label">Weighted RMSE</div></div>
      <div class="metric-card"><div class="value">{fmt_num(weighted['mae'], 4)}</div><div class="label">Weighted MAE</div></div>
      <div class="metric-card"><div class="value">{fmt_num(weighted['pearson'], 4)}</div><div class="label">Weighted Pearson</div></div>
      <div class="metric-card"><div class="value">{fmt_num(weighted['r2'], 4)}</div><div class="label">Weighted R²</div></div>
      <div class="metric-card"><div class="value">{fmt_num(ensemble['total_elapsed_sec'] / 60, 1)}m</div><div class="label">Total Time</div></div>
    </div>
    <div class="split" style="margin-top:16px;">
      <div>
        <div class="callout" style="margin-bottom:12px;"><b>선택된 base model:</b> CatBoost, XGBoost, LightGBM, TabNet, ResidualMLP, WideDeep</div>
        <table>
          <thead><tr><th>Model</th><th>Family</th><th>Weight</th><th>Sp</th><th>RMSE</th><th>Time</th></tr></thead>
          <tbody>{render_weights_table(ensemble['weights'], ensemble['selected_model_meta'])}</tbody>
        </table>
      </div>
      <div>
        <div class="callout">
          <b>Diversity summary</b><br>
          avg prediction Pearson: <b>{fmt_num(diversity['avg_prediction_pearson'], 4)}</b><br>
          avg prediction Spearman: <b>{fmt_num(diversity['avg_prediction_spearman'], 4)}</b><br>
          avg residual Pearson: <b>{fmt_num(diversity['avg_residual_pearson'], 4)}</b><br>
          avg residual Spearman: <b>{fmt_num(diversity['avg_residual_spearman'], 4)}</b><br>
          avg mean abs gap: <b>{fmt_num(diversity['avg_mean_abs_prediction_gap'], 4)}</b>
        </div>
        <div class="footnote">sample 3-fold에서도 residual correlation은 여전히 높은 편이라, 앙상블은 주로 평균화 이득을 주고 완전히 다른 오류 패턴을 만드는 단계는 아닙니다.</div>
      </div>
    </div>
  </section>

  <section class="section" id="step6">
    <h2>Step 6. METABRIC 외부 검증</h2>
    <div class="sub">sample 3-fold weighted ensemble top30을 기준으로 METABRIC BRCA 외부 검증을 수행했습니다.</div>
    <div class="metric-grid">
      <div class="metric-card"><div class="value">{step6['method_a']['n_targets_expressed']}/{step6['method_a']['n_total']}</div><div class="label">Target Expressed</div></div>
      <div class="metric-card"><div class="value">{step6['method_a']['n_brca_pathway']}/{step6['method_a']['n_total']}</div><div class="label">BRCA Pathway</div></div>
      <div class="metric-card"><div class="value">{step6['method_b']['n_significant']}/{step6['method_a']['n_total']}</div><div class="label">Survival Significant</div></div>
      <div class="metric-card"><div class="value">{fmt_pct(step6['method_c']['precision_at_k']['P@15']['precision'], 1)}</div><div class="label">P@15</div></div>
      <div class="metric-card"><div class="value">{fmt_num(step6['method_b']['rsf_c_index'], 4)}</div><div class="label">RSF C-index</div></div>
      <div class="metric-card"><div class="value">{fmt_num(step6['method_c']['graphsage_p20'], 2)}</div><div class="label">GraphSAGE P@20</div></div>
    </div>
    <div class="callout" style="margin-top:16px;">대표 validated 상위 후보는 <b>Bortezomib</b>, <b>Romidepsin</b>, <b>Sepantronium bromide</b>, <b>Docetaxel</b>, <b>Dactinomycin</b> 입니다.</div>
    <table style="margin-top:16px;">
      <thead><tr><th>#</th><th>Drug</th><th>Target</th><th>Pathway</th><th>Pred IC50</th><th>Sensitivity</th><th>Expr</th><th>BRCA Pathway</th><th>Survival</th><th>Validation</th></tr></thead>
      <tbody>{render_step6_table(step6_top15)}</tbody>
    </table>
  </section>

  <section class="section" id="step7">
    <h2>Step 7. ADMET 및 post-ADMET 분류</h2>
    <div class="sub">raw ADMET 결과와 팀원 레포 형식의 post-ADMET 임상 분류를 함께 정리했습니다.</div>
    <div class="metric-grid">
      <div class="metric-card"><div class="value">{step7_summary['n_approved_raw']}</div><div class="label">Approved</div></div>
      <div class="metric-card"><div class="value">{step7_summary['n_candidate_raw']}</div><div class="label">Candidate</div></div>
      <div class="metric-card"><div class="value">{step7_summary['n_caution_raw']}</div><div class="label">Caution</div></div>
      <div class="metric-card"><div class="value">{step7_summary['n_current_use']}</div><div class="label">유방암 현재 사용</div></div>
      <div class="metric-card"><div class="value">{step7_summary['n_expansion']}</div><div class="label">적응증 확장/연구 중</div></div>
      <div class="metric-card"><div class="value">{step7_summary['n_novel']}</div><div class="label">유방암 미사용</div></div>
    </div>
    <div class="callout" style="margin-top:16px;">Top 5 final candidate: <b>{html.escape(', '.join(step7_summary['top5_final']))}</b></div>
    <table style="margin-top:16px;">
      <thead><tr><th>#</th><th>Drug</th><th>임상 분류</th><th>Target</th><th>Pathway</th><th>Pred IC50</th><th>Validation</th><th>Safety</th><th>Combined</th><th>Flags</th><th>Recommendation</th></tr></thead>
      <tbody>{render_step7_table(step7_top15)}</tbody>
    </table>
  </section>

  <section class="section" id="step7plus">
    <h2>Step 7+. KG/API 근거 수집</h2>
    <div class="sub">ClinicalTrials, PubMed, FAERS, target/pathway 근거를 final top15 후보에 대해 수집해 사람이 설명 가능한 후보로 정리했습니다.</div>
    <div class="metric-grid">
      <div class="metric-card"><div class="value">15</div><div class="label">KG/API Candidate Count</div></div>
      <div class="metric-card"><div class="value">{html.escape(step7_summary['top5_final'][0])}</div><div class="label">Top Candidate</div></div>
      <div class="metric-card"><div class="value">7</div><div class="label">Response Types</div></div>
      <div class="metric-card"><div class="value">{fmt_num(max(int(r['api_success_count']) for r in step7plus_rows), 0)}</div><div class="label">Max API Success Count</div></div>
    </div>
    <div class="callout" style="margin-top:16px;">KG/API 요약 상위권 대표 후보는 <b>Romidepsin</b>, <b>Sepantronium bromide</b>, <b>Staurosporine</b>입니다.</div>
    <table style="margin-top:16px;">
      <thead><tr><th>Drug</th><th>임상 분류</th><th>Combined</th><th>API Success</th><th>FAERS</th><th>Trials</th><th>Targets</th><th>Pathways</th><th>PubMed</th><th>PubMed BRCA</th></tr></thead>
      <tbody>{render_step7plus_table(step7plus_rows, step7_top15)}</tbody>
    </table>
  </section>
</div>
</body>
</html>
"""

    OUT.write_text(html_text, encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
