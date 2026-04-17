from __future__ import annotations

import csv
import html
import json
from pathlib import Path


ROOT = Path("/Users/skku_aws2_18/pre_project/say2_preproject")
IMPROVING = ROOT / "Improving GroupCV"
RESULTS = IMPROVING / "results"
MODELS = ROOT / "models"
OUT = ROOT / "exact_slim_strong_context_smiles_detail.html"


ML_JSONS = [
    RESULTS / "exact_repo_slim_strong_context_smiles_ml_groupcv_catboost_v1.json",
    RESULTS / "exact_repo_slim_strong_context_smiles_ml_groupcv_lightgbm_dart_v1.json",
    RESULTS / "exact_repo_slim_strong_context_smiles_ml_groupcv_randomforest_v1.json",
    RESULTS / "exact_repo_slim_strong_context_smiles_ml_groupcv_xgboost_v1.json",
    RESULTS / "exact_repo_slim_strong_context_smiles_ml_groupcv_lightgbm_v1.json",
    RESULTS / "exact_repo_slim_strong_context_smiles_ml_groupcv_extratrees_v1.json",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


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


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


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


def rank_class(index: int) -> str:
    if index == 1:
        return "rank-1"
    if index == 2:
        return "rank-2"
    if index == 3:
        return "rank-3"
    return ""


def badge_class(category: str) -> str:
    return {
        "Approved": "tag-pass",
        "Candidate": "tag-candidate",
        "Caution": "tag-fail",
    }.get(category, "tag-tbd")


def quality_class(value: float, mode: str) -> str:
    if mode == "spearman":
        if value >= 0.60:
            return "val-good"
        if value >= 0.56:
            return "val-warn"
        return "val-neutral"
    if mode == "rmse":
        if value <= 2.08:
            return "val-good"
        if value <= 2.17:
            return "val-warn"
        return "val-neutral"
    return "val-neutral"


def build_ml_rows() -> list[dict]:
    rows: list[dict] = []
    for path in ML_JSONS:
        payload = load_json(path)
        model = payload["models"][0]
        summary = model["summary"]
        overall = model["overall_metrics"]
        rows.append(
            {
                "family": "ML",
                "model": model["model"],
                "spearman_mean": summary["spearman_mean"],
                "spearman_std": summary["spearman_std"],
                "rmse_mean": summary["rmse_mean"],
                "rmse_std": summary["rmse_std"],
                "mae_mean": summary["mae_mean"],
                "pearson": overall["pearson"],
                "r2": overall["r2"],
                "ndcg20": overall["ndcg@20"],
                "gap_spearman": summary["gap_spearman_mean"],
                "time": fmt_elapsed(model.get("elapsed_sec")),
                "note": "GroupCV OOF",
            }
        )
    return sorted(rows, key=lambda row: row["spearman_mean"], reverse=True)


def build_dl_rows() -> list[dict]:
    rows: list[dict] = []

    top3_payload = load_json(RESULTS / "exact_repo_slim_smiles_ab_top3_v1.json")
    for model_block in top3_payload["models"]:
        variant = model_block["variants"]["strong_context_plus_smiles"]
        rows.append(
            {
                "family": "DL",
                "model": model_block["model"],
                "spearman_mean": variant["spearman_mean"],
                "spearman_std": variant["spearman_std"],
                "rmse_mean": variant["rmse_mean"],
                "rmse_std": variant["rmse_std"],
                "mae_mean": variant["mae_mean"],
                "pearson": variant["pearson_mean"],
                "r2": variant["r2_mean"],
                "ndcg20": variant["ndcg@20_mean"],
                "gap_spearman": variant["gap_spearman_mean"],
                "time": fmt_elapsed(variant.get("elapsed_sec")),
                "note": "standalone",
            }
        )

    more_payload = load_json(RESULTS / "exact_repo_slim_strong_context_smiles_more_dl_oof_v2.json")
    for model_block in more_payload["models"]:
        note = "OOF rerun"
        if model_block.get("stopped_early"):
            note = f"early stop ({model_block.get('executed_folds', 0)} folds)"
        rows.append(
            {
                "family": "DL",
                "model": model_block["model"],
                "spearman_mean": model_block["spearman_mean"],
                "spearman_std": model_block["spearman_std"],
                "rmse_mean": model_block["rmse_mean"],
                "rmse_std": model_block["rmse_std"],
                "mae_mean": model_block["mae_mean"],
                "pearson": model_block["overall_metrics"]["pearson"],
                "r2": model_block["overall_metrics"]["r2"],
                "ndcg20": model_block["overall_metrics"]["ndcg@20"],
                "gap_spearman": (
                    sum(f["gap_spearman"] for f in model_block["fold_metrics"]) / len(model_block["fold_metrics"])
                ),
                "time": fmt_elapsed(model_block.get("elapsed_sec")),
                "note": note,
            }
        )
    return sorted(rows, key=lambda row: row["spearman_mean"], reverse=True)


def build_model_rows(rows: list[dict]) -> str:
    parts: list[str] = []
    for idx, row in enumerate(rows, start=1):
        parts.append(
            "<tr>"
            f"<td><span class='rank {rank_class(idx)}'>{idx}</span></td>"
            f"<td><b>{html.escape(row['model'])}</b></td>"
            f"<td class='{quality_class(row['spearman_mean'], 'spearman')}'>{fmt_num(row['spearman_mean'])}</td>"
            f"<td class='val-dim'>{fmt_num(row['spearman_std'])}</td>"
            f"<td class='{quality_class(row['rmse_mean'], 'rmse')}'>{fmt_num(row['rmse_mean'])}</td>"
            f"<td class='val-dim'>{fmt_num(row['rmse_std'])}</td>"
            f"<td>{fmt_num(row['mae_mean'])}</td>"
            f"<td>{fmt_num(row['pearson'])}</td>"
            f"<td>{fmt_num(row['r2'])}</td>"
            f"<td>{fmt_num(row['ndcg20'])}</td>"
            f"<td>{fmt_num(row['gap_spearman'])}</td>"
            f"<td class='val-dim'>{html.escape(row['time'])}</td>"
            f"<td class='val-dim'>{html.escape(row['note'])}</td>"
            "</tr>"
        )
    return "\n".join(parts)


def build_ensemble_rows() -> tuple[str, dict]:
    frc = load_json(RESULTS / "exact_repo_slim_strong_context_smiles_frc_ensemble_oof_v2.json")
    top3 = load_json(RESULTS / "exact_repo_slim_strong_context_smiles_top3_ensemble_v1.json")
    fle = load_json(RESULTS / "exact_repo_slim_strong_context_smiles_fle_ensemble_v1.json")

    ensemble_rows = [
        {
            "name": "FRC weighted",
            "members": "FlatMLP + ResidualMLP + CrossAttention",
            "metrics": frc["weighted_overall_metrics"],
            "best": True,
            "note": "current best",
            "diversity": frc["diversity"]["summary"],
        },
        {
            "name": "Top3 weighted",
            "members": "FlatMLP + WideDeep + CrossAttention",
            "metrics": top3["weighted_overall_metrics"],
            "best": False,
            "note": "standalone top3",
            "diversity": None,
        },
        {
            "name": "FLE weighted",
            "members": "FlatMLP + LightGBM_DART + ExtraTrees",
            "metrics": fle["weighted_overall_metrics"],
            "best": False,
            "note": "mixed ML+DL",
            "diversity": fle["diversity"]["summary"],
        },
    ]

    ensemble_rows.sort(key=lambda row: row["metrics"]["spearman"], reverse=True)

    parts: list[str] = []
    for idx, row in enumerate(ensemble_rows, start=1):
        tag = "tag-pass" if row["best"] else "tag-tbd"
        m = row["metrics"]
        parts.append(
            "<tr>"
            f"<td><span class='rank {rank_class(idx)}'>{idx}</span></td>"
            f"<td><b>{html.escape(row['name'])}</b></td>"
            f"<td>{html.escape(row['members'])}</td>"
            f"<td class='{quality_class(m['spearman'], 'spearman')}'>{fmt_num(m['spearman'])}</td>"
            f"<td class='{quality_class(m['rmse'], 'rmse')}'>{fmt_num(m['rmse'])}</td>"
            f"<td>{fmt_num(m['mae'])}</td>"
            f"<td>{fmt_num(m['pearson'])}</td>"
            f"<td>{fmt_num(m['r2'])}</td>"
            f"<td>{fmt_num(m['ndcg@20'])}</td>"
            f"<td><span class='ensemble-tag {tag}'>{html.escape(row['note'])}</span></td>"
            "</tr>"
        )

    diversity_rows = []
    for row in ensemble_rows:
        if row["diversity"] is None:
            continue
        d = row["diversity"]
        diversity_rows.append(
            "<tr>"
            f"<td><b>{html.escape(row['name'])}</b></td>"
            f"<td>{fmt_num(d['avg_prediction_pearson'])}</td>"
            f"<td>{fmt_num(d['avg_prediction_spearman'])}</td>"
            f"<td>{fmt_num(d['avg_residual_pearson'])}</td>"
            f"<td>{fmt_num(d['avg_residual_spearman'])}</td>"
            f"<td>{fmt_num(d['avg_mean_abs_prediction_gap'])}</td>"
            "</tr>"
        )

    return ("\n".join(parts), "\n".join(diversity_rows), ensemble_rows[0])


def build_top15_rows(rows: list[dict]) -> str:
    parts: list[str] = []
    for row in rows:
        flags = row["flags"] if row["flags"] not in {"[]", ""} else "-"
        parts.append(
            "<tr>"
            f"<td>{html.escape(row['final_rank'])}</td>"
            f"<td><b>{html.escape(row['drug_name'])}</b></td>"
            f"<td><span class='ensemble-tag {badge_class(row['category'])}'>{html.escape(row['category'])}</span></td>"
            f"<td>{html.escape(row['target'])}</td>"
            f"<td>{html.escape(row['pathway'])}</td>"
            f"<td class='val-good'>{fmt_num(row['predicted_ic50'], 3)}</td>"
            f"<td>{fmt_num(row['validation_score'], 2)}</td>"
            f"<td>{fmt_num(row['safety_score'], 2)}</td>"
            f"<td class='val-good'>{fmt_num(row['combined_score'], 2)}</td>"
            f"<td>{html.escape(row['recommendation_note'])}</td>"
            f"<td>{html.escape(flags)}</td>"
            "</tr>"
        )
    return "\n".join(parts)


def build_kg_rows(rows: list[dict]) -> str:
    parts: list[str] = []
    for row in rows:
        parts.append(
            "<tr>"
            f"<td><b>{html.escape(row['drug_name'])}</b></td>"
            f"<td><span class='ensemble-tag {badge_class(row['category'])}'>{html.escape(row['category'])}</span></td>"
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
    exact_summary = load_json(
        IMPROVING / "v3_input_reproduction" / "exact_repo_match" / "exact_repo_match_summary.json"
    )
    frc_summary = load_json(MODELS / "post_admet_summary_frc_strong_context_smiles" / "summary.json")
    top15 = load_csv(MODELS / "post_admet_summary_frc_strong_context_smiles" / "top15_comprehensive_table.csv")
    kg_rows = load_csv(MODELS / "kg_api_results_frc_strong_context_smiles" / "kg_api_summary.csv")
    step6 = load_json(MODELS / "metabric_results_frc_strong_context_smiles" / "step6_metabric_results.json")
    step7 = load_json(MODELS / "admet_results_frc_strong_context_smiles" / "step7_admet_results.json")

    ml_rows = build_ml_rows()
    dl_rows = build_dl_rows()
    ensemble_rows_html, diversity_rows_html, best_ensemble = build_ensemble_rows()

    ml_top = ml_rows[0]
    dl_top = dl_rows[0]
    best_ensemble_metrics = best_ensemble["metrics"]
    p20 = step6["method_c"]["precision_at_k"]["P@20"]
    top5_final = ", ".join(frc_summary["top5_final"])

    html_text = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>exact slim + strong context + SMILES 상세 결과</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }}
  .header {{ background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 20px 32px; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; gap: 16px; }}
  .header h1 {{ font-size: 1.3rem; color: #f8fafc; }}
  .header a {{ color: #38bdf8; text-decoration: none; font-size: 0.85rem; }}
  .container {{ max-width: 1380px; margin: 0 auto; padding: 24px; }}
  .section {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
  .section-title {{ font-size: 1.1rem; font-weight: 700; margin-bottom: 4px; display: flex; align-items: center; gap: 10px; }}
  .section-sub {{ font-size: 0.75rem; color: #64748b; margin-bottom: 16px; line-height: 1.6; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 600; }}
  .badge-done {{ background: #22c55e20; color: #22c55e; border: 1px solid #22c55e40; }}
  .benchmark {{ background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 12px 16px; margin-bottom: 16px; display: flex; gap: 24px; flex-wrap: wrap; }}
  .bench-item {{ font-size: 0.78rem; }}
  .bench-label {{ color: #64748b; }}
  .bench-val {{ color: #fbbf24; font-weight: 700; margin-left: 4px; }}
  .highlight {{ background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.24); border-radius: 8px; padding: 14px 16px; margin-bottom: 16px; line-height: 1.7; }}
  .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-top: 12px; }}
  .metric-card {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px; }}
  .metric-card .value {{ font-size: 1.35rem; font-weight: 800; }}
  .metric-card .label {{ font-size: 0.72rem; color: #64748b; margin-top: 6px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.78rem; }}
  th {{ text-align: left; padding: 8px 10px; background: #0f172a; color: #94a3b8; border-bottom: 2px solid #334155; font-weight: 600; white-space: nowrap; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #1e293b80; white-space: nowrap; vertical-align: top; }}
  tr:hover {{ background: rgba(59,130,246,0.05); }}
  .val-good {{ color: #4ade80; font-weight: 700; }}
  .val-warn {{ color: #fbbf24; font-weight: 700; }}
  .val-neutral {{ color: #e2e8f0; }}
  .val-dim {{ color: #64748b; font-size: 0.72rem; }}
  .ensemble-tag {{ display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 0.65rem; font-weight: 700; }}
  .tag-pass {{ background: #22c55e; color: #052e16; }}
  .tag-candidate {{ background: #60a5fa; color: #082f49; }}
  .tag-fail {{ background: #ef4444; color: #fff; }}
  .tag-tbd {{ background: #64748b; color: #fff; }}
  .rank {{ font-size: 0.85rem; font-weight: 800; }}
  .rank-1 {{ color: #fbbf24; }}
  .rank-2 {{ color: #94a3b8; }}
  .rank-3 {{ color: #cd7f32; }}
  .stack {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .list-card {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px; }}
  .list-card h3 {{ font-size: 0.92rem; margin-bottom: 8px; }}
  .list-card p {{ color: #cbd5e1; line-height: 1.7; font-size: 0.8rem; }}
  @media (max-width: 960px) {{
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
  <h1>exact slim + strong context + SMILES 상세 결과</h1>
  <a href="dashboard.html">&larr; 파이프라인 대시보드로 돌아가기</a>
</div>
<div class="container">

  <div class="highlight">
    <b>실험 트랙 요약:</b> 팀원 slim 입력을 exact-column match로 재현한 뒤, strong context 5개와 SMILES 표현을 추가한 GroupCV 트랙입니다.
    이번 페이지는 <b>모델 성능</b>, <b>앙상블/diversity</b>, <b>METABRIC</b>, <b>ADMET</b>, <b>KG/API</b>까지 현재 저장된 산출물을 한 화면에 모아 정리합니다.
  </div>

  <div class="benchmark">
    <div class="bench-item"><span class="bench-label">입력셋:</span><span class="bench-val">exact slim + strong context + SMILES</span></div>
    <div class="bench-item"><span class="bench-label">분할:</span><span class="bench-val">3-fold drug GroupCV</span></div>
    <div class="bench-item"><span class="bench-label">rows:</span><span class="bench-val">{exact_summary['repo_slim_shape'][0]:,}</span></div>
    <div class="bench-item"><span class="bench-label">exact slim numeric:</span><span class="bench-val">{exact_summary['repo_slim_numeric_shape'][1]:,}</span></div>
    <div class="bench-item"><span class="bench-label">ML feature dim:</span><span class="bench-val">5,625</span></div>
    <div class="bench-item"><span class="bench-label">strong context:</span><span class="bench-val">5 cols / 100% coverage</span></div>
    <div class="bench-item"><span class="bench-label">SMILES vocab:</span><span class="bench-val">38</span></div>
    <div class="bench-item"><span class="bench-label">best ensemble:</span><span class="bench-val">FRC weighted</span></div>
  </div>

  <div class="section">
    <div class="section-title">핵심 요약 <span class="badge badge-done">완료</span></div>
    <div class="section-sub">현재 strongest single model, best ensemble, 그리고 downstream shortlist 상태를 한 번에 보여줍니다.</div>
    <div class="metric-grid">
      <div class="metric-card"><div class="value">{html.escape(dl_top['model'])}</div><div class="label">Best Single Model</div></div>
      <div class="metric-card"><div class="value">{fmt_num(dl_top['spearman_mean'])}</div><div class="label">Best Single Spearman</div></div>
      <div class="metric-card"><div class="value">{fmt_num(best_ensemble_metrics['spearman'])}</div><div class="label">Best Ensemble Spearman</div></div>
      <div class="metric-card"><div class="value">{fmt_num(best_ensemble_metrics['rmse'])}</div><div class="label">Best Ensemble RMSE</div></div>
      <div class="metric-card"><div class="value">{frc_summary['n_approved']}</div><div class="label">Approved</div></div>
      <div class="metric-card"><div class="value">{frc_summary['n_candidate']}</div><div class="label">Candidate</div></div>
      <div class="metric-card"><div class="value">{frc_summary['n_caution']}</div><div class="label">Caution</div></div>
      <div class="metric-card"><div class="value">{top5_final}</div><div class="label">Top 5 Final Candidates</div></div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">ML 모델 (6개) <span class="badge badge-done">완료</span></div>
    <div class="section-sub">exact slim + strong context + SMILES 공통 숫자 행렬(5,625 dim)로 돌린 ML GroupCV 결과입니다.</div>
    <table>
      <thead>
        <tr>
          <th>#</th><th>모델</th><th>Spearman</th><th>&plusmn; std</th><th>RMSE</th><th>&plusmn; std</th>
          <th>MAE</th><th>Pearson</th><th>R&sup2;</th><th>NDCG@20</th><th>Gap(Sp)</th><th>소요시간</th><th>비고</th>
        </tr>
      </thead>
      <tbody>
        {build_model_rows(ml_rows)}
      </tbody>
    </table>
  </div>

  <div class="section">
    <div class="section-title">DL 모델 (7개) <span class="badge badge-done">완료</span></div>
    <div class="section-sub">5529 numeric + strong context 5개 + SMILES branch 조합으로 돌린 DL GroupCV 결과입니다.</div>
    <table>
      <thead>
        <tr>
          <th>#</th><th>모델</th><th>Spearman</th><th>&plusmn; std</th><th>RMSE</th><th>&plusmn; std</th>
          <th>MAE</th><th>Pearson</th><th>R&sup2;</th><th>NDCG@20</th><th>Gap(Sp)</th><th>소요시간</th><th>비고</th>
        </tr>
      </thead>
      <tbody>
        {build_model_rows(dl_rows)}
      </tbody>
    </table>
  </div>

  <div class="section">
    <div class="section-title">앙상블 비교 <span class="badge badge-done">완료</span></div>
    <div class="section-sub">FRC는 현재 best route, Top3는 standalone 상위 3개, FLE는 mixed ML+DL 조합입니다.</div>
    <table>
      <thead>
        <tr>
          <th>#</th><th>앙상블</th><th>구성</th><th>Spearman</th><th>RMSE</th><th>MAE</th><th>Pearson</th><th>R&sup2;</th><th>NDCG@20</th><th>비고</th>
        </tr>
      </thead>
      <tbody>
        {ensemble_rows_html}
      </tbody>
    </table>
  </div>

  <div class="section">
    <div class="section-title">앙상블 Diversity 요약</div>
    <div class="section-sub">prediction/residual 상관과 평균 예측 격차를 비교해 조합이 얼마나 서로 다른 오류를 내는지 요약했습니다.</div>
    <table>
      <thead>
        <tr>
          <th>앙상블</th><th>Avg Pred Pearson</th><th>Avg Pred Spearman</th><th>Avg Residual Pearson</th><th>Avg Residual Spearman</th><th>Avg |Pred Gap|</th>
        </tr>
      </thead>
      <tbody>
        {diversity_rows_html}
      </tbody>
    </table>
  </div>

  <div class="section">
    <div class="section-title">Downstream 검증 요약</div>
    <div class="section-sub">best ensemble(FRC weighted) 기준으로 Step 6 METABRIC, Step 7 ADMET, Step 7+ KG/API 흐름을 정리했습니다.</div>
    <div class="metric-grid">
      <div class="metric-card"><div class="value">{step6['method_a']['n_targets_expressed']}/{step6['method_a']['n_total']}</div><div class="label">Target Expressed</div></div>
      <div class="metric-card"><div class="value">{step6['method_a']['n_brca_pathway']}/{step6['method_a']['n_total']}</div><div class="label">BRCA Pathway Relevant</div></div>
      <div class="metric-card"><div class="value">{step6['method_b']['n_significant']}/{step6['method_a']['n_total']}</div><div class="label">Survival Significant</div></div>
      <div class="metric-card"><div class="value">{fmt_pct(p20['precision'])}</div><div class="label">P@20 ({p20['hits']}/{p20['total']})</div></div>
      <div class="metric-card"><div class="value">{step7['n_assays']}</div><div class="label">ADMET Assays</div></div>
      <div class="metric-card"><div class="value">{len(kg_rows)}/{len(kg_rows)}</div><div class="label">KG/API Coverage</div></div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">최종 Top 15 후보</div>
    <div class="section-sub">validation score, safety score, combined score, 최종 recommendation note를 한 번에 볼 수 있는 통합표입니다.</div>
    <table>
      <thead>
        <tr>
          <th>#</th><th>Drug</th><th>Category</th><th>Target</th><th>Pathway</th><th>Pred IC50</th><th>Validation</th><th>Safety</th><th>Combined</th><th>Recommendation</th><th>Flags</th>
        </tr>
      </thead>
      <tbody>
        {build_top15_rows(top15)}
      </tbody>
    </table>
  </div>

  <div class="section">
    <div class="section-title">KG/API 수집 요약</div>
    <div class="section-sub">FAERS, ClinicalTrials, target/pathway, PubMed 근거 집계를 최종 후보별로 붙였습니다.</div>
    <table>
      <thead>
        <tr>
          <th>Drug</th><th>Category</th><th>Combined</th><th>API Success</th><th>FAERS</th><th>Trials</th><th>Targets</th><th>Pathways</th><th>PubMed</th><th>PubMed BRCA</th>
        </tr>
      </thead>
      <tbody>
        {build_kg_rows(kg_rows)}
      </tbody>
    </table>
  </div>

</div>
</body>
</html>
"""

    OUT.write_text(html_text, encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
