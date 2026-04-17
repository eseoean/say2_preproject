from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/Users/skku_aws2_18/pre_project/say2_preproject")
RESULTS = ROOT / "Improving GroupCV" / "results"
OUT = ROOT / "exact_slim_strong_context_smiles_random3_detail.html"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def fmt_num(value: object, digits: int = 4) -> str:
    if value in {None, "", "-"}:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
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


def rank_class(index: int) -> str:
    if index == 1:
        return "rank-1"
    if index == 2:
        return "rank-2"
    if index == 3:
        return "rank-3"
    return ""


def status_info(spearman: float, rmse: float) -> tuple[str, str]:
    if spearman >= 0.86 and rmse <= 1.16:
        return ("TOP", "tag-pass")
    if spearman >= 0.85 and rmse <= 1.20:
        return ("STRONG", "tag-warn")
    return ("REFERENCE", "tag-tbd")


def quality_class(value: float, mode: str) -> str:
    if mode == "spearman":
        if value >= 0.87:
            return "val-good"
        if value >= 0.85:
            return "val-warn"
        return "val-neutral"
    if mode == "rmse":
        if value <= 1.12:
            return "val-good"
        if value <= 1.20:
            return "val-warn"
        return "val-neutral"
    return "val-neutral"


def build_rows(models: list[dict]) -> list[dict]:
    rows = []
    for model in models:
        overall = model["overall_metrics"]
        rows.append(
            {
                "model": model["model"],
                "spearman_mean": model.get("spearman_mean", overall["spearman"]),
                "spearman_std": model.get("spearman_std"),
                "rmse_mean": model.get("rmse_mean", overall["rmse"]),
                "rmse_std": model.get("rmse_std"),
                "mae_mean": model.get("mae_mean", overall["mae"]),
                "pearson": overall["pearson"],
                "r2": overall["r2"],
                "ndcg20": overall["ndcg@20"],
                "gap_spearman": model.get("summary", {}).get("gap_spearman_mean"),
                "time": fmt_elapsed(model.get("elapsed_sec")),
                "note": (
                    f"{model.get('executed_folds', 3)} folds"
                    if "executed_folds" in model
                    else "3 folds"
                ),
            }
        )
    return sorted(rows, key=lambda row: row["spearman_mean"], reverse=True)


def render_rows(rows: list[dict]) -> str:
    parts = []
    for idx, row in enumerate(rows, start=1):
        status_label, status_class = status_info(row["spearman_mean"], row["rmse_mean"])
        parts.append(
            "<tr>"
            f"<td><span class='rank {rank_class(idx)}'>{idx}</span></td>"
            f"<td><b>{row['model']}</b></td>"
            f"<td class='{quality_class(row['spearman_mean'], 'spearman')}'>{fmt_num(row['spearman_mean'])}</td>"
            f"<td class='val-dim'>{fmt_num(row['spearman_std'])}</td>"
            f"<td class='{quality_class(row['rmse_mean'], 'rmse')}'>{fmt_num(row['rmse_mean'])}</td>"
            f"<td class='val-dim'>{fmt_num(row['rmse_std'])}</td>"
            f"<td>{fmt_num(row['mae_mean'])}</td>"
            f"<td>{fmt_num(row['pearson'])}</td>"
            f"<td>{fmt_num(row['r2'])}</td>"
            f"<td>{fmt_num(row['ndcg20'])}</td>"
            f"<td>{fmt_num(row['gap_spearman'])}</td>"
            f"<td class='val-dim'>{row['time']}</td>"
            f"<td><span class='ensemble-tag {status_class}'>{status_label}</span></td>"
            f"<td class='val-dim'>{row['note']}</td>"
            "</tr>"
        )
    return "\n".join(parts)


def main() -> None:
    ml = load_json(RESULTS / "exact_repo_random3_strong_context_smiles_ml_v1.json")
    dl = load_json(RESULTS / "exact_repo_random3_strong_context_smiles_dl_v1.json")
    ensemble = load_json(RESULTS / "exact_repo_random3_strong_context_smiles_ensemble_v1.json")
    step6 = load_json(ROOT / "models" / "metabric_results_random3_strong_context_smiles" / "step6_metabric_results.json")
    step7 = load_json(ROOT / "models" / "admet_results_random3_strong_context_smiles" / "step7_admet_results.json")

    ml_rows = build_rows(ml["models"])
    dl_rows = build_rows(dl["models"])

    best_ml = ml_rows[0]
    best_dl = dl_rows[0]
    best_overall = best_ml if best_ml["spearman_mean"] >= best_dl["spearman_mean"] else best_dl

    html_text = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>exact slim + strong context + SMILES - random sample 3-fold 상세 결과</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }}
  .header {{ background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 20px 32px; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; gap: 16px; }}
  .header h1 {{ font-size: 1.3rem; color: #f8fafc; }}
  .header a {{ color: #38bdf8; text-decoration: none; font-size: 0.85rem; }}
  .flow-nav {{ display:flex; gap:10px; flex-wrap:wrap; margin: 0 0 16px; }}
  .flow-btn {{ text-decoration:none; padding:8px 12px; border-radius:999px; border:1px solid #334155; background:#0f172a; color:#cbd5e1; font-size:0.78rem; font-weight:700; }}
  .flow-btn.active {{ background:#1d4ed8; color:#eff6ff; border-color:#3b82f6; }}
  .container {{ max-width: 1320px; margin: 0 auto; padding: 24px; }}
  .section {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
  .section-title {{ font-size: 1.1rem; font-weight: 700; margin-bottom: 4px; display: flex; align-items: center; gap: 10px; }}
  .section-sub {{ font-size: 0.75rem; color: #64748b; margin-bottom: 16px; line-height: 1.6; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 600; }}
  .badge-done {{ background: #22c55e20; color: #22c55e; border: 1px solid #22c55e40; }}
  .badge-warn {{ background: #f59e0b20; color: #f59e0b; border: 1px solid #f59e0b40; }}
  .highlight {{ background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.24); border-radius: 8px; padding: 14px 16px; margin-bottom: 16px; line-height: 1.7; }}
  .benchmark {{ background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 12px 16px; margin-bottom: 16px; display: flex; gap: 24px; flex-wrap: wrap; }}
  .bench-item {{ font-size: 0.78rem; }}
  .bench-label {{ color: #64748b; }}
  .bench-val {{ color: #fbbf24; font-weight: 700; margin-left: 4px; }}
  .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-top: 12px; }}
  .metric-card {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px; }}
  .metric-card .value {{ font-size: 1.35rem; font-weight: 800; }}
  .metric-card .label {{ font-size: 0.72rem; color: #64748b; margin-top: 6px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.78rem; }}
  th {{ text-align: left; padding: 8px 10px; background: #0f172a; color: #94a3b8; border-bottom: 2px solid #334155; font-weight: 600; white-space: nowrap; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #1e293b80; white-space: nowrap; }}
  tr:hover {{ background: rgba(59,130,246,0.05); }}
  .val-good {{ color: #4ade80; font-weight: 700; }}
  .val-warn {{ color: #fbbf24; font-weight: 700; }}
  .val-neutral {{ color: #e2e8f0; }}
  .val-dim {{ color: #64748b; font-size: 0.72rem; }}
  .ensemble-tag {{ display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 0.65rem; font-weight: 700; }}
  .tag-pass {{ background: #22c55e; color: #052e16; }}
  .tag-warn {{ background: #f59e0b; color: #422006; }}
  .tag-tbd {{ background: #64748b; color: #fff; }}
  .rank {{ font-size: 0.85rem; font-weight: 800; }}
  .rank-1 {{ color: #fbbf24; }}
  .rank-2 {{ color: #94a3b8; }}
  .rank-3 {{ color: #cd7f32; }}
  .stack {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .list-card {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px; }}
  .list-card h3 {{ font-size: 0.92rem; margin-bottom: 8px; }}
  .list-card p, .list-card ul {{ color: #cbd5e1; line-height: 1.7; font-size: 0.8rem; }}
  .list-card ul {{ padding-left: 18px; }}
  .note-box {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px 16px; line-height: 1.7; color: #cbd5e1; }}
  .legend {{ display: flex; gap: 16px; margin-top: 12px; font-size: 0.72rem; color: #64748b; flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .legend-chip {{ width: 14px; height: 14px; border-radius: 4px; }}
  .legend-top {{ background: #22c55e; }}
  .legend-strong {{ background: #f59e0b; }}
  .legend-ref {{ background: #64748b; }}
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
  <h1>exact slim + strong context + SMILES - random sample 3-fold 상세 결과</h1>
  <a href="dashboard.html">&larr; 파이프라인 대시보드로 돌아가기</a>
</div>
<div class="container">

  <div class="flow-nav">
    <a class="flow-btn active" href="exact_slim_strong_context_smiles_random3_detail.html#step4">Step 4 모델</a>
    <a class="flow-btn" href="exact_slim_strong_context_smiles_random3_detail.html#step5">Step 5 앙상블</a>
    <a class="flow-btn" href="exact_slim_strong_context_smiles_random3_step6_detail.html">Step 6 METABRIC</a>
    <a class="flow-btn" href="exact_slim_strong_context_smiles_random3_step7_detail.html">Step 7 ADMET</a>
    <a class="flow-btn" href="exact_slim_strong_context_smiles_random3_step7plus_detail.html">Step 7+ KG/API</a>
  </div>

  <div class="highlight">
    <b>실험 트랙 요약:</b> `exact slim + strong context + SMILES` 입력을 기준으로, `random sample 3-fold KFold (seed=42)`에서 ML/DL 성능을 정리한 페이지입니다.
    같은 약물이 train/valid에 동시에 들어갈 수 있는 분할이므로, <b>GroupCV보다 훨씬 낙관적으로 나오는 수치</b>라는 점을 같이 봐야 합니다.
  </div>

  <div class="benchmark">
    <div class="bench-item"><span class="bench-label">입력셋:</span><span class="bench-val">exact slim + strong context + SMILES</span></div>
    <div class="bench-item"><span class="bench-label">분할:</span><span class="bench-val">random sample 3-fold</span></div>
    <div class="bench-item"><span class="bench-label">seed:</span><span class="bench-val">{ml['seed']}</span></div>
    <div class="bench-item"><span class="bench-label">rows:</span><span class="bench-val">{ml['rows']:,}</span></div>
    <div class="bench-item"><span class="bench-label">ML feature dim:</span><span class="bench-val">{ml['feature_dim']:,}</span></div>
    <div class="bench-item"><span class="bench-label">DL numeric dim:</span><span class="bench-val">{dl['x_shape'][1]:,}</span></div>
    <div class="bench-item"><span class="bench-label">strong context:</span><span class="bench-val">{len(dl['strong_context_columns'])} cols</span></div>
    <div class="bench-item"><span class="bench-label">SMILES vocab:</span><span class="bench-val">{dl['smiles_vocab_size']}</span></div>
  </div>

  <div class="section" id="step4">
    <div class="section-title">핵심 요약 <span class="badge badge-done">완료</span></div>
    <div class="section-sub">random sample 기준 최고 ML, 최고 DL, 전체 최고 수치를 먼저 요약했습니다.</div>
    <div class="metric-grid">
      <div class="metric-card"><div class="value">{best_ml['model']}</div><div class="label">Best ML</div></div>
      <div class="metric-card"><div class="value">{fmt_num(best_ml['spearman_mean'])}</div><div class="label">Best ML Spearman</div></div>
      <div class="metric-card"><div class="value">{best_dl['model']}</div><div class="label">Best DL</div></div>
      <div class="metric-card"><div class="value">{fmt_num(best_dl['spearman_mean'])}</div><div class="label">Best DL Spearman</div></div>
      <div class="metric-card"><div class="value">{best_overall['model']}</div><div class="label">Best Overall</div></div>
      <div class="metric-card"><div class="value">{fmt_num(best_overall['rmse_mean'])}</div><div class="label">Best Overall RMSE</div></div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">입력 구성과 해석 포인트</div>
    <div class="section-sub">이번 random track이 어떤 입력을 쓰는지와, 왜 GroupCV보다 성능이 높게 나오는지를 같이 적어뒀습니다.</div>
    <div class="stack">
      <div class="list-card">
        <h3>입력 구성</h3>
        <ul>
          <li>exact slim numeric: 5,529 dim</li>
          <li>ML: 5,529 numeric + 32 strong context one-hot + 64 SMILES SVD = 5,625 dim</li>
          <li>DL: 5,529 numeric + strong context 5개 + SMILES branch</li>
          <li>strong context: TCGA_DESC, PATHWAY_NAME_NORMALIZED, classification, drug_bridge_strength, stage3_resolution_status</li>
        </ul>
      </div>
      <div class="list-card">
        <h3>해석 메모</h3>
        <p>
          random sample split은 같은 약물이 train/valid 양쪽에 등장할 수 있어서,
          unseen-drug generalization을 보는 GroupCV보다 점수가 훨씬 높게 나옵니다.
          그래서 이 페이지는 <b>ceiling / interpolation 성능 확인용</b>으로 해석하는 게 가장 적절합니다.
        </p>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">ML 모델 (6개) <span class="badge badge-done">완료</span></div>
    <div class="section-sub">strong context + SMILES가 one-hot / numeric 형태로 합쳐진 공통 ML 입력에서의 random sample 3-fold 결과입니다.</div>
    <table>
      <thead>
        <tr>
          <th>#</th><th>모델</th><th>Spearman</th><th>&plusmn; std</th><th>RMSE</th><th>&plusmn; std</th><th>MAE</th><th>Pearson</th><th>R&sup2;</th><th>NDCG@20</th><th>Gap(Sp)</th><th>소요시간</th><th>Status</th><th>비고</th>
        </tr>
      </thead>
      <tbody>
        {render_rows(ml_rows)}
      </tbody>
    </table>
    <div class="legend">
      <div class="legend-item"><span class="legend-chip legend-top"></span> TOP: Spearman ≥ 0.86 and RMSE ≤ 1.16</div>
      <div class="legend-item"><span class="legend-chip legend-strong"></span> STRONG: Spearman ≥ 0.85 and RMSE ≤ 1.20</div>
      <div class="legend-item"><span class="legend-chip legend-ref"></span> REFERENCE: 그 외</div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">DL 모델 (6개) <span class="badge badge-done">완료</span></div>
    <div class="section-sub">5529 numeric + strong context + SMILES branch로 학습한 random sample 3-fold 결과입니다.</div>
    <table>
      <thead>
        <tr>
          <th>#</th><th>모델</th><th>Spearman</th><th>&plusmn; std</th><th>RMSE</th><th>&plusmn; std</th><th>MAE</th><th>Pearson</th><th>R&sup2;</th><th>NDCG@20</th><th>Gap(Sp)</th><th>소요시간</th><th>Status</th><th>비고</th>
        </tr>
      </thead>
      <tbody>
        {render_rows(dl_rows)}
      </tbody>
    </table>
    <div class="legend">
      <div class="legend-item"><span class="legend-chip legend-top"></span> TOP: Spearman ≥ 0.86 and RMSE ≤ 1.16</div>
      <div class="legend-item"><span class="legend-chip legend-strong"></span> STRONG: Spearman ≥ 0.85 and RMSE ≤ 1.20</div>
      <div class="legend-item"><span class="legend-chip legend-ref"></span> REFERENCE: 그 외</div>
    </div>
  </div>

  <div class="section" id="step5">
    <div class="section-title">앙상블 결과 (random3) <span class="badge badge-done">완료</span></div>
    <div class="section-sub">top 3 ML + top 3 DL 조합으로 equal / Spearman-weighted ensemble을 재구성했습니다.</div>
    <div class="highlight">
      <b>선택 조합:</b> {' + '.join(ensemble['selected_models'])}<br>
      <b>Weighted ensemble:</b> Spearman {fmt_num(ensemble['weighted_overall_metrics']['spearman'])}, RMSE {fmt_num(ensemble['weighted_overall_metrics']['rmse'])}, MAE {fmt_num(ensemble['weighted_overall_metrics']['mae'])}, Pearson {fmt_num(ensemble['weighted_overall_metrics']['pearson'])}, R² {fmt_num(ensemble['weighted_overall_metrics']['r2'])}
    </div>
    <div class="metric-grid">
      <div class="metric-card"><div class="value">{fmt_num(ensemble['weighted_overall_metrics']['spearman'])}</div><div class="label">Weighted Spearman</div></div>
      <div class="metric-card"><div class="value">{fmt_num(ensemble['weighted_overall_metrics']['rmse'])}</div><div class="label">Weighted RMSE</div></div>
      <div class="metric-card"><div class="value">{fmt_num(ensemble['equal_overall_metrics']['spearman'])}</div><div class="label">Equal Spearman</div></div>
      <div class="metric-card"><div class="value">{fmt_num(ensemble['equal_overall_metrics']['rmse'])}</div><div class="label">Equal RMSE</div></div>
      <div class="metric-card"><div class="value">{fmt_elapsed(ensemble['total_elapsed_sec'])}</div><div class="label">Base Model Total Time</div></div>
      <div class="metric-card"><div class="value">{fmt_num(ensemble['diversity']['summary']['avg_prediction_pearson'])}</div><div class="label">Avg Pred Pearson</div></div>
    </div>
    <table style="margin-top:16px;">
      <thead>
        <tr>
          <th>모델</th><th>Family</th><th>가중치</th><th>OOF Spearman</th><th>OOF RMSE</th><th>소요시간</th>
        </tr>
      </thead>
      <tbody>
        {''.join(
            "<tr>"
            f"<td><b>{name}</b></td>"
            f"<td>{ensemble['selected_model_meta'][name]['family']}</td>"
            f"<td>{fmt_num(ensemble['weights'][name])}</td>"
            f"<td>{fmt_num(ensemble['selected_model_meta'][name]['overall_metrics']['spearman'])}</td>"
            f"<td>{fmt_num(ensemble['selected_model_meta'][name]['overall_metrics']['rmse'])}</td>"
            f"<td class='val-dim'>{fmt_elapsed(ensemble['selected_model_meta'][name]['elapsed_sec'])}</td>"
            "</tr>"
            for name in ensemble['selected_models']
        )}
      </tbody>
    </table>
  </div>

  <div class="section">
    <div class="section-title">METABRIC 외부 검증 (Step 6) <span class="badge badge-done">완료</span></div>
    <div class="section-sub">방금 계산한 random3 앙상블 top30을 기준으로 METABRIC BRCA 코호트 외부 검증을 다시 수행했습니다. 상세 페이지는 <a href="exact_slim_strong_context_smiles_random3_step6_detail.html" style="color:#38bdf8;">여기</a>서 볼 수 있습니다.</div>
    <div class="metric-grid">
      <div class="metric-card"><div class="value">{step6['method_a']['n_targets_expressed']}/{step6['method_a']['n_total']}</div><div class="label">Target Expressed</div></div>
      <div class="metric-card"><div class="value">{step6['method_a']['n_brca_pathway']}/{step6['method_a']['n_total']}</div><div class="label">BRCA Pathway Relevant</div></div>
      <div class="metric-card"><div class="value">{step6['method_b']['n_significant']}/{step6['method_a']['n_total']}</div><div class="label">Survival Significant</div></div>
      <div class="metric-card"><div class="value">{fmt_num(step6['method_c']['precision_at_k']['P@15']['precision'] * 100, 1)}%</div><div class="label">P@15 Known BRCA Precision</div></div>
      <div class="metric-card"><div class="value">{fmt_num(step6['method_b']['rsf_c_index'], 3)}</div><div class="label">RSF C-index</div></div>
      <div class="metric-card"><div class="value">{fmt_num(step6['method_c']['graphsage_p20'], 2)}</div><div class="label">GraphSAGE P@20</div></div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">ADMET 게이트 (Step 7) <span class="badge badge-done">완료</span></div>
    <div class="section-sub">random3 Step 6 Top15를 기준으로 ADMET 22개 항목을 평가했습니다. 상세 페이지는 <a href="exact_slim_strong_context_smiles_random3_step7_detail.html" style="color:#38bdf8;">여기</a>서 볼 수 있습니다.</div>
    <div class="metric-grid">
      <div class="metric-card"><div class="value">{sum(1 for row in step7['final_candidates'] if row['category'] == 'Approved')}</div><div class="label">Approved</div></div>
      <div class="metric-card"><div class="value">{sum(1 for row in step7['final_candidates'] if row['category'] == 'Candidate')}</div><div class="label">Candidate</div></div>
      <div class="metric-card"><div class="value">{sum(1 for row in step7['final_candidates'] if row['category'] == 'Caution')}</div><div class="label">Caution</div></div>
      <div class="metric-card"><div class="value">{step7['n_assays']}</div><div class="label">ADMET Assays</div></div>
      <div class="metric-card"><div class="value">{step7['final_candidates'][0]['drug_name']}</div><div class="label">Top Final Candidate</div></div>
      <div class="metric-card"><div class="value">{fmt_num(step7['final_candidates'][0]['combined_score'], 2)}</div><div class="label">Top Combined Score</div></div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">해석 메모 <span class="badge badge-warn">주의</span></div>
    <div class="section-sub">이 페이지는 의도적으로 random sample 결과만 모아 보여줍니다.</div>
    <div class="note-box">
      1. 이 분할은 같은 약물이 train/valid에 동시에 등장할 수 있어서, unseen-drug 일반화 성능을 보는 GroupCV보다 훨씬 높게 나옵니다.<br>
      2. 그래서 여기 수치는 <b>모델의 ceiling과 interpolation 성능</b>을 보는 참고용으로 해석하는 게 맞습니다.<br>
      3. 실제 약물 추천 파이프라인과 METABRIC/ADMET/KG/API는 계속 <b>GroupCV 기반 best route</b>를 기준으로 유지하는 게 안전합니다.
    </div>
  </div>

</div>
</body>
</html>
"""

    OUT.write_text(html_text, encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
