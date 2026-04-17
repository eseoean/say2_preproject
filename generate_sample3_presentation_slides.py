from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path("/Users/skku_aws2_18/pre_project/say2_preproject")
RESULTS = ROOT / "Improving GroupCV" / "results"
MODELS = ROOT / "models"
OUT = ROOT / "sample3_pipeline_presentation_20260416.html"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def bar_chart_svg(
    items: list[tuple[str, float, str]],
    title: str,
    value_fmt: str = "{:.4f}",
    width: int = 560,
    height: int = 270,
    ymin: float | None = None,
    ymax: float | None = None,
) -> str:
    if not items:
        return ""
    vals = [v for _, v, _ in items]
    lo = min(vals) if ymin is None else ymin
    hi = max(vals) if ymax is None else ymax
    if math.isclose(hi, lo):
        hi = lo + 1.0
    margin = {"l": 54, "r": 20, "t": 34, "b": 62}
    cw = width - margin["l"] - margin["r"]
    ch = height - margin["t"] - margin["b"]
    bar_w = cw / max(len(items), 1) * 0.62
    gap = cw / max(len(items), 1)

    ticks = 4
    tick_lines = []
    label_lines = []
    for i in range(ticks + 1):
        val = lo + (hi - lo) * i / ticks
        y = margin["t"] + ch - ch * i / ticks
        tick_lines.append(
            f"<line x1='{margin['l']}' y1='{y:.1f}' x2='{width - margin['r']}' y2='{y:.1f}' stroke='rgba(148,163,184,0.18)' stroke-width='1'/>"
        )
        label_lines.append(
            f"<text x='{margin['l'] - 10}' y='{y + 4:.1f}' text-anchor='end' fill='#94a3b8' font-size='11'>{val:.2f}</text>"
        )

    bars = []
    for idx, (label, value, color) in enumerate(items):
        x = margin["l"] + gap * idx + (gap - bar_w) / 2
        h = (value - lo) / (hi - lo) * ch
        y = margin["t"] + ch - h
        bars.append(
            f"""
            <rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="8" fill="{color}" />
            <text x="{x + bar_w/2:.1f}" y="{y - 8:.1f}" text-anchor="middle" fill="#f8fafc" font-size="11" font-weight="700">{value_fmt.format(value)}</text>
            <text x="{x + bar_w/2:.1f}" y="{height - 26:.1f}" text-anchor="middle" fill="#cbd5e1" font-size="11">{label}</text>
            """
        )

    return f"""
    <div class="chart-card">
      <div class="chart-title">{title}</div>
      <svg viewBox="0 0 {width} {height}" class="svg-chart" role="img" aria-label="{title}">
        {''.join(tick_lines)}
        {''.join(label_lines)}
        {''.join(bars)}
      </svg>
    </div>
    """


def horizontal_weight_svg(weights: list[tuple[str, float, str]], width: int = 540, height: int = 250) -> str:
    margin = {"l": 150, "r": 26, "t": 20, "b": 16}
    cw = width - margin["l"] - margin["r"]
    row_h = 32
    rows = []
    for idx, (label, value, color) in enumerate(weights):
        y = margin["t"] + idx * row_h
        bar_w = cw * value / max(w for _, w, _ in weights)
        rows.append(
            f"""
            <text x="{margin['l'] - 10}" y="{y + 16}" text-anchor="end" fill="#cbd5e1" font-size="11">{label}</text>
            <rect x="{margin['l']}" y="{y + 4}" width="{cw}" height="16" rx="8" fill="rgba(148,163,184,0.12)" />
            <rect x="{margin['l']}" y="{y + 4}" width="{bar_w:.1f}" height="16" rx="8" fill="{color}" />
            <text x="{margin['l'] + bar_w + 8:.1f}" y="{y + 17}" fill="#f8fafc" font-size="11" font-weight="700">{value:.4f}</text>
            """
        )
    return f"""
    <div class="chart-card">
      <div class="chart-title">Step 5 weighted ensemble 구성 비중</div>
      <svg viewBox="0 0 {width} {height}" class="svg-chart" role="img" aria-label="Ensemble Weights">
        {''.join(rows)}
      </svg>
    </div>
    """


def donut_svg(values: list[tuple[str, float, str]], size: int = 260) -> str:
    total = sum(v for _, v, _ in values) or 1
    cx = cy = size / 2
    r = 72
    stroke = 24
    circumference = 2 * math.pi * r
    offset = 0.0
    rings = []
    legend = []
    for idx, (label, value, color) in enumerate(values):
        frac = value / total
        dash = circumference * frac
        rings.append(
            f"<circle cx='{cx}' cy='{cy}' r='{r}' fill='none' stroke='{color}' stroke-width='{stroke}' stroke-linecap='round' stroke-dasharray='{dash:.2f} {circumference - dash:.2f}' stroke-dashoffset='{-offset:.2f}' transform='rotate(-90 {cx} {cy})'/>"
        )
        offset += dash
        legend.append(
            f"<div class='legend-row'><span class='legend-dot' style='background:{color}'></span><span>{label}</span><b>{int(value)}</b></div>"
        )
    return f"""
    <div class="donut-wrap">
      <svg viewBox="0 0 {size} {size}" class="donut">
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="rgba(148,163,184,0.14)" stroke-width="{stroke}"/>
        {''.join(rings)}
        <text x="{cx}" y="{cy - 4}" text-anchor="middle" fill="#f8fafc" font-size="34" font-weight="800">{int(total)}</text>
        <text x="{cx}" y="{cy + 20}" text-anchor="middle" fill="#94a3b8" font-size="12">Final Top15</text>
      </svg>
      <div class="donut-legend">{''.join(legend)}</div>
    </div>
    """


def progress_card(label: str, num: int, den: int, color: str, note: str = "") -> str:
    pct = num / den * 100 if den else 0
    return f"""
    <div class="progress-card">
      <div class="progress-head">
        <span>{label}</span>
        <b>{num}/{den}</b>
      </div>
      <div class="progress-track"><div class="progress-fill" style="width:{pct:.1f}%; background:{color}"></div></div>
      <div class="progress-foot">{pct:.1f}% {note}</div>
    </div>
    """


def pipeline_boxes() -> str:
    steps = [
        ("Step 1", "수집 & 정합성", "GDSC / DepMap / LINCS / TCGA-BRCA / DrugBank / ChEMBL"),
        ("Step 2", "Feature Eng.", "sample-drug pair 생성\nexact slim + context + SMILES"),
        ("Step 3", "모델 학습", "ML 6개 + DL 6개\nrandom sample 3-fold"),
        ("Step 4", "앙상블", "CatBoost + XGBoost + LightGBM + TabNet + ResidualMLP + WideDeep"),
        ("Step 5", "외부 검증", "METABRIC 타깃 발현 / survival / precision"),
        ("Step 6", "안전성 평가", "ADMET raw gate + post-ADMET 임상 분류"),
        ("Step 7", "근거 수집", "KG/API (Trials, PubMed, FAERS, target/pathway)"),
    ]
    boxes = []
    for idx, (kicker, title, desc) in enumerate(steps):
        cls = "flow-box box-main" if idx < 4 else "flow-box box-final"
        boxes.append(
            f"""
            <div class="{cls}">
              <div class="flow-kicker">{kicker}</div>
              <div class="flow-title">{title}</div>
              <div class="flow-desc">{desc}</div>
            </div>
            """
        )
    return "".join(boxes)


def table_rows(rows: list[list[str]]) -> str:
    return "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)


def main() -> None:
    ml = load_json(RESULTS / "exact_repo_random3_strong_context_smiles_ml_v1.json")
    dl = load_json(RESULTS / "exact_repo_random3_strong_context_smiles_dl_v1.json")
    ens = load_json(RESULTS / "exact_repo_random3_strong_context_smiles_ensemble_v1.json")
    step6 = load_json(MODELS / "metabric_results_random3_strong_context_smiles" / "step6_metabric_results.json")
    step7 = load_json(MODELS / "post_admet_summary_random3_strong_context_smiles" / "summary.json")
    top15 = load_csv(MODELS / "post_admet_summary_random3_strong_context_smiles" / "top15_comprehensive_table.csv")
    kg = load_csv(MODELS / "kg_api_results_random3_strong_context_smiles" / "kg_api_summary.csv")

    top_ml = sorted(ml["models"], key=lambda x: x["overall_metrics"]["spearman"], reverse=True)[:3]
    top_dl = sorted(dl["models"], key=lambda x: x["overall_metrics"]["spearman"], reverse=True)[:3]

    step4_chart = bar_chart_svg(
        [
            ("CatBoost", top_ml[0]["overall_metrics"]["spearman"], "#22d3ee"),
            ("XGBoost", top_ml[1]["overall_metrics"]["spearman"], "#38bdf8"),
            ("LightGBM", top_ml[2]["overall_metrics"]["spearman"], "#60a5fa"),
            ("TabNet", top_dl[0]["overall_metrics"]["spearman"], "#34d399"),
            ("ResidualMLP", top_dl[1]["overall_metrics"]["spearman"], "#a78bfa"),
            ("WideDeep", top_dl[2]["overall_metrics"]["spearman"], "#fbbf24"),
            ("Ensemble", ens["weighted_overall_metrics"]["spearman"], "#f97316"),
        ],
        "Step 4~5 Spearman 비교",
        value_fmt="{:.4f}",
        ymin=0.84,
        ymax=0.875,
    )

    step4_rmse_chart = bar_chart_svg(
        [
            ("CatBoost", top_ml[0]["overall_metrics"]["rmse"], "#22d3ee"),
            ("XGBoost", top_ml[1]["overall_metrics"]["rmse"], "#38bdf8"),
            ("LightGBM", top_ml[2]["overall_metrics"]["rmse"], "#60a5fa"),
            ("TabNet", top_dl[0]["overall_metrics"]["rmse"], "#34d399"),
            ("ResidualMLP", top_dl[1]["overall_metrics"]["rmse"], "#a78bfa"),
            ("WideDeep", top_dl[2]["overall_metrics"]["rmse"], "#fbbf24"),
            ("Ensemble", ens["weighted_overall_metrics"]["rmse"], "#f97316"),
        ],
        "Step 4~5 RMSE 비교",
        value_fmt="{:.3f}",
        ymin=1.09,
        ymax=1.18,
    )

    weight_items = [
        (name, weight, color)
        for (name, weight), color in zip(
            sorted(ens["weights"].items(), key=lambda kv: kv[1], reverse=True),
            ["#22d3ee", "#38bdf8", "#60a5fa", "#34d399", "#a78bfa", "#fbbf24"],
        )
    ]
    weight_chart = horizontal_weight_svg(weight_items)

    validated_top5 = [
        ("Bortezomib", "Proteasome", "12.79"),
        ("Romidepsin", "HDAC1/2/3/8", "19.00"),
        ("Sepantronium bromide", "BIRC5", "16.50"),
        ("Docetaxel", "Microtubule stabiliser", "12.83"),
        ("Dactinomycin", "RNA polymerase", "13.00"),
    ]
    validated_rows = table_rows([[str(i + 1), d, t, s] for i, (d, t, s) in enumerate(validated_top5)])

    top5_rows = table_rows(
        [
            [str(i + 1), row["drug_name"], row["clinical_bucket"], row["combined_score"]]
            for i, row in enumerate(top15[:5])
        ]
    )

    kg_top5 = kg[:5]
    kg_rows = table_rows(
        [
            [
                row["drug_name"],
                row["trial_count"],
                row["pubmed_general_count"],
                row["pubmed_breast_cancer_count"],
                row["target_count"],
                row["pathway_count"],
            ]
            for row in kg_top5
        ]
    )

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>유방암 관련 약물 재창출 파이프라인 구축 - 발표 슬라이드</title>
  <style>
    :root {{
      --bg: #0a1628;
      --panel: #102238;
      --panel-2: #152b45;
      --line: rgba(87, 143, 198, 0.22);
      --text: #eef4fb;
      --muted: #8ea3bb;
      --cyan: #30c6ff;
      --cyan-2: #22d3ee;
      --green: #34d399;
      --yellow: #fbbf24;
      --orange: #f97316;
      --purple: #a78bfa;
      --pink: #fb7185;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #f4f6fa;
      font-family: 'Pretendard', 'Apple SD Gothic Neo', 'Noto Sans KR', 'Segoe UI', sans-serif;
      color: var(--text);
    }}
    .deck {{
      width: min(1400px, calc(100vw - 48px));
      margin: 18px auto 60px;
    }}
    .slide-no {{
      font-size: 14px;
      color: #475569;
      margin: 16px 0 6px 8px;
    }}
    .slide {{
      position: relative;
      min-height: 768px;
      background: linear-gradient(180deg, #0d1f34 0%, #0b1a2d 100%);
      border-radius: 20px;
      padding: 54px 58px;
      overflow: hidden;
      box-shadow: 0 22px 60px rgba(15, 23, 42, 0.18);
      page-break-after: always;
    }}
    .slide::before, .slide::after {{
      content: "";
      position: absolute;
      border: 1px solid rgba(48,198,255,0.10);
      border-radius: 999px;
      pointer-events: none;
    }}
    .slide::before {{ width: 360px; height: 360px; right: -90px; top: -50px; }}
    .slide::after {{ width: 420px; height: 420px; left: -120px; bottom: -180px; }}
    .kicker {{
      font-size: 15px;
      letter-spacing: 0.28em;
      color: var(--cyan);
      font-weight: 800;
      margin-bottom: 18px;
    }}
    .title {{
      font-size: 60px;
      font-weight: 900;
      line-height: 1.18;
      margin: 0 0 18px;
    }}
    .subtitle {{
      font-size: 28px;
      color: var(--cyan);
      font-weight: 700;
      margin-bottom: 24px;
    }}
    .hero-pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: rgba(148,163,184,0.12);
      border: 1px solid rgba(148,163,184,0.20);
      color: #dbeafe;
      border-radius: 999px;
      padding: 14px 26px;
      font-size: 20px;
      margin: 14px 0 28px;
    }}
    .people {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 18px;
      margin-top: 38px;
      padding-top: 22px;
      border-top: 2px solid rgba(48,198,255,0.18);
    }}
    .person {{
      text-align: center;
    }}
    .person .name {{ font-size: 20px; font-weight: 800; margin-bottom: 4px; }}
    .person .role {{ font-size: 15px; color: var(--muted); }}
    .section-title {{
      font-size: 48px;
      font-weight: 900;
      margin: 0 0 8px;
    }}
    .section-sub {{
      color: var(--cyan);
      font-size: 22px;
      font-weight: 700;
      margin-bottom: 22px;
    }}
    .underline {{
      width: 150px;
      height: 4px;
      background: var(--cyan);
      border-radius: 999px;
      margin-bottom: 26px;
    }}
    .two-col {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }}
    .panel {{
      background: rgba(21,43,69,0.92);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 22px 24px;
    }}
    .panel h3 {{
      font-size: 28px;
      margin: 0 0 16px;
      display: flex;
      gap: 12px;
      align-items: center;
    }}
    .panel p, .panel li {{
      font-size: 20px;
      line-height: 1.65;
      color: #d4deeb;
    }}
    .panel ul {{
      margin: 0;
      padding-left: 24px;
    }}
    .quote-box {{
      margin-top: 24px;
      background: rgba(48,198,255,0.08);
      border: 1px solid rgba(48,198,255,0.22);
      border-radius: 16px;
      padding: 20px 22px;
    }}
    .quote-label {{ color: var(--yellow); font-size: 40px; line-height: 1; }}
    .quote-text {{
      font-size: 22px;
      line-height: 1.6;
      font-weight: 800;
      margin-top: 8px;
    }}
    .quote-text .emph {{ color: var(--yellow); }}
    .flow-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 18px;
      margin-bottom: 24px;
    }}
    .flow-grid.bottom {{
      grid-template-columns: repeat(3, 1fr);
      width: 72%;
      margin: 28px auto 0;
    }}
    .flow-box {{
      background: rgba(21,43,69,0.94);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      min-height: 190px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      text-align: center;
    }}
    .flow-box.box-main {{ box-shadow: inset 0 0 0 1px rgba(48,198,255,0.10); }}
    .flow-box.box-final {{ box-shadow: inset 0 0 0 1px rgba(168,139,250,0.18); }}
    .flow-step {{ font-size: 18px; color: var(--cyan); font-weight: 800; margin-bottom: 10px; }}
    .flow-main-title {{ font-size: 28px; font-weight: 800; margin-bottom: 8px; }}
    .flow-desc {{ font-size: 16px; color: #cbd5e1; line-height: 1.55; white-space: pre-line; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-bottom: 18px;
    }}
    .metric {{
      background: rgba(21,43,69,0.92);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px 18px;
    }}
    .metric .value {{ font-size: 38px; font-weight: 900; }}
    .metric .label {{ margin-top: 8px; color: var(--muted); font-size: 16px; }}
    .wide-grid {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 18px;
      align-items: start;
    }}
    .chart-card {{
      background: rgba(21,43,69,0.92);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px 18px 18px;
    }}
    .chart-title {{ font-size: 20px; font-weight: 800; margin-bottom: 10px; }}
    .svg-chart {{ width: 100%; height: auto; display: block; }}
    .mini-note {{
      background: rgba(248,250,252,0.04);
      border: 1px solid rgba(148,163,184,0.16);
      border-radius: 14px;
      padding: 16px 18px;
      font-size: 18px;
      line-height: 1.65;
      color: #dbe5f0;
    }}
    .progress-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 14px;
      margin-top: 16px;
    }}
    .progress-card {{
      background: rgba(21,43,69,0.92);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px 16px;
    }}
    .progress-head {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 10px;
      font-size: 18px;
      font-weight: 700;
    }}
    .progress-track {{
      height: 12px;
      background: rgba(148,163,184,0.15);
      border-radius: 999px;
      overflow: hidden;
    }}
    .progress-fill {{ height: 100%; border-radius: 999px; }}
    .progress-foot {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 14px;
    }}
    .table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 16px;
    }}
    .table th {{
      text-align: left;
      color: var(--muted);
      font-size: 15px;
      padding: 10px 12px;
      border-bottom: 1px solid rgba(148,163,184,0.18);
    }}
    .table td {{
      padding: 11px 12px;
      border-bottom: 1px solid rgba(148,163,184,0.10);
      color: #e6edf6;
      vertical-align: top;
    }}
    .chip {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 14px;
      font-weight: 800;
    }}
    .chip.current {{ background: rgba(34,197,94,0.16); color: #86efac; }}
    .chip.expansion {{ background: rgba(59,130,246,0.16); color: #93c5fd; }}
    .chip.novel {{ background: rgba(168,139,250,0.16); color: #c4b5fd; }}
    .chip.caution {{ background: rgba(251,113,133,0.16); color: #fda4af; }}
    .donut-wrap {{
      display: grid;
      grid-template-columns: 260px 1fr;
      gap: 18px;
      align-items: center;
      background: rgba(21,43,69,0.92);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px 18px;
    }}
    .donut {{ width: 100%; height: auto; }}
    .legend-row {{
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 18px;
      padding: 8px 0;
      border-bottom: 1px solid rgba(148,163,184,0.08);
    }}
    .legend-row:last-child {{ border-bottom: 0; }}
    .legend-row b {{ margin-left: auto; }}
    .legend-dot {{
      width: 12px; height: 12px; border-radius: 999px; display: inline-block;
    }}
    .footer-note {{
      margin-top: 18px;
      color: #a5b7cb;
      font-size: 17px;
      line-height: 1.7;
    }}
    @media print {{
      body {{ background: white; }}
      .deck {{ width: auto; margin: 0; }}
      .slide-no {{ margin: 0 0 4px 0; }}
      .slide {{ box-shadow: none; break-after: page; width: 100%; }}
    }}
  </style>
</head>
<body>
  <div class="deck">
    <div class="slide-no">1 / 9</div>
    <section class="slide">
      <div class="kicker">PROJECT PRESENTATION</div>
      <h1 class="title">유방암 관련 약물 재창출<br>파이프라인 구축</h1>
      <div class="subtitle">데이터 기반 신약 재창출 연구</div>
      <div class="hero-pill">유방암(BRCA)에서 기존 약물의 재적용 후보를 <b>end-to-end 파이프라인</b>으로 도출</div>
      <div class="people">
        <div class="person"><div class="name">최현석 <span style="font-size:16px;color:var(--muted)">(PM)</span></div><div class="role">프로젝트 관리 / 모델 고도화</div></div>
        <div class="person"><div class="name">소병임 <span style="font-size:16px;color:var(--muted)">(기획)</span></div><div class="role">파이프라인 설계 및 구현</div></div>
        <div class="person"><div class="name">홍주영 <span style="font-size:16px;color:var(--muted)">(데이터)</span></div><div class="role">데이터 수집 / 전처리 / Feature Eng.</div></div>
        <div class="person"><div class="name">조창현 <span style="font-size:16px;color:var(--muted)">(프론트/문서)</span></div><div class="role">프론트엔드 구축 / 근거 문서 탐색</div></div>
      </div>
      <div style="position:absolute;left:58px;bottom:34px;color:#6b7c93;font-size:16px;">2026-04-16</div>
    </section>

    <div class="slide-no">2 / 9</div>
    <section class="slide">
      <h2 class="section-title">프로젝트 배경과 목표</h2>
      <div class="section-sub">약물 재창출의 필요성과 파이프라인 구축의 핵심 도전과제</div>
      <div class="underline"></div>
      <div class="two-col">
        <div class="panel">
          <h3>💡 왜 약물 재창출인가?</h3>
          <p>신약 개발 대비 비용과 시간을 줄이면서, 이미 임상/독성 정보가 축적된 물질을 활용해 <b>유방암 적응증 재검토</b> 가능성을 탐색하는 것이 목표입니다.</p>
          <div class="quote-box">
            <div class="quote-label">“</div>
            <div class="quote-text">“유방암에서 현재 사용되지 않거나 덜 쓰이는 약 중, 실제로 다시 <span class="emph">재검토할 가치가 있는 약물</span>을 찾을 수 있을까?”</div>
          </div>
        </div>
        <div class="panel">
          <h3>⚠️ 파이프라인 구축의 주요 도전과제</h3>
          <ul>
            <li><b>데이터 연결성</b>: GDSC, DepMap, LINCS, TCGA, METABRIC, DrugBank/ChEMBL 연결</li>
            <li><b>임상 번역 가능성</b>: 세포주 성능이 실제 환자 코호트(METABRIC)에서도 유지되는지 검증</li>
            <li><b>후보 안전성</b>: ADMET 기반 독성·약동학 필터로 후보 현실성 확보</li>
            <li><b>설명 가능성</b>: KG/API로 타깃, 문헌, 임상시험, 부작용 근거 매핑</li>
          </ul>
        </div>
      </div>
    </section>

    <div class="slide-no">3 / 9</div>
    <section class="slide">
      <h2 class="section-title">전체 파이프라인 흐름 (7단계)</h2>
      <div class="section-sub">데이터 수집부터 KG/API 근거 수집까지의 end-to-end 프로세스</div>
      <div class="underline"></div>
      <div class="flow-grid">
        {pipeline_boxes()[:]}
      </div>
      <div class="footer-note">현재 발표 수치는 <b>exact slim + strong context + SMILES</b> 입력과 <b>random sample 3-fold</b> 기준으로 정리했습니다. 이 경로는 ceiling/interpolation 성능을 보여주는 reference route입니다.</div>
    </section>

    <div class="slide-no">4 / 9</div>
    <section class="slide">
      <h2 class="section-title">입력 데이터와 모델 입력 구성</h2>
      <div class="section-sub">정답(y), feature(X), 외부 검증, 안전성/설명 데이터의 역할 분리</div>
      <div class="underline"></div>
      <div class="metrics">
        <div class="metric"><div class="value">13,388</div><div class="label">GDSC2 BRCA 라벨 rows</div></div>
        <div class="metric"><div class="value">52</div><div class="label">Cell lines</div></div>
        <div class="metric"><div class="value">295</div><div class="label">Drugs</div></div>
        <div class="metric"><div class="value">6,366</div><div class="label">Sample 3-fold training rows</div></div>
      </div>
      <div class="two-col">
        <div class="panel">
          <h3>🧬 데이터 소스와 역할</h3>
          <ul>
            <li><b>GDSC2 BRCA</b>: 약물 반응 정답(y)</li>
            <li><b>DepMap CRISPR</b>: 세포주 gene dependency</li>
            <li><b>LINCS MCF7</b>: 약물 perturbation signature</li>
            <li><b>DrugBank / ChEMBL</b>: SMILES, 타깃, pathway 표준화</li>
            <li><b>METABRIC</b>: 외부 검증 코호트</li>
            <li><b>ADMET / KG/API</b>: 안전성 및 설명 근거</li>
          </ul>
        </div>
        <div class="panel">
          <h3>📦 이번 실험의 최종 입력</h3>
          <ul>
            <li><b>exact slim numeric</b>: 5,529 dim</li>
            <li><b>strong context</b>: 5개 컬럼
              <br><span style="color:var(--muted)">TCGA_DESC, PATHWAY_NAME_NORMALIZED, classification, drug_bridge_strength, stage3_resolution_status</span></li>
            <li><b>SMILES</b>: 토큰 기반 branch + ML용 64-dim SVD</li>
            <li><b>ML input dim</b>: 5,625</li>
          </ul>
          <div class="mini-note" style="margin-top:16px;">random sample 3-fold는 같은 약물이 train/valid 양쪽에 등장할 수 있어 GroupCV보다 높게 나오는 <b>상한선(reference ceiling)</b>으로 해석해야 합니다.</div>
        </div>
      </div>
    </section>

    <div class="slide-no">5 / 9</div>
    <section class="slide">
      <h2 class="section-title">Step 4. 단일 모델 비교</h2>
      <div class="section-sub">sample 3-fold 기준 최고 ML은 CatBoost, 최고 DL은 TabNet</div>
      <div class="underline"></div>
      <div class="metrics">
        <div class="metric"><div class="value">0.8706</div><div class="label">CatBoost Spearman</div></div>
        <div class="metric"><div class="value">1.1083</div><div class="label">CatBoost RMSE</div></div>
        <div class="metric"><div class="value">0.8613</div><div class="label">TabNet Spearman</div></div>
        <div class="metric"><div class="value">1.1560</div><div class="label">TabNet RMSE</div></div>
      </div>
      <div class="wide-grid">
        {step4_chart}
        <div class="mini-note">
          <b>해석 포인트</b><br><br>
          • ML 상위권: <b>CatBoost, XGBoost, LightGBM</b><br>
          • DL 상위권: <b>TabNet, ResidualMLP, WideDeep</b><br>
          • 단일모델 기준에서는 ML이 조금 더 안정적이었고, DL은 근접한 후보군으로 경쟁했습니다.<br><br>
          발표에서는 “모델 family를 넓게 비교한 뒤, Step 5에서 ML+DL 혼합 앙상블로 ceiling을 조금 더 밀어 올렸다”는 흐름으로 설명하면 좋습니다.
        </div>
      </div>
      <div style="margin-top:18px;">
        {step4_rmse_chart}
      </div>
    </section>

    <div class="slide-no">6 / 9</div>
    <section class="slide">
      <h2 class="section-title">Step 5. 혼합 앙상블</h2>
      <div class="section-sub">ML 3개 + DL 3개를 섞은 weighted ensemble이 전체 최고 성능</div>
      <div class="underline"></div>
      <div class="metrics">
        <div class="metric"><div class="value">0.8720</div><div class="label">Weighted Spearman</div></div>
        <div class="metric"><div class="value">1.1016</div><div class="label">Weighted RMSE</div></div>
        <div class="metric"><div class="value">0.8281</div><div class="label">Weighted MAE</div></div>
        <div class="metric"><div class="value">0.9137</div><div class="label">Weighted Pearson</div></div>
      </div>
      <div class="wide-grid">
        {weight_chart}
        <div class="mini-note">
          <b>선택된 base model</b><br><br>
          CatBoost / XGBoost / LightGBM / TabNet / ResidualMLP / WideDeep<br><br>
          <b>Diversity summary</b><br>
          • avg prediction Pearson: <b>{fmt(ens['diversity']['summary']['avg_prediction_pearson'])}</b><br>
          • avg residual Pearson: <b>{fmt(ens['diversity']['summary']['avg_residual_pearson'])}</b><br>
          • avg mean abs gap: <b>{fmt(ens['diversity']['summary']['avg_mean_abs_prediction_gap'])}</b><br><br>
          residual correlation이 여전히 높아서 “완전히 다른 오류 패턴”보다는 평균화 이득이 중심이었습니다.
        </div>
      </div>
    </section>

    <div class="slide-no">7 / 9</div>
    <section class="slide">
      <h2 class="section-title">Step 6. METABRIC 외부 검증</h2>
      <div class="section-sub">sample 3-fold top30이 환자 코호트에서도 생물학적 타당성을 가지는지 검증</div>
      <div class="underline"></div>
      <div class="progress-grid">
        {progress_card("Target expressed", 29, 30, "#22d3ee", "타깃 발현 확인")}
        {progress_card("BRCA pathway relevant", 23, 30, "#34d399", "경로 정합성")}
        {progress_card("Survival significant", 28, 30, "#fbbf24", "생존 유의")}
        <div class="progress-card">
          <div class="progress-head"><span>P@15 / RSF / Graph</span><b>80.0%</b></div>
          <div class="progress-track"><div class="progress-fill" style="width:80%; background:#a78bfa"></div></div>
          <div class="progress-foot">RSF C-index 0.8209 · GraphSAGE P@20 0.94</div>
        </div>
      </div>
      <div class="wide-grid" style="margin-top:18px;">
        <div class="chart-card">
          <div class="chart-title">Validated Top 5</div>
          <table class="table">
            <thead><tr><th>#</th><th>Drug</th><th>Target</th><th>Combined</th></tr></thead>
            <tbody>{validated_rows}</tbody>
          </table>
        </div>
        <div class="mini-note">
          <b>핵심 메시지</b><br><br>
          • 단순히 sample 3-fold에서 숫자가 높은 후보를 고른 것이 아니라,<br>
          • <b>METABRIC BRCA</b> 환자 코호트에서 타깃 발현과 생존, 기존 BRCA 약물 정밀도로 다시 검증했습니다.<br><br>
          대표적으로 <b>Bortezomib</b>, <b>Romidepsin</b>, <b>Sepantronium bromide</b>, <b>Docetaxel</b>, <b>Dactinomycin</b>가 상위 validated 후보였습니다.
        </div>
      </div>
    </section>

    <div class="slide-no">8 / 9</div>
    <section class="slide">
      <h2 class="section-title">Step 7. ADMET 및 post-ADMET 분류</h2>
      <div class="section-sub">raw 안전성 게이트와 유방암 임상 분류를 함께 적용해 최종 15개 후보를 정리</div>
      <div class="underline"></div>
      <div class="two-col">
        {donut_svg([("유방암 현재 사용", step7['n_current_use'], "#34d399"), ("적응증 확장/연구 중", step7['n_expansion'], "#60a5fa"), ("유방암 미사용", step7['n_novel'], "#a78bfa")])}
        <div>
          <div class="metrics" style="grid-template-columns: repeat(3, 1fr);">
            <div class="metric"><div class="value">8</div><div class="label">Raw Approved</div></div>
            <div class="metric"><div class="value">6</div><div class="label">Raw Candidate</div></div>
            <div class="metric"><div class="value">1</div><div class="label">Raw Caution</div></div>
          </div>
          <div class="mini-note">
            <b>Top 5 final candidate</b><br>
            Romidepsin · Sepantronium bromide · Staurosporine · SN-38 · Docetaxel<br><br>
            <b>Caution</b>: Epirubicin (Ames / DILI 플래그)<br><br>
            팀원 레포 형식 기준으로는 <b>현재 사용 5 / 확장·연구 중 6 / 미사용 4</b>로 다시 정리했습니다.
          </div>
        </div>
      </div>
      <div style="margin-top:20px;" class="chart-card">
        <div class="chart-title">Top 5 최종 후보와 임상 분류</div>
        <table class="table">
          <thead><tr><th>#</th><th>Drug</th><th>임상 분류</th><th>Combined Score</th></tr></thead>
          <tbody>{top5_rows}</tbody>
        </table>
      </div>
    </section>

    <div class="slide-no">9 / 9</div>
    <section class="slide">
      <h2 class="section-title">Step 7+. KG/API 근거 수집 및 최종 메시지</h2>
      <div class="section-sub">ClinicalTrials, PubMed, FAERS, target/pathway를 붙여 설명 가능한 최종 후보로 변환</div>
      <div class="underline"></div>
      <div class="metrics">
        <div class="metric"><div class="value">15</div><div class="label">KG/API Candidate Count</div></div>
        <div class="metric"><div class="value">Romidepsin</div><div class="label">Top Candidate</div></div>
        <div class="metric"><div class="value">Trials</div><div class="label">ClinicalTrials.gov</div></div>
        <div class="metric"><div class="value">PubMed</div><div class="label">문헌 근거</div></div>
      </div>
      <div class="wide-grid">
        <div class="chart-card">
          <div class="chart-title">KG/API evidence top 5</div>
          <table class="table">
            <thead><tr><th>Drug</th><th>Trials</th><th>PubMed</th><th>PubMed BRCA</th><th>Targets</th><th>Pathways</th></tr></thead>
            <tbody>{kg_rows}</tbody>
          </table>
        </div>
        <div class="mini-note">
          <b>발표용 최종 메시지</b><br><br>
          1. 이 프로젝트는 데이터 연결 → 모델링 → 외부 검증 → 안전성 평가 → KG/API 근거 수집까지 이어지는 <b>end-to-end 유방암 약물 재창출 파이프라인</b>입니다.<br><br>
          2. sample 3-fold 기준 최고 단일 모델은 <b>CatBoost</b>, 최고 DL은 <b>TabNet</b>, 최고 전체 성능은 <b>혼합 weighted ensemble</b>이었습니다.<br><br>
          3. 최종적으로는 <b>Romidepsin</b>, <b>Sepantronium bromide</b>, <b>Staurosporine</b>, <b>SN-38</b>, <b>Docetaxel</b>이 강한 후보로 남았습니다.<br><br>
          4. 다만 이 경로는 <b>sample 3-fold ceiling</b>이므로, 실제 일반화 성능 논의에서는 GroupCV와 같이 읽어야 합니다.
        </div>
      </div>
      <div class="footer-note">참고 문서: 프로젝트 이해 문서, random3 Step 4~7+ 상세 통합 대시보드, Step 6/7/7+ 세부 결과 파일</div>
    </section>
  </div>
</body>
</html>
"""

    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
