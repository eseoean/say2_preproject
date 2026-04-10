#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DASHBOARD_PATH = ROOT / "dashboard.html"
INDEX_PATH = ROOT / "index.html"

PROJECT_META = {
    "title": "약물 재창출 파이프라인 - ESEO 재현 대시보드",
    "subtitle": "팀원 대시보드 스타일을 기준으로 현재 재현 결과를 반영한 실행 보드",
    "started_at": "2026-04-09",
    "tag": "20260409_eseo",
    "fe_run_id": "20260409_newfe_v8_eseo",
    "s3_base": "s3://say2-4team/20260409_eseo",
    "aws_queue": "team4-fe-queue-cpu",
    "python": "3.10.20",
    "java": "17",
    "conda": "drug4",
    "numpy": "1.26.4",
    "step2_note": "reference LINCS parquet 복사 사용",
    "step3_runtime": "8분 7초",
}

THRESHOLDS = {
    "spearman": 0.713,
    "rmse": 1.385,
}

MODEL_ROLE = {
    "CatBoost": "6-model",
    "LightGBM": "6-model + lite",
    "XGBoost": "6-model",
    "FlatMLP": "6-model + lite",
    "ResidualMLP": "6-model",
    "Cross-Attention": "6-model + lite",
    "Stacking Ridge": "meta",
    "RSF": "step6",
}


def read_json(*parts: str):
    return json.loads((MODELS_DIR / Path(*parts)).read_text())


def model_label(raw: str) -> str:
    name = re.sub(r"^\d+_", "", raw)
    name = name.replace("_", " ")
    replacements = {
        "Cross Attention": "Cross-Attention",
        "Stacking Ridge": "Stacking Ridge",
        "ResidualMLP": "ResidualMLP",
        "FlatMLP": "FlatMLP",
        "FT Transformer": "FT-Transformer",
        "GraphSAGE": "GraphSAGE",
    }
    return replacements.get(name, name)


def model_family(raw: str) -> str:
    idx = int(raw.split("_", 1)[0])
    if idx <= 8:
        return "ML"
    if idx <= 13:
        return "DL"
    return "Graph"


def status_for_model(row: dict) -> str:
    name = row["label"]
    if name == "RSF":
        return "METABRIC"
    if name == "Stacking Ridge":
        return "META"
    if row["spearman_mean"] >= THRESHOLDS["spearman"] and row["rmse_mean"] <= THRESHOLDS["rmse"]:
        return "PASS"
    if row["family"] == "Graph":
        return "AUX"
    return "FAIL"


def fmt_num(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def fmt_elapsed(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = int(minutes // 60)
    remain = minutes - (hours * 60)
    return f"{hours}h {remain:.1f}m"


def esc(value: object) -> str:
    return html.escape(str(value))


def pill(text: str, variant: str = "neutral") -> str:
    return f'<span class="pill pill-{variant}">{esc(text)}</span>'


def load_model_rows() -> list[dict]:
    rows = []
    for rel in [
        ("ml_results", "ml_results.json"),
        ("dl_results", "dl_results.json"),
        ("graph_results", "graph_results.json"),
    ]:
        for item in read_json(*rel):
            label = model_label(item["model"])
            row = {
                **item,
                "label": label,
                "family": model_family(item["model"]),
            }
            row["status"] = status_for_model(row)
            rows.append(row)
    return sorted(rows, key=lambda x: x["spearman_mean"], reverse=True)


def step7_category_counts(step7: dict) -> Counter:
    return Counter(candidate["category"] for candidate in step7["final_candidates"])


def build_step_cards(
    model_rows: list[dict],
    ensemble_full: dict,
    ensemble_lite: dict,
    step6_full: dict,
    step7_lite: dict,
) -> str:
    pass_rows = [row for row in model_rows if row["status"] == "PASS"]
    meta_rows = [row for row in model_rows if row["status"] == "META"]
    best_ml = next(row for row in model_rows if row["label"] == "CatBoost")
    best_dl = next(row for row in model_rows if row["label"] == "FlatMLP")
    categories = step7_category_counts(step7_lite)

    cards = [
        {
            "num": 1,
            "title": "환경 설정",
            "metrics": [
                f"Python {PROJECT_META['python']} / conda {PROJECT_META['conda']}",
                f"Java {PROJECT_META['java']} / AWS Batch {PROJECT_META['aws_queue']}",
                f"NumPy {PROJECT_META['numpy']} + RDKit 호환성 정리",
                "GitHub 원격 + AWS 자격증명 확인 완료",
            ],
        },
        {
            "num": 2,
            "title": "데이터 준비",
            "metrics": [
                "gdsc_ic50: 13,388 rows / 52 cell lines / 295 drugs",
                "depmap_crispr_long: 20.44M rows / 18,443 genes",
                "drug_features_catalog: 243 / 295 matched (82.4%)",
                f"LINCS fast track + {PROJECT_META['step2_note']}",
            ],
        },
        {
            "num": 3,
            "title": "특성 공학",
            "metrics": [
                f"Nextflow on AWS Batch 완료 ({PROJECT_META['step3_runtime']})",
                "features.parquet + labels.parquet + pair_features 생성",
                "7,730 pair 학습셋 기준 FE 산출물 사용",
                f"fe_output/{PROJECT_META['fe_run_id']} 업로드 완료",
            ],
        },
        {
            "num": 4,
            "title": "모델 학습",
            "metrics": [
                f"총 15개 모델 실행, PASS {len(pass_rows)}개 + META {len(meta_rows)}개",
                f"최고 ML: {best_ml['label']} ({fmt_num(best_ml['spearman_mean'])} / {fmt_num(best_ml['rmse_mean'])})",
                f"최고 DL: {best_dl['label']} ({fmt_num(best_dl['spearman_mean'])} / {fmt_num(best_dl['rmse_mean'])})",
                "Graph는 보조 검증용으로만 유지",
            ],
        },
        {
            "num": 5,
            "title": "앙상블",
            "metrics": [
                f"6-model: Sp {fmt_num(ensemble_full['ensemble_metrics']['spearman_mean'])} / RMSE {fmt_num(ensemble_full['ensemble_metrics']['rmse_mean'])}",
                f"Lite 3-model: Sp {fmt_num(ensemble_lite['ensemble_metrics']['spearman_mean'])} / RMSE {fmt_num(ensemble_lite['ensemble_metrics']['rmse_mean'])}",
                "Top15 drug set overlap 15 / 15",
                "선택 경로: LightGBM + FlatMLP + Cross-Attention",
            ],
        },
        {
            "num": 6,
            "title": "METABRIC 외부 검증",
            "metrics": [
                f"Target expressed {step6_full['method_a']['n_targets_expressed']} / {step6_full['method_a']['n_total']}",
                f"BRCA pathway {step6_full['method_a']['n_brca_pathway']} / {step6_full['method_a']['n_total']}",
                f"Survival significant {step6_full['method_b']['n_significant']} / {step6_full['method_a']['n_total']}",
                f"P@20 {fmt_pct(step6_full['method_c']['precision_at_k']['P@20']['precision'])}",
            ],
        },
        {
            "num": 7,
            "title": "ADMET 최종 선정",
            "metrics": [
                f"Approved {categories['Approved']} / Candidate {categories['Candidate']} / Caution {categories['Caution']}",
                f"Top1 {step7_lite['final_candidates'][0]['drug_name']}",
                "최종 후보는 경량형 트랙 기준으로 채택",
                "drug_name 기준 고유 약물만 유지",
            ],
        },
    ]

    parts = []
    for card in cards:
        metrics = "".join(f"<li>{esc(metric)}</li>" for metric in card["metrics"])
        parts.append(
            f"""
            <article class="step-card complete">
              <div class="step-kicker">Step {card['num']}</div>
              <h3>{esc(card['title'])}</h3>
              <ul>{metrics}</ul>
            </article>
            """
        )
    return "".join(parts)


def build_model_table(model_rows: list[dict]) -> str:
    parts = []
    for row in model_rows:
        status_variant = {
            "PASS": "good",
            "META": "warn",
            "METABRIC": "info",
            "AUX": "info",
            "FAIL": "bad",
        }[row["status"]]
        role = MODEL_ROLE.get(row["label"], "-")
        parts.append(
            f"""
            <tr>
              <td>{esc(row['family'])}</td>
              <td class="strong">{esc(row['label'])}</td>
              <td>{fmt_num(row['spearman_mean'])}</td>
              <td>{fmt_num(row['rmse_mean'])}</td>
              <td>{fmt_num(row['pearson_mean'])}</td>
              <td>{fmt_num(row['r2_mean'])}</td>
              <td>{fmt_num(row['gap_spearman_mean'])}</td>
              <td>{fmt_elapsed(row.get('elapsed_sec'))}</td>
              <td>{pill(row['status'], status_variant)}</td>
              <td>{esc(role)}</td>
            </tr>
            """
        )
    return "".join(parts)


def build_weights(weights: dict[str, float], variant: str) -> str:
    return "".join(
        f'<div class="weight-chip weight-{variant}"><span>{esc(name)}</span><strong>{fmt_pct(value, 2)}</strong></div>'
        for name, value in weights.items()
    )


def build_step5_compare(full: dict, lite: dict) -> str:
    full_m = full["ensemble_metrics"]
    lite_m = lite["ensemble_metrics"]
    rows = [
        ("모델 수", str(full["n_models"]), str(lite["n_models"])),
        ("Spearman", f"{fmt_num(full_m['spearman_mean'])} ± {fmt_num(full_m['spearman_std'])}", f"{fmt_num(lite_m['spearman_mean'])} ± {fmt_num(lite_m['spearman_std'])}"),
        ("RMSE", f"{fmt_num(full_m['rmse_mean'])} ± {fmt_num(full_m['rmse_std'])}", f"{fmt_num(lite_m['rmse_mean'])} ± {fmt_num(lite_m['rmse_std'])}"),
        ("Pearson", fmt_num(full_m["pearson"]), fmt_num(lite_m["pearson"])),
        ("R²", fmt_num(full_m["r2"]), fmt_num(lite_m["r2"])),
        ("Gap Spearman", fmt_num(full_m["gap_spearman_mean"]), fmt_num(lite_m["gap_spearman_mean"])),
        ("Top15 overlap", "15 / 15", "15 / 15"),
    ]
    return "".join(
        f"<tr><th>{esc(label)}</th><td>{left}</td><td>{right}</td></tr>" for label, left, right in rows
    )


def build_step6_compare(full: dict, lite: dict) -> str:
    rows = [
        ("Target expressed", f"{full['method_a']['n_targets_expressed']} / {full['method_a']['n_total']}", f"{lite['method_a']['n_targets_expressed']} / {lite['method_a']['n_total']}"),
        ("BRCA pathway", f"{full['method_a']['n_brca_pathway']} / {full['method_a']['n_total']}", f"{lite['method_a']['n_brca_pathway']} / {lite['method_a']['n_total']}"),
        ("Survival significant", f"{full['method_b']['n_significant']} / {full['method_a']['n_total']}", f"{lite['method_b']['n_significant']} / {lite['method_a']['n_total']}"),
        ("RSF C-index", fmt_num(full['method_b']['rsf_c_index']), fmt_num(lite['method_b']['rsf_c_index'])),
        ("RSF AUROC", fmt_num(full['method_b']['rsf_auroc']), fmt_num(lite['method_b']['rsf_auroc'])),
        ("P@5", fmt_pct(full['method_c']['precision_at_k']['P@5']['precision']), fmt_pct(lite['method_c']['precision_at_k']['P@5']['precision'])),
        ("P@10", fmt_pct(full['method_c']['precision_at_k']['P@10']['precision']), fmt_pct(lite['method_c']['precision_at_k']['P@10']['precision'])),
        ("P@15", fmt_pct(full['method_c']['precision_at_k']['P@15']['precision']), fmt_pct(lite['method_c']['precision_at_k']['P@15']['precision'])),
        ("P@20", fmt_pct(full['method_c']['precision_at_k']['P@20']['precision']), fmt_pct(lite['method_c']['precision_at_k']['P@20']['precision'])),
        ("GraphSAGE P@20", fmt_pct(full['method_c']['graphsage_p20']), fmt_pct(lite['method_c']['graphsage_p20'])),
    ]
    return "".join(
        f"<tr><th>{esc(label)}</th><td>{left}</td><td>{right}</td></tr>" for label, left, right in rows
    )


def build_step7_compare(full: dict, lite: dict) -> str:
    counts_full = step7_category_counts(full)
    counts_lite = step7_category_counts(lite)
    rows = [
        ("Final candidates", str(len(full["final_candidates"])), str(len(lite["final_candidates"]))),
        ("Approved", str(counts_full["Approved"]), str(counts_lite["Approved"])),
        ("Candidate", str(counts_full["Candidate"]), str(counts_lite["Candidate"])),
        ("Caution", str(counts_full["Caution"]), str(counts_lite["Caution"])),
        ("Top15 overlap", "15 / 15", "15 / 15"),
        (
            "Top5",
            ", ".join(candidate["drug_name"] for candidate in full["final_candidates"][:5]),
            ", ".join(candidate["drug_name"] for candidate in lite["final_candidates"][:5]),
        ),
    ]
    return "".join(
        f"<tr><th>{esc(label)}</th><td>{esc(left)}</td><td>{esc(right)}</td></tr>" for label, left, right in rows
    )


def build_rank_compare(full: dict, lite: dict) -> str:
    full_by_name = {item["drug_name"]: item for item in full["final_candidates"]}
    lite_by_name = {item["drug_name"]: item for item in lite["final_candidates"]}
    rows = []
    for name in [item["drug_name"] for item in lite["final_candidates"]]:
        left = full_by_name[name]
        right = lite_by_name[name]
        delta = right["final_rank"] - left["final_rank"]
        if delta == 0:
            diff = "-"
        elif delta < 0:
            diff = f"▲ {abs(delta)}"
        else:
            diff = f"▼ {delta}"
        rows.append(
            f"""
            <tr>
              <td class="strong">{esc(name)}</td>
              <td>{left['final_rank']}</td>
              <td>{right['final_rank']}</td>
              <td>{esc(diff)}</td>
              <td>{fmt_num(left['combined_score'], 2)}</td>
              <td>{fmt_num(right['combined_score'], 2)}</td>
              <td>{pill(right['category'], right['category'].lower())}</td>
            </tr>
            """
        )
    return "".join(rows)


def build_final_candidates(step7: dict) -> str:
    rows = []
    for candidate in step7["final_candidates"]:
        flags = ", ".join(candidate["flags"]) if candidate["flags"] else "-"
        rows.append(
            f"""
            <tr>
              <td>{candidate['final_rank']}</td>
              <td class="strong">{esc(candidate['drug_name'])}</td>
              <td>{esc(candidate['target'])}</td>
              <td>{esc(candidate['pathway'])}</td>
              <td>{fmt_num(candidate['pred_ic50'])}</td>
              <td>{fmt_num(candidate['combined_score'], 2)}</td>
              <td>{pill(candidate['category'], candidate['category'].lower())}</td>
              <td>{esc(flags)}</td>
            </tr>
            """
        )
    return "".join(rows)


def generate_dashboard() -> str:
    model_rows = load_model_rows()
    ensemble_full = read_json("ensemble_results", "ensemble_results.json")
    ensemble_lite = read_json("ensemble_results_lightweight", "ensemble_results.json")
    step6_full = read_json("metabric_results", "step6_metabric_results.json")
    step6_lite = read_json("metabric_results_lightweight", "step6_metabric_results.json")
    step7_full = read_json("admet_results", "step7_admet_results.json")
    step7_lite = read_json("admet_results_lightweight", "step7_admet_results.json")

    categories = step7_category_counts(step7_lite)
    updated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")

    hero_cards = f"""
      <div class="hero-card">
        <div class="eyebrow">Selected Track</div>
        <div class="hero-value">Lightweight 3-model</div>
        <p>Step 6-7은 LightGBM + FlatMLP + Cross-Attention 조합으로 채택했습니다.</p>
      </div>
      <div class="hero-card">
        <div class="eyebrow">Final Winner</div>
        <div class="hero-value">{esc(step7_lite['final_candidates'][0]['drug_name'])}</div>
        <p>Combined score {fmt_num(step7_lite['final_candidates'][0]['combined_score'], 2)} / category {esc(step7_lite['final_candidates'][0]['category'])}</p>
      </div>
      <div class="hero-card">
        <div class="eyebrow">ADMET Summary</div>
        <div class="hero-value">{categories['Approved']} / {categories['Candidate']} / {categories['Caution']}</div>
        <p>Approved / Candidate / Caution 분포입니다.</p>
      </div>
      <div class="hero-card">
        <div class="eyebrow">S3 Base</div>
        <div class="hero-value mono">{esc(PROJECT_META['s3_base'])}</div>
        <p>FE, 모델, METABRIC, ADMET 산출물이 이 경로 아래에 정리돼 있습니다.</p>
      </div>
    """

    sources = [
        "models/ml_results/ml_results.json",
        "models/dl_results/dl_results.json",
        "models/graph_results/graph_results.json",
        "models/ensemble_results/ensemble_results.json",
        "models/ensemble_results_lightweight/ensemble_results.json",
        "models/metabric_results/step6_metabric_results.json",
        "models/metabric_results_lightweight/step6_metabric_results.json",
        "models/admet_results/step7_admet_results.json",
        "models/admet_results_lightweight/step7_admet_results.json",
    ]
    source_links = "".join(
        f'<li><code>{esc(path)}</code></li>' for path in sources
    )

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{esc(PROJECT_META['title'])}</title>
  <style>
    :root {{
      --bg: #09111f;
      --panel: #111c2f;
      --panel-2: #17253d;
      --line: #2d4061;
      --text: #e5eefb;
      --muted: #95a7c5;
      --accent: #53b4ff;
      --accent-2: #7ef0c5;
      --good: #37d67a;
      --warn: #f6b94f;
      --bad: #ff6d7a;
      --info: #77b3ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top right, rgba(83,180,255,0.14), transparent 28%),
        radial-gradient(circle at top left, rgba(126,240,197,0.10), transparent 26%),
        linear-gradient(180deg, #0b1424 0%, #09111f 100%);
    }}
    a {{ color: inherit; }}
    .shell {{
      max-width: 1380px;
      margin: 0 auto;
      padding: 28px 24px 56px;
    }}
    .masthead {{
      display: grid;
      gap: 18px;
      grid-template-columns: minmax(0, 1.8fr) minmax(280px, 1fr);
      align-items: stretch;
      margin-bottom: 26px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(23,37,61,0.96), rgba(12,21,36,0.96));
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 28px;
      box-shadow: 0 18px 48px rgba(0,0,0,0.26);
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: clamp(2rem, 4vw, 3rem);
      line-height: 1.05;
      letter-spacing: -0.03em;
    }}
    .hero p {{
      margin: 0 0 18px;
      color: var(--muted);
      line-height: 1.65;
      max-width: 70ch;
    }}
    .hero-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .hero-meta .pill {{
      font-size: 0.8rem;
    }}
    .status-panel {{
      background: linear-gradient(180deg, rgba(17,28,47,0.95), rgba(9,17,31,0.95));
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 24px;
      display: grid;
      gap: 14px;
    }}
    .status-panel h2 {{
      margin: 0;
      font-size: 1rem;
      color: var(--muted);
      font-weight: 600;
    }}
    .status-big {{
      font-size: 3rem;
      font-weight: 800;
      letter-spacing: -0.04em;
    }}
    .progress {{
      background: rgba(255,255,255,0.05);
      border-radius: 999px;
      overflow: hidden;
      height: 12px;
      border: 1px solid rgba(255,255,255,0.06);
    }}
    .progress > span {{
      display: block;
      width: 100%;
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
    }}
    .hero-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 22px;
    }}
    .hero-card {{
      background: rgba(17,28,47,0.92);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
    }}
    .eyebrow {{
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 10px;
    }}
    .hero-value {{
      font-size: 1.3rem;
      font-weight: 800;
      line-height: 1.2;
      margin-bottom: 8px;
    }}
    .hero-card p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
      font-size: 0.92rem;
    }}
    .mono {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.95rem;
      word-break: break-all;
    }}
    .section {{
      background: rgba(17,28,47,0.94);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 22px;
      margin-bottom: 18px;
      box-shadow: 0 14px 34px rgba(0,0,0,0.18);
    }}
    .section-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      margin-bottom: 18px;
    }}
    .section-head h2 {{
      margin: 0;
      font-size: 1.2rem;
      letter-spacing: -0.02em;
    }}
    .section-head p {{
      margin: 6px 0 0;
      color: var(--muted);
      line-height: 1.5;
    }}
    .step-grid {{
      display: grid;
      grid-template-columns: repeat(7, minmax(0, 1fr));
      gap: 12px;
    }}
    .step-card {{
      padding: 16px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(23,37,61,0.85), rgba(11,18,32,0.96));
    }}
    .step-card.complete {{
      box-shadow: inset 0 0 0 1px rgba(55,214,122,0.20);
    }}
    .step-kicker {{
      font-size: 0.74rem;
      text-transform: uppercase;
      letter-spacing: 0.10em;
      color: var(--accent-2);
      margin-bottom: 10px;
    }}
    .step-card h3 {{
      margin: 0 0 10px;
      font-size: 1rem;
    }}
    .step-card ul {{
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
      line-height: 1.55;
      font-size: 0.88rem;
    }}
    .comparison-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 16px;
    }}
    .compare-card {{
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(10,18,31,0.55);
    }}
    .compare-card h3 {{
      margin: 0 0 8px;
      font-size: 1rem;
    }}
    .compare-card p {{
      margin: 0 0 14px;
      color: var(--muted);
      line-height: 1.5;
      font-size: 0.9rem;
    }}
    .weights {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }}
    .weight-chip {{
      display: inline-flex;
      gap: 10px;
      align-items: center;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.08);
      font-size: 0.82rem;
    }}
    .weight-6 {{
      background: rgba(83,180,255,0.10);
    }}
    .weight-lite {{
      background: rgba(126,240,197,0.10);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }}
    thead th {{
      text-align: left;
      color: var(--muted);
      font-weight: 600;
      padding: 11px 10px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,0.02);
    }}
    tbody td, tbody th {{
      padding: 10px;
      border-bottom: 1px solid rgba(45,64,97,0.55);
      vertical-align: top;
    }}
    tbody tr:hover {{
      background: rgba(83,180,255,0.04);
    }}
    .strong {{
      font-weight: 700;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 0.75rem;
      font-weight: 700;
      border: 1px solid transparent;
      white-space: nowrap;
    }}
    .pill-neutral {{ background: rgba(255,255,255,0.08); color: var(--text); }}
    .pill-good {{ background: rgba(55,214,122,0.15); color: var(--good); border-color: rgba(55,214,122,0.24); }}
    .pill-warn {{ background: rgba(246,185,79,0.14); color: var(--warn); border-color: rgba(246,185,79,0.22); }}
    .pill-bad {{ background: rgba(255,109,122,0.14); color: var(--bad); border-color: rgba(255,109,122,0.20); }}
    .pill-info {{ background: rgba(119,179,255,0.14); color: var(--info); border-color: rgba(119,179,255,0.22); }}
    .pill-approved {{ background: rgba(55,214,122,0.15); color: var(--good); border-color: rgba(55,214,122,0.24); }}
    .pill-candidate {{ background: rgba(83,180,255,0.14); color: var(--accent); border-color: rgba(83,180,255,0.24); }}
    .pill-caution {{ background: rgba(246,185,79,0.16); color: var(--warn); border-color: rgba(246,185,79,0.26); }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      text-decoration: none;
      padding: 10px 14px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 0.85rem;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.05);
    }}
    .button.primary {{
      background: linear-gradient(90deg, rgba(83,180,255,0.16), rgba(126,240,197,0.16));
      border-color: rgba(126,240,197,0.22);
    }}
    .footnote {{
      color: var(--muted);
      line-height: 1.6;
      font-size: 0.88rem;
    }}
    .source-list {{
      columns: 2;
      padding-left: 18px;
      color: var(--muted);
      line-height: 1.6;
      margin: 0;
    }}
    @media (max-width: 1200px) {{
      .step-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
      .hero-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 900px) {{
      .masthead, .comparison-grid {{ grid-template-columns: 1fr; }}
      .step-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .source-list {{ columns: 1; }}
    }}
    @media (max-width: 640px) {{
      .shell {{ padding: 18px 14px 40px; }}
      .hero, .status-panel, .section {{ padding: 18px; border-radius: 18px; }}
      .hero-grid, .step-grid {{ grid-template-columns: 1fr; }}
      .section-head {{ flex-direction: column; align-items: start; }}
      table {{ display: block; overflow-x: auto; white-space: nowrap; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="masthead">
      <div class="hero">
        <div class="toolbar" style="margin-bottom:14px;">
          {pill("7 / 7 steps complete", "good")}
          {pill(f"Updated {updated_at}", "info")}
          {pill(f"TAG {PROJECT_META['tag']}", "neutral")}
        </div>
        <h1>{esc(PROJECT_META['title'])}</h1>
        <p>{esc(PROJECT_META['subtitle'])}</p>
        <div class="hero-meta">
          {pill(f"Started {PROJECT_META['started_at']}", "neutral")}
          {pill(f"FE run {PROJECT_META['fe_run_id']}", "neutral")}
          {pill("Duplicate drug_name removed", "warn")}
          {pill("Step 6-7 selected route = lightweight", "good")}
        </div>
      </div>
      <aside class="status-panel">
        <h2>Overall Progress</h2>
        <div class="status-big">100%</div>
        <div class="progress"><span></span></div>
        <div class="footnote">
          팀원 원본 대시보드 구조를 참고하되, 현재 재현한 산출물과 6-model / lightweight 비교를 같은 화면에서 확인할 수 있게 정리했습니다.
        </div>
        <div class="toolbar">
          <a class="button primary" href="dashboard_reference.html">원본 참고 대시보드</a>
          <a class="button" href="step7_detail.html">기존 Step 7 상세</a>
        </div>
      </aside>
    </section>

    <section class="hero-grid">
      {hero_cards}
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Pipeline Snapshot</h2>
          <p>현재 재현 흐름의 핵심 상태만 7단계로 압축했습니다.</p>
        </div>
      </div>
      <div class="step-grid">
        {build_step_cards(model_rows, ensemble_full, ensemble_lite, step6_full, step7_lite)}
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Step 4 Model Leaderboard</h2>
          <p>기준선은 Spearman ≥ {THRESHOLDS['spearman']} and RMSE ≤ {THRESHOLDS['rmse']} 입니다. `META`는 base ensemble 후보가 아닌 메타 모델, `METABRIC`은 Step 6 보조용 모델입니다.</p>
        </div>
      </div>
      <table>
        <thead>
          <tr>
            <th>Family</th>
            <th>Model</th>
            <th>Spearman</th>
            <th>RMSE</th>
            <th>Pearson</th>
            <th>R²</th>
            <th>Gap</th>
            <th>Time</th>
            <th>Status</th>
            <th>Step 5 Role</th>
          </tr>
        </thead>
        <tbody>
          {build_model_table(model_rows)}
        </tbody>
      </table>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Step 5 Ensemble Comparison</h2>
          <p>6-model 기준 성능과 실제 채택한 lightweight 3-model 트랙을 나란히 비교합니다.</p>
        </div>
      </div>
      <table>
        <thead>
          <tr>
            <th>Metric</th>
            <th>6-model</th>
            <th>Lightweight</th>
          </tr>
        </thead>
        <tbody>
          {build_step5_compare(ensemble_full, ensemble_lite)}
        </tbody>
      </table>
      <div class="comparison-grid" style="margin-top:18px;">
        <div class="compare-card">
          <h3>6-model Weights</h3>
          <p>CatBoost, LightGBM, XGBoost, FlatMLP, ResidualMLP, Cross-Attention</p>
          <div class="weights">
            {build_weights(ensemble_full['weights'], "6")}
          </div>
        </div>
        <div class="compare-card">
          <h3>Lightweight Weights</h3>
          <p>LightGBM, FlatMLP, Cross-Attention. 성능 저하는 작고 Top15 세트는 동일합니다.</p>
          <div class="weights">
            {build_weights(ensemble_lite['weights'], "lite")}
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Step 6 METABRIC Comparison</h2>
          <p>중복 제거 이후에는 6-model과 lightweight의 검증 지표가 사실상 동일합니다.</p>
        </div>
      </div>
      <table>
        <thead>
          <tr>
            <th>Metric</th>
            <th>6-model</th>
            <th>Lightweight</th>
          </tr>
        </thead>
        <tbody>
          {build_step6_compare(step6_full, step6_lite)}
        </tbody>
      </table>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Step 7 ADMET Comparison</h2>
          <p>최종 후보 drug set은 15 / 15 동일하고, 차이는 일부 순위 이동뿐입니다.</p>
        </div>
      </div>
      <table style="margin-bottom:18px;">
        <thead>
          <tr>
            <th>Metric</th>
            <th>6-model</th>
            <th>Lightweight</th>
          </tr>
        </thead>
        <tbody>
          {build_step7_compare(step7_full, step7_lite)}
        </tbody>
      </table>
      <table>
        <thead>
          <tr>
            <th>Drug</th>
            <th>6-model rank</th>
            <th>Lite rank</th>
            <th>Shift</th>
            <th>6-model score</th>
            <th>Lite score</th>
            <th>Category</th>
          </tr>
        </thead>
        <tbody>
          {build_rank_compare(step7_full, step7_lite)}
        </tbody>
      </table>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Final Candidates (Selected Lightweight Route)</h2>
          <p>Step 7 최종 선정 결과입니다. 중복 약물은 제거된 상태이며, flags는 안전성 주의 포인트만 남겼습니다.</p>
        </div>
      </div>
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Drug</th>
            <th>Target</th>
            <th>Pathway</th>
            <th>Pred IC50</th>
            <th>Combined</th>
            <th>Category</th>
            <th>Flags</th>
          </tr>
        </thead>
        <tbody>
          {build_final_candidates(step7_lite)}
        </tbody>
      </table>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Data Sources</h2>
          <p>이 대시보드는 아래 산출물을 직접 읽어 생성했습니다.</p>
        </div>
      </div>
      <ul class="source-list">
        {source_links}
      </ul>
    </section>
  </div>
</body>
</html>
"""


def generate_index() -> str:
    updated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{esc(PROJECT_META['title'])}</title>
  <style>
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background:
        radial-gradient(circle at top right, rgba(83,180,255,0.18), transparent 28%),
        linear-gradient(180deg, #0b1424 0%, #09111f 100%);
      color: #e5eefb;
      font-family: "Segoe UI", system-ui, sans-serif;
      padding: 24px;
    }}
    .card {{
      width: min(780px, 100%);
      background: rgba(17,28,47,0.94);
      border: 1px solid #2d4061;
      border-radius: 24px;
      padding: 28px;
      box-shadow: 0 22px 48px rgba(0,0,0,0.28);
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(2rem, 4vw, 2.8rem);
      letter-spacing: -0.03em;
    }}
    p {{
      margin: 0;
      color: #95a7c5;
      line-height: 1.7;
    }}
    .meta {{
      margin-top: 16px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 0.76rem;
      font-weight: 700;
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.10);
    }}
    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 24px;
    }}
    .button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      text-decoration: none;
      color: inherit;
      padding: 12px 16px;
      border-radius: 999px;
      border: 1px solid #2d4061;
      font-weight: 700;
      background: rgba(255,255,255,0.04);
    }}
    .button.primary {{
      background: linear-gradient(90deg, rgba(83,180,255,0.16), rgba(126,240,197,0.16));
      border-color: rgba(126,240,197,0.24);
    }}
  </style>
</head>
<body>
  <main class="card">
    <h1>{esc(PROJECT_META['title'])}</h1>
    <p>현재 실행 기준 대시보드는 `dashboard.html`에, 팀원 참고 원본은 `dashboard_reference.html`에 남겨두었습니다. 이 페이지는 GitHub Pages나 로컬 진입점으로 사용할 수 있게 간단하게 정리했습니다.</p>
    <div class="meta">
      <span class="pill">Updated {esc(updated_at)}</span>
      <span class="pill">Selected route: lightweight</span>
      <span class="pill">S3: {esc(PROJECT_META['s3_base'])}</span>
    </div>
    <div class="actions">
      <a class="button primary" href="dashboard.html">현재 재현 대시보드 열기</a>
      <a class="button" href="dashboard_reference.html">팀원 참고 대시보드 보기</a>
      <a class="button" href="step7_detail.html">기존 상세 페이지 보기</a>
    </div>
  </main>
</body>
</html>
"""


def main() -> None:
    DASHBOARD_PATH.write_text(generate_dashboard())
    INDEX_PATH.write_text(generate_index())
    print(f"Wrote {DASHBOARD_PATH}")
    print(f"Wrote {INDEX_PATH}")


if __name__ == "__main__":
    main()
