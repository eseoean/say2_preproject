#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path


ROOT = Path("/Users/skku_aws2_18/pre_project/say2_preproject/Improving GroupCV")
RESULTS = ROOT / "results"
MD_OUT = RESULTS / "oof_metrics_diversity_report_20260415.md"
HTML_OUT = RESULTS / "oof_metrics_diversity_report_20260415.html"


def load_json(path: Path):
    return json.loads(path.read_text())


def maybe_load(path: Path):
    if not path.exists():
        return None
    return load_json(path)


def fmt(v, digits: int = 4):
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    return f"{float(v):.{digits}f}"


def fmt_time_seconds(v):
    if v is None:
        return "-"
    return f"{float(v) / 60.0:.1f}m"


def sort_rows(rows, spearman_idx: int):
    def key(row):
        try:
            return float(row[spearman_idx])
        except Exception:
            return float("-inf")

    return sorted(rows, key=key, reverse=True)


def inline_html(text):
    if text is None:
        return "-"
    raw = str(text)
    escaped = escape(raw)
    return re.sub(r"`([^`]+)`", lambda m: f"<code>{escape(m.group(1))}</code>", escaped)


def md_table(headers, rows):
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def html_table(headers, rows):
    head = "".join(f"<th>{inline_html(h)}</th>" for h in headers)
    body = []
    for row in rows:
        cells = "".join(f"<td>{inline_html(cell)}</td>" for cell in row)
        body.append(f"<tr>{cells}</tr>")
    return (
        '<div class="table-wrap">'
        f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
        "</div>"
    )


def model_row(name, metrics, elapsed=None, note=""):
    return [
        name,
        fmt(metrics.get("spearman")),
        fmt(metrics.get("rmse")),
        fmt(metrics.get("mae")),
        fmt(metrics.get("pearson")),
        fmt(metrics.get("r2")),
        fmt(metrics.get("ndcg@20")),
        fmt_time_seconds(elapsed),
        note,
    ]


def foldmean_row(
    name,
    spearman,
    rmse,
    mae,
    pearson,
    r2,
    ndcg,
    elapsed,
    note="legacy fold-mean",
):
    return [
        name,
        fmt(spearman),
        fmt(rmse),
        fmt(mae),
        fmt(pearson),
        fmt(r2),
        fmt(ndcg),
        fmt_time_seconds(elapsed),
        note,
    ]


def diversity_rows(obj):
    diversity = obj.get("diversity")
    if not diversity:
        return []
    pair_rows = []
    for item in diversity.get("pairwise", []):
        pair_rows.append(
            [
                item["pair"],
                fmt(item.get("prediction_pearson")),
                fmt(item.get("prediction_spearman")),
                fmt(item.get("residual_pearson")),
                fmt(item.get("residual_spearman")),
                fmt(item.get("mean_abs_prediction_gap")),
            ]
        )
    summary = diversity.get("summary", {})
    if summary:
        pair_rows.append(
            [
                "AVG",
                fmt(summary.get("avg_prediction_pearson")),
                fmt(summary.get("avg_prediction_spearman")),
                fmt(summary.get("avg_residual_pearson")),
                fmt(summary.get("avg_residual_spearman")),
                fmt(summary.get("avg_mean_abs_prediction_gap")),
            ]
        )
    return pair_rows


def labeled_diversity_rows(obj, label):
    rows = diversity_rows(obj)
    if not rows:
        return []
    labeled = []
    for idx, row in enumerate(rows):
        row = row.copy()
        if idx == 0:
            row[0] = f"[{label}] {row[0]}"
        elif row[0] == "AVG":
            row[0] = f"[{label}] AVG"
        else:
            row[0] = f"{label} :: {row[0]}"
        labeled.append(row)
    return labeled


def numeric_only_ml_rows():
    rows = []
    file_specs = [
        RESULTS / "exact_repo_slim_numeric_ml_groupcv_oof_v1.json",
        RESULTS / "exact_repo_slim_numeric_ml_groupcv_rf_et_oof_v1.json",
        RESULTS / "exact_repo_slim_numeric_ml_groupcv_catboost_oof_v2.json",
    ]
    for path in file_specs:
        obj = maybe_load(path)
        if not obj:
            continue
        for m in obj.get("models", []):
            rows.append(model_row(m["model"], m["overall_metrics"], m.get("elapsed_sec"), "OOF"))
    return sort_rows(rows, 1)


def numeric_only_dl_rows():
    rows = []
    for path in [
        RESULTS / "exact_repo_slim_groupcv_3models_v1.json",
        RESULTS / "exact_repo_slim_groupcv_moremodels_v1.json",
        RESULTS / "exact_repo_slim_groupcv_tab_wd_v1.json",
    ]:
        obj = maybe_load(path)
        if not obj:
            continue
        for m in obj.get("models", []):
            rows.append(
                foldmean_row(
                    m["model"],
                    m.get("spearman_mean"),
                    m.get("rmse_mean"),
                    m.get("mae_mean"),
                    m.get("pearson_mean"),
                    m.get("r2_mean"),
                    m.get("ndcg@20_mean"),
                    m.get("elapsed_sec"),
                )
            )
    return sort_rows(rows, 1)


def numeric_only_ensemble_rows():
    obj = maybe_load(RESULTS / "exact_repo_slim_top3_ensemble_weighted_oof_v2.json")
    if not obj:
        return []
    rows = [
        [
            "Equal",
            "WideDeep + CrossAttention + FlatMLP",
            fmt(obj["equal_overall_metrics"].get("spearman")),
            fmt(obj["equal_overall_metrics"].get("rmse")),
            fmt(obj["equal_overall_metrics"].get("mae")),
            fmt(obj["equal_overall_metrics"].get("pearson")),
            fmt(obj["equal_overall_metrics"].get("r2")),
            fmt(obj["equal_overall_metrics"].get("ndcg@20")),
        ],
        [
            "Weighted",
            "WideDeep + CrossAttention + FlatMLP",
            fmt(obj["weighted_overall_metrics"].get("spearman")),
            fmt(obj["weighted_overall_metrics"].get("rmse")),
            fmt(obj["weighted_overall_metrics"].get("mae")),
            fmt(obj["weighted_overall_metrics"].get("pearson")),
            fmt(obj["weighted_overall_metrics"].get("r2")),
            fmt(obj["weighted_overall_metrics"].get("ndcg@20")),
        ],
    ]
    return sort_rows(rows, 2)


def smiles_ml_rows():
    obj = maybe_load(RESULTS / "exact_repo_slim_smiles_ml_groupcv_v1.json")
    rows = []
    if not obj:
        return rows
    for m in obj.get("models", []):
        rows.append(model_row(m["model"], m["overall_metrics"], m.get("elapsed_sec"), "OOF"))
    return sort_rows(rows, 1)


def smiles_dl_rows():
    rows = []
    obj = maybe_load(RESULTS / "exact_repo_slim_smiles_all_dl_v1.json")
    if obj:
        for m in obj.get("models", []):
            rows.append(
                foldmean_row(
                    m["model"],
                    m["overall_metrics"].get("spearman"),
                    m["overall_metrics"].get("rmse"),
                    m["overall_metrics"].get("mae"),
                    m["overall_metrics"].get("pearson"),
                    m["overall_metrics"].get("r2"),
                    m["overall_metrics"].get("ndcg@20"),
                    m.get("elapsed_sec"),
                    "OOF",
                )
            )
    tt = maybe_load(RESULTS / "exact_repo_slim_smiles_tabtransformer_earlystop_v1.json")
    if tt and tt.get("models"):
        m = tt["models"][0]
        rows.append(
            foldmean_row(
                m["model"],
                m["overall_metrics"].get("spearman"),
                m["overall_metrics"].get("rmse"),
                m["overall_metrics"].get("mae"),
                m["overall_metrics"].get("pearson"),
                m["overall_metrics"].get("r2"),
                m["overall_metrics"].get("ndcg@20"),
                m.get("elapsed_sec"),
                "OOF, 2-fold early stop",
            )
        )
    return sort_rows(rows, 1)


def smiles_ensemble_rows():
    obj = maybe_load(RESULTS / "exact_repo_slim_smiles_frc_ensemble_oof_v3.json")
    if not obj:
        obj = maybe_load(RESULTS / "exact_repo_slim_smiles_frc_ensemble_v1.json")
    if not obj:
        return []
    rows = [
        [
            "Equal",
            "FlatMLP + ResidualMLP + CrossAttention",
            fmt(obj["equal_overall_metrics"].get("spearman")),
            fmt(obj["equal_overall_metrics"].get("rmse")),
            fmt(obj["equal_overall_metrics"].get("mae")),
            fmt(obj["equal_overall_metrics"].get("pearson")),
            fmt(obj["equal_overall_metrics"].get("r2")),
            fmt(obj["equal_overall_metrics"].get("ndcg@20")),
        ],
        [
            "Weighted",
            "FlatMLP + ResidualMLP + CrossAttention",
            fmt(obj["weighted_overall_metrics"].get("spearman")),
            fmt(obj["weighted_overall_metrics"].get("rmse")),
            fmt(obj["weighted_overall_metrics"].get("mae")),
            fmt(obj["weighted_overall_metrics"].get("pearson")),
            fmt(obj["weighted_overall_metrics"].get("r2")),
            fmt(obj["weighted_overall_metrics"].get("ndcg@20")),
        ],
    ]
    return sort_rows(rows, 2)


def strong_ml_rows():
    rows = []
    for name in [
        "lightgbm",
        "lightgbm_dart",
        "xgboost",
        "catboost",
        "randomforest",
        "extratrees",
    ]:
        obj = maybe_load(RESULTS / f"exact_repo_slim_strong_context_smiles_ml_groupcv_{name}_v1.json")
        if not obj:
            continue
        m = obj["models"][0]
        rows.append(model_row(m["model"], m["overall_metrics"], m.get("elapsed_sec"), "OOF"))
    return sort_rows(rows, 1)


def strong_dl_rows():
    rows = []
    top3 = maybe_load(RESULTS / "exact_repo_slim_smiles_ab_top3_v1.json")
    if top3:
        for m in top3.get("models", []):
            v = m["variants"]["strong_context_plus_smiles"]
            rows.append(
                foldmean_row(
                    m["model"],
                    v.get("spearman_mean"),
                    v.get("rmse_mean"),
                    v.get("mae_mean"),
                    v.get("pearson_mean"),
                    v.get("r2_mean"),
                    v.get("ndcg@20_mean"),
                    v.get("elapsed_sec"),
                    "OOF-ish standalone summary",
                )
            )
    extra = maybe_load(RESULTS / "exact_repo_slim_strong_context_smiles_more_dl_oof_v2.json")
    if extra:
        seen = {row[0] for row in rows}
        for m in extra.get("models", []):
            if m["model"] in seen:
                continue
            rows.append(model_row(m["model"], m["overall_metrics"], m.get("elapsed_sec"), "OOF"))
    else:
        frc = maybe_load(RESULTS / "exact_repo_slim_strong_context_smiles_frc_ensemble_v1.json")
        if frc and "ResidualMLP" not in {row[0] for row in rows}:
            residual = frc["base_model_metrics"]["ResidualMLP"]
            rows.append(
                foldmean_row(
                    "ResidualMLP",
                    residual.get("spearman_mean"),
                    residual.get("rmse_mean"),
                    residual.get("mae_mean"),
                    residual.get("pearson_mean"),
                    residual.get("r2_mean"),
                    residual.get("ndcg@20_mean"),
                    None,
                    "base run from ensemble",
                )
            )
        rows.extend(
            [
                ["TabNet", "0.5543", "2.1695", "1.6232", "-", "-", "0.8163", "-", "legacy log summary"],
                ["FTTransformer", "0.5550", "2.1911", "1.6389", "-", "-", "0.8223", "-", "legacy log summary"],
                ["TabTransformer", "-", "-", "-", "-", "-", "-", "-", "pending / not saved"],
            ]
        )
    return sort_rows(rows, 1)


def strong_ensemble_rows():
    rows = []
    frc = maybe_load(RESULTS / "exact_repo_slim_strong_context_smiles_frc_ensemble_oof_v2.json")
    if not frc:
        frc = maybe_load(RESULTS / "exact_repo_slim_strong_context_smiles_frc_ensemble_v1.json")
    if frc:
        rows.extend(
            [
                [
                    "Equal",
                    "FlatMLP + ResidualMLP + CrossAttention",
                    fmt(frc["equal_overall_metrics"].get("spearman")),
                    fmt(frc["equal_overall_metrics"].get("rmse")),
                    fmt(frc["equal_overall_metrics"].get("mae")),
                    fmt(frc["equal_overall_metrics"].get("pearson")),
                    fmt(frc["equal_overall_metrics"].get("r2")),
                    fmt(frc["equal_overall_metrics"].get("ndcg@20")),
                ],
                [
                    "Weighted",
                    "FlatMLP + ResidualMLP + CrossAttention",
                    fmt(frc["weighted_overall_metrics"].get("spearman")),
                    fmt(frc["weighted_overall_metrics"].get("rmse")),
                    fmt(frc["weighted_overall_metrics"].get("mae")),
                    fmt(frc["weighted_overall_metrics"].get("pearson")),
                    fmt(frc["weighted_overall_metrics"].get("r2")),
                    fmt(frc["weighted_overall_metrics"].get("ndcg@20")),
                ],
            ]
        )
    old_top3 = maybe_load(RESULTS / "exact_repo_slim_strong_context_smiles_top3_ensemble_v1.json")
    if old_top3:
        rows.extend(
            [
                [
                    "Equal",
                    "FlatMLP + WideDeep + CrossAttention",
                    fmt(old_top3["equal_overall_metrics"].get("spearman")),
                    fmt(old_top3["equal_overall_metrics"].get("rmse")),
                    fmt(old_top3["equal_overall_metrics"].get("mae")),
                    fmt(old_top3["equal_overall_metrics"].get("pearson")),
                    fmt(old_top3["equal_overall_metrics"].get("r2")),
                    fmt(old_top3["equal_overall_metrics"].get("ndcg@20")),
                ],
                [
                    "Weighted",
                    "FlatMLP + WideDeep + CrossAttention",
                    fmt(old_top3["weighted_overall_metrics"].get("spearman")),
                    fmt(old_top3["weighted_overall_metrics"].get("rmse")),
                    fmt(old_top3["weighted_overall_metrics"].get("mae")),
                    fmt(old_top3["weighted_overall_metrics"].get("pearson")),
                    fmt(old_top3["weighted_overall_metrics"].get("r2")),
                    fmt(old_top3["weighted_overall_metrics"].get("ndcg@20")),
                ],
            ]
        )
    fle = maybe_load(RESULTS / "exact_repo_slim_strong_context_smiles_fle_ensemble_v1.json")
    if fle and "weighted_overall_metrics" in fle:
        rows.extend(
            [
                [
                    "Equal",
                    "FlatMLP + LightGBM_DART + ExtraTrees",
                    fmt(fle["equal_overall_metrics"].get("spearman")),
                    fmt(fle["equal_overall_metrics"].get("rmse")),
                    fmt(fle["equal_overall_metrics"].get("mae")),
                    fmt(fle["equal_overall_metrics"].get("pearson")),
                    fmt(fle["equal_overall_metrics"].get("r2")),
                    fmt(fle["equal_overall_metrics"].get("ndcg@20")),
                ],
                [
                    "Weighted",
                    "FlatMLP + LightGBM_DART + ExtraTrees",
                    fmt(fle["weighted_overall_metrics"].get("spearman")),
                    fmt(fle["weighted_overall_metrics"].get("rmse")),
                    fmt(fle["weighted_overall_metrics"].get("mae")),
                    fmt(fle["weighted_overall_metrics"].get("pearson")),
                    fmt(fle["weighted_overall_metrics"].get("r2")),
                    fmt(fle["weighted_overall_metrics"].get("ndcg@20")),
                ],
            ]
        )
    return sort_rows(rows, 2)


def strong_diversity_rows():
    rows = []
    frc = maybe_load(RESULTS / "exact_repo_slim_strong_context_smiles_frc_ensemble_oof_v2.json")
    if not frc:
        frc = maybe_load(RESULTS / "exact_repo_slim_strong_context_smiles_frc_ensemble_v1.json")
    fle = maybe_load(RESULTS / "exact_repo_slim_strong_context_smiles_fle_ensemble_v1.json")
    if frc:
        rows.extend(labeled_diversity_rows(frc, "FRC ensemble"))
    if fle and fle.get("diversity"):
        if rows:
            rows.append(["---", "---", "---", "---", "---", "---"])
        rows.extend(labeled_diversity_rows(fle, "FLE mixed ensemble"))
    return rows


def dataset_sections():
    num_ens = maybe_load(RESULTS / "exact_repo_slim_top3_ensemble_weighted_oof_v2.json")
    smiles_ens = maybe_load(RESULTS / "exact_repo_slim_smiles_frc_ensemble_oof_v3.json")
    strong_ens = maybe_load(RESULTS / "exact_repo_slim_strong_context_smiles_frc_ensemble_oof_v2.json")
    if not smiles_ens:
        smiles_ens = maybe_load(RESULTS / "exact_repo_slim_smiles_frc_ensemble_v1.json")
    if not strong_ens:
        strong_ens = maybe_load(RESULTS / "exact_repo_slim_strong_context_smiles_frc_ensemble_v1.json")

    return [
        {
            "title": "A. exact slim (numeric-only)",
            "ml_rows": numeric_only_ml_rows(),
            "dl_rows": numeric_only_dl_rows(),
            "ensemble_rows": numeric_only_ensemble_rows(),
            "diversity_rows": diversity_rows(num_ens) if num_ens else [],
            "notes": [
                "ML은 OOF 로컬 재실행 기준입니다. DL은 기존 numeric-only GroupCV 요약본을 사용했습니다.",
                "Ensemble diversity는 `WideDeep + CrossAttention + FlatMLP` 기준입니다.",
            ],
        },
        {
            "title": "B. exact slim + SMILES",
            "ml_rows": smiles_ml_rows(),
            "dl_rows": smiles_dl_rows(),
            "ensemble_rows": smiles_ensemble_rows(),
            "diversity_rows": diversity_rows(smiles_ens) if smiles_ens else [],
            "notes": [
                "TabTransformer는 조기 종료 규칙으로 2-fold만 실행되었습니다.",
                "Ensemble diversity는 `FlatMLP + ResidualMLP + CrossAttention` 기준입니다.",
            ],
        },
        {
            "title": "C. exact slim + strong context + SMILES",
            "ml_rows": strong_ml_rows(),
            "dl_rows": strong_dl_rows(),
            "ensemble_rows": strong_ensemble_rows(),
            "diversity_rows": strong_diversity_rows(),
            "notes": [
                "DL 추가 OOF 재실행 결과를 반영한 최신본입니다.",
                "Ensemble에는 `FlatMLP + ResidualMLP + CrossAttention`와 `FlatMLP + LightGBM_DART + ExtraTrees`를 함께 반영했습니다.",
                "Diversity 표는 `FRC ensemble`과 `FLE mixed ensemble`을 한 섹션에서 같이 보여줍니다.",
            ],
        },
    ]


def build_md():
    out = []
    out.append("# OOF Metrics and Ensemble Diversity Report")
    out.append("")
    out.append("세 가지 입력셋 기준으로 ML, DL, Ensemble 성능을 같은 형식으로 정리했다. 기본 정렬 기준은 `Spearman` 내림차순이다.")
    out.append("")

    headers_model = ["Model", "Spearman", "RMSE", "MAE", "Pearson", "R²", "NDCG@20", "Time", "Note"]
    headers_ens = ["Type", "구성", "Spearman", "RMSE", "MAE", "Pearson", "R²", "NDCG@20"]
    headers_div = ["Pair", "Pred Pearson", "Pred Spearman", "Resid Pearson", "Resid Spearman", "Mean Abs Gap"]

    for section in dataset_sections():
        out.append(f"## {section['title']}")
        out.append("")
        out.append("### ML")
        out.append("")
        out.append(md_table(headers_model, section["ml_rows"]) if section["ml_rows"] else "_No rows_")
        out.append("")
        out.append("### DL")
        out.append("")
        out.append(md_table(headers_model, section["dl_rows"]) if section["dl_rows"] else "_No rows_")
        out.append("")
        out.append("### Ensemble")
        out.append("")
        out.append(md_table(headers_ens, section["ensemble_rows"]) if section["ensemble_rows"] else "_No rows_")
        out.append("")
        out.append("### Diversity")
        out.append("")
        out.append(md_table(headers_div, section["diversity_rows"]) if section["diversity_rows"] else "_No diversity rows yet_")
        out.append("")
        out.append("### Notes")
        out.append("")
        for note in section["notes"]:
            out.append(f"- {note}")
        out.append("")

    return "\n".join(out)


def build_html():
    sections_html = []
    headers_model = ["Model", "Spearman", "RMSE", "MAE", "Pearson", "R²", "NDCG@20", "Time", "Note"]
    headers_ens = ["Type", "구성", "Spearman", "RMSE", "MAE", "Pearson", "R²", "NDCG@20"]
    headers_div = ["Pair", "Pred Pearson", "Pred Spearman", "Resid Pearson", "Resid Spearman", "Mean Abs Gap"]

    for section in dataset_sections():
        notes_html = "".join(f"<li>{inline_html(note)}</li>" for note in section["notes"])
        sections_html.append(
            f"""
            <section class="dataset-section">
              <h2>{inline_html(section['title'])}</h2>
              <section class="subsection"><h3>ML</h3>{html_table(headers_model, section['ml_rows']) if section['ml_rows'] else '<p class="plain-note">No rows</p>'}</section>
              <section class="subsection"><h3>DL</h3>{html_table(headers_model, section['dl_rows']) if section['dl_rows'] else '<p class="plain-note">No rows</p>'}</section>
              <section class="subsection"><h3>Ensemble</h3>{html_table(headers_ens, section['ensemble_rows']) if section['ensemble_rows'] else '<p class="plain-note">No rows</p>'}</section>
              <section class="subsection"><h3>Diversity</h3>{html_table(headers_div, section['diversity_rows']) if section['diversity_rows'] else '<p class="plain-note">No diversity rows yet</p>'}</section>
              <section class="subsection"><h3>Notes</h3><ul>{notes_html}</ul></section>
            </section>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OOF Metrics and Diversity Report</title>
  <style>
    :root {{
      --bg: #f6f8fc;
      --panel: #ffffff;
      --ink: #1e2a39;
      --muted: #5b6777;
      --line: #d9e1ec;
      --head: #eaf2ff;
      --accent: #2f6fdd;
      --shadow: 0 10px 30px rgba(31, 45, 61, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 32px;
      background: linear-gradient(180deg, #f6f8fc 0%, #eef3fb 100%);
      color: var(--ink);
      font-family: "Pretendard", "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
      line-height: 1.55;
    }}
    .page {{
      max-width: 1480px;
      margin: 0 auto;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
      padding: 32px;
    }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    h1 {{ font-size: 32px; }}
    h2 {{ font-size: 24px; padding-top: 8px; }}
    h3 {{ font-size: 18px; color: var(--accent); }}
    p.lead {{ color: var(--muted); margin: 0 0 24px; }}
    .dataset-section {{
      margin-top: 28px;
      padding: 24px;
      border: 1px solid var(--line);
      border-radius: 20px;
      background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
    }}
    .subsection {{ margin-top: 20px; }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: white;
    }}
    table {{ width: 100%; border-collapse: collapse; min-width: 900px; }}
    th, td {{
      padding: 11px 14px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }}
    th {{
      position: sticky;
      top: 0;
      background: var(--head);
      color: #17345f;
      font-weight: 700;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    code {{
      background: #f3f7ff;
      border: 1px solid #dbe7ff;
      border-radius: 8px;
      padding: 1px 6px;
      font-size: 0.95em;
    }}
    ul {{ margin: 0; padding-left: 20px; }}
    .plain-note {{ color: var(--muted); }}
  </style>
</head>
<body>
  <div class="page">
    <h1>OOF Metrics and Ensemble Diversity Report</h1>
    <p class="lead">세 가지 입력셋 기준으로 ML, DL, Ensemble 성능과 ensemble diversity를 같은 형식으로 정리했다. 표는 기본적으로 <code>Spearman</code> 내림차순이다.</p>
    {''.join(sections_html)}
  </div>
</body>
</html>"""


def main():
    MD_OUT.write_text(build_md())
    HTML_OUT.write_text(build_html())
    print(MD_OUT)
    print(HTML_OUT)


if __name__ == "__main__":
    main()
