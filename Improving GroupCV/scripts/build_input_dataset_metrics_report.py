#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path


ROOT = Path("/Users/skku_aws2_18/pre_project/say2_preproject/Improving GroupCV")
RESULTS = ROOT / "results"
OUT = RESULTS / "input_dataset_metrics_report_20260415.md"
HTML_OUT = RESULTS / "input_dataset_metrics_report_20260415.html"
REPO_STEP4 = Path("/tmp/biso_myprotocol_repo/20260414_re_pre_project_v3/step4_results")


def load_json(path: Path):
    return json.loads(path.read_text())


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


def sort_rows_by_spearman(rows, spearman_idx: int):
    def parse_value(row):
        try:
            return float(row[spearman_idx])
        except (TypeError, ValueError):
            return float("-inf")

    return sorted(rows, key=parse_value, reverse=True)


def md_table(headers, rows):
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def inline_html(text):
    if text is None:
        return "-"
    raw = str(text)
    escaped = escape(raw)
    return re.sub(r"`([^`]+)`", lambda m: f"<code>{escape(m.group(1))}</code>", escaped)


def html_table(headers, rows):
    head = "".join(f"<th>{inline_html(h)}</th>" for h in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{inline_html(cell)}</td>" for cell in row)
        body_rows.append(f"<tr>{cells}</tr>")
    body = "".join(body_rows)
    return (
        '<div class="table-wrap">'
        f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
        "</div>"
    )


def numeric_only_dl_rows():
    rows = []
    for file in [
        RESULTS / "exact_repo_slim_groupcv_3models_v1.json",
        RESULTS / "exact_repo_slim_groupcv_moremodels_v1.json",
        RESULTS / "exact_repo_slim_groupcv_tab_wd_v1.json",
    ]:
        obj = load_json(file)
        for m in obj["models"]:
            rows.append(
                [
                    m["model"],
                    fmt(m.get("spearman_mean")),
                    fmt(m.get("rmse_mean")),
                    fmt(m.get("mae_mean")),
                    fmt(m.get("pearson_mean")),
                    fmt(m.get("r2_mean")),
                    fmt(m.get("ndcg@20_mean")),
                    fmt_time_seconds(m.get("elapsed_sec")),
                ]
            )
    return sort_rows_by_spearman(rows, 1)


def numeric_only_ensemble_rows():
    local_top3 = load_json(RESULTS / "exact_repo_slim_top3_ensemble_weighted_v1.json")
    repo_weighted = load_json(REPO_STEP4 / "step2_groupkfold_ensemble_weighted_results.json")
    rows = [
        [
            "Local top3 weighted",
            "WideDeep + CrossAttention + FlatMLP",
            fmt(local_top3["weighted_overall_metrics"].get("spearman")),
            fmt(local_top3["weighted_overall_metrics"].get("rmse")),
            fmt(local_top3["weighted_overall_metrics"].get("mae")),
            fmt(local_top3["weighted_overall_metrics"].get("pearson")),
            fmt(local_top3["weighted_overall_metrics"].get("r2")),
            "-",
        ],
        [
            "Saved 6-model weighted",
            "1/2/4/10/12/13 weighted",
            fmt(repo_weighted["overall_metrics"].get("spearman")),
            fmt(repo_weighted["overall_metrics"].get("rmse")),
            fmt(repo_weighted["overall_metrics"].get("mae")),
            fmt(repo_weighted["overall_metrics"].get("pearson")),
            fmt(repo_weighted["overall_metrics"].get("r2")),
            fmt(repo_weighted["overall_metrics"].get("ndcg@20")),
        ],
    ]
    return sort_rows_by_spearman(rows, 2)


def smiles_only_ml_rows():
    obj = load_json(RESULTS / "exact_repo_slim_smiles_ml_groupcv_v1.json")
    rows = []
    for m in obj["models"]:
        o = m["overall_metrics"]
        rows.append(
            [
                m["model"],
                fmt(o.get("spearman")),
                fmt(o.get("rmse")),
                fmt(o.get("mae")),
                fmt(o.get("pearson")),
                fmt(o.get("r2")),
                fmt(o.get("ndcg@20")),
                fmt_time_seconds(m.get("elapsed_sec")),
            ]
        )
    return sort_rows_by_spearman(rows, 1)


def smiles_only_dl_rows():
    obj = load_json(RESULTS / "exact_repo_slim_smiles_all_dl_v1.json")
    rows = []
    for m in obj["models"]:
        rows.append(
            [
                m["model"],
                fmt(m.get("spearman_mean")),
                fmt(m.get("rmse_mean")),
                fmt(m.get("mae_mean")),
                fmt(m.get("pearson_mean")),
                fmt(m.get("r2_mean")),
                fmt(m.get("ndcg@20_mean")),
                fmt_time_seconds(m.get("elapsed_sec")),
                "",
            ]
        )
    tt = load_json(RESULTS / "exact_repo_slim_smiles_tabtransformer_earlystop_v1.json")["models"][0]
    rows.append(
        [
            tt["model"],
            fmt(tt.get("spearman_mean")),
            fmt(tt.get("rmse_mean")),
            fmt(tt.get("mae_mean")),
            fmt(tt.get("pearson_mean")),
            fmt(tt.get("r2_mean")),
            fmt(tt.get("ndcg@20_mean")),
            fmt_time_seconds(tt.get("elapsed_sec")),
            "2-fold early stop",
        ]
    )
    return sort_rows_by_spearman(rows, 1)


def smiles_only_ensemble_rows():
    obj = load_json(RESULTS / "exact_repo_slim_smiles_frc_ensemble_v1.json")
    return sort_rows_by_spearman([
        [
            "FRC weighted",
            "FlatMLP + ResidualMLP + CrossAttention",
            fmt(obj["weighted_overall_metrics"].get("spearman")),
            fmt(obj["weighted_overall_metrics"].get("rmse")),
            fmt(obj["weighted_overall_metrics"].get("mae")),
            fmt(obj["weighted_overall_metrics"].get("pearson")),
            fmt(obj["weighted_overall_metrics"].get("r2")),
            fmt(obj["weighted_overall_metrics"].get("ndcg@20")),
        ]
    ], 2)


def strong_context_ml_rows():
    files = [
        ("LightGBM", "exact_repo_slim_strong_context_smiles_ml_groupcv_lightgbm_v1.json"),
        ("LightGBM_DART", "exact_repo_slim_strong_context_smiles_ml_groupcv_lightgbm_dart_v1.json"),
        ("XGBoost", "exact_repo_slim_strong_context_smiles_ml_groupcv_xgboost_v1.json"),
        ("CatBoost", "exact_repo_slim_strong_context_smiles_ml_groupcv_catboost_v1.json"),
        ("RandomForest", "exact_repo_slim_strong_context_smiles_ml_groupcv_randomforest_v1.json"),
        ("ExtraTrees", "exact_repo_slim_strong_context_smiles_ml_groupcv_extratrees_v1.json"),
    ]
    rows = []
    for model, name in files:
        obj = load_json(RESULTS / name)
        m = obj["models"][0]
        o = m["overall_metrics"]
        rows.append(
            [
                model,
                fmt(o.get("spearman")),
                fmt(o.get("rmse")),
                fmt(o.get("mae")),
                fmt(o.get("pearson")),
                fmt(o.get("r2")),
                fmt(o.get("ndcg@20")),
                fmt_time_seconds(m.get("elapsed_sec")),
            ]
        )
    return sort_rows_by_spearman(rows, 1)


def strong_context_dl_rows():
    top3 = load_json(RESULTS / "exact_repo_slim_smiles_ab_top3_v1.json")
    rows = []
    for m in top3["models"]:
        v = m["variants"]["strong_context_plus_smiles"]
        rows.append(
            [
                m["model"],
                fmt(v.get("spearman_mean")),
                fmt(v.get("rmse_mean")),
                fmt(v.get("mae_mean")),
                fmt(v.get("pearson_mean")),
                fmt(v.get("r2_mean")),
                fmt(v.get("ndcg@20_mean")),
                fmt_time_seconds(v.get("elapsed_sec")),
                "standalone",
            ]
        )

    frc = load_json(RESULTS / "exact_repo_slim_strong_context_smiles_frc_ensemble_v1.json")
    residual = frc["base_model_metrics"]["ResidualMLP"]
    rows.append(
        [
            "ResidualMLP",
            fmt(residual.get("spearman_mean")),
            fmt(residual.get("rmse_mean")),
            fmt(residual.get("mae_mean")),
            fmt(residual.get("pearson_mean")),
            fmt(residual.get("r2_mean")),
            fmt(residual.get("ndcg@20_mean")),
            "-",
            "base run from FRC ensemble",
        ]
    )

    # Earlier standalone summaries were not persisted to JSON.
    rows.extend(
        [
            ["TabNet", "0.5543", "2.1695", "1.6232", "-", "-", "0.8163", "-", "log summary only"],
            ["FTTransformer", "0.5550", "2.1911", "1.6389", "-", "-", "0.8223", "-", "log summary only"],
            ["TabTransformer", "-", "-", "-", "-", "-", "-", "-", "not run / not saved"],
        ]
    )

    return sort_rows_by_spearman(rows, 1)


def strong_context_ensemble_rows():
    frc = load_json(RESULTS / "exact_repo_slim_strong_context_smiles_frc_ensemble_v1.json")
    old_top3 = load_json(RESULTS / "exact_repo_slim_strong_context_smiles_top3_ensemble_v1.json")
    return sort_rows_by_spearman([
        [
            "FRC weighted",
            "FlatMLP + ResidualMLP + CrossAttention",
            fmt(frc["weighted_overall_metrics"].get("spearman")),
            fmt(frc["weighted_overall_metrics"].get("rmse")),
            fmt(frc["weighted_overall_metrics"].get("mae")),
            fmt(frc["weighted_overall_metrics"].get("pearson")),
            fmt(frc["weighted_overall_metrics"].get("r2")),
            fmt(frc["weighted_overall_metrics"].get("ndcg@20")),
        ],
        [
            "Top3 weighted",
            "FlatMLP + WideDeep + CrossAttention",
            fmt(old_top3["weighted_overall_metrics"].get("spearman")),
            fmt(old_top3["weighted_overall_metrics"].get("rmse")),
            fmt(old_top3["weighted_overall_metrics"].get("mae")),
            fmt(old_top3["weighted_overall_metrics"].get("pearson")),
            fmt(old_top3["weighted_overall_metrics"].get("r2")),
            fmt(old_top3["weighted_overall_metrics"].get("ndcg@20")),
        ],
    ], 2)


def build_html_report():
    variant_rows = [
        [
            "A",
            "`exact slim (numeric-only)`",
            "`5529` numeric features",
            "`5529` numeric features",
        ],
        [
            "B",
            "`exact slim + SMILES`",
            "`5529 numeric + 64 SMILES SVD = 5593`",
            "`5529 numeric + SMILES branch`",
        ],
        [
            "C",
            "`exact slim + strong context + SMILES`",
            "`5529 numeric + 32 context one-hot + 64 SMILES SVD = 5625`",
            "`5529 numeric + 5 categorical context + SMILES branch`",
        ],
    ]

    sections = [
        (
            "A. exact slim (numeric-only)",
            [
                ("DL", ["Model", "Spearman", "RMSE", "MAE", "Pearson", "R²", "NDCG@20", "Time"], numeric_only_dl_rows()),
                (
                    "Ensemble",
                    ["Ensemble", "구성", "Spearman", "RMSE", "MAE", "Pearson", "R²", "NDCG@20"],
                    numeric_only_ensemble_rows(),
                ),
                ("ML", None, "현재 워크스페이스에는 `exact slim (numeric-only)` 조건의 ML full rerun 저장본이 없습니다."),
            ],
        ),
        (
            "B. exact slim + SMILES",
            [
                ("ML", ["Model", "Spearman", "RMSE", "MAE", "Pearson", "R²", "NDCG@20", "Time"], smiles_only_ml_rows()),
                (
                    "DL",
                    ["Model", "Spearman", "RMSE", "MAE", "Pearson", "R²", "NDCG@20", "Time", "비고"],
                    smiles_only_dl_rows(),
                ),
                (
                    "Ensemble",
                    ["Ensemble", "구성", "Spearman", "RMSE", "MAE", "Pearson", "R²", "NDCG@20"],
                    smiles_only_ensemble_rows(),
                ),
            ],
        ),
        (
            "C. exact slim + strong context + SMILES",
            [
                ("ML", ["Model", "Spearman", "RMSE", "MAE", "Pearson", "R²", "NDCG@20", "Time"], strong_context_ml_rows()),
                (
                    "DL",
                    ["Model", "Spearman", "RMSE", "MAE", "Pearson", "R²", "NDCG@20", "Time", "비고"],
                    strong_context_dl_rows(),
                ),
                (
                    "Ensemble",
                    ["Ensemble", "구성", "Spearman", "RMSE", "MAE", "Pearson", "R²", "NDCG@20"],
                    strong_context_ensemble_rows(),
                ),
            ],
        ),
    ]

    note_items = [
        "`exact slim + SMILES`의 `TabTransformer`는 사용자 지정 조기 종료 규칙으로 2개 fold만 실행되었습니다.",
        "`exact slim + strong context + SMILES`의 `TabNet`, `FTTransformer`는 이전 실행 로그 요약값만 남아 있어 `Pearson`, `R²`, `Time`은 `-`로 표시했습니다.",
        "`exact slim + strong context + SMILES`에서 현재 best ensemble은 `FlatMLP + ResidualMLP + CrossAttention` weighted 입니다.",
    ]

    section_html = []
    for title, blocks in sections:
        block_html = []
        for name, headers, payload in blocks:
            if headers is None:
                content = f'<p class="plain-note">{inline_html(payload)}</p>'
            else:
                content = html_table(headers, payload)
            block_html.append(
                f'<section class="subsection"><h3>{inline_html(name)}</h3>{content}</section>'
            )
        section_html.append(
            f'<section class="dataset-section"><h2>{inline_html(title)}</h2>{"".join(block_html)}</section>'
        )

    notes_html = "".join(f"<li>{inline_html(item)}</li>" for item in note_items)

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Input Dataset Metrics Report</title>
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
      max-width: 1400px;
      margin: 0 auto;
    }}
    .hero {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: var(--shadow);
      padding: 28px 32px;
      margin-bottom: 24px;
    }}
    h1, h2, h3 {{
      margin: 0;
      line-height: 1.25;
    }}
    h1 {{
      font-size: 2rem;
      margin-bottom: 8px;
    }}
    h2 {{
      font-size: 1.45rem;
      margin-bottom: 16px;
    }}
    h3 {{
      font-size: 1.1rem;
      margin-bottom: 12px;
      color: var(--accent);
    }}
    p {{
      margin: 8px 0 0;
      color: var(--muted);
    }}
    code {{
      background: #edf3ff;
      color: #204f9e;
      border-radius: 6px;
      padding: 0.12rem 0.38rem;
      font-size: 0.92em;
    }}
    .dataset-section, .notes {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
      padding: 24px;
      margin-bottom: 20px;
    }}
    .subsection + .subsection {{
      margin-top: 20px;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fbfdff;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 760px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 12px 14px;
      text-align: left;
      vertical-align: top;
      font-size: 0.95rem;
    }}
    th {{
      background: var(--head);
      color: #24406c;
      font-weight: 700;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    tbody tr:nth-child(even) td {{
      background: #f8fbff;
    }}
    tbody tr:hover td {{
      background: #eef5ff;
    }}
    .plain-note {{
      margin: 0;
      background: #f8fbff;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px 16px;
      color: var(--muted);
    }}
    ul {{
      margin: 8px 0 0;
      padding-left: 20px;
    }}
    li + li {{
      margin-top: 8px;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      padding: 8px 12px;
      border-radius: 999px;
      background: #eef5ff;
      color: #24406c;
      font-weight: 600;
      font-size: 0.92rem;
      border: 1px solid #d7e5ff;
    }}
    @media print {{
      body {{
        padding: 0;
        background: white;
      }}
      .hero, .dataset-section, .notes {{
        box-shadow: none;
        break-inside: avoid;
      }}
      th {{
        position: static;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <h1>Input Dataset Metrics Report</h1>
      <p>기준 날짜: 2026-04-15</p>
      <p>이 문서는 현재 저장된 GroupCV 결과를 입력셋 3종 기준으로 묶은 요약 문서입니다.</p>
      <div class="meta">
        <span class="chip">Variant A: exact slim</span>
        <span class="chip">Variant B: exact slim + SMILES</span>
        <span class="chip">Variant C: exact slim + strong context + SMILES</span>
      </div>
    </section>

    <section class="dataset-section">
      <h2>Input Variants</h2>
      {html_table(["Variant", "설명", "ML 입력", "DL 입력"], variant_rows)}
      <p><code>-</code> 표시는 해당 지표가 저장되지 않았거나 해당 조합이 실행되지 않았음을 뜻합니다.</p>
    </section>

    {"".join(section_html)}

    <section class="notes">
      <h2>Notes</h2>
      <ul>{notes_html}</ul>
    </section>
  </main>
</body>
</html>
"""


def build_report():
    parts = []
    parts.append("# Input Dataset Metrics Report")
    parts.append("")
    parts.append("기준 날짜: 2026-04-15")
    parts.append("")
    parts.append("이 문서는 현재 저장된 GroupCV 결과를 입력셋 3종 기준으로 묶은 요약 문서입니다.")
    parts.append("")
    parts.append("## Input Variants")
    parts.append("")
    parts.append(
        md_table(
            ["Variant", "설명", "ML 입력", "DL 입력"],
            [
                [
                    "A",
                    "`exact slim (numeric-only)`",
                    "`5529` numeric features",
                    "`5529` numeric features",
                ],
                [
                    "B",
                    "`exact slim + SMILES`",
                    "`5529 numeric + 64 SMILES SVD = 5593`",
                    "`5529 numeric + SMILES branch`",
                ],
                [
                    "C",
                    "`exact slim + strong context + SMILES`",
                    "`5529 numeric + 32 context one-hot + 64 SMILES SVD = 5625`",
                    "`5529 numeric + 5 categorical context + SMILES branch`",
                ],
            ],
        )
    )
    parts.append("")
    parts.append("`-` 표시는 해당 지표가 저장되지 않았거나 해당 조합이 실행되지 않았음을 뜻합니다.")
    parts.append("")

    # Dataset A
    parts.append("## A. exact slim (numeric-only)")
    parts.append("")
    parts.append("### DL")
    parts.append("")
    parts.append(
        md_table(
            ["Model", "Spearman", "RMSE", "MAE", "Pearson", "R²", "NDCG@20", "Time"],
            numeric_only_dl_rows(),
        )
    )
    parts.append("")
    parts.append("### Ensemble")
    parts.append("")
    parts.append(
        md_table(
            ["Ensemble", "구성", "Spearman", "RMSE", "MAE", "Pearson", "R²", "NDCG@20"],
            numeric_only_ensemble_rows(),
        )
    )
    parts.append("")
    parts.append("### ML")
    parts.append("")
    parts.append("현재 워크스페이스에는 `exact slim (numeric-only)` 조건의 ML full rerun 저장본이 없습니다.")
    parts.append("")

    # Dataset B
    parts.append("## B. exact slim + SMILES")
    parts.append("")
    parts.append("### ML")
    parts.append("")
    parts.append(
        md_table(
            ["Model", "Spearman", "RMSE", "MAE", "Pearson", "R²", "NDCG@20", "Time"],
            smiles_only_ml_rows(),
        )
    )
    parts.append("")
    parts.append("### DL")
    parts.append("")
    parts.append(
        md_table(
            ["Model", "Spearman", "RMSE", "MAE", "Pearson", "R²", "NDCG@20", "Time", "비고"],
            smiles_only_dl_rows(),
        )
    )
    parts.append("")
    parts.append("### Ensemble")
    parts.append("")
    parts.append(
        md_table(
            ["Ensemble", "구성", "Spearman", "RMSE", "MAE", "Pearson", "R²", "NDCG@20"],
            smiles_only_ensemble_rows(),
        )
    )
    parts.append("")

    # Dataset C
    parts.append("## C. exact slim + strong context + SMILES")
    parts.append("")
    parts.append("### ML")
    parts.append("")
    parts.append(
        md_table(
            ["Model", "Spearman", "RMSE", "MAE", "Pearson", "R²", "NDCG@20", "Time"],
            strong_context_ml_rows(),
        )
    )
    parts.append("")
    parts.append("### DL")
    parts.append("")
    parts.append(
        md_table(
            ["Model", "Spearman", "RMSE", "MAE", "Pearson", "R²", "NDCG@20", "Time", "비고"],
            strong_context_dl_rows(),
        )
    )
    parts.append("")
    parts.append("### Ensemble")
    parts.append("")
    parts.append(
        md_table(
            ["Ensemble", "구성", "Spearman", "RMSE", "MAE", "Pearson", "R²", "NDCG@20"],
            strong_context_ensemble_rows(),
        )
    )
    parts.append("")

    parts.append("## Notes")
    parts.append("")
    parts.append("- `exact slim + SMILES`의 `TabTransformer`는 사용자 지정 조기 종료 규칙으로 2개 fold만 실행되었습니다.")
    parts.append("- `exact slim + strong context + SMILES`의 `TabNet`, `FTTransformer`는 이전 실행 로그 요약값만 남아 있어 `Pearson`, `R²`, `Time`은 `-`로 표시했습니다.")
    parts.append("- `exact slim + strong context + SMILES`에서 현재 best ensemble은 `FlatMLP + ResidualMLP + CrossAttention` weighted 입니다.")
    parts.append("")

    return "\n".join(parts)


def main():
    OUT.write_text(build_report())
    HTML_OUT.write_text(build_html_report())
    print(OUT)
    print(HTML_OUT)


if __name__ == "__main__":
    main()
