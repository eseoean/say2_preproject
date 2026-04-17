#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
WORK_ROOT = SCRIPT_PATH.parents[1]
RESULT_ROOT = WORK_ROOT / "results"
OUT_MD = RESULT_ROOT / "random3_metrics_report_20260415.md"
OUT_HTML = RESULT_ROOT / "random3_metrics_report_20260415.html"


INPUTS = [
    ("exact slim (numeric-only)", "numeric"),
    ("exact slim + SMILES", "smiles"),
    ("exact slim + strong context + SMILES", "strong_context_smiles"),
]


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def load_models(path: Path) -> list[dict]:
    if not path.exists():
        return []
    obj = json.loads(path.read_text())
    models = obj.get("models", [])
    if not isinstance(models, list):
        return []
    return sorted(models, key=lambda m: m.get("overall_metrics", {}).get("spearman", -1), reverse=True)


def table_rows(models: list[dict]) -> str:
    rows = []
    for m in models:
        om = m.get("overall_metrics", {})
        time_min = m.get("elapsed_sec")
        rows.append(
            f"| {m.get('model')} | {fmt(om.get('spearman'))} | {fmt(om.get('rmse'))} | {fmt(om.get('mae'))} | "
            f"{fmt(om.get('pearson'))} | {fmt(om.get('r2'))} | {fmt(om.get('ndcg@20'))} | "
            f"{fmt(time_min / 60 if time_min is not None else None)}m |"
        )
    return "\n".join(rows) if rows else "| - | - | - | - | - | - | - | - |"


def build_md() -> str:
    parts = ["# Random Sample 3-Fold Metrics Report", "", "현재 저장된 결과 기준으로 정리했습니다.", ""]
    for title, stem in INPUTS:
        ml_path = RESULT_ROOT / f"exact_repo_random3_{stem}_ml_v1.json"
        dl_path = RESULT_ROOT / f"exact_repo_random3_{stem}_dl_v1.json"
        ml_models = load_models(ml_path)
        dl_models = load_models(dl_path)

        parts.extend(
            [
                f"## {title}",
                "",
                "### ML",
                "",
                "| Model | Spearman | RMSE | MAE | Pearson | R² | NDCG@20 | Time |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
                table_rows(ml_models),
                "",
                "### DL",
                "",
                "| Model | Spearman | RMSE | MAE | Pearson | R² | NDCG@20 | Time |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
                table_rows(dl_models),
                "",
            ]
        )

        if stem in {"smiles", "strong_context_smiles"}:
            parts.extend(
                [
                    f"비고: `{stem}` DL 런은 `TabTransformer` 구간에서 중단하여 현재 저장된 모델까지만 반영했습니다.",
                    "",
                ]
            )
    return "\n".join(parts)


def build_html(md: str) -> str:
    body = []
    for line in md.splitlines():
        if line.startswith("# "):
            body.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            body.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("| "):
            body.append(line)
        elif line.strip() == "":
            body.append("")
        else:
            body.append(f"<p>{line}</p>")
    html = []
    i = 0
    while i < len(body):
        line = body[i]
        if line.startswith("| "):
            table_lines = []
            while i < len(body) and body[i].startswith("| "):
                table_lines.append(body[i])
                i += 1
            headers = [x.strip() for x in table_lines[0].strip("|").split("|")]
            rows = [[x.strip() for x in r.strip("|").split("|")] for r in table_lines[2:]]
            html.append("<table>")
            html.append("<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>")
            html.append("<tbody>")
            for row in rows:
                html.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
            html.append("</tbody></table>")
            continue
        html.append(line)
        i += 1
    return """<!doctype html><html><head><meta charset="utf-8"><title>Random3 Metrics Report</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1200px;margin:40px auto;padding:0 20px;line-height:1.6;color:#1f2937}
h1,h2,h3{color:#111827} table{border-collapse:collapse;width:100%;margin:16px 0 28px 0;font-size:14px}
th,td{border:1px solid #d1d5db;padding:8px 10px;text-align:left} thead{background:#f3f4f6}
</style></head><body>""" + "\n".join(html) + "</body></html>"


def main() -> None:
    md = build_md()
    OUT_MD.write_text(md)
    OUT_HTML.write_text(build_html(md))
    print(OUT_MD)
    print(OUT_HTML)


if __name__ == "__main__":
    main()
