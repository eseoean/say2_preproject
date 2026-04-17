#!/usr/bin/env python3
"""
Build a small report for the FlatMLP + LightGBM_DART + ExtraTrees mixed ensemble.
"""

from __future__ import annotations

import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
WORK_ROOT = SCRIPT_PATH.parents[1]
RESULT_ROOT = WORK_ROOT / "results"
INPUT_JSON = RESULT_ROOT / "exact_repo_slim_strong_context_smiles_fle_ensemble_v1.json"
OUT_MD = RESULT_ROOT / "exact_repo_slim_strong_context_smiles_fle_ensemble_report_20260415.md"
OUT_HTML = RESULT_ROOT / "exact_repo_slim_strong_context_smiles_fle_ensemble_report_20260415.html"


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, (int, float)):
        return f"{v:.4f}"
    return str(v)


def build_md(obj: dict) -> str:
    rows = []
    for name, meta in obj["base_model_metrics"].items():
        om = meta["overall_metrics"]
        rows.append(
            f"| {name} | {fmt(om['spearman'])} | {fmt(om['rmse'])} | {fmt(om['mae'])} | "
            f"{fmt(om['pearson'])} | {fmt(om['r2'])} | {fmt(om['ndcg@20'])} | {fmt(meta['elapsed_sec']/60 if meta.get('elapsed_sec') is not None else None)}m |"
        )

    pair_rows = []
    diversity = obj.get("diversity")
    for row in diversity.get("pairwise", []) if diversity else []:
        pair_rows.append(
            f"| {row['pair']} | {fmt(row['prediction_pearson'])} | {fmt(row['prediction_spearman'])} | "
            f"{fmt(row['residual_pearson'])} | {fmt(row['residual_spearman'])} | {fmt(row['mean_abs_prediction_gap'])} |"
        )

    if not pair_rows:
        pair_rows.append("| - | - | - | - | - | - |")

    equal_metrics = obj.get("equal_overall_metrics", {})
    weighted_metrics = obj.get("weighted_overall_metrics", {})
    weights = obj.get("weights", {})
    diversity_summary = diversity.get("summary", {}) if diversity else {}
    gain = obj.get("ensemble_gain_vs_best_base", {})

    md = f"""# FLE Mixed Ensemble Report

입력셋: `exact slim + strong context + SMILES`  
모델 조합: `FlatMLP + LightGBM_DART + ExtraTrees`

## Base Models

| Model | Spearman | RMSE | MAE | Pearson | R² | NDCG@20 | Time |
| --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

## Ensemble

| Type | Spearman | RMSE | MAE | Pearson | R² | NDCG@20 |
| --- | --- | --- | --- | --- | --- | --- |
| Equal | {fmt(equal_metrics.get('spearman'))} | {fmt(equal_metrics.get('rmse'))} | {fmt(equal_metrics.get('mae'))} | {fmt(equal_metrics.get('pearson'))} | {fmt(equal_metrics.get('r2'))} | {fmt(equal_metrics.get('ndcg@20'))} |
| Weighted | {fmt(weighted_metrics.get('spearman'))} | {fmt(weighted_metrics.get('rmse'))} | {fmt(weighted_metrics.get('mae'))} | {fmt(weighted_metrics.get('pearson'))} | {fmt(weighted_metrics.get('r2'))} | {fmt(weighted_metrics.get('ndcg@20'))} |

## Weights

| Model | Weight |
| --- | --- |
| FlatMLP | {fmt(weights.get('FlatMLP'))} |
| LightGBM_DART | {fmt(weights.get('LightGBM_DART'))} |
| ExtraTrees | {fmt(weights.get('ExtraTrees'))} |

## Diversity

| Pair | Pred Pearson | Pred Spearman | Resid Pearson | Resid Spearman | Mean Abs Gap |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(pair_rows)}

## Diversity Summary

| Metric | Value |
| --- | --- |
| Avg prediction Pearson | {fmt(diversity_summary.get('avg_prediction_pearson'))} |
| Avg prediction Spearman | {fmt(diversity_summary.get('avg_prediction_spearman'))} |
| Avg residual Pearson | {fmt(diversity_summary.get('avg_residual_pearson'))} |
| Avg residual Spearman | {fmt(diversity_summary.get('avg_residual_spearman'))} |
| Avg mean abs prediction gap | {fmt(diversity_summary.get('avg_mean_abs_prediction_gap'))} |

## Gain Vs Best Base

| Metric | Delta |
| --- | --- |
| Weighted Spearman gain | {fmt(gain.get('weighted_spearman_gain'))} |
| Weighted RMSE gain | {fmt(gain.get('weighted_rmse_gain'))} |
"""
    return md


def build_html(md: str) -> str:
    body = []
    for line in md.splitlines():
        if line.startswith("# "):
            body.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("| "):
            body.append(line)
        elif line.strip() == "":
            body.append("")
        else:
            body.append(f"<p>{line}</p>")
    # crude markdown table support
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
    return """<!doctype html><html><head><meta charset="utf-8"><title>FLE Mixed Ensemble Report</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;line-height:1.6;color:#1f2937}
h1,h2{color:#111827} table{border-collapse:collapse;width:100%;margin:16px 0 28px 0;font-size:14px}
th,td{border:1px solid #d1d5db;padding:8px 10px;text-align:left} thead{background:#f3f4f6}
p code{background:#f3f4f6;padding:2px 6px;border-radius:6px}
</style></head><body>""" + "\n".join(html) + "</body></html>"


def main() -> None:
    obj = json.loads(INPUT_JSON.read_text())
    md = build_md(obj)
    OUT_MD.write_text(md)
    OUT_HTML.write_text(build_html(md))
    print(OUT_MD)
    print(OUT_HTML)


if __name__ == "__main__":
    main()
