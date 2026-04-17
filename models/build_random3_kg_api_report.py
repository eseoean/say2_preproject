#!/usr/bin/env python3
"""
Build an HTML/CSV summary report from random3 KG/API collection outputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).parent
KG_DIR = BASE_DIR / "kg_api_results_random3_strong_context_smiles"
OUT_HTML = KG_DIR / "kg_api_summary.html"


def main() -> None:
    summary_csv = KG_DIR / "kg_api_summary.csv"
    results_json = KG_DIR / "kg_api_results.json"
    if not summary_csv.exists() or not results_json.exists():
        raise SystemExit("KG/API result files not found. Run collector first.")

    df = pd.read_csv(summary_csv)
    meta = json.loads(results_json.read_text())

    html = f"""<html>
<head>
  <meta charset="utf-8">
  <title>random3 KG/API Summary</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; padding: 24px; line-height: 1.5; }}
    .meta {{ background: #f6f8fa; border: 1px solid #d0d7de; padding: 12px 16px; border-radius: 8px; margin-bottom: 20px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 8px 10px; text-align: left; }}
    th {{ background: #f6f8fa; }}
    tr:nth-child(even) {{ background: #fcfcfd; }}
  </style>
</head>
<body>
  <h1>random3 Ensemble KG/API Summary</h1>
  <div class="meta">
    <div><strong>API base</strong>: {meta.get("api_base")}</div>
    <div><strong>Mode</strong>: {meta.get("mode")}</div>
    <div><strong>Drug count</strong>: {meta.get("n_drugs")}</div>
  </div>
  {df.to_html(index=False)}
</body>
</html>"""
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Saved report: {OUT_HTML}")


if __name__ == "__main__":
    main()
