#!/usr/bin/env python3
"""
Step 7+ KG/API collection for the FRC ensemble
(FlatMLP + ResidualMLP + CrossAttention).

This follows the teammate repo flow, but targets the local
FRC post-ADMET outputs.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).parent
DEFAULT_INPUT = BASE_DIR / "admet_results_frc_strong_context_smiles" / "final_drug_candidates.csv"
DEFAULT_OUTPUT = BASE_DIR / "kg_api_results_frc_strong_context_smiles"
DEFAULT_API = "http://localhost:8000"


def call_api(api_base: str, endpoint: str, params: dict | None = None):
    try:
        if params:
            query = urllib.parse.urlencode(params)
            url = f"{api_base}{endpoint}?{query}"
        else:
            url = f"{api_base}{endpoint}"

        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"_error": f"HTTP {e.code}: {e.reason}", "_ok": False}
    except urllib.error.URLError as e:
        return {"_error": f"URL error: {e.reason}", "_ok": False}
    except Exception as e:  # pragma: no cover - defensive
        return {"_error": str(e), "_ok": False}


def select_drugs(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "all":
        return df.copy()
    if mode == "non_approved":
        return df[df["category"].isin(["Candidate", "Caution"])].copy()
    if mode == "approved":
        return df[df["category"].eq("Approved")].copy()
    raise ValueError(f"Unknown mode: {mode}")


def count_payload_items(payload) -> int:
    if not isinstance(payload, dict):
        return 0
    data = payload.get("data")
    if isinstance(data, list):
        return len(data)
    if data:
        return 1
    return 0


def collect_for_drug(api_base: str, row: pd.Series) -> dict:
    drug_name = str(row["drug_name"])
    result = {
        "drug_id": int(row["drug_id"]),
        "drug_name": drug_name,
        "category": row["category"],
        "combined_score": float(row["combined_score"]),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_collected": {},
        "errors": [],
    }

    specs = [
        ("basic_info", f"/api/drug/{drug_name}", None),
        ("side_effects", f"/api/drug/{drug_name}/side_effects", None),
        ("trials", f"/api/drug/{drug_name}/trials", None),
        ("targets", f"/api/drug/{drug_name}/targets", None),
        ("pathways", f"/api/drug/{drug_name}/pathways", None),
        ("pubmed_general", "/api/pubmed", {"query": drug_name, "max_results": 50}),
        ("pubmed_breast_cancer", "/api/pubmed", {"query": f"{drug_name} breast cancer", "max_results": 50}),
    ]

    for key, endpoint, params in specs:
        payload = call_api(api_base, endpoint, params)
        if payload.get("_ok", True) is False:
            result["errors"].append(f"{key}: {payload['_error']}")
        else:
            result["data_collected"][key] = payload

    return result


def build_summary_rows(results: list[dict]) -> list[dict]:
    rows = []
    for res in results:
        data = res["data_collected"]
        rows.append(
            {
                "drug_name": res["drug_name"],
                "category": res["category"],
                "combined_score": res["combined_score"],
                "api_success_count": len(data),
                "api_error_count": len(res["errors"]),
                "basic_info": "Yes" if "basic_info" in data else "No",
                "faers_count": count_payload_items(data.get("side_effects", {})),
                "trial_count": count_payload_items(data.get("trials", {})),
                "target_count": count_payload_items(data.get("targets", {})),
                "pathway_count": count_payload_items(data.get("pathways", {})),
                "pubmed_general_count": count_payload_items(data.get("pubmed_general", {})),
                "pubmed_breast_cancer_count": count_payload_items(data.get("pubmed_breast_cancer", {})),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--api-base", default=DEFAULT_API)
    parser.add_argument(
        "--mode",
        choices=["all", "non_approved", "approved"],
        default="all",
        help="Which post-ADMET subset to collect.",
    )
    args = parser.parse_args()

    args.output.mkdir(exist_ok=True)

    df = pd.read_csv(args.input)
    selected = select_drugs(df, args.mode).sort_values(["final_rank", "combined_score"], ascending=[True, False])

    # Quick health check before we fan out.
    health = call_api(args.api_base, "/health")
    if health.get("_ok", True) is False:
        raise SystemExit(
            "KG/API server is not reachable. "
            f"Checked {args.api_base}. Error: {health['_error']}"
        )

    all_results: list[dict] = []
    for _, row in selected.iterrows():
        res = collect_for_drug(args.api_base, row)
        safe_name = str(row["drug_name"]).replace(" ", "_").replace("/", "_")
        (args.output / f"{safe_name}_kg_data.json").write_text(
            json.dumps(res, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        all_results.append(res)

    summary_rows = build_summary_rows(all_results)
    summary_df = pd.DataFrame(summary_rows).sort_values(["combined_score"], ascending=[False])
    summary_df.to_csv(args.output / "kg_api_summary.csv", index=False)

    final = {
        "api_base": args.api_base,
        "mode": args.mode,
        "n_drugs": int(len(selected)),
        "drugs": selected["drug_name"].tolist(),
        "results": all_results,
    }
    (args.output / "kg_api_results.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved KG/API collection to: {args.output}")


if __name__ == "__main__":
    main()
