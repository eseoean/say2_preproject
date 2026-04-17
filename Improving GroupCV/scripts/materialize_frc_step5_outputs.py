#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/Users/skku_aws2_18/pre_project/say2_preproject")
WORK_ROOT = ROOT / "Improving GroupCV"
RESULTS = WORK_ROOT / "results"
MODELS_DIR = ROOT / "models"
OUT_DIR = MODELS_DIR / "ensemble_results_frc_strong_context_smiles"
OUT_DIR.mkdir(exist_ok=True)


def dedupe_drug_candidates(drug_summary: pd.DataFrame) -> pd.DataFrame:
    out = drug_summary.sort_values(["mean_pred_ic50", "drug_id"], ascending=[True, True]).copy()
    before = len(out)
    out = out.drop_duplicates(subset=["drug_name"], keep="first").reset_index(drop=True)
    removed = before - len(out)
    if removed:
        print(f"Removed {removed} duplicate candidate rows by drug name")
    out["rank"] = range(1, len(out) + 1)
    return out


def main() -> None:
    frc_path = RESULTS / "exact_repo_slim_strong_context_smiles_frc_ensemble_oof_v2.json"
    obj = json.loads(frc_path.read_text())

    keys = pd.read_parquet(obj["keys_path"])
    y = np.load(obj["input_y_path"]).astype(np.float32)

    oof = np.zeros_like(y, dtype=np.float64)
    for name, w in obj["weights"].items():
        pred = np.load(obj["base_model_metrics"][name]["oof_path"]).astype(np.float64)
        oof += float(w) * pred

    features = pd.read_parquet(obj["input_features_path"], columns=["sample_id", "canonical_drug_id", "drug__drug_name_norm"])
    merged = keys.merge(
        features.drop_duplicates(subset=["sample_id", "canonical_drug_id"]),
        on=["sample_id", "canonical_drug_id"],
        how="left",
    )

    df_pred = pd.DataFrame(
        {
            "sample_id": keys["sample_id"].astype(str).values,
            "drug_id": keys["canonical_drug_id"].astype(str).values,
            "y_true": y,
            "y_pred_ensemble": oof.astype(np.float32),
            "drug_name": merged["drug__drug_name_norm"].astype(str).values,
        }
    )

    drug_summary = (
        df_pred.groupby(["drug_id", "drug_name"])
        .agg(
            mean_pred_ic50=("y_pred_ensemble", "mean"),
            mean_true_ic50=("y_true", "mean"),
            std_pred_ic50=("y_pred_ensemble", "std"),
            n_samples=("y_pred_ensemble", "count"),
            sensitivity_rate=("y_true", lambda x: (x < np.median(y)).mean()),
        )
        .reset_index()
    )

    drug_summary = dedupe_drug_candidates(drug_summary)
    top30 = drug_summary.head(30).copy()
    top30["category"] = top30["sensitivity_rate"].apply(lambda x: "Validated" if x > 0.5 else "Recommended")
    top30["score"] = (
        -top30["mean_pred_ic50"].rank()
        + top30["sensitivity_rate"].rank() * 2
        + (top30["n_samples"] >= 5).astype(int) * 5
    )
    top15 = (
        top30.sort_values(["score", "mean_pred_ic50"], ascending=[False, True])
        .drop_duplicates(subset=["drug_name"], keep="first")
        .head(15)
        .sort_values("mean_pred_ic50", ascending=True)
        .reset_index(drop=True)
    )
    top15["final_rank"] = range(1, len(top15) + 1)

    results = {
        "ensemble_method": "spearman_weighted_average",
        "n_models": 3,
        "selected_models": list(obj["weights"].keys()),
        "weights": {k: float(v) for k, v in obj["weights"].items()},
        "ensemble_metrics": obj["weighted_overall_metrics"],
        "individual_models": {
            name: {
                "spearman": float(obj["base_model_metrics"][name]["spearman_mean"]),
                "rmse": float(obj["base_model_metrics"][name]["rmse_mean"]),
                "weight": float(obj["weights"][name]),
            }
            for name in obj["weights"]
        },
        "top30_drugs": top30[
            ["rank", "drug_id", "drug_name", "mean_pred_ic50", "mean_true_ic50", "sensitivity_rate", "n_samples", "category"]
        ].to_dict(orient="records"),
        "top15_drugs": top15[
            ["final_rank", "drug_id", "drug_name", "mean_pred_ic50", "mean_true_ic50", "sensitivity_rate", "n_samples", "category"]
        ].to_dict(orient="records"),
    }

    (OUT_DIR / "top30_drugs.csv").write_text(top30.to_csv(index=False))
    (OUT_DIR / "top15_drugs.csv").write_text(top15.to_csv(index=False))
    (OUT_DIR / "ensemble_results.json").write_text(json.dumps(results, indent=2))
    print(OUT_DIR / "ensemble_results.json")
    print(OUT_DIR / "top30_drugs.csv")
    print(OUT_DIR / "top15_drugs.csv")


if __name__ == "__main__":
    main()
