#!/usr/bin/env python3
"""Summarize delta between two progressive GroupCV result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HIGHER_IS_BETTER = {"spearman_mean", "pearson_mean", "r2_mean", "ndcg@20_mean"}
LOWER_IS_BETTER = {"rmse_mean", "mae_mean", "gap_spearman_mean"}


def load_results(path: Path):
    data = json.loads(path.read_text())
    return {m["model"]: m for m in data.get("models", [])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--test", required=True)
    args = parser.parse_args()

    baseline = load_results(Path(args.baseline))
    test = load_results(Path(args.test))
    models = sorted(set(baseline) | set(test))

    metrics = [
        "spearman_mean",
        "rmse_mean",
        "mae_mean",
        "pearson_mean",
        "r2_mean",
        "ndcg@20_mean",
        "gap_spearman_mean",
    ]

    print(
        "Model\t"
        + "\t".join(
            [
                "Sp(base)",
                "Sp(test)",
                "dSp",
                "RMSE(base)",
                "RMSE(test)",
                "dRMSE",
                "MAE(base)",
                "MAE(test)",
                "dMAE",
                "NDCG20(base)",
                "NDCG20(test)",
                "dNDCG20",
            ]
        )
    )

    for model in models:
        b = baseline.get(model)
        t = test.get(model)
        if not b or not t:
            continue
        print(
            f"{model}\t"
            f"{b['spearman_mean']:.4f}\t{t['spearman_mean']:.4f}\t{t['spearman_mean']-b['spearman_mean']:+.4f}\t"
            f"{b['rmse_mean']:.4f}\t{t['rmse_mean']:.4f}\t{t['rmse_mean']-b['rmse_mean']:+.4f}\t"
            f"{b['mae_mean']:.4f}\t{t['mae_mean']:.4f}\t{t['mae_mean']-b['mae_mean']:+.4f}\t"
            f"{b['ndcg@20_mean']:.4f}\t{t['ndcg@20_mean']:.4f}\t{t['ndcg@20_mean']-b['ndcg@20_mean']:+.4f}"
        )


if __name__ == "__main__":
    main()
