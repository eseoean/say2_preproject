#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_ROOT = ROOT / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
(CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FIG_DIR = ROOT / "docs" / "report_figures"
MODEL_DIR = ROOT / "models"

BENCH_SPEARMAN = 0.713
BENCH_RMSE = 1.385

MODEL_NAME_MAP = {
    "1_LightGBM": "LightGBM",
    "2_LightGBM_DART": "LightGBM-DART",
    "3_XGBoost": "XGBoost",
    "4_CatBoost": "CatBoost",
    "5_RandomForest": "RandomForest",
    "6_ExtraTrees": "ExtraTrees",
    "7_Stacking_Ridge": "Stacking Ridge",
    "8_RSF": "RSF",
    "9_ResidualMLP": "ResidualMLP",
    "10_FlatMLP": "FlatMLP",
    "11_TabNet": "TabNet",
    "12_FT_Transformer": "FT-Transformer",
    "13_Cross_Attention": "Cross-Attention",
    "14_GraphSAGE": "GraphSAGE",
    "15_GAT": "GAT",
}

FAMILY_COLORS = {
    "ML": "#2F6BFF",
    "DL": "#00A88F",
    "Graph": "#F25F5C",
}

CATEGORY_COLORS = {
    "Approved": "#2E8B57",
    "Candidate": "#F4A259",
    "Caution": "#D1495B",
}


def setup_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )


def load_json(path: Path):
    return json.loads(path.read_text())


def load_model_results() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sources = [
        ("ML", MODEL_DIR / "ml_results" / "ml_results.json"),
        ("DL", MODEL_DIR / "dl_results" / "dl_results.json"),
        ("Graph", MODEL_DIR / "graph_results" / "graph_results.json"),
    ]
    for family, path in sources:
        for item in load_json(path):
            raw = item["model"]
            rows.append(
                {
                    "raw_model": raw,
                    "model": MODEL_NAME_MAP.get(raw, raw),
                    "family": family,
                    "spearman": float(item["spearman_mean"]),
                    "rmse": float(item["rmse_mean"]),
                    "pearson": float(item["pearson_mean"]),
                    "r2": float(item["r2_mean"]),
                    "gap": float(item["gap_spearman_mean"]),
                    "elapsed_min": float(item["elapsed_sec"]) / 60.0,
                    "pass": float(item["spearman_mean"]) >= BENCH_SPEARMAN
                    and float(item["rmse_mean"]) <= BENCH_RMSE,
                    "p_at_20": float(item.get("p_at_20_mean", np.nan)),
                    "auroc": float(item.get("auroc_mean", np.nan)),
                }
            )
    return pd.DataFrame(rows)


def load_ensemble() -> dict:
    return load_json(MODEL_DIR / "ensemble_results" / "ensemble_results.json")


def load_metabric() -> dict:
    return load_json(MODEL_DIR / "metabric_results" / "step6_metabric_results.json")


def load_admet_candidates() -> list[dict[str, str]]:
    with open(MODEL_DIR / "admet_results" / "final_drug_candidates.csv", newline="") as f:
        return list(csv.DictReader(f))


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
    )


def save(fig: plt.Figure, filename: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def plot_model_performance_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.axvline(BENCH_RMSE, color="#9CA3AF", linestyle="--", linewidth=1.2, label="RMSE threshold")
    ax.axhline(BENCH_SPEARMAN, color="#6B7280", linestyle="--", linewidth=1.2, label="Spearman threshold")
    ax.axvspan(0, BENCH_RMSE, ymin=(BENCH_SPEARMAN + 1) / 2, ymax=1, color="#ECFDF5", alpha=0.65)

    for family, sub in df.groupby("family"):
        ax.scatter(
            sub["rmse"],
            sub["spearman"],
            s=120 + sub["elapsed_min"] * 7,
            alpha=0.85,
            color=FAMILY_COLORS[family],
            edgecolor="white",
            linewidth=0.8,
            label=family,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                row["model"],
                (row["rmse"], row["spearman"]),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=9,
            )

    ax.set_title("Model Performance Map")
    ax.set_xlabel("RMSE (lower is better)")
    ax.set_ylabel("Spearman (higher is better)")
    ax.legend(loc="lower left", frameon=True)
    save(fig, "model_performance_scatter.png")


def plot_model_metric_leaderboard(df: pd.DataFrame) -> None:
    plot_df = df.sort_values("spearman", ascending=True).copy()
    colors = plot_df["family"].map(FAMILY_COLORS)

    fig, axes = plt.subplots(1, 3, figsize=(16, 10), sharey=True)

    axes[0].barh(plot_df["model"], plot_df["spearman"], color=colors)
    axes[0].axvline(BENCH_SPEARMAN, color="#6B7280", linestyle="--", linewidth=1.2)
    axes[0].set_title("Spearman")
    axes[0].set_xlabel("Higher is better")

    axes[1].barh(plot_df["model"], plot_df["rmse"], color=colors)
    axes[1].axvline(BENCH_RMSE, color="#6B7280", linestyle="--", linewidth=1.2)
    axes[1].set_title("RMSE")
    axes[1].set_xlabel("Lower is better")

    axes[2].barh(plot_df["model"], plot_df["elapsed_min"], color=colors)
    axes[2].set_title("Elapsed Time")
    axes[2].set_xlabel("Minutes")

    for ax in axes:
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle("Model Leaderboard", fontsize=15, fontweight="bold", y=1.01)
    save(fig, "model_metric_leaderboard.png")


def plot_ensemble_weights(ensemble: dict) -> None:
    weights = ensemble["weights"]
    metrics = ensemble["ensemble_metrics"]
    df = (
        pd.DataFrame(
            [{"model": k, "weight": float(v)} for k, v in weights.items()]
        )
        .sort_values("weight", ascending=True)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(df["model"], df["weight"], color="#4F46E5")
    ax.set_title("Final Ensemble Weights")
    ax.set_xlabel("Weight")

    note = (
        f"Spearman: {metrics['spearman_mean']:.4f}\n"
        f"RMSE: {metrics['rmse_mean']:.4f}\n"
        f"Pearson: {metrics['pearson']:.4f}\n"
        f"R²: {metrics['r2']:.4f}\n"
        f"Gap: {metrics['gap_spearman_mean']:.4f}"
    )
    ax.text(
        0.98,
        0.15,
        note,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#EEF2FF", "edgecolor": "#C7D2FE"},
        fontsize=10,
    )
    save(fig, "ensemble_weights.png")


def plot_candidate_screening_flow(admet_rows: list[dict[str, str]]) -> None:
    final_total = len(admet_rows)
    approved = sum(1 for r in admet_rows if r["category"] == "Approved")
    candidate = sum(1 for r in admet_rows if r["category"] == "Candidate")
    caution = sum(1 for r in admet_rows if r["category"] == "Caution")

    stages = ["Input drugs", "Top30 ranked", "Top15 validated", "Final selected"]
    counts = [295, 30, 15, final_total]
    colors = ["#DCEAFE", "#BFDBFE", "#93C5FD", "#60A5FA"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    axes[0].barh(stages[::-1], counts[::-1], color=colors[::-1])
    axes[0].set_title("Candidate Reduction Flow")
    axes[0].set_xlabel("Number of drugs")
    for idx, value in enumerate(counts[::-1]):
        axes[0].text(value + 4, idx, str(value), va="center", fontsize=10)

    cat_labels = ["Approved", "Candidate", "Caution"]
    cat_values = [approved, candidate, caution]
    cat_colors = [CATEGORY_COLORS[c] for c in cat_labels]
    axes[1].bar(cat_labels, cat_values, color=cat_colors)
    axes[1].set_title("Final Candidate Categories")
    axes[1].set_ylabel("Count")
    for idx, value in enumerate(cat_values):
        axes[1].text(idx, value + 0.15, str(value), ha="center", fontsize=10)

    save(fig, "candidate_screening_flow.png")


def plot_external_validation_summary(metabric: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    total = metabric["method_a"]["n_total"]
    method_a_labels = ["Target\nexpressed", "BRCA\npathway", "Survival\nsignificant"]
    method_a_values = [
        metabric["method_a"]["n_targets_expressed"] / total * 100.0,
        metabric["method_a"]["n_brca_pathway"] / total * 100.0,
        metabric["method_b"]["n_significant"] / total * 100.0,
    ]
    axes[0].bar(method_a_labels, method_a_values, color="#38BDF8")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("Validation Coverage")
    axes[0].set_ylabel("Percent")
    for idx, value in enumerate(method_a_values):
        axes[0].text(idx, value + 2, f"{value:.1f}%", ha="center", fontsize=9)

    pk = metabric["method_c"]["precision_at_k"]
    pk_labels = ["P@5", "P@10", "P@15", "P@20"]
    pk_values = [pk[label]["precision"] * 100.0 for label in pk_labels]
    axes[1].plot(pk_labels, pk_values, marker="o", linewidth=2.2, color="#2563EB")
    axes[1].fill_between(pk_labels, pk_values, color="#DBEAFE", alpha=0.55)
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Known BRCA Drug Precision")
    axes[1].set_ylabel("Percent")

    summary_labels = ["RSF C-index", "RSF AUROC", "GraphSAGE P@20"]
    summary_values = [
        metabric["method_b"]["rsf_c_index"] * 100.0,
        metabric["method_b"]["rsf_auroc"] * 100.0,
        metabric["method_c"]["graphsage_p20"] * 100.0,
    ]
    axes[2].bar(summary_labels, summary_values, color=["#818CF8", "#A78BFA", "#FB7185"])
    axes[2].set_ylim(0, 100)
    axes[2].set_title("Auxiliary Validation Metrics")
    axes[2].set_ylabel("Percent")
    for idx, value in enumerate(summary_values):
        axes[2].text(idx, value + 2, f"{value:.1f}", ha="center", fontsize=9)

    save(fig, "external_validation_summary.png")


def plot_final_candidates(admet_rows: list[dict[str, str]]) -> None:
    top10 = pd.DataFrame(admet_rows[:10]).copy()
    top10["combined_score"] = top10["combined_score"].astype(float)
    top10 = top10.sort_values("combined_score", ascending=True)
    colors = top10["category"].map(CATEGORY_COLORS)

    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.barh(top10["drug_name"], top10["combined_score"], color=colors)
    ax.set_title("Top 10 Final Drug Candidates")
    ax.set_xlabel("Combined score")

    for _, row in top10.iterrows():
        ax.text(
            row["combined_score"] + 0.15,
            row["drug_name"],
            row["category"],
            va="center",
            fontsize=9,
        )

    handles = [
        plt.Line2D([0], [0], color=color, lw=8, label=label)
        for label, color in CATEGORY_COLORS.items()
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True)
    save(fig, "final_candidates_top10.png")


def main() -> None:
    setup_style()
    df = load_model_results()
    ensemble = load_ensemble()
    metabric = load_metabric()
    admet_rows = load_admet_candidates()

    plot_model_performance_scatter(df)
    plot_model_metric_leaderboard(df)
    plot_ensemble_weights(ensemble)
    plot_candidate_screening_flow(admet_rows)
    plot_external_validation_summary(metabric)
    plot_final_candidates(admet_rows)

    print(f"Saved figures to {FIG_DIR}")
    for path in sorted(FIG_DIR.glob('*.png')):
        print(path.name)


if __name__ == "__main__":
    main()
