from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score, r2_score
from sklearn.model_selection import KFold


ROOT = Path("/Users/skku_aws2_18/pre_project/say2_preproject")
RESULTS = ROOT / "Improving GroupCV" / "results"
OUT = RESULTS / "exact_repo_random3_strong_context_smiles_ensemble_v1.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    spearman = spearmanr(y_true, y_pred).statistic
    pearson = pearsonr(y_true, y_pred).statistic
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    relevance = -y_true.astype(float)
    relevance = relevance - relevance.min() + 1e-8
    score = -y_pred.astype(float)
    ndcg20 = float(ndcg_score(relevance.reshape(1, -1), score.reshape(1, -1), k=min(20, len(y_true))))

    return {
        "spearman": float(spearman),
        "rmse": rmse,
        "mae": mae,
        "pearson": float(pearson),
        "r2": r2,
        "ndcg@20": ndcg20,
    }


def avg_pairwise_diversity(pred_map: dict[str, np.ndarray], y_true: np.ndarray) -> dict:
    names = list(pred_map)
    pairwise = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            a = pred_map[left]
            b = pred_map[right]
            ra = y_true - a
            rb = y_true - b
            pairwise.append(
                {
                    "pair": f"{left} vs {right}",
                    "prediction_pearson": float(pearsonr(a, b).statistic),
                    "prediction_spearman": float(spearmanr(a, b).statistic),
                    "residual_pearson": float(pearsonr(ra, rb).statistic),
                    "residual_spearman": float(spearmanr(ra, rb).statistic),
                    "mean_abs_prediction_gap": float(np.mean(np.abs(a - b))),
                }
            )

    summary = {
        "avg_prediction_pearson": float(np.mean([x["prediction_pearson"] for x in pairwise])),
        "avg_prediction_spearman": float(np.mean([x["prediction_spearman"] for x in pairwise])),
        "avg_residual_pearson": float(np.mean([x["residual_pearson"] for x in pairwise])),
        "avg_residual_spearman": float(np.mean([x["residual_spearman"] for x in pairwise])),
        "avg_mean_abs_prediction_gap": float(np.mean([x["mean_abs_prediction_gap"] for x in pairwise])),
    }
    return {"pairwise": pairwise, "summary": summary}


def main() -> None:
    ml_payload = load_json(RESULTS / "exact_repo_random3_strong_context_smiles_ml_v1.json")
    dl_payload = load_json(RESULTS / "exact_repo_random3_strong_context_smiles_dl_v1.json")

    selected_ml = ["CatBoost", "XGBoost", "LightGBM"]
    selected_dl = ["TabNet", "ResidualMLP", "WideDeep"]
    selected = selected_ml + selected_dl

    model_meta = {}
    for payload, family in [(ml_payload, "ML"), (dl_payload, "DL")]:
        for model in payload["models"]:
            if model["model"] in selected:
                model_meta[model["model"]] = {
                    "family": family,
                    "overall_metrics": model["overall_metrics"],
                    "elapsed_sec": model.get("elapsed_sec"),
                    "oof_path": model["oof_path"],
                }

    y = np.load(ml_payload["input_y_path"])
    pred_map = {name: np.load(meta["oof_path"]) for name, meta in model_meta.items()}

    weight_basis = {
        name: float(meta["overall_metrics"]["spearman"])
        for name, meta in model_meta.items()
    }
    total_weight = sum(weight_basis.values())
    weights = {name: weight_basis[name] / total_weight for name in selected}

    equal_pred = np.mean([pred_map[name] for name in selected], axis=0)
    weighted_pred = np.sum([pred_map[name] * weights[name] for name in selected], axis=0)

    fold_metrics_equal = []
    fold_metrics_weighted = []
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for fold, (_, valid_idx) in enumerate(kf.split(np.arange(len(y)))):
        y_fold = y[valid_idx]
        equal_fold = equal_pred[valid_idx]
        weighted_fold = weighted_pred[valid_idx]
        eq = compute_metrics(y_fold, equal_fold)
        wt = compute_metrics(y_fold, weighted_fold)
        eq["fold"] = fold
        wt["fold"] = fold
        fold_metrics_equal.append(eq)
        fold_metrics_weighted.append(wt)

    output = {
        "split_mode": "random_sample_kfold",
        "seed": 42,
        "input_mode": "strong_context_smiles",
        "folds": 3,
        "rows": int(len(y)),
        "selected_models": selected,
        "selected_model_meta": model_meta,
        "weights": weights,
        "equal_overall_metrics": compute_metrics(y, equal_pred),
        "weighted_overall_metrics": compute_metrics(y, weighted_pred),
        "equal_fold_metrics": fold_metrics_equal,
        "weighted_fold_metrics": fold_metrics_weighted,
        "diversity": avg_pairwise_diversity({name: pred_map[name] for name in selected}, y),
        "total_elapsed_sec": float(sum(meta.get("elapsed_sec") or 0.0 for meta in model_meta.values())),
    }

    OUT.write_text(json.dumps(output, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
