#!/usr/bin/env python3
"""
Step 4-A: ML Model Training (8 models) with 5-fold CV
Models: LightGBM, LightGBM-DART, XGBoost, CatBoost, RF, ExtraTrees, Stacking(Ridge), RSF

Input:  FE output from S3 (features.parquet + pair_features_newfe_v2.parquet + labels.parquet)
Output: Per-model per-fold metrics + summary table
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import json
import time
import os
import sys
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from scipy.stats import spearmanr, pearsonr

# ── Config ──
S3_BASE = os.getenv("S3_BASE", "s3://say2-4team/20260409_eseo")
FE_RUN_ID = os.getenv("FE_RUN_ID", "20260409_newfe_v8_eseo")
FEATURES_URI = f"{S3_BASE}/fe_output/{FE_RUN_ID}/features/features.parquet"
PAIR_FEATURES_URI = f"{S3_BASE}/fe_output/{FE_RUN_ID}/pair_features/pair_features_newfe_v2.parquet"
LABELS_URI = f"{S3_BASE}/fe_output/{FE_RUN_ID}/features/labels.parquet"

N_FOLDS = 5
SEED = 42
OUTPUT_DIR = Path(__file__).parent / "ml_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Benchmarks (Team 4 reference) ──
BENCH_SPEARMAN = 0.713
BENCH_RMSE = 1.385

# Use conservative thread defaults to avoid native-library deadlocks on local reruns.
CPU_THREADS = int(os.getenv("MODEL_CPU_THREADS", "4"))
CATBOOST_THREAD_COUNT = int(os.getenv("CATBOOST_THREAD_COUNT", "1"))

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, str(CPU_THREADS))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def load_data():
    """Load and merge feature matrices + labels."""
    print("=" * 70)
    print("Loading data from S3...")
    t0 = time.time()

    features = pd.read_parquet(FEATURES_URI)
    pair_features = pd.read_parquet(PAIR_FEATURES_URI)
    labels = pd.read_parquet(LABELS_URI)

    # Merge features + pair_features on shared keys
    merged = features.merge(pair_features, on=["sample_id", "canonical_drug_id"], how="inner")

    # Align labels
    labels = labels.set_index(["sample_id", "canonical_drug_id"])
    merged = merged.set_index(["sample_id", "canonical_drug_id"])
    labels = labels.loc[merged.index]

    # Separate numeric features only
    X = merged.select_dtypes(include=[np.number]).copy()
    y_reg = labels["label_regression"].values.astype(np.float64)
    y_bin = labels["label_binary"].values.astype(np.int32)

    # Fill any remaining NaN
    if X.isnull().any().any():
        X = X.fillna(0.0)

    feature_names = list(X.columns)
    X_np = X.values.astype(np.float32)

    dt = time.time() - t0
    print(f"  Loaded: {X_np.shape[0]} samples x {X_np.shape[1]} features ({dt:.1f}s)")
    print(f"  y_reg: mean={y_reg.mean():.3f}, std={y_reg.std():.3f}")
    print(f"  y_bin: {y_bin.sum()}/{len(y_bin)} positive ({y_bin.mean()*100:.1f}%)")
    print("=" * 70)
    return X_np, y_reg, y_bin, feature_names


def compute_metrics(y_true, y_pred, y_train_true=None, y_train_pred=None):
    """Compute regression metrics."""
    sp, _ = spearmanr(y_true, y_pred)
    pe, _ = pearsonr(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    m = {"spearman": sp, "pearson": pe, "rmse": rmse, "r2": r2}
    if y_train_true is not None and y_train_pred is not None:
        tr_sp, _ = spearmanr(y_train_true, y_train_pred)
        tr_rmse = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
        m["train_spearman"] = tr_sp
        m["train_rmse"] = tr_rmse
        m["gap_spearman"] = tr_sp - sp
        m["gap_rmse"] = rmse - tr_rmse
    return m


def _json_default(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_results_snapshot(results, filename):
    path = OUTPUT_DIR / filename
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    return path


def run_cv(model_name, model_fn, X, y_reg, y_bin, feature_names):
    """Run 5-fold CV for a model, return per-fold metrics."""
    print(f"\n{'─'*60}")
    print(f"  [{model_name}] Training with 5-fold CV...")
    print(f"{'─'*60}")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        t0 = time.time()
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y_reg[train_idx], y_reg[val_idx]

        # Train model
        model, y_pred_val, y_pred_tr = model_fn(
            X_tr, y_tr, X_val, y_val, fold_idx, feature_names
        )

        # Compute metrics
        m = compute_metrics(y_val, y_pred_val, y_tr, y_pred_tr)
        m["fold"] = fold_idx
        fold_metrics.append(m)

        dt = time.time() - t0
        print(f"  Fold {fold_idx}: Spearman={m['spearman']:.4f}  RMSE={m['rmse']:.4f}  "
              f"R2={m['r2']:.4f}  Gap(Sp)={m['gap_spearman']:.4f}  ({dt:.1f}s)")

    # Summary
    df = pd.DataFrame(fold_metrics)
    summary = {
        "model": model_name,
        "spearman_mean": df["spearman"].mean(),
        "spearman_std": df["spearman"].std(),
        "rmse_mean": df["rmse"].mean(),
        "rmse_std": df["rmse"].std(),
        "pearson_mean": df["pearson"].mean(),
        "r2_mean": df["r2"].mean(),
        "r2_std": df["r2"].std(),
        "train_spearman_mean": df["train_spearman"].mean(),
        "gap_spearman_mean": df["gap_spearman"].mean(),
        "gap_rmse_mean": df["gap_rmse"].mean(),
        "folds": fold_metrics,
    }

    beat_sp = "PASS" if summary["spearman_mean"] >= BENCH_SPEARMAN else "FAIL"
    beat_rm = "PASS" if summary["rmse_mean"] <= BENCH_RMSE else "FAIL"
    print(f"\n  >>> {model_name} SUMMARY:")
    print(f"      Spearman: {summary['spearman_mean']:.4f} +/- {summary['spearman_std']:.4f}  [{beat_sp} vs {BENCH_SPEARMAN}]")
    print(f"      RMSE:     {summary['rmse_mean']:.4f} +/- {summary['rmse_std']:.4f}  [{beat_rm} vs {BENCH_RMSE}]")
    print(f"      Pearson:  {summary['pearson_mean']:.4f}")
    print(f"      R2:       {summary['r2_mean']:.4f} +/- {summary['r2_std']:.4f}")
    print(f"      Train Sp: {summary['train_spearman_mean']:.4f}  Gap: {summary['gap_spearman_mean']:.4f}")

    return summary


# ── Model Definitions ──

def lgbm_model(X_tr, y_tr, X_val, y_val, fold_idx, feat_names):
    import lightgbm as lgb
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_names)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=feat_names, reference=dtrain)
    params = {
        "objective": "regression", "metric": "rmse",
        "boosting_type": "gbdt", "learning_rate": 0.05,
        "num_leaves": 63, "max_depth": 7, "min_child_samples": 20,
        "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 5,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "verbose": -1, "seed": SEED + fold_idx, "n_jobs": CPU_THREADS,
    }
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    model = lgb.train(params, dtrain, num_boost_round=1500, valid_sets=[dval], callbacks=callbacks)
    return model, model.predict(X_val), model.predict(X_tr)


def lgbm_dart_model(X_tr, y_tr, X_val, y_val, fold_idx, feat_names):
    import lightgbm as lgb
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_names)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=feat_names, reference=dtrain)
    params = {
        "objective": "regression", "metric": "rmse",
        "boosting_type": "dart", "learning_rate": 0.05,
        "num_leaves": 63, "max_depth": 7, "min_child_samples": 20,
        "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 5,
        "drop_rate": 0.1, "skip_drop": 0.5,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "verbose": -1, "seed": SEED + fold_idx, "n_jobs": CPU_THREADS,
    }
    # DART: no early stopping, use fixed rounds
    model = lgb.train(params, dtrain, num_boost_round=500, valid_sets=[dval],
                      callbacks=[lgb.log_evaluation(0)])
    return model, model.predict(X_val), model.predict(X_tr)


def xgboost_model(X_tr, y_tr, X_val, y_val, fold_idx, feat_names):
    import xgboost as xgb
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feat_names)
    params = {
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "max_depth": 7, "learning_rate": 0.05,
        "colsample_bytree": 0.7, "subsample": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 1.0, "min_child_weight": 5,
        "tree_method": "hist", "seed": SEED + fold_idx, "nthread": CPU_THREADS,
        "verbosity": 0,
    }
    model = xgb.train(params, dtrain, num_boost_round=1500,
                      evals=[(dval, "val")], early_stopping_rounds=50,
                      verbose_eval=False)
    return model, model.predict(dval), model.predict(dtrain)


def catboost_model(X_tr, y_tr, X_val, y_val, fold_idx, feat_names):
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(
        iterations=1500, learning_rate=0.05, depth=7,
        l2_leaf_reg=3.0, rsm=0.7, subsample=0.8,
        early_stopping_rounds=50, random_seed=SEED + fold_idx,
        verbose=0, thread_count=CATBOOST_THREAD_COUNT,
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
    return model, model.predict(X_val), model.predict(X_tr)


def rf_model(X_tr, y_tr, X_val, y_val, fold_idx, feat_names):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators=500, max_depth=None, max_features="sqrt",
        min_samples_leaf=5, n_jobs=CPU_THREADS, random_state=SEED + fold_idx,
    )
    model.fit(X_tr, y_tr)
    return model, model.predict(X_val), model.predict(X_tr)


def extratrees_model(X_tr, y_tr, X_val, y_val, fold_idx, feat_names):
    from sklearn.ensemble import ExtraTreesRegressor
    model = ExtraTreesRegressor(
        n_estimators=500, max_depth=None, max_features="sqrt",
        min_samples_leaf=5, n_jobs=CPU_THREADS, random_state=SEED + fold_idx,
    )
    model.fit(X_tr, y_tr)
    return model, model.predict(X_val), model.predict(X_tr)


def stacking_model(X_tr, y_tr, X_val, y_val, fold_idx, feat_names):
    """Stacking: LightGBM + XGBoost + RF base → Ridge meta."""
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestRegressor, StackingRegressor
    from sklearn.linear_model import Ridge

    base_lgb = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=63, max_depth=7,
        colsample_bytree=0.7, subsample=0.8, reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=CPU_THREADS, verbose=-1, random_state=SEED + fold_idx,
    )
    base_rf = RandomForestRegressor(
        n_estimators=300, max_features="sqrt", min_samples_leaf=5,
        n_jobs=CPU_THREADS, random_state=SEED + fold_idx,
    )
    # Use XGBoost sklearn API for stacking compatibility
    import xgboost as xgb
    base_xgb = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=7,
        colsample_bytree=0.7, subsample=0.8, reg_alpha=0.1, reg_lambda=1.0,
        tree_method="hist", n_jobs=CPU_THREADS, verbosity=0, random_state=SEED + fold_idx,
    )

    model = StackingRegressor(
        estimators=[("lgb", base_lgb), ("xgb", base_xgb), ("rf", base_rf)],
        final_estimator=Ridge(alpha=1.0),
        cv=3, n_jobs=1, passthrough=False,
    )
    model.fit(X_tr, y_tr)
    return model, model.predict(X_val), model.predict(X_tr)


def rsf_model(X_tr, y_tr, X_val, y_val, fold_idx, feat_names):
    """Random Survival Forest using scikit-survival.
    Converts regression label to survival format:
      time = ln_IC50 shifted to positive
      event = binary label (1=sensitive)
    Uses top-500 features from variance to keep computation feasible.
    """
    from sksurv.ensemble import RandomSurvivalForest
    from sklearn.feature_selection import VarianceThreshold

    # Feature selection: top 500 by variance (RSF is slow with 20K features)
    variances = np.var(X_tr, axis=0)
    top_k = min(500, X_tr.shape[1])
    top_idx = np.argsort(variances)[-top_k:]
    X_tr_sub = X_tr[:, top_idx]
    X_val_sub = X_val[:, top_idx]

    # Convert to survival format
    # time must be positive; shift ln_IC50 so min=1
    all_y = np.concatenate([y_tr, y_val])
    shift = -all_y.min() + 1.0

    # For train, need binary event indicator
    # We load binary labels via index
    # Since y_tr is passed as regression, we need to derive binary
    # Use quantile(0.3) threshold from FE pipeline
    threshold = np.quantile(y_tr, 0.3)
    event_tr = (y_tr <= threshold).astype(bool)
    event_val = (y_val <= threshold).astype(bool)

    time_tr = y_tr + shift
    time_val = y_val + shift

    # Structured array for sksurv
    y_surv_tr = np.array([(e, t) for e, t in zip(event_tr, time_tr)],
                         dtype=[("event", bool), ("time", float)])
    y_surv_val = np.array([(e, t) for e, t in zip(event_val, time_val)],
                          dtype=[("event", bool), ("time", float)])

    model = RandomSurvivalForest(
        n_estimators=100, max_depth=None, max_features="sqrt",
        min_samples_leaf=10, n_jobs=CPU_THREADS, random_state=SEED + fold_idx,
    )
    model.fit(X_tr_sub, y_surv_tr)

    # Risk score (higher = higher risk = more sensitive)
    risk_val = model.predict(X_val_sub)
    risk_tr = model.predict(X_tr_sub)

    # Invert risk to match IC50 direction (lower risk score = lower IC50 = sensitive)
    # Actually, RSF.predict returns cumulative hazard function sum -> higher = more events
    # For correlation with ln_IC50: lower IC50 = sensitive, higher risk = sensitive
    # So risk should negatively correlate with IC50
    # We negate risk to align with IC50 direction for Spearman
    pred_val = -risk_val
    pred_tr = -risk_tr

    return model, pred_val, pred_tr


def rsf_extra_metrics(model, X_tr, y_tr, X_val, y_val, fold_idx):
    """Compute C-index and AUROC for RSF."""
    from sksurv.metrics import concordance_index_censored

    variances = np.var(X_tr, axis=0)
    top_k = min(500, X_tr.shape[1])
    top_idx = np.argsort(variances)[-top_k:]
    X_val_sub = X_val[:, top_idx]

    all_y = np.concatenate([y_tr, y_val])
    shift = -all_y.min() + 1.0
    threshold = np.quantile(y_tr, 0.3)
    event_val = (y_val <= threshold).astype(bool)
    time_val = y_val + shift

    risk_val = model.predict(X_val_sub)

    # C-index
    c_idx = concordance_index_censored(event_val, time_val, risk_val)

    # AUROC using risk scores vs binary event
    try:
        auroc = roc_auc_score(event_val.astype(int), risk_val)
    except:
        auroc = float("nan")

    return {"c_index": c_idx[0], "auroc": auroc}


# ── Main ──

def main():
    X, y_reg, y_bin, feat_names = load_data()
    print(f"Thread safety mode: MODEL_CPU_THREADS={CPU_THREADS}, CATBOOST_THREAD_COUNT={CATBOOST_THREAD_COUNT}")

    all_results = []
    models_config = [
        ("1_LightGBM",      lgbm_model),
        ("2_LightGBM_DART", lgbm_dart_model),
        ("3_XGBoost",       xgboost_model),
        ("4_CatBoost",      catboost_model),
        ("5_RandomForest",  rf_model),
        ("6_ExtraTrees",    extratrees_model),
        ("7_Stacking_Ridge", stacking_model),
        ("8_RSF",           rsf_model),
    ]

    total_start = time.time()

    for name, fn in models_config:
        t0 = time.time()
        result = run_cv(name, fn, X, y_reg, y_bin, feat_names)
        result["elapsed_sec"] = time.time() - t0

        # Extra metrics for RSF
        if name == "8_RSF":
            print("  Computing RSF extra metrics (C-index, AUROC)...")
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            c_indices, aurocs = [], []
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y_reg[train_idx], y_reg[val_idx]
                # Re-train a quick RSF for C-index
                model, _, _ = rsf_model(X_tr, y_tr, X_val, y_val, fold_idx, feat_names)
                extra = rsf_extra_metrics(model, X_tr, y_tr, X_val, y_val, fold_idx)
                c_indices.append(extra["c_index"])
                aurocs.append(extra["auroc"])
            result["c_index_mean"] = np.mean(c_indices)
            result["c_index_std"] = np.std(c_indices)
            result["auroc_mean"] = np.nanmean(aurocs)
            result["auroc_std"] = np.nanstd(aurocs)
            print(f"      C-index: {result['c_index_mean']:.4f} +/- {result['c_index_std']:.4f}")
            print(f"      AUROC:   {result['auroc_mean']:.4f} +/- {result['auroc_std']:.4f}")

        all_results.append(result)
        snapshot_path = save_results_snapshot(all_results, "ml_results.partial.json")
        print(f"  Snapshot saved to {snapshot_path}")

    total_elapsed = time.time() - total_start

    # ── Summary Table ──
    print("\n" + "=" * 90)
    print(f"  ML MODELS SUMMARY TABLE  (Total: {total_elapsed/60:.1f} min)")
    print("=" * 90)
    print(f"{'Model':<22} {'Spearman':>10} {'  std':>6} {'RMSE':>8} {'  std':>6} "
          f"{'Pearson':>8} {'R2':>8} {'Gap(Sp)':>8} {'Time':>6}")
    print("-" * 90)

    for r in all_results:
        sp_flag = " *" if r["spearman_mean"] >= BENCH_SPEARMAN else ""
        rm_flag = " *" if r["rmse_mean"] <= BENCH_RMSE else ""
        print(f"{r['model']:<22} {r['spearman_mean']:>8.4f}{sp_flag:>2} "
              f"{r['spearman_std']:>6.4f} {r['rmse_mean']:>8.4f}{rm_flag:>2} "
              f"{r['rmse_std']:>6.4f} {r['pearson_mean']:>8.4f} {r['r2_mean']:>8.4f} "
              f"{r['gap_spearman_mean']:>8.4f} {r['elapsed_sec']/60:>5.1f}m")

    print("-" * 90)
    print(f"  Benchmark: Spearman >= {BENCH_SPEARMAN}, RMSE <= {BENCH_RMSE}   (* = meets benchmark)")
    print("=" * 90)

    results_path = OUTPUT_DIR / "ml_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_json_default)
    print(f"\nResults saved to {results_path}")

    # Upload to S3
    s3_path = f"{S3_BASE}/model_results/ml_results.json"
    os.system(f"aws s3 cp {results_path} {s3_path} 2>/dev/null")
    print(f"Uploaded to {s3_path}")


if __name__ == "__main__":
    main()
