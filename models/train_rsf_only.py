#!/usr/bin/env python3
"""RSF (Random Survival Forest) only - 5-fold CV"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import json
import time
import os
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from scipy.stats import spearmanr, pearsonr
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

S3_BASE = os.getenv("S3_BASE", "s3://say2-4team/20260409_eseo")
FE_RUN_ID = os.getenv("FE_RUN_ID", "20260409_newfe_v8_eseo")
FEATURES_URI = f"{S3_BASE}/fe_output/{FE_RUN_ID}/features/features.parquet"
PAIR_FEATURES_URI = f"{S3_BASE}/fe_output/{FE_RUN_ID}/pair_features/pair_features_newfe_v2.parquet"
LABELS_URI = f"{S3_BASE}/fe_output/{FE_RUN_ID}/features/labels.parquet"
SEED = 42

print("Loading data from S3...")
features = pd.read_parquet(FEATURES_URI)
pair_features = pd.read_parquet(PAIR_FEATURES_URI)
labels = pd.read_parquet(LABELS_URI)

merged = features.merge(pair_features, on=["sample_id", "canonical_drug_id"], how="inner")
labels = labels.set_index(["sample_id", "canonical_drug_id"])
merged = merged.set_index(["sample_id", "canonical_drug_id"])
labels = labels.loc[merged.index]

X = merged.select_dtypes(include=[np.number]).fillna(0.0).values.astype(np.float32)
y_reg = labels["label_regression"].values.astype(np.float64)
y_bin = labels["label_binary"].values.astype(np.int32)
print(f"Loaded: {X.shape[0]} samples x {X.shape[1]} features")

# Feature selection: top 500 by variance
variances = np.var(X, axis=0)
top_idx = np.argsort(variances)[-500:]
X_sub = X[:, top_idx]
print(f"RSF using top-500 features by variance")

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
fold_metrics = []

print("\n" + "─" * 60)
print("  [8_RSF] Training with 5-fold CV...")
print("─" * 60)

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_sub)):
    t0 = time.time()
    X_tr, X_val = X_sub[train_idx], X_sub[val_idx]
    y_tr, y_val = y_reg[train_idx], y_reg[val_idx]

    # Survival format
    shift = -np.concatenate([y_tr, y_val]).min() + 1.0
    threshold = np.quantile(y_tr, 0.3)
    event_tr = (y_tr <= threshold).astype(bool)
    event_val = (y_val <= threshold).astype(bool)
    time_tr = y_tr + shift
    time_val = y_val + shift

    y_surv_tr = np.array([(e, t) for e, t in zip(event_tr, time_tr)],
                         dtype=[("event", bool), ("time", float)])

    model = RandomSurvivalForest(
        n_estimators=100, max_depth=None, max_features="sqrt",
        min_samples_leaf=10, n_jobs=-1, random_state=SEED + fold_idx,
    )
    model.fit(X_tr, y_surv_tr)

    risk_val = model.predict(X_val)
    risk_tr = model.predict(X_tr)

    # Metrics: correlate -risk with IC50
    pred_val = -risk_val
    pred_tr = -risk_tr

    sp_val, _ = spearmanr(y_val, pred_val)
    sp_tr, _ = spearmanr(y_tr, pred_tr)
    pe_val, _ = pearsonr(y_val, pred_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, pred_val))
    r2_val = r2_score(y_val, pred_val)

    # C-index
    c_idx = concordance_index_censored(event_val, time_val, risk_val)[0]

    # AUROC
    try:
        auroc = roc_auc_score(event_val.astype(int), risk_val)
    except:
        auroc = float("nan")

    dt = time.time() - t0
    m = {
        "fold": fold_idx, "spearman": sp_val, "pearson": pe_val,
        "rmse": rmse_val, "r2": r2_val,
        "train_spearman": sp_tr, "gap_spearman": sp_tr - sp_val,
        "c_index": c_idx, "auroc": auroc,
    }
    fold_metrics.append(m)
    print(f"  Fold {fold_idx}: Sp={sp_val:.4f}  RMSE={rmse_val:.4f}  "
          f"C-idx={c_idx:.4f}  AUROC={auroc:.4f}  Gap(Sp)={sp_tr-sp_val:.4f}  ({dt:.1f}s)")

df = pd.DataFrame(fold_metrics)
print(f"\n  >>> 8_RSF SUMMARY:")
print(f"      Spearman: {df['spearman'].mean():.4f} +/- {df['spearman'].std():.4f}")
print(f"      RMSE:     {df['rmse'].mean():.4f} +/- {df['rmse'].std():.4f}")
print(f"      Pearson:  {df['pearson'].mean():.4f}")
print(f"      R2:       {df['r2'].mean():.4f} +/- {df['r2'].std():.4f}")
print(f"      Train Sp: {df['train_spearman'].mean():.4f}  Gap: {df['gap_spearman'].mean():.4f}")
print(f"      C-index:  {df['c_index'].mean():.4f} +/- {df['c_index'].std():.4f}")
print(f"      AUROC:    {df['auroc'].mean():.4f} +/- {df['auroc'].std():.4f}")

# Save
result = {
    "model": "8_RSF", "folds": fold_metrics,
    "spearman_mean": float(df["spearman"].mean()),
    "rmse_mean": float(df["rmse"].mean()),
    "c_index_mean": float(df["c_index"].mean()),
    "auroc_mean": float(df["auroc"].mean()),
}
out = Path(__file__).parent / "ml_results" / "rsf_result.json"
out.parent.mkdir(exist_ok=True)
with open(out, "w") as f:
    json.dump(result, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
print(f"\nSaved to {out}")
