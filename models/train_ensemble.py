#!/usr/bin/env python3
"""
Step 5: Ensemble (Track 2) - Spearman-weighted average
6 models: CatBoost, FlatMLP, LightGBM, XGBoost, ResidualMLP, Cross-Attention
Output: Ensemble metrics + Top 30 drugs → Top 15 selection
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import json
import time
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

S3_BASE = os.getenv("S3_BASE", "s3://say2-4team/20260409_eseo")
FE_RUN_ID = os.getenv("FE_RUN_ID", "20260409_newfe_v8_eseo")
FEATURES_URI = f"{S3_BASE}/fe_output/{FE_RUN_ID}/features/features.parquet"
PAIR_FEATURES_URI = f"{S3_BASE}/fe_output/{FE_RUN_ID}/pair_features/pair_features_newfe_v2.parquet"
LABELS_URI = f"{S3_BASE}/fe_output/{FE_RUN_ID}/features/labels.parquet"
SEED = 42
N_FOLDS = 5
BENCH_SP = 0.713
BENCH_RMSE = 1.385
OUTPUT_DIR = Path(__file__).parent / "ensemble_results"
OUTPUT_DIR.mkdir(exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")


# ── Data Loading ──
def load_data():
    print("Loading data from S3...")
    t0 = time.time()
    features = pd.read_parquet(FEATURES_URI)
    pair_features = pd.read_parquet(PAIR_FEATURES_URI)
    labels = pd.read_parquet(LABELS_URI)

    merged = features.merge(pair_features, on=["sample_id", "canonical_drug_id"], how="inner")
    labels = labels.set_index(["sample_id", "canonical_drug_id"])
    merged = merged.set_index(["sample_id", "canonical_drug_id"])
    labels = labels.loc[merged.index]

    sample_ids = merged.index.get_level_values("sample_id").values
    drug_ids = merged.index.get_level_values("canonical_drug_id").values
    X = merged.select_dtypes(include=[np.number]).fillna(0.0).values.astype(np.float32)
    y = labels["label_regression"].values.astype(np.float32)
    print(f"  Loaded: {X.shape[0]} x {X.shape[1]} features ({time.time()-t0:.1f}s)")
    return X, y, sample_ids, drug_ids


# ── DL Model Definitions (same as train_dl_models.py) ──

class FlatMLP(nn.Module):
    def __init__(self, in_dim, layers=[1024, 512, 256], dropout=0.3):
        super().__init__()
        modules = []
        prev = in_dim
        for h in layers:
            modules += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        modules.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ResidualMLP(nn.Module):
    def __init__(self, in_dim, hidden=512, n_blocks=3, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.Dropout(dropout),
            ))
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = h + block(h)
        return self.head(h).squeeze(-1)


class CrossAttentionNet(nn.Module):
    def __init__(self, in_dim, sample_dim=18311, d_model=128, nhead=4, dropout=0.2):
        super().__init__()
        drug_dim = in_dim - sample_dim
        self.sample_dim = sample_dim
        self.sample_proj = nn.Sequential(nn.Linear(sample_dim, d_model), nn.GELU())
        self.drug_proj = nn.Sequential(nn.Linear(drug_dim, d_model), nn.GELU())
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, d_model),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1),
        )

    def forward(self, x):
        sample_x = x[:, :self.sample_dim]
        drug_x = x[:, self.sample_dim:]
        s = self.sample_proj(sample_x).unsqueeze(1)
        d = self.drug_proj(drug_x).unsqueeze(1)
        attn_out, _ = self.cross_attn(s, d, d)
        combined = torch.cat([attn_out.squeeze(1), s.squeeze(1)], dim=1)
        return self.ffn(combined).squeeze(-1)


def train_dl_model(model, X_tr, y_tr, X_val, y_val, epochs=100, lr=1e-3, batch_size=256, patience=15):
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    X_tr_t = torch.tensor(X_tr, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, device=DEVICE)
    X_val_t = torch.tensor(X_val, device=DEVICE)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).cpu().numpy()
            val_loss = mean_squared_error(y_val, val_pred)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred_val = model(X_val_t).cpu().numpy()
        y_pred_tr = model(X_tr_t).cpu().numpy()
    return y_pred_val, y_pred_tr


# ── ML Model Trainers ──

def train_catboost(X_tr, y_tr, X_val, y_val):
    model = CatBoostRegressor(
        iterations=2000, learning_rate=0.05, depth=8, l2_leaf_reg=3,
        random_seed=SEED, verbose=0, task_type="CPU",
        early_stopping_rounds=50,
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
    return model.predict(X_val).astype(np.float32), model.predict(X_tr).astype(np.float32)


def train_lightgbm(X_tr, y_tr, X_val, y_val):
    params = {
        "objective": "regression", "metric": "rmse", "boosting_type": "gbdt",
        "num_leaves": 127, "learning_rate": 0.05, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 5, "min_child_samples": 20,
        "verbose": -1, "seed": SEED, "n_jobs": -1,
    }
    dtrain = lgb.Dataset(X_tr, y_tr)
    dval = lgb.Dataset(X_val, y_val, reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=2000,
                      valid_sets=[dval], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    return model.predict(X_val).astype(np.float32), model.predict(X_tr).astype(np.float32)


def train_xgboost(X_tr, y_tr, X_val, y_val):
    params = {
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "max_depth": 8, "learning_rate": 0.05, "subsample": 0.8,
        "colsample_bytree": 0.8, "min_child_weight": 5,
        "seed": SEED, "n_jobs": -1, "verbosity": 0,
    }
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(params, dtrain, num_boost_round=2000,
                      evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=False)
    return model.predict(dval).astype(np.float32), model.predict(dtrain).astype(np.float32)


# ── Ensemble Pipeline ──

def main():
    X, y, sample_ids, drug_ids = load_data()
    in_dim = X.shape[1]
    sample_dim = 18311

    # Model configs: (name, type, train_func_or_class)
    model_configs = [
        ("CatBoost", "ml", train_catboost),
        ("LightGBM", "ml", train_lightgbm),
        ("XGBoost", "ml", train_xgboost),
        ("FlatMLP", "dl", FlatMLP),
        ("ResidualMLP", "dl", ResidualMLP),
        ("Cross-Attention", "dl", CrossAttentionNet),
    ]

    # DL model kwargs
    dl_kwargs = {
        "FlatMLP": ({"in_dim": in_dim, "layers": [1024, 512, 256], "dropout": 0.3},
                    {"epochs": 100, "lr": 1e-3, "batch_size": 256}),
        "ResidualMLP": ({"in_dim": in_dim, "hidden": 512, "n_blocks": 3, "dropout": 0.3},
                        {"epochs": 100, "lr": 1e-3, "batch_size": 256}),
        "Cross-Attention": ({"in_dim": in_dim, "sample_dim": sample_dim, "d_model": 128, "nhead": 4, "dropout": 0.2},
                            {"epochs": 80, "lr": 5e-4, "batch_size": 256}),
    }

    # 5-fold CV: collect OOF predictions for each model
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    n_samples = len(y)

    oof_preds = {name: np.zeros(n_samples, dtype=np.float32) for name, _, _ in model_configs}
    oof_train_preds = {name: np.zeros(n_samples, dtype=np.float32) for name, _, _ in model_configs}
    model_spearman = {}

    print(f"\n{'='*60}")
    print(f"  Step 5: Ensemble Training (6 models x 5-fold CV)")
    print(f"{'='*60}")

    total_t0 = time.time()

    for name, mtype, trainer in model_configs:
        print(f"\n  Training {name}...")
        t0 = time.time()
        fold_sp = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            if mtype == "ml":
                pred_val, pred_tr = trainer(X_tr, y_tr, X_val, y_val)
            else:
                # DL model
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
                X_val_s = scaler.transform(X_val).astype(np.float32)

                torch.manual_seed(SEED + fold_idx)
                if DEVICE.type == "mps":
                    torch.mps.empty_cache()

                model_kw, train_kw = dl_kwargs[name]
                model = trainer(**model_kw)
                pred_val, pred_tr = train_dl_model(model, X_tr_s, y_tr, X_val_s, y_val, **train_kw)

                del model
                if DEVICE.type == "mps":
                    torch.mps.empty_cache()

            oof_preds[name][val_idx] = pred_val
            # For train predictions, average across folds where this sample was in training
            sp, _ = spearmanr(y_val, pred_val)
            fold_sp.append(sp)

        mean_sp = np.mean(fold_sp)
        model_spearman[name] = mean_sp
        dt = time.time() - t0
        print(f"    {name}: Mean Sp={mean_sp:.4f} ({dt/60:.1f} min)")

    # ── Spearman-weighted ensemble ──
    print(f"\n{'─'*60}")
    print(f"  Computing Spearman-weighted ensemble...")
    print(f"{'─'*60}")

    # Weights proportional to Spearman
    total_sp = sum(model_spearman.values())
    weights = {name: sp / total_sp for name, sp in model_spearman.items()}

    print("\n  Model weights (Spearman-proportional):")
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"    {name:<20}: {w:.4f} (Sp={model_spearman[name]:.4f})")

    # Weighted average of OOF predictions
    ensemble_pred = np.zeros(n_samples, dtype=np.float64)
    for name, w in weights.items():
        ensemble_pred += w * oof_preds[name]

    # ── Ensemble metrics ──
    ens_sp, _ = spearmanr(y, ensemble_pred)
    ens_pe, _ = pearsonr(y, ensemble_pred)
    ens_rmse = np.sqrt(mean_squared_error(y, ensemble_pred))
    ens_r2 = r2_score(y, ensemble_pred)

    # Per-fold ensemble metrics for std calculation
    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        y_val = y[val_idx]
        ens_val = ensemble_pred[val_idx]
        sp_f, _ = spearmanr(y_val, ens_val)
        rmse_f = np.sqrt(mean_squared_error(y_val, ens_val))

        # Train ensemble prediction
        ens_tr = ensemble_pred[train_idx]
        sp_tr, _ = spearmanr(y[train_idx], ens_tr)

        fold_metrics.append({
            "fold": fold_idx, "spearman": sp_f, "rmse": rmse_f,
            "train_spearman": sp_tr, "gap_spearman": sp_tr - sp_f,
        })
        print(f"  Fold {fold_idx}: Sp={sp_f:.4f}  RMSE={rmse_f:.4f}  "
              f"Train Sp={sp_tr:.4f}  Gap={sp_tr-sp_f:.4f}")

    fm_df = pd.DataFrame(fold_metrics)

    sp_flag = "PASS" if fm_df["spearman"].mean() >= BENCH_SP else "FAIL"
    rm_flag = "PASS" if fm_df["rmse"].mean() <= BENCH_RMSE else "FAIL"

    print(f"\n  >>> ENSEMBLE SUMMARY:")
    print(f"      Spearman: {fm_df['spearman'].mean():.4f} +/- {fm_df['spearman'].std():.4f}  [{sp_flag} vs {BENCH_SP}]")
    print(f"      RMSE:     {fm_df['rmse'].mean():.4f} +/- {fm_df['rmse'].std():.4f}  [{rm_flag} vs {BENCH_RMSE}]")
    print(f"      Pearson:  {ens_pe:.4f}")
    print(f"      R2:       {ens_r2:.4f}")
    print(f"      Train Sp: {fm_df['train_spearman'].mean():.4f}  Gap: {fm_df['gap_spearman'].mean():.4f}")
    print(f"      Time:     {(time.time()-total_t0)/60:.1f} min")

    # ── Individual model comparison ──
    print(f"\n{'─'*60}")
    print(f"  Individual Model vs Ensemble Comparison")
    print(f"{'─'*60}")
    print(f"  {'Model':<20} {'OOF Spearman':>14} {'OOF RMSE':>10}")
    print(f"  {'-'*46}")
    for name in sorted(model_spearman, key=lambda x: -model_spearman[x]):
        sp, _ = spearmanr(y, oof_preds[name])
        rmse = np.sqrt(mean_squared_error(y, oof_preds[name]))
        print(f"  {name:<20} {sp:>14.4f} {rmse:>10.4f}")
    print(f"  {'─'*46}")
    print(f"  {'ENSEMBLE':<20} {ens_sp:>14.4f} {ens_rmse:>10.4f}")

    # ── Top 30 drugs → Top 15 ──
    print(f"\n{'='*60}")
    print(f"  Top 30 Drug Extraction & Top 15 Selection")
    print(f"{'='*60}")

    # Build a drug-level summary: average predicted IC50 per drug
    df_pred = pd.DataFrame({
        "sample_id": sample_ids,
        "drug_id": drug_ids,
        "y_true": y,
        "y_pred_ensemble": ensemble_pred.astype(np.float32),
    })

    # Lower IC50 (ln_IC50) = more sensitive = better drug
    drug_summary = df_pred.groupby("drug_id").agg(
        mean_pred_ic50=("y_pred_ensemble", "mean"),
        mean_true_ic50=("y_true", "mean"),
        std_pred_ic50=("y_pred_ensemble", "std"),
        n_samples=("y_pred_ensemble", "count"),
        sensitivity_rate=("y_true", lambda x: (x < np.median(y)).mean()),
    ).reset_index()

    # Rank by predicted sensitivity (lower predicted IC50 = more effective)
    drug_summary = drug_summary.sort_values("mean_pred_ic50", ascending=True)
    drug_summary["rank"] = range(1, len(drug_summary) + 1)

    # Top 30
    top30 = drug_summary.head(30).copy()

    # Categorize: validated (sensitivity_rate > 0.5) vs recommended (model-based)
    top30["category"] = top30["sensitivity_rate"].apply(
        lambda x: "Validated" if x > 0.5 else "Recommended"
    )

    print(f"\n  Top 30 Drugs (ranked by ensemble predicted IC50):")
    print(f"  {'Rank':<5} {'Drug ID':<25} {'Pred IC50':>10} {'True IC50':>10} "
          f"{'Sens Rate':>10} {'N':>4} {'Category':<12}")
    print(f"  {'-'*80}")
    for _, row in top30.iterrows():
        print(f"  {int(row['rank']):<5} {row['drug_id']:<25} {row['mean_pred_ic50']:>10.3f} "
              f"{row['mean_true_ic50']:>10.3f} {row['sensitivity_rate']:>10.1%} "
              f"{int(row['n_samples']):>4} {row['category']:<12}")

    # Top 15 selection strategy:
    # Prefer drugs with: low predicted IC50 + high sensitivity_rate + enough samples
    top30["score"] = (
        -top30["mean_pred_ic50"].rank()  # lower IC50 = better
        + top30["sensitivity_rate"].rank() * 2  # higher sensitivity = better
        + (top30["n_samples"] >= 5).astype(int) * 5  # bonus for sufficient samples
    )
    top15 = top30.nlargest(15, "score")
    top15 = top15.sort_values("mean_pred_ic50", ascending=True)
    top15["final_rank"] = range(1, 16)

    print(f"\n  {'='*80}")
    print(f"  TOP 15 SELECTED DRUGS")
    print(f"  {'='*80}")
    print(f"  {'#':<4} {'Drug ID':<25} {'Pred IC50':>10} {'True IC50':>10} "
          f"{'Sens Rate':>10} {'N':>4} {'Category':<12}")
    print(f"  {'-'*80}")
    for _, row in top15.iterrows():
        print(f"  {int(row['final_rank']):<4} {row['drug_id']:<25} {row['mean_pred_ic50']:>10.3f} "
              f"{row['mean_true_ic50']:>10.3f} {row['sensitivity_rate']:>10.1%} "
              f"{int(row['n_samples']):>4} {row['category']:<12}")

    # ── Save results ──
    def convert(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    results = {
        "ensemble_method": "spearman_weighted_average",
        "n_models": 6,
        "weights": {k: float(v) for k, v in weights.items()},
        "ensemble_metrics": {
            "spearman_mean": float(fm_df["spearman"].mean()),
            "spearman_std": float(fm_df["spearman"].std()),
            "rmse_mean": float(fm_df["rmse"].mean()),
            "rmse_std": float(fm_df["rmse"].std()),
            "pearson": float(ens_pe),
            "r2": float(ens_r2),
            "train_spearman_mean": float(fm_df["train_spearman"].mean()),
            "gap_spearman_mean": float(fm_df["gap_spearman"].mean()),
        },
        "individual_models": {
            name: {
                "spearman": float(spearmanr(y, oof_preds[name])[0]),
                "rmse": float(np.sqrt(mean_squared_error(y, oof_preds[name]))),
                "weight": float(weights[name]),
            } for name in model_spearman
        },
        "fold_metrics": fold_metrics,
        "top30_drugs": top30[["rank", "drug_id", "mean_pred_ic50", "mean_true_ic50",
                              "sensitivity_rate", "n_samples", "category"]].to_dict(orient="records"),
        "top15_drugs": top15[["final_rank", "drug_id", "mean_pred_ic50", "mean_true_ic50",
                              "sensitivity_rate", "n_samples", "category"]].to_dict(orient="records"),
    }

    out_path = OUTPUT_DIR / "ensemble_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")

    # Save Top 15 as CSV
    top15_csv = OUTPUT_DIR / "top15_drugs.csv"
    top15.to_csv(top15_csv, index=False)
    print(f"Top 15 drugs saved to {top15_csv}")

    # Save Top 30 as CSV
    top30_csv = OUTPUT_DIR / "top30_drugs.csv"
    top30.to_csv(top30_csv, index=False)
    print(f"Top 30 drugs saved to {top30_csv}")


if __name__ == "__main__":
    main()
