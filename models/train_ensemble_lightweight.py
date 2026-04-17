#!/usr/bin/env python3
"""
Step 5 lightweight ensemble runner.
Models: LightGBM + FlatMLP + Cross-Attention

This keeps the original Step 5 "Spearman-weighted average" logic, but loads
LightGBM before importing torch to avoid native runtime conflicts on macOS.
"""
import warnings
warnings.filterwarnings("ignore")

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

S3_BASE = os.getenv("S3_BASE", "s3://say2-4team/20260409_eseo")
FE_RUN_ID = os.getenv("FE_RUN_ID", "20260409_newfe_v8_eseo")
FEATURES_URI = f"{S3_BASE}/fe_output/{FE_RUN_ID}/features/features.parquet"
PAIR_FEATURES_URI = f"{S3_BASE}/fe_output/{FE_RUN_ID}/pair_features/pair_features_newfe_v2.parquet"
LABELS_URI = f"{S3_BASE}/fe_output/{FE_RUN_ID}/features/labels.parquet"
DRUG_ANN_URI = f"{S3_BASE}/data/gsdc/gdsc2_drug_annotation_master_20260406.parquet"
SEED = 42
N_FOLDS = 5
BENCH_SP = 0.713
BENCH_RMSE = 1.385
CPU_THREADS = int(os.getenv("MODEL_CPU_THREADS", "4"))
OUTPUT_DIR = Path(__file__).parent / os.getenv("ENSEMBLE_OUTPUT_DIRNAME", "ensemble_results_lightweight")
OUTPUT_DIR.mkdir(exist_ok=True)
STAGE = os.getenv("ENSEMBLE_STAGE", "all").strip().lower()

os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(CPU_THREADS))

np.random.seed(SEED)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def convert(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


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


def save_oof_predictions(model_name, preds):
    safe_name = model_name.lower().replace("-", "_").replace(" ", "_")
    out_path = OUTPUT_DIR / f"oof_{safe_name}.npy"
    np.save(out_path, preds.astype(np.float32))
    return out_path


def load_oof_predictions(model_name):
    safe_name = model_name.lower().replace("-", "_").replace(" ", "_")
    return np.load(OUTPUT_DIR / f"oof_{safe_name}.npy").astype(np.float32)


def dedupe_drug_candidates(drug_summary):
    drug_ann = pd.read_parquet(DRUG_ANN_URI)[["DRUG_ID", "DRUG_NAME"]].drop_duplicates("DRUG_ID")
    out = drug_summary.merge(drug_ann, left_on="drug_id", right_on="DRUG_ID", how="left")
    out["drug_name"] = out["DRUG_NAME"].fillna(out["drug_id"].map(lambda x: f"Drug_{x}"))
    out = out.drop(columns=["DRUG_ID", "DRUG_NAME"])
    out = out.sort_values(["mean_pred_ic50", "drug_id"], ascending=[True, True])
    before = len(out)
    out = out.drop_duplicates(subset=["drug_name"], keep="first").reset_index(drop=True)
    removed = before - len(out)
    if removed:
        print(f"  Removed {removed} duplicate candidate rows by drug name")
    out["rank"] = range(1, len(out) + 1)
    return out


def train_lightgbm_oof(X, y, kf):
    import lightgbm as lgb

    print("\n  Training LightGBM...")
    t0 = time.time()
    oof_pred = np.zeros(len(y), dtype=np.float32)
    fold_sps = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 127,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "verbose": -1,
            "seed": SEED,
            "n_jobs": CPU_THREADS,
        }
        dtrain = lgb.Dataset(X_tr, y_tr)
        dval = lgb.Dataset(X_val, y_val, reference=dtrain)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        pred_val = model.predict(X_val).astype(np.float32)
        oof_pred[val_idx] = pred_val

        sp, _ = spearmanr(y_val, pred_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred_val))
        fold_sps.append(float(sp))
        print(f"    Fold {fold_idx}: Sp={sp:.4f}  RMSE={rmse:.4f}")

    oof_path = save_oof_predictions("LightGBM", oof_pred)
    dt = time.time() - t0
    mean_sp = float(np.mean(fold_sps))
    print(f"    LightGBM: Mean Sp={mean_sp:.4f} ({dt/60:.1f} min)")
    print(f"    OOF saved to {oof_path}")
    return oof_pred, mean_sp, dt


def ensure_torch_runtime():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(SEED)
    torch.set_num_threads(CPU_THREADS)

    device_override = os.getenv("ENSEMBLE_DEVICE", "").strip().lower()
    if device_override:
        device = torch.device(device_override)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return torch, nn, DataLoader, TensorDataset, device


def build_dl_model(model_name, in_dim, sample_dim, torch, nn):
    class FlatMLP(nn.Module):
        def __init__(self, in_dim, layers=None, dropout=0.3):
            super().__init__()
            layers = layers or [1024, 512, 256]
            modules = []
            prev = in_dim
            for h in layers:
                modules += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
                prev = h
            modules.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*modules)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    class CrossAttentionNet(nn.Module):
        def __init__(self, in_dim, sample_dim=18311, d_model=128, nhead=4, dropout=0.2):
            super().__init__()
            drug_dim = in_dim - sample_dim
            self.sample_dim = sample_dim
            self.sample_proj = nn.Sequential(nn.Linear(sample_dim, d_model), nn.GELU())
            self.drug_proj = nn.Sequential(nn.Linear(drug_dim, d_model), nn.GELU())
            self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.ffn = nn.Sequential(
                nn.LayerNorm(d_model * 2),
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

        def forward(self, x):
            sample_x = x[:, :self.sample_dim]
            drug_x = x[:, self.sample_dim:]
            s = self.sample_proj(sample_x).unsqueeze(1)
            d = self.drug_proj(drug_x).unsqueeze(1)
            attn_out, _ = self.cross_attn(s, d, d)
            combined = torch.cat([attn_out.squeeze(1), s.squeeze(1)], dim=1)
            return self.ffn(combined).squeeze(-1)

    if model_name == "FlatMLP":
        return FlatMLP(in_dim=in_dim, layers=[1024, 512, 256], dropout=0.3), {
            "epochs": 100,
            "lr": 1e-3,
            "batch_size": 256,
        }
    if model_name == "Cross-Attention":
        return CrossAttentionNet(in_dim=in_dim, sample_dim=sample_dim, d_model=128, nhead=4, dropout=0.2), {
            "epochs": 80,
            "lr": 5e-4,
            "batch_size": 256,
        }
    raise ValueError(f"Unsupported DL model: {model_name}")


def train_dl_fold(model, X_tr, y_tr, X_val, y_val, torch, DataLoader, TensorDataset, device, epochs, lr, batch_size, patience=15):
    if device.type == "mps":
        torch.mps.empty_cache()

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.MSELoss()

    X_tr_t = torch.tensor(X_tr, device=device)
    y_tr_t = torch.tensor(y_tr, device=device)
    X_val_t = torch.tensor(X_val, device=device)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for _ in range(epochs):
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
        pred_val = model(X_val_t).cpu().numpy().astype(np.float32)
    return pred_val


def train_dl_oof(model_name, X, y, kf, in_dim, sample_dim):
    torch, nn, DataLoader, TensorDataset, device = ensure_torch_runtime()
    print(f"\n  Training {model_name} on {device}...")
    t0 = time.time()
    oof_pred = np.zeros(len(y), dtype=np.float32)
    fold_sps = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
        X_val_s = scaler.transform(X_val).astype(np.float32)

        torch.manual_seed(SEED + fold_idx)
        model, train_kwargs = build_dl_model(model_name, in_dim, sample_dim, torch, nn)
        pred_val = train_dl_fold(
            model,
            X_tr_s,
            y_tr,
            X_val_s,
            y_val,
            torch,
            DataLoader,
            TensorDataset,
            device,
            **train_kwargs,
        )
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

        oof_pred[val_idx] = pred_val
        sp, _ = spearmanr(y_val, pred_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred_val))
        fold_sps.append(float(sp))
        print(f"    Fold {fold_idx}: Sp={sp:.4f}  RMSE={rmse:.4f}")

    oof_path = save_oof_predictions(model_name, oof_pred)
    dt = time.time() - t0
    mean_sp = float(np.mean(fold_sps))
    print(f"    {model_name}: Mean Sp={mean_sp:.4f} ({dt/60:.1f} min)")
    print(f"    OOF saved to {oof_path}")
    return oof_pred, mean_sp, dt


def build_ensemble_outputs(y, sample_ids, drug_ids, selected_models, oof_preds, model_elapsed, total_t0):
    print(f"\n{'─'*60}")
    print("  Computing Spearman-weighted ensemble...")
    print(f"{'─'*60}")

    model_spearman = {}
    for name in selected_models:
        model_spearman[name] = float(spearmanr(y, oof_preds[name])[0])

    total_sp = sum(model_spearman.values())
    weights = {name: sp / total_sp for name, sp in model_spearman.items()}

    print("\n  Model weights (Spearman-proportional):")
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"    {name:<20}: {w:.4f} (OOF Sp={model_spearman[name]:.4f})")

    ensemble_pred = np.zeros(len(y), dtype=np.float64)
    for name, w in weights.items():
        ensemble_pred += w * oof_preds[name]

    ens_sp, _ = spearmanr(y, ensemble_pred)
    ens_pe, _ = pearsonr(y, ensemble_pred)
    ens_rmse = np.sqrt(mean_squared_error(y, ensemble_pred))
    ens_r2 = r2_score(y, ensemble_pred)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(y)):
        y_val = y[val_idx]
        ens_val = ensemble_pred[val_idx]
        sp_f, _ = spearmanr(y_val, ens_val)
        rmse_f = np.sqrt(mean_squared_error(y_val, ens_val))
        ens_tr = ensemble_pred[train_idx]
        sp_tr, _ = spearmanr(y[train_idx], ens_tr)
        fold_metrics.append({
            "fold": fold_idx,
            "spearman": float(sp_f),
            "rmse": float(rmse_f),
            "train_spearman": float(sp_tr),
            "gap_spearman": float(sp_tr - sp_f),
        })
        print(f"  Fold {fold_idx}: Sp={sp_f:.4f}  RMSE={rmse_f:.4f}  Train Sp={sp_tr:.4f}  Gap={sp_tr-sp_f:.4f}")

    fm_df = pd.DataFrame(fold_metrics)
    sp_flag = "PASS" if fm_df["spearman"].mean() >= BENCH_SP else "FAIL"
    rm_flag = "PASS" if fm_df["rmse"].mean() <= BENCH_RMSE else "FAIL"

    print("\n  >>> ENSEMBLE SUMMARY:")
    print(f"      Spearman: {fm_df['spearman'].mean():.4f} +/- {fm_df['spearman'].std():.4f}  [{sp_flag} vs {BENCH_SP}]")
    print(f"      RMSE:     {fm_df['rmse'].mean():.4f} +/- {fm_df['rmse'].std():.4f}  [{rm_flag} vs {BENCH_RMSE}]")
    print(f"      Pearson:  {ens_pe:.4f}")
    print(f"      R2:       {ens_r2:.4f}")
    print(f"      Train Sp: {fm_df['train_spearman'].mean():.4f}  Gap: {fm_df['gap_spearman'].mean():.4f}")
    print(f"      Time:     {(time.time()-total_t0)/60:.1f} min")

    print(f"\n{'─'*60}")
    print("  Individual Model vs Ensemble Comparison")
    print(f"{'─'*60}")
    print(f"  {'Model':<20} {'OOF Spearman':>14} {'OOF RMSE':>10}")
    print(f"  {'-'*46}")
    for name in sorted(model_spearman, key=lambda x: -model_spearman[x]):
        rmse = np.sqrt(mean_squared_error(y, oof_preds[name]))
        print(f"  {name:<20} {model_spearman[name]:>14.4f} {rmse:>10.4f}")
    print(f"  {'─'*46}")
    print(f"  {'ENSEMBLE':<20} {ens_sp:>14.4f} {ens_rmse:>10.4f}")

    df_pred = pd.DataFrame({
        "sample_id": sample_ids,
        "drug_id": drug_ids,
        "y_true": y,
        "y_pred_ensemble": ensemble_pred.astype(np.float32),
    })

    drug_summary = df_pred.groupby("drug_id").agg(
        mean_pred_ic50=("y_pred_ensemble", "mean"),
        mean_true_ic50=("y_true", "mean"),
        std_pred_ic50=("y_pred_ensemble", "std"),
        n_samples=("y_pred_ensemble", "count"),
        sensitivity_rate=("y_true", lambda x: (x < np.median(y)).mean()),
    ).reset_index()
    drug_summary = dedupe_drug_candidates(drug_summary)

    top30 = drug_summary.head(30).copy()
    top30["category"] = top30["sensitivity_rate"].apply(lambda x: "Validated" if x > 0.5 else "Recommended")
    top30["score"] = (
        -top30["mean_pred_ic50"].rank()
        + top30["sensitivity_rate"].rank() * 2
        + (top30["n_samples"] >= 5).astype(int) * 5
    )
    top15 = top30.sort_values(["score", "mean_pred_ic50"], ascending=[False, True]) \
                 .drop_duplicates(subset=["drug_name"], keep="first") \
                 .head(15) \
                 .sort_values("mean_pred_ic50", ascending=True)
    top15["final_rank"] = range(1, 16)

    results = {
        "ensemble_method": "spearman_weighted_average",
        "n_models": len(selected_models),
        "selected_models": selected_models,
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
            "elapsed_sec": float(time.time() - total_t0),
        },
        "individual_models": {
            name: {
                "spearman": float(model_spearman[name]),
                "rmse": float(np.sqrt(mean_squared_error(y, oof_preds[name]))),
                "weight": float(weights[name]),
                "elapsed_sec": float(model_elapsed.get(name, 0.0)),
            } for name in selected_models
        },
        "fold_metrics": fold_metrics,
        "top30_drugs": top30[["rank", "drug_id", "drug_name", "mean_pred_ic50", "mean_true_ic50",
                              "sensitivity_rate", "n_samples", "category"]].to_dict(orient="records"),
        "top15_drugs": top15[["final_rank", "drug_id", "drug_name", "mean_pred_ic50", "mean_true_ic50",
                              "sensitivity_rate", "n_samples", "category"]].to_dict(orient="records"),
    }

    out_path = OUTPUT_DIR / "ensemble_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")

    top15_csv = OUTPUT_DIR / "top15_drugs.csv"
    top15.to_csv(top15_csv, index=False)
    print(f"Top 15 drugs saved to {top15_csv}")

    top30_csv = OUTPUT_DIR / "top30_drugs.csv"
    top30.to_csv(top30_csv, index=False)
    print(f"Top 30 drugs saved to {top30_csv}")


def main():
    X, y, sample_ids, drug_ids = load_data()
    in_dim = X.shape[1]
    sample_dim = 18311
    selected_models = ["LightGBM", "FlatMLP", "Cross-Attention"]

    print(f"\n{'='*60}")
    print(f"  Step 5 Lightweight Ensemble ({len(selected_models)} models x {N_FOLDS}-fold CV)")
    print(f"  Selected models: {', '.join(selected_models)}")
    print(f"  Stage: {STAGE}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"{'='*60}")

    total_t0 = time.time()
    oof_preds = {}
    model_elapsed = {}
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    if STAGE in {"all", "lightgbm_only"}:
        oof_preds["LightGBM"], _, model_elapsed["LightGBM"] = train_lightgbm_oof(X, y, kf)
        if STAGE == "lightgbm_only":
            return
    else:
        oof_preds["LightGBM"] = load_oof_predictions("LightGBM")

    if STAGE in {"all", "dl_only"}:
        oof_preds["FlatMLP"], _, model_elapsed["FlatMLP"] = train_dl_oof("FlatMLP", X, y, kf, in_dim, sample_dim)
        oof_preds["Cross-Attention"], _, model_elapsed["Cross-Attention"] = train_dl_oof("Cross-Attention", X, y, kf, in_dim, sample_dim)
        if STAGE == "dl_only":
            return
    else:
        oof_preds["FlatMLP"] = load_oof_predictions("FlatMLP")
        oof_preds["Cross-Attention"] = load_oof_predictions("Cross-Attention")

    build_ensemble_outputs(y, sample_ids, drug_ids, selected_models, oof_preds, model_elapsed, total_t0)


if __name__ == "__main__":
    main()
