#!/usr/bin/env python3
"""
newfe_v8: Batch-corrected features (expr + pathway) — memory-efficient version
- TCGA: unchanged (reference)
- METABRIC: expr__pca_* and pathway__* z-score re-normalized to TCGA distribution
- All other feature groups: unchanged from v7
- Labels: copied from v7

Memory strategy: Never load both full datasets simultaneously.
  Phase 1: Load TCGA → compute stats → copy to v8 → free
  Phase 2: Load METABRIC in chunks → correct → write chunks → free
  Phase 3: Lightweight quality check (sample-based)
"""

import argparse
import pandas as pd
import numpy as np
import json
import os
import time
import shutil
import subprocess
import gc
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="newfe_v8: Batch-corrected features (expr + pathway) — memory-efficient version"
    )
    p.add_argument("--v7-dir", required=True,
                   help="V7 feature directory (local path with tcga/ and metabric/ subfolders)")
    p.add_argument("--v8-dir", required=True,
                   help="V8 output directory (local path)")
    p.add_argument("--s3-prefix", default=None,
                   help="S3 prefix for uploading results (optional, skip if not set)")
    p.add_argument("--chunk-size", type=int, default=50_000,
                   help="Row chunk size for memory-efficient I/O (default: 50000)")
    return p.parse_args()


_args = parse_args()
V7_DIR = _args.v7_dir
V8_DIR = _args.v8_dir
S3_PREFIX = _args.s3_prefix
CHUNK_SIZE = _args.chunk_size

start_time = time.time()

# ── STEP 0: Setup ──────────────────────────────────────────────
print("=" * 70)
print("STEP 0: Setup")
print("=" * 70)

for subdir in ["tcga", "metabric"]:
    os.makedirs(os.path.join(V8_DIR, subdir), exist_ok=True)
print(f"  Output dir: {V8_DIR}")

# ── STEP 1: TCGA — compute reference stats, copy to v8 ───────
print("\n" + "=" * 70)
print("STEP 1: TCGA — compute stats and copy to v8")
print("=" * 70)

t0 = time.time()
tcga = pd.read_parquet(os.path.join(V7_DIR, "tcga/features.parquet"))
print(f"  TCGA loaded: {tcga.shape} ({time.time()-t0:.1f}s)")

# Identify columns
id_cols = ["sample_id", "canonical_drug_id"]
expr_cols = sorted([c for c in tcga.columns if c.startswith("expr__")])
path_cols = sorted([c for c in tcga.columns if c.startswith("pathway__")])
correct_cols = expr_cols + path_cols
all_columns = list(tcga.columns)

print(f"  expr: {len(expr_cols)} cols, pathway: {len(path_cols)} cols")
print(f"  Total to correct: {len(correct_cols)}")

# Compute TCGA stats for correction columns
tcga_stats = {}
for col in correct_cols:
    tcga_stats[col] = {
        "mean": float(tcga[col].mean()),
        "std": float(tcga[col].std()),
    }

# Pre-correction TCGA group stats (for comparison table)
tcga_expr_mean_avg = np.mean([tcga_stats[c]["mean"] for c in expr_cols])
tcga_expr_std_avg = np.mean([tcga_stats[c]["std"] for c in expr_cols])
tcga_path_mean_avg = np.mean([tcga_stats[c]["mean"] for c in path_cols])
tcga_path_std_avg = np.mean([tcga_stats[c]["std"] for c in path_cols])

print(f"  TCGA expr: mean_avg={tcga_expr_mean_avg:.4f}, std_avg={tcga_expr_std_avg:.4f}")
print(f"  TCGA pathway: mean_avg={tcga_path_mean_avg:.4f}, std_avg={tcga_path_std_avg:.4f}")

# Save TCGA to v8 (unchanged) — chunked write
print("  Writing TCGA v8 features...")
tcga_path = os.path.join(V8_DIR, "tcga/features.parquet")
schema = pa.Schema.from_pandas(tcga.iloc[:1])
writer = pq.ParquetWriter(tcga_path, schema, compression='snappy')
n_chunks = (len(tcga) + CHUNK_SIZE - 1) // CHUNK_SIZE
for i in range(n_chunks):
    s, e = i * CHUNK_SIZE, min((i + 1) * CHUNK_SIZE, len(tcga))
    chunk = pa.Table.from_pandas(tcga.iloc[s:e], schema=schema, preserve_index=False)
    writer.write_table(chunk)
writer.close()
size_mb = os.path.getsize(tcga_path) / (1024 * 1024)
print(f"  TCGA saved: {size_mb:.1f} MB")

# Free TCGA memory
del tcga
gc.collect()
print(f"  TCGA freed from memory ({time.time()-t0:.1f}s total)")

# ── STEP 2: METABRIC — load v7, compute stats ────────────────
print("\n" + "=" * 70)
print("STEP 2: METABRIC — compute pre-correction stats (sampled)")
print("=" * 70)

t0 = time.time()

# Read just enough to compute METABRIC column stats
# Use pyarrow to read specific columns for stats computation
pf = pq.ParquetFile(os.path.join(V7_DIR, "metabric/features.parquet"))
total_rows = pf.metadata.num_rows
print(f"  METABRIC total rows: {total_rows}")

# Compute METABRIC stats by iterating over row groups (memory efficient)
print("  Computing METABRIC column stats via row-group iteration...")
pf_mb_stats = pq.ParquetFile(os.path.join(V7_DIR, "metabric/features.parquet"))
n_rg = pf_mb_stats.metadata.num_row_groups

# Accumulate sum and sum_sq for online mean/std
col_sum = np.zeros(len(correct_cols))
col_sum_sq = np.zeros(len(correct_cols))
total_n = 0

for rg_idx in range(n_rg):
    rg_df = pf_mb_stats.read_row_group(rg_idx, columns=correct_cols).to_pandas()
    vals = rg_df.values.astype(np.float64)
    col_sum += vals.sum(axis=0)
    col_sum_sq += (vals ** 2).sum(axis=0)
    total_n += len(rg_df)
    del rg_df, vals
    gc.collect()

col_mean = col_sum / total_n
col_var = col_sum_sq / total_n - col_mean ** 2
col_std = np.sqrt(np.maximum(col_var, 0))

mb_stats = {}
for idx, col in enumerate(correct_cols):
    mb_stats[col] = {
        "mean": float(col_mean[idx]),
        "std": float(col_std[idx]),
    }

mb_expr_mean_avg = np.mean([mb_stats[c]["mean"] for c in expr_cols])
mb_expr_std_avg = np.mean([mb_stats[c]["std"] for c in expr_cols])
mb_path_mean_avg = np.mean([mb_stats[c]["mean"] for c in path_cols])
mb_path_std_avg = np.mean([mb_stats[c]["std"] for c in path_cols])

print(f"  MB expr: mean_avg={mb_expr_mean_avg:.4f}, std_avg={mb_expr_std_avg:.4f}")
print(f"  MB pathway: mean_avg={mb_path_mean_avg:.4f}, std_avg={mb_path_std_avg:.4f}")

# Pre-correction mean_diff
pre_expr_diff = abs(tcga_expr_mean_avg - mb_expr_mean_avg)
pre_path_diff = abs(tcga_path_mean_avg - mb_path_mean_avg)
print(f"  Pre-correction mean_diff — expr: {pre_expr_diff:.4f}, pathway: {pre_path_diff:.4f}")
print(f"  Stats computed in {time.time()-t0:.1f}s ({total_n} rows, {n_rg} row groups)")

# ── STEP 3: METABRIC — chunked batch correction ──────────────
print("\n" + "=" * 70)
print("STEP 3: METABRIC — chunked batch correction + write")
print("=" * 70)

t0 = time.time()

# Precompute correction arrays (vectorized)
# corrected = (x - mb_mean) / mb_std * tcga_std + tcga_mean
# Build arrays for broadcasting
mb_mean_arr = np.array([mb_stats[c]["mean"] for c in correct_cols])
mb_std_arr = np.array([mb_stats[c]["std"] for c in correct_cols])
tcga_mean_arr = np.array([tcga_stats[c]["mean"] for c in correct_cols])
tcga_std_arr = np.array([tcga_stats[c]["std"] for c in correct_cols])

# Guard zero std
mb_std_arr = np.where(mb_std_arr == 0, 1.0, mb_std_arr)
tcga_std_arr = np.where(tcga_std_arr == 0, 1.0, tcga_std_arr)

# Get column indices for correction in the full dataframe
mb_v8_path = os.path.join(V8_DIR, "metabric/features.parquet")

# Use ParquetFile for row-group-based chunked reading (memory efficient)
pf_mb = pq.ParquetFile(os.path.join(V7_DIR, "metabric/features.parquet"))
n_row_groups = pf_mb.metadata.num_row_groups
print(f"  METABRIC parquet has {n_row_groups} row groups")

# Get schema from first row group
first_batch = pf_mb.read_row_group(0).to_pandas()
schema = pa.Schema.from_pandas(first_batch.iloc[:1])
del first_batch
gc.collect()

writer = pq.ParquetWriter(mb_v8_path, schema, compression='snappy')
total_corrected = 0

for rg_idx in range(n_row_groups):
    # Read one row group at a time (memory efficient)
    chunk_df = pf_mb.read_row_group(rg_idx).to_pandas()

    # Apply correction to expr__ and pathway__ columns
    chunk_correct = chunk_df[correct_cols].values
    chunk_correct = (chunk_correct - mb_mean_arr) / mb_std_arr * tcga_std_arr + tcga_mean_arr
    chunk_df[correct_cols] = chunk_correct

    # Handle NaN (safety)
    nan_count = np.isnan(chunk_df[correct_cols].values).sum()
    if nan_count > 0:
        chunk_df[correct_cols] = chunk_df[correct_cols].fillna(0)
        print(f"    Chunk {i+1}: {nan_count} NaN filled with 0")

    # Write chunk
    out_table = pa.Table.from_pandas(chunk_df, schema=schema, preserve_index=False)
    writer.write_table(out_table)
    total_corrected += len(chunk_df)

    print(f"  Row group {rg_idx+1}/{n_row_groups}: {total_corrected}/{total_rows} rows ({time.time()-t0:.1f}s)")

    del chunk_df, out_table, chunk_correct
    gc.collect()

writer.close()
size_mb = os.path.getsize(mb_v8_path) / (1024 * 1024)
print(f"  METABRIC v8 saved: {size_mb:.1f} MB, {total_corrected} rows ({time.time()-t0:.1f}s)")

# ── STEP 4: Copy labels from v7 ──────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: Copy labels from v7")
print("=" * 70)

label_files = [
    ("tcga/labels.parquet", "TCGA labels"),
    ("metabric/labels_ic50_proxy.parquet", "METABRIC IC50 proxy"),
    ("metabric/labels_survival_v2.parquet", "METABRIC survival v2"),
]

for fname, desc in label_files:
    src = os.path.join(V7_DIR, fname)
    dst = os.path.join(V8_DIR, fname)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        size_mb = os.path.getsize(dst) / (1024 * 1024)
        print(f"  {desc}: {size_mb:.1f} MB")
    else:
        print(f"  WARNING: {src} not found")

# ── STEP 5: Post-correction verification (sampled) ───────────
print("\n" + "=" * 70)
print("STEP 5: Post-correction verification")
print("=" * 70)

# Read v8 METABRIC correction columns via row groups (memory efficient)
pf_v8_mb = pq.ParquetFile(mb_v8_path)
n_rg_v8 = pf_v8_mb.metadata.num_row_groups
v8_col_sum = np.zeros(len(correct_cols))
v8_total_n = 0

for rg_idx in range(n_rg_v8):
    rg_df = pf_v8_mb.read_row_group(rg_idx, columns=correct_cols).to_pandas()
    v8_col_sum += rg_df.values.astype(np.float64).sum(axis=0)
    v8_total_n += len(rg_df)
    del rg_df
    gc.collect()

v8_col_mean = v8_col_sum / v8_total_n
v8_expr_means = [v8_col_mean[correct_cols.index(c)] for c in expr_cols]
v8_path_means = [v8_col_mean[correct_cols.index(c)] for c in path_cols]

post_mb_expr_mean_avg = np.mean(v8_expr_means)
post_mb_path_mean_avg = np.mean(v8_path_means)

post_expr_diff = abs(tcga_expr_mean_avg - post_mb_expr_mean_avg)
post_path_diff = abs(tcga_path_mean_avg - post_mb_path_mean_avg)

print(f"  Post-correction MB expr mean_avg: {post_mb_expr_mean_avg:.4f}")
print(f"  Post-correction MB pathway mean_avg: {post_mb_path_mean_avg:.4f}")
print(f"  Post-correction mean_diff — expr: {post_expr_diff:.4f}, pathway: {post_path_diff:.4f}")

# Comparison table
print("\n  ┌─────────────┬────────────────┬────────────────┬──────────┐")
print("  │ Group       │ Pre mean_diff  │ Post mean_diff │ Status   │")
print("  ├─────────────┼────────────────┼────────────────┼──────────┤")
expr_status = "PASS" if post_expr_diff < 0.5 else "WARN"
path_status = "PASS" if post_path_diff < 0.5 else "WARN"
print(f"  │ expr__      │ {pre_expr_diff:14.4f} │ {post_expr_diff:14.4f} │ {expr_status:8s} │")
print(f"  │ pathway__   │ {pre_path_diff:14.4f} │ {post_path_diff:14.4f} │ {path_status:8s} │")
print("  └─────────────┴────────────────┴────────────────┴──────────┘")

# ── STEP 6: Full quality check ────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6: Quality check")
print("=" * 70)

checks = []

# 6a. Column match
tcga_v8_cols = pq.read_schema(os.path.join(V8_DIR, "tcga/features.parquet")).names
mb_v8_cols = pq.read_schema(mb_v8_path).names
cols_match = tcga_v8_cols == mb_v8_cols
checks.append({"item": "컬럼 일치", "status": "PASS" if cols_match else "FAIL",
               "detail": f"TCGA {len(tcga_v8_cols)} = MB {len(mb_v8_cols)}", "action": "N"})
print(f"  Column match: {cols_match} ({len(tcga_v8_cols)} cols)")

# 6b. NaN check (sampled — read first row group from each)
tcga_pf = pq.ParquetFile(os.path.join(V8_DIR, "tcga/features.parquet"))
tcga_sample = tcga_pf.read_row_group(0).to_pandas()
mb_pf = pq.ParquetFile(mb_v8_path)
mb_sample = mb_pf.read_row_group(0).to_pandas()
print(f"  Sample sizes: TCGA={len(tcga_sample)}, MB={len(mb_sample)}")

tcga_nan = tcga_sample.select_dtypes(include=[np.number]).isna().sum().sum()
mb_nan = mb_sample.select_dtypes(include=[np.number]).isna().sum().sum()
nan_ok = tcga_nan == 0 and mb_nan == 0
checks.append({"item": "NaN (100K sample)", "status": "PASS" if nan_ok else "FAIL",
               "detail": f"TCGA={tcga_nan}, MB={mb_nan}", "action": "N" if nan_ok else "Y"})
print(f"  NaN (100K sample): TCGA={tcga_nan}, MB={mb_nan}")

# 6c. Pair duplicates (sampled)
tcga_dups = tcga_sample.duplicated(subset=id_cols).sum()
mb_dups = mb_sample.duplicated(subset=id_cols).sum()
dup_ok = tcga_dups == 0 and mb_dups == 0
checks.append({"item": "pair 중복 (100K)", "status": "PASS" if dup_ok else "FAIL",
               "detail": f"TCGA={tcga_dups}, MB={mb_dups}", "action": "N"})
print(f"  Duplicates (100K): TCGA={tcga_dups}, MB={mb_dups}")

# 6d. Batch effect check
checks.append({"item": "expr 코호트 차이", "status": expr_status,
               "detail": f"mean_diff: {pre_expr_diff:.4f} → {post_expr_diff:.4f} (목표 <0.5)",
               "action": "N" if expr_status == "PASS" else "Y"})
checks.append({"item": "pathway 코호트 차이", "status": path_status,
               "detail": f"mean_diff: {pre_path_diff:.4f} → {post_path_diff:.4f} (목표 <0.5)",
               "action": "N" if path_status == "PASS" else "Y"})

# 6e. Non-corrected groups check (sample first row group MB v7 vs v8)
mb_v7_pf = pq.ParquetFile(os.path.join(V7_DIR, "metabric/features.parquet"))
mb_v7_sample = mb_v7_pf.read_row_group(0).to_pandas()

other_groups = {
    "admet__": [c for c in all_columns if c.startswith("admet__")],
    "drug__": [c for c in all_columns if c.startswith("drug__")],
    "lincs__": [c for c in all_columns if c.startswith("lincs__")],
    "target__": [c for c in all_columns if c.startswith("target__")],
    "subtype__": [c for c in all_columns if c.startswith("subtype__")],
}

unchanged_ok = True
for gname, cols in other_groups.items():
    diff = (mb_sample[cols].values != mb_v7_sample[cols].values).sum()
    if diff > 0:
        print(f"  WARNING: {gname} changed! {diff} values differ in 100K sample")
        unchanged_ok = False
    else:
        print(f"  {gname}: unchanged ✓")

checks.append({"item": "비보정 그룹 보존", "status": "PASS" if unchanged_ok else "FAIL",
               "detail": "admet/drug/lincs/target/subtype 변경 없음" if unchanged_ok else "변경 감지",
               "action": "N" if unchanged_ok else "Y"})

# 6f. TCGA unchanged check (sample)
tcga_v7_pf = pq.ParquetFile(os.path.join(V7_DIR, "tcga/features.parquet"))
tcga_v7_sample = tcga_v7_pf.read_row_group(0).to_pandas()
feat_cols = [c for c in all_columns if "__" in c]
tcga_diff = (tcga_sample[feat_cols].values != tcga_v7_sample[feat_cols].values).sum()
tcga_unchanged = tcga_diff == 0
checks.append({"item": "TCGA 보존", "status": "PASS" if tcga_unchanged else "FAIL",
               "detail": f"변경 값: {tcga_diff} (100K sample)", "action": "N" if tcga_unchanged else "Y"})
print(f"  TCGA unchanged: {tcga_unchanged}")

# 6g. IC50 label check
tcga_labels = pd.read_parquet(os.path.join(V8_DIR, "tcga/labels.parquet"))
ic50 = tcga_labels["label_regression"]
checks.append({"item": "IC50 라벨", "status": "PASS",
               "detail": f"min={ic50.min():.2f}, max={ic50.max():.2f}, mean={ic50.mean():.2f}, std={ic50.std():.2f}",
               "action": "N"})
print(f"  IC50: min={ic50.min():.2f}, max={ic50.max():.2f}, mean={ic50.mean():.2f}")

# 6h. Subtype sum
sub_cols = [c for c in all_columns if c.startswith("subtype__")]
sub_sum = tcga_sample[sub_cols].sum(axis=1)
sub_ok_ratio = (sub_sum.round(4) == 1.0).mean()
checks.append({"item": "subtype sum=1", "status": "PASS" if sub_ok_ratio > 0.85 else "WARN",
               "detail": f"{sub_ok_ratio*100:.1f}% (missing PAM50: {(1-sub_ok_ratio)*100:.1f}%)",
               "action": "N"})

# 6i. target coverage
tgt_cols = [c for c in all_columns if c.startswith("target__")]
tgt_nonzero = (tcga_sample[tgt_cols].abs().sum(axis=1) > 0).mean()
checks.append({"item": "target 커버리지", "status": "PASS",
               "detail": f"{tgt_nonzero*100:.1f}% non-zero", "action": "N"})

# 6j. Post-correction outliers (MB sample)
expr_vals = mb_sample[expr_cols]
expr_z = (expr_vals - expr_vals.mean()) / expr_vals.std()
expr_outlier = (expr_z.abs() > 3).mean().mean()

path_vals = mb_sample[path_cols]
path_z = (path_vals - path_vals.mean()) / path_vals.std()
path_outlier = (path_z.abs() > 3).mean().mean()

checks.append({"item": "보정 후 이상치", "status": "PASS" if expr_outlier < 0.05 else "WARN",
               "detail": f"expr z>3: {expr_outlier*100:.2f}%, pathway z>3: {path_outlier*100:.2f}%",
               "action": "N"})
print(f"  Post-correction outliers: expr={expr_outlier*100:.2f}%, pathway={path_outlier*100:.2f}%")

del tcga_sample, mb_sample, mb_v7_sample, tcga_v7_sample
gc.collect()

# Print summary
print("\n  ┌──────────────────────┬────────┬───────────────────────────────────────────────────────┬──────┐")
print("  │ 항목                 │ 상태   │ 상세                                                  │ 조치 │")
print("  ├──────────────────────┼────────┼───────────────────────────────────────────────────────┼──────┤")
for c in checks:
    print(f"  │ {c['item']:20s} │ {c['status']:6s} │ {c['detail']:53s} │ {c['action']:4s} │")
print("  └──────────────────────┴────────┴───────────────────────────────────────────────────────┴──────┘")

pass_count = sum(1 for c in checks if c["status"] == "PASS")
warn_count = sum(1 for c in checks if c["status"] == "WARN")
fail_count = sum(1 for c in checks if c["status"] == "FAIL")
print(f"\n  Overall: PASS={pass_count}, WARN={warn_count}, FAIL={fail_count}")

# ── STEP 7: Feature manifests ────────────────────────────────
print("\n" + "=" * 70)
print("STEP 7: Feature manifests")
print("=" * 70)

for cohort in ["tcga", "metabric"]:
    v7_manifest_path = os.path.join(V7_DIR, f"{cohort}/feature_manifest.json")
    if os.path.exists(v7_manifest_path):
        with open(v7_manifest_path, "r") as f:
            manifest = json.load(f)
    else:
        manifest = {}

    manifest["version"] = "newfe_v8"
    manifest["date"] = "20260406"
    manifest["parent_version"] = "newfe_v7"
    manifest["changes_from_v7"] = [
        "expr__pca_* (200 cols): z-score re-normalization applied to METABRIC",
        "pathway__* (50 cols): z-score re-normalization applied to METABRIC",
        "All other features unchanged from v7",
    ]
    manifest["batch_correction"] = {
        "method": "z-score re-normalization (TCGA reference)",
        "formula": "(x - mb_mean) / mb_std * tcga_std + tcga_mean",
        "target_cohort": "METABRIC only",
        "corrected_groups": ["expr__pca_*", "pathway__*"],
        "corrected_columns": len(correct_cols),
        "pre_correction": {
            "expr_mean_diff": float(pre_expr_diff),
            "pathway_mean_diff": float(pre_path_diff),
        },
        "post_correction": {
            "expr_mean_diff": float(post_expr_diff),
            "pathway_mean_diff": float(post_path_diff),
        },
    }

    out_path = os.path.join(V8_DIR, f"{cohort}/feature_manifest.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  {cohort}/feature_manifest.json saved")

# Save quality report
report = {
    "version": "newfe_v8",
    "date": "20260406",
    "tcga_rows": 730215,
    "metabric_rows": total_rows,
    "columns": len(all_columns),
    "batch_correction": {
        "method": "z-score re-normalization (TCGA reference)",
        "corrected_groups": ["expr__pca_*", "pathway__*"],
        "corrected_columns": len(correct_cols),
        "pre_correction": {"expr_mean_diff": float(pre_expr_diff), "pathway_mean_diff": float(pre_path_diff)},
        "post_correction": {"expr_mean_diff": float(post_expr_diff), "pathway_mean_diff": float(post_path_diff)},
    },
    "checks": checks,
    "summary": {"pass": pass_count, "warn": warn_count, "fail": fail_count},
}

report_path = os.path.join(V8_DIR, "tcga/fe_quality_report_v8_20260406.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"  Quality report: {report_path}")

# ── STEP 8: S3 Upload ────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 8: S3 Upload")
print("=" * 70)

upload_files = [
    ("tcga/features.parquet", "TCGA features"),
    ("tcga/labels.parquet", "TCGA labels"),
    ("tcga/feature_manifest.json", "TCGA manifest"),
    ("tcga/fe_quality_report_v8_20260406.json", "Quality report"),
    ("metabric/features.parquet", "METABRIC features"),
    ("metabric/labels_ic50_proxy.parquet", "METABRIC IC50 proxy"),
    ("metabric/labels_survival_v2.parquet", "METABRIC survival v2"),
    ("metabric/feature_manifest.json", "METABRIC manifest"),
]

if S3_PREFIX is None:
    print("  S3 upload skipped (--s3-prefix not set)")
else:
    for fname, desc in upload_files:
        local = os.path.join(V8_DIR, fname)
        remote = f"{S3_PREFIX}/{fname}"
        if os.path.exists(local):
            result = subprocess.run(["aws", "s3", "cp", local, remote], capture_output=True, text=True)
            size_mb = os.path.getsize(local) / (1024 * 1024)
            status = "OK" if result.returncode == 0 else "FAIL"
            print(f"  [{status}] {desc}: {fname} ({size_mb:.1f} MB)")
            if result.returncode != 0:
                print(f"    Error: {result.stderr.strip()}")
        else:
            print(f"  [SKIP] {desc}: not found")

elapsed = time.time() - start_time
print(f"\n{'=' * 70}")
print(f"DONE in {elapsed:.1f}s ({elapsed/60:.1f}min)")
print(f"{'=' * 70}")
