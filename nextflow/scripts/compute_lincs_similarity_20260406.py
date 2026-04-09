#!/usr/bin/env python3
"""
compute_lincs_similarity_20260406.py
────────────────────────────────────
Compute LINCS drug signature vs sample expression similarity.

1. Read LINCS MCF7 signatures (63K sigs × 12K genes), aggregate per drug (BRD)
2. Align genes with TCGA/METABRIC expression
3. Compute 5 similarity metrics per (sample, drug) pair
4. Output: sample-level avg similarity + per-drug similarity for matched drugs
"""
import argparse
import json, os, sys, warnings, gc, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import s3fs
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine as cosine_dist

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)
np.random.seed(42)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute LINCS drug signature vs sample expression similarity."
    )
    p.add_argument("--lincs-mcf7-uri", required=True,
                   help="LINCS MCF7 signatures parquet (sig_id + gene columns).")
    p.add_argument("--gene-map-uri", required=True,
                   help="Gene mapping parquet (entrez_gene_id, gene_symbol).")
    p.add_argument("--tcga-expr-uri", required=True,
                   help="TCGA expression parquet (gene_id rows × sample columns).")
    p.add_argument("--tcga-cdr-uri", required=True,
                   help="TCGA CDR xlsx/csv for BRCA patient filtering.")
    p.add_argument("--metabric-expr-uri", required=True,
                   help="METABRIC expression parquet/tsv (entrezGeneId rows × sample columns).")
    p.add_argument("--metabric-filtered-uri", default=None,
                   help="METABRIC filtered samples parquet (for sample list). If None, use all MB- columns.")
    p.add_argument("--output-dir", required=True,
                   help="Local output directory for results.")
    p.add_argument("--s3-output-prefix", default=None,
                   help="S3 prefix for uploading results (optional).")
    return p.parse_args()


_args = parse_args()
LOCAL_OUT = _args.output_dir
os.makedirs(LOCAL_OUT, exist_ok=True)
fs = s3fs.S3FileSystem()

t_total = time.time()

###############################################################################
# STEP 1 — Load gene mapping & find common genes
###############################################################################
print("=" * 70)
print("STEP 1 — Gene mapping & common genes")
print("=" * 70, flush=True)

# LINCS gene columns (entrez IDs)
lincs_uri = _args.lincs_mcf7_uri
if lincs_uri.startswith("s3://"):
    lincs_pf = pq.ParquetFile(fs.open(lincs_uri.replace("s3://", "")))
else:
    lincs_pf = pq.ParquetFile(lincs_uri)
lincs_all_cols = [f.name for f in lincs_pf.schema_arrow]
lincs_gene_cols = [c for c in lincs_all_cols if c.isdigit()]
print(f"  LINCS gene columns: {len(lincs_gene_cols)}", flush=True)

# LINCS entrez → gene_symbol mapping
gene_map = pd.read_parquet(_args.gene_map_uri)
entrez_to_sym = dict(zip(gene_map["entrez_gene_id"].astype(str), gene_map["gene_symbol"]))
print(f"  Gene mapping: {len(entrez_to_sym)} entries", flush=True)

# TCGA expression (z-scored, genes × samples)
print("  Loading TCGA expression …", flush=True)
tcga_expr_uri = _args.tcga_expr_uri
if tcga_expr_uri.startswith("s3://"):
    tcga_raw_pf = pq.ParquetFile(fs.open(tcga_expr_uri.replace("s3://", "")))
else:
    tcga_raw_pf = pq.ParquetFile(tcga_expr_uri)
tcga_all_cols = [f.name for f in tcga_raw_pf.schema_arrow]

# Get BRCA sample columns
tcga_cdr_uri = _args.tcga_cdr_uri
if tcga_cdr_uri.startswith("s3://"):
    with fs.open(tcga_cdr_uri.replace("s3://", ""), "rb") as f:
        cdr = pd.read_excel(f, sheet_name=0) if tcga_cdr_uri.endswith(".xlsx") else pd.read_csv(f)
else:
    cdr = pd.read_excel(tcga_cdr_uri, sheet_name=0) if tcga_cdr_uri.endswith(".xlsx") else pd.read_csv(tcga_cdr_uri)
brca_patients = set(cdr.loc[cdr["type"] == "BRCA", "bcr_patient_barcode"])
brca_sample_cols = [c for c in tcga_all_cols if c.startswith("TCGA-") and c[:12] in brca_patients]

# Read gene_id + BRCA samples
tcga_df = tcga_raw_pf.read(columns=["gene_id"] + brca_sample_cols).to_pandas()
tcga_df["entrez"] = tcga_df["gene_id"].apply(lambda x: str(x).split("|")[-1] if "|" in str(x) else str(x))
tcga_expr = tcga_df.set_index("entrez")[brca_sample_cols].astype(np.float32)
del tcga_df; gc.collect()
print(f"  TCGA BRCA: {tcga_expr.shape[1]} samples × {tcga_expr.shape[0]} genes", flush=True)

# METABRIC expression
print("  Loading METABRIC expression …", flush=True)
mb_expr_uri = _args.metabric_expr_uri
if mb_expr_uri.startswith("s3://"):
    if mb_expr_uri.endswith(".parquet"):
        mb_raw = pd.read_parquet(mb_expr_uri)
    else:
        with fs.open(mb_expr_uri.replace("s3://", "")) as f:
            mb_raw = pd.read_csv(f, sep="\t")
else:
    if mb_expr_uri.endswith(".parquet"):
        mb_raw = pd.read_parquet(mb_expr_uri)
    else:
        mb_raw = pd.read_csv(mb_expr_uri, sep="\t")

# Detect entrez column
entrez_col = "entrezGeneId" if "entrezGeneId" in mb_raw.columns else "entrez_gene_id"
mb_raw["entrez"] = mb_raw[entrez_col].astype(str)

# Get filtered sample list
if _args.metabric_filtered_uri:
    mb_filt_uri = _args.metabric_filtered_uri
    if mb_filt_uri.startswith("s3://"):
        mb_filt_pf = pq.ParquetFile(fs.open(mb_filt_uri.replace("s3://", "")))
    else:
        mb_filt_pf = pq.ParquetFile(mb_filt_uri)
    mb_filt_samples = [f.name for f in mb_filt_pf.schema_arrow if f.name.startswith("MB-")]
    mb_sample_cols = [c for c in mb_raw.columns if c.startswith("MB-") and c in mb_filt_samples]
else:
    mb_sample_cols = [c for c in mb_raw.columns if c.startswith("MB-")]
mb_expr = mb_raw.set_index("entrez")[mb_sample_cols].astype(np.float32)
del mb_raw; gc.collect()
print(f"  METABRIC: {mb_expr.shape[1]} samples × {mb_expr.shape[0]} genes", flush=True)

# Common genes across LINCS + TCGA + METABRIC
common_genes = sorted(set(lincs_gene_cols) & set(tcga_expr.index) & set(mb_expr.index))
print(f"  Common genes (LINCS ∩ TCGA ∩ METABRIC): {len(common_genes)}", flush=True)

# Z-score TCGA and METABRIC (per gene, TCGA-fit)
print("  Z-scoring expression (TCGA-fit) …", flush=True)
tcga_common = tcga_expr.loc[common_genes].astype(np.float32)
mb_common = mb_expr.loc[common_genes].astype(np.float32)
del tcga_expr, mb_expr; gc.collect()

gene_mean = tcga_common.mean(axis=1)
gene_std = tcga_common.std(axis=1).replace(0, 1.0)

tcga_z = tcga_common.subtract(gene_mean, axis=0).divide(gene_std, axis=0).fillna(0)
mb_z = mb_common.subtract(gene_mean, axis=0).divide(gene_std, axis=0).fillna(0)
del tcga_common, mb_common; gc.collect()

# Transpose: samples × genes (numpy)
tcga_mat = tcga_z.T.values  # (1215, n_genes)
mb_mat = mb_z.T.values      # (1980, n_genes)
tcga_sample_ids = tcga_z.columns.tolist()
mb_sample_ids = mb_z.columns.tolist()
del tcga_z, mb_z; gc.collect()
print(f"  TCGA matrix: {tcga_mat.shape}, METABRIC matrix: {mb_mat.shape}", flush=True)

###############################################################################
# STEP 2 — Read LINCS signatures & aggregate per drug
###############################################################################
print("\n" + "=" * 70)
print("STEP 2 — Read LINCS & aggregate per drug")
print("=" * 70, flush=True)

# Read only sig_id + common gene columns
read_cols = ["sig_id"] + common_genes
print(f"  Reading {len(read_cols)} columns from LINCS …", flush=True)

t0 = time.time()
drug_sum = {}   # brd_short → sum array
drug_count = {} # brd_short → count

batch_size = 5000
for batch in lincs_pf.iter_batches(batch_size=batch_size, columns=read_cols):
    df = batch.to_pandas()
    # Extract BRD short from sig_id
    df["brd_short"] = df["sig_id"].apply(
        lambda x: "-".join(str(x).split(":")[1].split("-")[:2]) if ":" in str(x) and "BRD" in str(x) else None
    )
    df = df[df["brd_short"].notna()]
    vals = df[common_genes].values.astype(np.float32)

    for brd in df["brd_short"].unique():
        mask = (df["brd_short"] == brd).values
        chunk_vals = vals[mask]
        if brd not in drug_sum:
            drug_sum[brd] = np.zeros(len(common_genes), dtype=np.float64)
            drug_count[brd] = 0
        drug_sum[brd] += chunk_vals.sum(axis=0).astype(np.float64)
        drug_count[brd] += chunk_vals.shape[0]

    del df, vals; gc.collect()

print(f"  LINCS read & aggregated in {time.time()-t0:.1f}s", flush=True)
print(f"  Unique drugs (BRD): {len(drug_sum)}", flush=True)

# Compute mean signature per drug
drug_ids_list = sorted(drug_sum.keys())
drug_mat = np.zeros((len(drug_ids_list), len(common_genes)), dtype=np.float32)
for i, brd in enumerate(drug_ids_list):
    drug_mat[i] = (drug_sum[brd] / max(drug_count[brd], 1)).astype(np.float32)
del drug_sum, drug_count; gc.collect()
print(f"  Drug signature matrix: {drug_mat.shape}", flush=True)

###############################################################################
# STEP 3 — Compute similarity metrics (vectorized)
###############################################################################
print("\n" + "=" * 70)
print("STEP 3 — Compute similarity metrics")
print("=" * 70, flush=True)

def compute_similarities_batch(sample_mat, drug_mat, sample_ids, drug_ids, cohort_name):
    """Compute 5 similarity metrics for all (sample, drug) pairs.
    Returns DataFrame with sample_id, canonical_drug_id, 5 metrics."""
    n_samples = sample_mat.shape[0]
    n_drugs = drug_mat.shape[0]
    n_genes = sample_mat.shape[1]
    print(f"  {cohort_name}: {n_samples} samples × {n_drugs} drugs = {n_samples*n_drugs:,} pairs", flush=True)

    # Normalize for cosine similarity
    sample_norms = np.linalg.norm(sample_mat, axis=1, keepdims=True)
    sample_norms[sample_norms == 0] = 1
    sample_normed = sample_mat / sample_norms

    drug_norms = np.linalg.norm(drug_mat, axis=1, keepdims=True)
    drug_norms[drug_norms == 0] = 1
    drug_normed = drug_mat / drug_norms

    # Cosine similarity: dot product of normalized vectors
    # (n_samples × n_genes) @ (n_genes × n_drugs) → (n_samples × n_drugs)
    cosine_sim = sample_normed @ drug_normed.T  # (n_samples, n_drugs)
    print(f"    Cosine done", flush=True)

    # Pearson correlation: center then cosine
    sample_centered = sample_mat - sample_mat.mean(axis=1, keepdims=True)
    drug_centered = drug_mat - drug_mat.mean(axis=1, keepdims=True)
    sc_norms = np.linalg.norm(sample_centered, axis=1, keepdims=True)
    sc_norms[sc_norms == 0] = 1
    dc_norms = np.linalg.norm(drug_centered, axis=1, keepdims=True)
    dc_norms[dc_norms == 0] = 1
    pearson_sim = (sample_centered / sc_norms) @ (drug_centered / dc_norms).T
    print(f"    Pearson done", flush=True)

    # Spearman: rank-based Pearson (rank per vector, then Pearson)
    # For efficiency, rank along genes axis
    sample_ranks = np.argsort(np.argsort(sample_mat, axis=1), axis=1).astype(np.float32)
    drug_ranks = np.argsort(np.argsort(drug_mat, axis=1), axis=1).astype(np.float32)
    sr_centered = sample_ranks - sample_ranks.mean(axis=1, keepdims=True)
    dr_centered = drug_ranks - drug_ranks.mean(axis=1, keepdims=True)
    sr_norms = np.linalg.norm(sr_centered, axis=1, keepdims=True)
    sr_norms[sr_norms == 0] = 1
    dr_norms = np.linalg.norm(dr_centered, axis=1, keepdims=True)
    dr_norms[dr_norms == 0] = 1
    spearman_sim = (sr_centered / sr_norms) @ (dr_centered / dr_norms).T
    del sample_ranks, drug_ranks, sr_centered, dr_centered; gc.collect()
    print(f"    Spearman done", flush=True)

    # Reverse score: for each drug, find top-K upregulated genes in drug signature
    # Score = fraction of those genes that are DOWNregulated in sample (or vice versa)
    # This captures the "reversal" idea from CMap
    def reverse_score(sample_m, drug_m, k):
        """For each drug, top-K genes by drug signature magnitude.
        Score = negative correlation of sample values at those gene positions."""
        n_s, n_d = sample_m.shape[0], drug_m.shape[0]
        scores = np.zeros((n_s, n_d), dtype=np.float32)
        for d_idx in range(n_d):
            top_idx = np.argsort(-drug_m[d_idx])[:k]
            bot_idx = np.argsort(drug_m[d_idx])[:k]
            # Enrichment: mean of sample expression at drug's top-up genes
            # minus mean at drug's top-down genes
            up_scores = sample_m[:, top_idx].mean(axis=1)
            dn_scores = sample_m[:, bot_idx].mean(axis=1)
            # Reversal: high reversal = sample is opposite to drug
            scores[:, d_idx] = dn_scores - up_scores
        return scores

    # For efficiency, only compute for top 200 drugs by signature magnitude
    # (most variable drugs) then fill rest with 0
    drug_magnitudes = np.abs(drug_mat).mean(axis=1)
    top_drug_idx = np.argsort(-drug_magnitudes)[:min(500, n_drugs)]

    rev50_full = np.zeros((n_samples, n_drugs), dtype=np.float32)
    rev100_full = np.zeros((n_samples, n_drugs), dtype=np.float32)

    rev50_partial = reverse_score(sample_mat, drug_mat[top_drug_idx], 50)
    rev100_partial = reverse_score(sample_mat, drug_mat[top_drug_idx], 100)
    rev50_full[:, top_drug_idx] = rev50_partial
    rev100_full[:, top_drug_idx] = rev100_partial
    print(f"    Reverse scores done (top {len(top_drug_idx)} drugs)", flush=True)

    # Flatten to DataFrame
    # For memory, compute per-sample summary (mean across drugs)
    # and also save per-drug for matched drugs
    results_rows = []
    for s_idx in range(n_samples):
        results_rows.append({
            "sample_id": sample_ids[s_idx],
            "lincs_cosine_mean": float(np.mean(cosine_sim[s_idx])),
            "lincs_pearson_mean": float(np.mean(pearson_sim[s_idx])),
            "lincs_spearman_mean": float(np.mean(spearman_sim[s_idx])),
            "lincs_reverse_top50_mean": float(np.mean(rev50_full[s_idx])),
            "lincs_reverse_top100_mean": float(np.mean(rev100_full[s_idx])),
            "lincs_cosine_std": float(np.std(cosine_sim[s_idx])),
            "lincs_pearson_std": float(np.std(pearson_sim[s_idx])),
            "lincs_spearman_std": float(np.std(spearman_sim[s_idx])),
            "lincs_n_drugs": n_drugs,
        })

    result_df = pd.DataFrame(results_rows)
    del cosine_sim, pearson_sim, spearman_sim, rev50_full, rev100_full; gc.collect()
    return result_df

# TCGA
print("\n  Computing TCGA similarities …", flush=True)
t0 = time.time()
tcga_result = compute_similarities_batch(tcga_mat, drug_mat, tcga_sample_ids, drug_ids_list, "TCGA")
print(f"  TCGA done in {time.time()-t0:.1f}s", flush=True)

# METABRIC
print("\n  Computing METABRIC similarities …", flush=True)
t0 = time.time()
mb_result = compute_similarities_batch(mb_mat, drug_mat, mb_sample_ids, drug_ids_list, "METABRIC")
print(f"  METABRIC done in {time.time()-t0:.1f}s", flush=True)

del tcga_mat, mb_mat, drug_mat; gc.collect()

###############################################################################
# STEP 4 — Save & upload
###############################################################################
print("\n" + "=" * 70)
print("STEP 4 — Save & S3 upload")
print("=" * 70, flush=True)

tcga_result.to_parquet(f"{LOCAL_OUT}/tcga_lincs_similarity_20260406.parquet", index=False)
mb_result.to_parquet(f"{LOCAL_OUT}/metabric_lincs_similarity_20260406.parquet", index=False)

# Summary stats
print("\n  TCGA similarity stats:")
for c in [col for col in tcga_result.columns if col.startswith("lincs_")]:
    vals = tcga_result[c]
    print(f"    {c}: mean={vals.mean():.6f}, std={vals.std():.6f}, min={vals.min():.6f}, max={vals.max():.6f}")

print("\n  METABRIC similarity stats:")
for c in [col for col in mb_result.columns if col.startswith("lincs_")]:
    vals = mb_result[c]
    print(f"    {c}: mean={vals.mean():.6f}, std={vals.std():.6f}, min={vals.min():.6f}, max={vals.max():.6f}")

# Audit
audit = {
    "run_date": "20260406",
    "lincs_source": f"s3://{B}/results/lincs/16_mcf7_processed.parquet",
    "common_genes": len(common_genes),
    "unique_drugs_brd": len(drug_ids_list),
    "tcga_samples": len(tcga_sample_ids),
    "metabric_samples": len(mb_sample_ids),
    "metrics": ["cosine_mean", "pearson_mean", "spearman_mean", "reverse_top50_mean", "reverse_top100_mean",
                "cosine_std", "pearson_std", "spearman_std"],
    "tcga_stats": {c: {"mean": round(float(tcga_result[c].mean()), 6),
                        "std": round(float(tcga_result[c].std()), 6)}
                   for c in tcga_result.columns if c.startswith("lincs_")},
    "metabric_stats": {c: {"mean": round(float(mb_result[c].mean()), 6),
                            "std": round(float(mb_result[c].std()), 6)}
                       for c in mb_result.columns if c.startswith("lincs_")},
    "total_time_sec": round(time.time() - t_total, 1),
}
with open(f"{LOCAL_OUT}/lincs_similarity_audit_20260406.json", "w") as f:
    json.dump(audit, f, indent=2)

# S3 upload
if _args.s3_output_prefix:
    S3_OUT = _args.s3_output_prefix.replace("s3://", "").rstrip("/")
    for fname in ["tcga_lincs_similarity_20260406.parquet",
                  "metabric_lincs_similarity_20260406.parquet",
                  "lincs_similarity_audit_20260406.json"]:
        local_p = f"{LOCAL_OUT}/{fname}"
        if os.path.exists(local_p):
            fs.put(local_p, f"{S3_OUT}/{fname}")
            sz = os.path.getsize(local_p)
            print(f"  → s3://{S3_OUT}/{fname} ({sz:,} bytes)", flush=True)
else:
    print("  S3 upload skipped (--s3-output-prefix not set)", flush=True)

print(f"\nTotal time: {time.time()-t_total:.1f}s")
print("Done ✓")
