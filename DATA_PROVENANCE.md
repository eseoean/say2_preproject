# Data Provenance — Step 2 Data Preparation

> Last updated: 2026-04-08

## Overview

Work folder: `s3://say2-4team/20260408_new_pre_project_biso/20260408_pre_project_biso_myprotocol/data/`

All data originates from two S3 locations:
| Source | Type | Access |
|--------|------|--------|
| `s3://say2-4team/curated_date/` | Curated parquet files | Read-only |
| `s3://say2-4team/oringinal_raw/` | Original raw data files | Read-only |

## File Sources

### 1. curated_date/ direct copy (157 files)

Subdirectories copied as-is (excluding `glue/`):

| Folder | Files | Notes |
|--------|-------|-------|
| `admet/` | 40+ | ADMET benchmark tasks (TDC) |
| `chembl/` | 8 | ChEMBL activity, compounds, targets, mechanisms |
| `depmap/` | 18 | DepMap CRISPR, model info, repurposing |
| `drugbank/` | 7 | DrugBank drugs, targets, external IDs |
| `gsdc/` | 15 | GDSC2 basic_clean, labels, annotations |
| `lincs/` | 11 | LINCS sig_info, cell_info, pert_info, gene_info (metadata only) |
| `metabric/` | 5 | METABRIC expression, clinical |
| `msigdb/` | 4 | MSigDB gene sets |
| `opentargets/` | 6 | OpenTargets associations, targets |
| `string/` | 5 | STRING PPI links, protein info |
| `tcga/` | 14 | TCGA clinical, BRCA RNA inventory |

### 2. gdsc_ic50.parquet (generated)

**NOT a direct copy.** Generated from curated GDSC2 data:

| Source | Records |
|--------|---------|
| `curated_date/gsdc/gdsc2_basic_clean_20260406.parquet` | 13,388 (BRCA + HCC1806) |

**Processing:**
1. Filtered: `TCGA_DESC == 'BRCA'` OR `CELL_LINE_NAME == 'HCC1806'` (triple-negative BRCA, classified as UNCLASSIFIED)
2. Column rename: `DATASET→gdsc_version`, `CELL_LINE_NAME→cell_line_name`, `DRUG_NAME→drug_name`, `LN_IC50→ln_IC50`

**Result:** 13,388 rows, 52 cell lines, 295 drugs

**GDSC2만 사용 (GDSC1 미포함) 사유:**
- GDSC2가 GDSC1 대비 실험 품질 우수 (newer screens, better curve fitting)
- ADMET 필터링(Step 7)으로 어차피 후보 약물 축소되므로 품질 우선 결정
- curated_date에 GDSC2만 전처리되어 있음 (GDSC1은 oringinal_raw에 xlsx 원본만 존재)

**Binary label (computed at FE stage, not stored here):**
- `ln_IC50 <= quantile(0.3)` → sensitive (1)
- quantile(0.3) = 2.2416 for GDSC2 BRCA dataset

### 3. lincs_mcf7.parquet (generated)

**NOT a direct copy.** Extracted from Level 5 GCTx matrix:

| Component | Source |
|-----------|--------|
| MCF7 sig_ids | `curated_date/lincs/lincs_sig_info_basic_20260406.parquet` (cell_id == 'MCF7') |
| Cell line validation | `curated_date/lincs/lincs_cell_info_basic_20260406.parquet` |
| Expression z-scores | `oringinal_raw/lincs/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz` |

**Processing:**
1. Identified 63,367 MCF7 signatures from curated sig_info
2. Extracted corresponding rows from Level 5 GCTx matrix (473,647 x 12,328 genes)
3. Saved as parquet with sig_id index + 12,328 gene columns (Entrez IDs)

**Result:** 63,367 signatures x 12,328 genes (z-score expression values)

**Why oringinal_raw:** curated_date/lincs/ contains only metadata tables (sig_info, cell_info, etc.). The actual gene expression z-score matrix is only available in the Level 5 GCTx file in oringinal_raw/.

## Data NOT included

| Item | Reason |
|------|--------|
| `glue/` | Other team member's area — access forbidden |
| `gdsc_binary.parquet` | Binary label computed at FE stage using quantile(0.3) |
| `id_mapping.parquet` | Generated internally by FE pipeline |

## Config

- `config/data_paths.yaml` — S3 paths for all data files
- `binary_quantile: 0.3` — IC50 sensitivity threshold
