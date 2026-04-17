# Step 7+ Frontend Integration Plan

## Goal
- 팀원 레포 흐름에 맞춰 `Step 7+ KG/API` 결과를 프론트엔드에 자연스럽게 연결한다.
- 현재 best route인 `FRC (FlatMLP + ResidualMLP + CrossAttention)` 기준 최종 후보, METABRIC, ADMET, KG/API 근거를 한 번에 보여준다.

## Recommended UI shape

### 1. New result page
- Recommended route: `/step7plus` or `/results/final-candidates`
- Primary content:
  - Ensemble / input bundle summary cards
  - Approved / Candidate / Caution counts
  - Top 15 final candidate table
  - KG/API evidence summary table

### 2. Drug detail page extension
- Existing `/drugs` page already exposes:
  - `targets`
  - `pathways`
  - `side_effects`
  - `trials`
- Extend with tabs:
  - `pubmed`
  - `validation`
  - `final gate`

## Local data sources
- Step 6 metrics:
  - `/Users/skku_aws2_18/pre_project/say2_preproject/models/metabric_results_frc_strong_context_smiles/step6_metabric_results.json`
- Step 7 results:
  - `/Users/skku_aws2_18/pre_project/say2_preproject/models/admet_results_frc_strong_context_smiles/step7_admet_results.json`
- Top 15 summary:
  - `/Users/skku_aws2_18/pre_project/say2_preproject/models/post_admet_summary_frc_strong_context_smiles/top15_comprehensive_table.csv`
- KG/API summary:
  - `/Users/skku_aws2_18/pre_project/say2_preproject/models/kg_api_results_frc_strong_context_smiles/kg_api_summary.csv`

## API shape to mirror
- `/api/drug/{name}`
- `/api/drug/{name}/targets`
- `/api/drug/{name}/pathways`
- `/api/drug/{name}/side_effects`
- `/api/drug/{name}/trials`
- `/api/pubmed?query=...`

## Best path from current repo state
- Current workspace contains static HTML outputs, not the deployed React/Vite source.
- Therefore:
  1. keep `step7plus_detail.html` as the local presentation artifact
  2. when the real frontend source repo is available, port the same sections into:
     - a new route page
     - the existing `/drugs` tabs

## Suggested component split for a React app
- `FinalCandidateSummaryCards`
- `FinalCandidateTable`
- `KgApiEvidenceTable`
- `DrugEvidenceTabs`
- `CategoryBadge`

## Minimum migration order
1. Add route page for final candidates
2. Add dashboard CTA card linking to Step 7+
3. Add `pubmed`, `validation`, `final gate` tabs in `/drugs`
