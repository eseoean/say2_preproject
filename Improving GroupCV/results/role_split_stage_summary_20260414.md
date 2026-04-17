## Role-Split Stage Summary (2026-04-14)

Current comparison basis:
- local-only execution under `Improving GroupCV`
- default `3-fold drug GroupCV`
- same current numeric baseline input
- reconstructed context added from local caches only

### Variants

- `baseline_numeric`
  - current common input only
- `reconstructed_context_full`
  - all 11 context columns added as one categorical pack
- `role_split_context_full`
  - semantic categorical:
    - `TCGA_DESC`
    - `PATHWAY_NAME_NORMALIZED`
    - `classification`
  - auxiliary context:
    - `drug_bridge_strength`
    - `stage3_resolution_status`
    - `WEBRELEASE`
    - `drugbank_match_rule`
    - `chembl_match_rule`
    - `lincs_match_rule`
    - `admet_match_rule`
    - `cell_bridge_match_rule`

### ResidualMLP

| Variant | Spearman | RMSE | MAE | NDCG@20 |
| --- | ---: | ---: | ---: | ---: |
| baseline_numeric | 0.3131 | 2.4406 | 1.8382 | 0.7469 |
| reconstructed_context_full | 0.3253 | 2.4301 | 1.8228 | 0.7447 |
| role_split_context_full | 0.3070 | 2.4486 | 1.8177 | 0.7313 |

Interpretation:
- simple reconstructed context improved slightly over baseline
- role split did not help ResidualMLP in this stage

### FTTransformer

| Variant | Spearman | RMSE | MAE | NDCG@20 |
| --- | ---: | ---: | ---: | ---: |
| baseline_numeric | 0.3367 | 2.4471 | 1.8272 | 0.7685 |
| reconstructed_context_full | 0.3650 | 2.3821 | 1.7957 | 0.7713 |
| role_split_context_full | 0.3602 | 2.4172 | 1.8170 | 0.7643 |

Interpretation:
- both context variants improved over baseline
- for FTTransformer, plain reconstructed-context packing currently outperformed role split
- role split still beat baseline, but not reconstructed_context_full

### Current takeaway

- role separation is not automatically better
- model architecture matters:
  - `ResidualMLP`: role split hurt
  - `FTTransformer`: role split helped vs baseline but lost to simple reconstructed context
- next likely high-value target:
  - `TabTransformer`
  - then `WideDeep`
