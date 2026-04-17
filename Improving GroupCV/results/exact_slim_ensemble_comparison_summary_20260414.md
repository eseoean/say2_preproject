# Exact Slim Ensemble Comparison

## Setup

- Input: exact repo-matched slim numeric input
- Context: strong context columns added
  - `TCGA_DESC`
  - `PATHWAY_NAME_NORMALIZED`
  - `classification`
  - `drug_bridge_strength`
  - `stage3_resolution_status`
- Split: 3-fold GroupKFold by `canonical_drug_id`

## Compared Ensembles

### Top-3 Weighted Ensemble

- Models
  - `FlatMLP`
  - `WideDeep`
  - `CrossAttention`
- Result
  - Spearman: `0.5866`
  - RMSE: `2.1322`
  - MAE: `1.5835`
  - NDCG@20: `0.8576`

### Top-5 Weighted Ensemble

- Models
  - `FlatMLP`
  - `WideDeep`
  - `CrossAttention`
  - `TabNet`
  - `FTTransformer`
- Result
  - Spearman: `0.5853`
  - RMSE: `2.1408`
  - MAE: `1.5890`
  - NDCG@20: `0.8381`

## Delta: Top-5 minus Top-3

- Spearman: `-0.0013`
- RMSE: `+0.0086`
- MAE: `+0.0055`
- NDCG@20: `-0.0195`

## Interpretation

- Adding more models did not improve the ensemble.
- The best three models already captured most of the useful signal in this setting.
- `TabNet` and `FTTransformer` likely added diversity, but not enough quality, so their predictions diluted the stronger base models.
- This is consistent with the single-model rankings under the same strong-context setting:
  - `FlatMLP`, `WideDeep`, and `CrossAttention` formed the strongest group.
  - `TabNet` and `FTTransformer` were weaker and more expensive.

## Recommendation

- Use the top-3 weighted ensemble as the current best GroupCV configuration.
- Keep the top-5 result as a negative control showing that "more models" did not automatically improve performance.
