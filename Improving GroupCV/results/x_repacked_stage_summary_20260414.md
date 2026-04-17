## X-Repacked Stage Summary (2026-04-14)

Experiment intent:
- do not add external context
- do not replace the current dataset
- rebuild the current numeric X into a smaller, block-aware representation

Repacked numeric design:
- input source is still the current common numeric X
- per train fold:
  - `sample__crispr__*` -> TruncatedSVD 256 dims + 5 row summary stats
  - `drug_morgan_*` -> TruncatedSVD 96 dims + 5 row summary stats
  - smaller drug/target/LINCS blocks -> passthrough
- resulting numeric dimension:
  - `388`
  - `256 sample svd + 5 sample summary + 96 drug svd + 5 drug summary + 26 passthrough`

### FTTransformer

| Variant | Spearman | RMSE | MAE | NDCG@20 |
| --- | ---: | ---: | ---: | ---: |
| baseline_numeric | 0.3367 | 2.4471 | 1.8272 | 0.7685 |
| reconstructed_context_full | 0.3650 | 2.3821 | 1.7957 | 0.7713 |
| role_split_context_full | 0.3602 | 2.4172 | 1.8170 | 0.7643 |
| x_repacked_blocksvd | 0.3793 | 2.4265 | 1.7992 | 0.7452 |

Interpretation:
- `x_repacked_blocksvd` gave the best Spearman among tested FTTransformer variants
- RMSE improved over baseline, but reconstructed context still had the best RMSE
- NDCG@20 decreased slightly versus baseline/reconstructed context

### ResidualMLP

| Variant | Spearman | RMSE | MAE | NDCG@20 |
| --- | ---: | ---: | ---: | ---: |
| baseline_numeric | 0.3131 | 2.4406 | 1.8382 | 0.7469 |
| reconstructed_context_full | 0.3253 | 2.4301 | 1.8228 | 0.7447 |
| role_split_context_full | 0.3070 | 2.4486 | 1.8177 | 0.7313 |
| x_repacked_blocksvd | 0.4646 | 2.3423 | 1.7261 | 0.8073 |

Interpretation:
- `x_repacked_blocksvd` clearly outperformed all previous ResidualMLP variants
- this suggests that changing the representation of X itself is much more effective than appending small context packs for MLP-like models

### Current takeaway

- for this project, changing X itself appears more promising than adding a small number of extra context columns
- best next candidates:
  - run `x_repacked_blocksvd` on `TabTransformer`
  - try a second repacked design with more aggressive drug-side compression or stronger sample summary statistics
