# FLE Mixed Ensemble Report

입력셋: `exact slim + strong context + SMILES`  
모델 조합: `FlatMLP + LightGBM_DART + ExtraTrees`

## Base Models

| Model | Spearman | RMSE | MAE | Pearson | R² | NDCG@20 | Time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FlatMLP | 0.6007 | 2.0346 | 1.5541 | 0.6615 | 0.4346 | 0.8410 | -m |
| LightGBM_DART | 0.5700 | 2.1410 | 1.6593 | 0.6179 | 0.3739 | 0.8469 | 1.3225m |
| ExtraTrees | 0.5252 | 2.2396 | 1.6597 | 0.5624 | 0.3148 | 0.8876 | 0.4059m |

## Ensemble

| Type | Spearman | RMSE | MAE | Pearson | R² | NDCG@20 |
| --- | --- | --- | --- | --- | --- | --- |
| Equal | 0.6121 | 2.0659 | 1.5736 | 0.6515 | 0.4170 | 0.8426 |
| Weighted | 0.6132 | 2.0606 | 1.5707 | 0.6536 | 0.4200 | 0.8422 |

## Weights

| Model | Weight |
| --- | --- |
| FlatMLP | 0.3542 |
| LightGBM_DART | 0.3361 |
| ExtraTrees | 0.3097 |

## Diversity

| Pair | Pred Pearson | Pred Spearman | Resid Pearson | Resid Spearman | Mean Abs Gap |
| --- | --- | --- | --- | --- | --- |
| FlatMLP vs LightGBM_DART | 0.8457 | 0.7798 | 0.8799 | 0.8550 | 0.7951 |
| FlatMLP vs ExtraTrees | 0.8459 | 0.7624 | 0.8853 | 0.8633 | 0.7981 |
| LightGBM_DART vs ExtraTrees | 0.8695 | 0.8060 | 0.9381 | 0.9271 | 0.5865 |

## Diversity Summary

| Metric | Value |
| --- | --- |
| Avg prediction Pearson | 0.8537 |
| Avg prediction Spearman | 0.7827 |
| Avg residual Pearson | 0.9011 |
| Avg residual Spearman | 0.8818 |
| Avg mean abs prediction gap | 0.7266 |

## Gain Vs Best Base

| Metric | Delta |
| --- | --- |
| Weighted Spearman gain | 0.0124 |
| Weighted RMSE gain | 0.0260 |
