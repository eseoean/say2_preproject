# OOF Metrics and Ensemble Diversity Report

세 가지 입력셋 기준으로 ML, DL, Ensemble 성능을 같은 형식으로 정리했다. 기본 정렬 기준은 `Spearman` 내림차순이다.

## A. exact slim (numeric-only)

### ML

| Model | Spearman | RMSE | MAE | Pearson | R² | NDCG@20 | Time | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBM | 0.5586 | 2.2100 | 1.6465 | 0.5891 | 0.3329 | 0.8312 | 3.0m | OOF |
| XGBoost | 0.5546 | 2.2153 | 1.6473 | 0.5829 | 0.3297 | 0.8376 | 6.1m | OOF |
| LightGBM_DART | 0.5495 | 2.1984 | 1.6653 | 0.5892 | 0.3398 | 0.8548 | 2.4m | OOF |
| RandomForest | 0.5494 | 2.2265 | 1.6545 | 0.5744 | 0.3229 | 0.8852 | 0.4m | OOF |
| CatBoost | 0.5481 | 2.1947 | 1.6327 | 0.5849 | 0.3420 | 0.8633 | 23.8m | OOF |
| ExtraTrees | 0.5000 | 2.2700 | 1.6755 | 0.5443 | 0.2961 | 0.8855 | 0.6m | OOF |

### DL

| Model | Spearman | RMSE | MAE | Pearson | R² | NDCG@20 | Time | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WideDeep | 0.5568 | 2.1979 | 1.6544 | 0.5862 | 0.3372 | - | 0.2m | legacy fold-mean |
| CrossAttention | 0.5439 | 2.2344 | 1.6477 | 0.5704 | 0.3155 | - | 0.5m | legacy fold-mean |
| FlatMLP | 0.5393 | 2.2695 | 1.6890 | 0.5685 | 0.2935 | - | 0.5m | legacy fold-mean |
| TabNet | 0.5332 | 2.3152 | 1.6811 | 0.5640 | 0.2643 | - | 0.2m | legacy fold-mean |
| FTTransformer | 0.5305 | 2.2327 | 1.6351 | 0.5683 | 0.3161 | - | 0.6m | legacy fold-mean |
| ResidualMLP | 0.5141 | 2.3194 | 1.7092 | 0.5496 | 0.2607 | - | 0.4m | legacy fold-mean |
| TabTransformer | 0.4323 | 2.3862 | 1.8134 | 0.4961 | 0.2186 | - | 3.8m | legacy fold-mean |

### Ensemble

| Type | 구성 | Spearman | RMSE | MAE | Pearson | R² | NDCG@20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Weighted | WideDeep + CrossAttention + FlatMLP | 0.5636 | 2.1868 | 1.6215 | 0.5892 | 0.3468 | - |
| Equal | WideDeep + CrossAttention + FlatMLP | 0.5634 | 2.1872 | 1.6214 | 0.5890 | 0.3466 | - |

### Diversity

| Pair | Pred Pearson | Pred Spearman | Resid Pearson | Resid Spearman | Mean Abs Gap |
| --- | --- | --- | --- | --- | --- |
| WideDeep vs CrossAttention | 0.9252 | 0.9076 | 0.9615 | 0.9429 | 0.4762 |
| WideDeep vs FlatMLP | 0.9097 | 0.8584 | 0.9397 | 0.9151 | 0.6235 |
| CrossAttention vs FlatMLP | 0.8666 | 0.8262 | 0.9128 | 0.8812 | 0.6816 |
| AVG | 0.9005 | 0.8641 | 0.9380 | 0.9130 | 0.5938 |

### Notes

- ML은 OOF 로컬 재실행 기준입니다. DL은 기존 numeric-only GroupCV 요약본을 사용했습니다.
- Ensemble diversity는 `WideDeep + CrossAttention + FlatMLP` 기준입니다.

## B. exact slim + SMILES

### ML

| Model | Spearman | RMSE | MAE | Pearson | R² | NDCG@20 | Time | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CatBoost | 0.5683 | 2.1573 | 1.6209 | 0.6047 | 0.3643 | 0.8501 | 22.6m | OOF |
| LightGBM | 0.5632 | 2.1379 | 1.6408 | 0.6142 | 0.3757 | 0.8584 | 4.9m | OOF |
| LightGBM_DART | 0.5586 | 2.1526 | 1.6694 | 0.6121 | 0.3671 | 0.8579 | 2.7m | OOF |
| RandomForest | 0.5573 | 2.2168 | 1.6640 | 0.5859 | 0.3288 | 0.8755 | 1.1m | OOF |
| XGBoost | 0.5482 | 2.1596 | 1.6655 | 0.6041 | 0.3629 | 0.8550 | 17.4m | OOF |
| ExtraTrees | 0.5124 | 2.2481 | 1.6697 | 0.5575 | 0.3097 | 0.8887 | 1.3m | OOF |

### DL

| Model | Spearman | RMSE | MAE | Pearson | R² | NDCG@20 | Time | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FlatMLP | 0.5899 | 2.0441 | 1.5845 | 0.6620 | 0.4292 | 0.8238 | 9.9m | OOF |
| CrossAttention | 0.5767 | 2.1395 | 1.6123 | 0.6125 | 0.3748 | 0.8236 | 11.7m | OOF |
| ResidualMLP | 0.5588 | 2.1519 | 1.6302 | 0.6077 | 0.3675 | 0.8669 | 9.7m | OOF |
| TabNet | 0.5542 | 2.1323 | 1.6319 | 0.6163 | 0.3790 | 0.8310 | 7.3m | OOF |
| FTTransformer | 0.5523 | 2.1487 | 1.6093 | 0.6084 | 0.3694 | 0.7617 | 10.0m | OOF |
| WideDeep | 0.5500 | 2.1622 | 1.6239 | 0.6025 | 0.3614 | 0.8442 | 13.0m | OOF |
| TabTransformer | 0.2454 | 3.0380 | 2.3774 | 0.2751 | -0.2607 | 0.8346 | 10.6m | OOF, 2-fold early stop |

### Ensemble

| Type | 구성 | Spearman | RMSE | MAE | Pearson | R² | NDCG@20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Weighted | FlatMLP + ResidualMLP + CrossAttention | 0.6067 | 2.0480 | 1.5593 | 0.6543 | 0.4271 | 0.8458 |
| Equal | FlatMLP + ResidualMLP + CrossAttention | 0.6059 | 2.0505 | 1.5608 | 0.6532 | 0.4257 | 0.8495 |

### Diversity

| Pair | Pred Pearson | Pred Spearman | Resid Pearson | Resid Spearman | Mean Abs Gap |
| --- | --- | --- | --- | --- | --- |
| FlatMLP vs ResidualMLP | 0.8819 | 0.8192 | 0.9005 | 0.8850 | 0.7258 |
| FlatMLP vs CrossAttention | 0.8586 | 0.8071 | 0.8787 | 0.8680 | 0.8069 |
| ResidualMLP vs CrossAttention | 0.8775 | 0.8341 | 0.9297 | 0.9262 | 0.6140 |
| AVG | 0.8727 | 0.8202 | 0.9029 | 0.8931 | 0.7156 |

### Notes

- TabTransformer는 조기 종료 규칙으로 2-fold만 실행되었습니다.
- Ensemble diversity는 `FlatMLP + ResidualMLP + CrossAttention` 기준입니다.

## C. exact slim + strong context + SMILES

### ML

| Model | Spearman | RMSE | MAE | Pearson | R² | NDCG@20 | Time | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CatBoost | 0.5742 | 2.1415 | 1.6170 | 0.6130 | 0.3736 | 0.8413 | 20.6m | OOF |
| LightGBM_DART | 0.5700 | 2.1410 | 1.6593 | 0.6179 | 0.3739 | 0.8469 | 2.7m | OOF |
| RandomForest | 0.5696 | 2.2129 | 1.6597 | 0.5891 | 0.3311 | 0.8812 | 1.5m | OOF |
| XGBoost | 0.5614 | 2.1463 | 1.6505 | 0.6118 | 0.3708 | 0.8482 | 15.5m | OOF |
| LightGBM | 0.5585 | 2.1555 | 1.6457 | 0.6067 | 0.3654 | 0.8515 | 5.1m | OOF |
| ExtraTrees | 0.5207 | 2.2431 | 1.6634 | 0.5605 | 0.3127 | 0.8848 | 1.8m | OOF |

### DL

| Model | Spearman | RMSE | MAE | Pearson | R² | NDCG@20 | Time | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FlatMLP | 0.6052 | 2.0515 | 1.5608 | 0.6554 | 0.4208 | 0.8037 | 9.5m | OOF-ish standalone summary |
| ResidualMLP | 0.6019 | 2.1396 | 1.6134 | 0.6138 | 0.3747 | 0.8419 | 15.6m | OOF |
| CrossAttention | 0.5931 | 2.1455 | 1.5895 | 0.6132 | 0.3680 | 0.7836 | 8.8m | OOF-ish standalone summary |
| WideDeep | 0.5707 | 2.1674 | 1.6246 | 0.6025 | 0.3551 | 0.8252 | 10.0m | OOF-ish standalone summary |
| TabNet | 0.5535 | 2.1697 | 1.6231 | 0.5977 | 0.3569 | 0.8465 | 8.3m | OOF |
| FTTransformer | 0.5532 | 2.1923 | 1.6389 | 0.5884 | 0.3435 | 0.8118 | 12.9m | OOF |
| TabTransformer | 0.2901 | 2.9713 | 2.3330 | 0.3414 | -0.2059 | 0.8203 | 15.2m | OOF |

### Ensemble

| Type | 구성 | Spearman | RMSE | MAE | Pearson | R² | NDCG@20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Equal | FlatMLP + ResidualMLP + CrossAttention | 0.6280 | 2.0223 | 1.5297 | 0.6645 | 0.4414 | 0.8281 |
| Weighted | FlatMLP + ResidualMLP + CrossAttention | 0.6280 | 2.0220 | 1.5296 | 0.6646 | 0.4415 | 0.8282 |
| Weighted | FlatMLP + WideDeep + CrossAttention | 0.6179 | 2.0568 | 1.5430 | 0.6506 | 0.4221 | 0.8356 |
| Equal | FlatMLP + WideDeep + CrossAttention | 0.6175 | 2.0584 | 1.5439 | 0.6499 | 0.4212 | 0.8356 |
| Weighted | FlatMLP + LightGBM_DART + ExtraTrees | 0.6132 | 2.0606 | 1.5707 | 0.6536 | 0.4200 | 0.8422 |
| Equal | FlatMLP + LightGBM_DART + ExtraTrees | 0.6121 | 2.0659 | 1.5736 | 0.6515 | 0.4170 | 0.8426 |

### Diversity

| Pair | Pred Pearson | Pred Spearman | Resid Pearson | Resid Spearman | Mean Abs Gap |
| --- | --- | --- | --- | --- | --- |
| [FRC ensemble] FlatMLP vs ResidualMLP | 0.8856 | 0.8320 | 0.9065 | 0.8843 | 0.6934 |
| FRC ensemble :: FlatMLP vs CrossAttention | 0.8519 | 0.8045 | 0.8805 | 0.8548 | 0.8100 |
| FRC ensemble :: ResidualMLP vs CrossAttention | 0.8942 | 0.8669 | 0.9217 | 0.9038 | 0.6384 |
| [FRC ensemble] AVG | 0.8773 | 0.8345 | 0.9029 | 0.8810 | 0.7140 |
| --- | --- | --- | --- | --- | --- |
| [FLE mixed ensemble] FlatMLP vs LightGBM_DART | 0.8457 | 0.7798 | 0.8799 | 0.8550 | 0.7951 |
| FLE mixed ensemble :: FlatMLP vs ExtraTrees | 0.8459 | 0.7624 | 0.8853 | 0.8633 | 0.7981 |
| FLE mixed ensemble :: LightGBM_DART vs ExtraTrees | 0.8695 | 0.8060 | 0.9381 | 0.9271 | 0.5865 |
| [FLE mixed ensemble] AVG | 0.8537 | 0.7827 | 0.9011 | 0.8818 | 0.7266 |

### Notes

- DL 추가 OOF 재실행 결과를 반영한 최신본입니다.
- Ensemble에는 `FlatMLP + ResidualMLP + CrossAttention`와 `FlatMLP + LightGBM_DART + ExtraTrees`를 함께 반영했습니다.
- Diversity 표는 `FRC ensemble`과 `FLE mixed ensemble`을 한 섹션에서 같이 보여줍니다.
