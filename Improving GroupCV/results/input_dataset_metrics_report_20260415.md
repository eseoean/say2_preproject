# Input Dataset Metrics Report

기준 날짜: 2026-04-15

이 문서는 현재 저장된 GroupCV 결과를 입력셋 3종 기준으로 묶은 요약 문서입니다.

## Input Variants

| Variant | 설명 | ML 입력 | DL 입력 |
| --- | --- | --- | --- |
| A | `exact slim (numeric-only)` | `5529` numeric features | `5529` numeric features |
| B | `exact slim + SMILES` | `5529 numeric + 64 SMILES SVD = 5593` | `5529 numeric + SMILES branch` |
| C | `exact slim + strong context + SMILES` | `5529 numeric + 32 context one-hot + 64 SMILES SVD = 5625` | `5529 numeric + 5 categorical context + SMILES branch` |

`-` 표시는 해당 지표가 저장되지 않았거나 해당 조합이 실행되지 않았음을 뜻합니다.

## A. exact slim (numeric-only)

### DL

| Model | Spearman | RMSE | MAE | Pearson | R² | NDCG@20 | Time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WideDeep | 0.5568 | 2.1979 | 1.6544 | 0.5862 | 0.3372 | - | 0.2m |
| CrossAttention | 0.5439 | 2.2344 | 1.6477 | 0.5704 | 0.3155 | - | 0.5m |
| FlatMLP | 0.5393 | 2.2695 | 1.6890 | 0.5685 | 0.2935 | - | 0.5m |
| TabNet | 0.5332 | 2.3152 | 1.6811 | 0.5640 | 0.2643 | - | 0.2m |
| FTTransformer | 0.5305 | 2.2327 | 1.6351 | 0.5683 | 0.3161 | - | 0.6m |
| ResidualMLP | 0.5141 | 2.3194 | 1.7092 | 0.5496 | 0.2607 | - | 0.4m |
| TabTransformer | 0.4323 | 2.3862 | 1.8134 | 0.4961 | 0.2186 | - | 3.8m |

### Ensemble

| Ensemble | 구성 | Spearman | RMSE | MAE | Pearson | R² | NDCG@20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Local top3 weighted | WideDeep + CrossAttention + FlatMLP | 0.5636 | 2.1868 | 1.6215 | 0.5892 | 0.3468 | - |
| Saved 6-model weighted | 1/2/4/10/12/13 weighted | 0.5483 | 2.2171 | 1.6641 | 0.5784 | 0.3286 | - |

### ML

현재 워크스페이스에는 `exact slim (numeric-only)` 조건의 ML full rerun 저장본이 없습니다.

## B. exact slim + SMILES

### ML

| Model | Spearman | RMSE | MAE | Pearson | R² | NDCG@20 | Time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CatBoost | 0.5683 | 2.1573 | 1.6209 | 0.6047 | 0.3643 | 0.8501 | 22.6m |
| LightGBM | 0.5632 | 2.1379 | 1.6408 | 0.6142 | 0.3757 | 0.8584 | 4.9m |
| LightGBM_DART | 0.5586 | 2.1526 | 1.6694 | 0.6121 | 0.3671 | 0.8579 | 2.7m |
| RandomForest | 0.5573 | 2.2168 | 1.6640 | 0.5859 | 0.3288 | 0.8755 | 1.1m |
| XGBoost | 0.5482 | 2.1596 | 1.6655 | 0.6041 | 0.3629 | 0.8550 | 17.4m |
| ExtraTrees | 0.5124 | 2.2481 | 1.6697 | 0.5575 | 0.3097 | 0.8887 | 1.3m |

### DL

| Model | Spearman | RMSE | MAE | Pearson | R² | NDCG@20 | Time | 비고 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FlatMLP | 0.5941 | 2.0441 | 1.5845 | 0.6604 | 0.4252 | 0.8147 | 9.9m |  |
| CrossAttention | 0.5795 | 2.1378 | 1.6123 | 0.6146 | 0.3723 | 0.8029 | 11.7m |  |
| ResidualMLP | 0.5642 | 2.1514 | 1.6302 | 0.6071 | 0.3648 | 0.8203 | 9.7m |  |
| TabNet | 0.5618 | 2.1321 | 1.6319 | 0.6194 | 0.3757 | 0.8201 | 7.3m |  |
| FTTransformer | 0.5540 | 2.1481 | 1.6093 | 0.6071 | 0.3666 | 0.7968 | 10.0m |  |
| WideDeep | 0.5506 | 2.1611 | 1.6239 | 0.6014 | 0.3595 | 0.7988 | 13.0m |  |
| TabTransformer | 0.4446 | 2.3666 | 1.7701 | 0.5234 | 0.2601 | 0.8018 | 10.6m | 2-fold early stop |

### Ensemble

| Ensemble | 구성 | Spearman | RMSE | MAE | Pearson | R² | NDCG@20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FRC weighted | FlatMLP + ResidualMLP + CrossAttention | 0.6067 | 2.0480 | 1.5593 | 0.6543 | 0.4271 | 0.8458 |

## C. exact slim + strong context + SMILES

### ML

| Model | Spearman | RMSE | MAE | Pearson | R² | NDCG@20 | Time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CatBoost | 0.5742 | 2.1415 | 1.6170 | 0.6130 | 0.3736 | 0.8413 | 20.6m |
| LightGBM_DART | 0.5700 | 2.1410 | 1.6593 | 0.6179 | 0.3739 | 0.8469 | 2.7m |
| RandomForest | 0.5696 | 2.2129 | 1.6597 | 0.5891 | 0.3311 | 0.8812 | 1.5m |
| XGBoost | 0.5614 | 2.1463 | 1.6505 | 0.6118 | 0.3708 | 0.8482 | 15.5m |
| LightGBM | 0.5585 | 2.1555 | 1.6457 | 0.6067 | 0.3654 | 0.8515 | 5.1m |
| ExtraTrees | 0.5207 | 2.2431 | 1.6634 | 0.5605 | 0.3127 | 0.8848 | 1.8m |

### DL

| Model | Spearman | RMSE | MAE | Pearson | R² | NDCG@20 | Time | 비고 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FlatMLP | 0.6052 | 2.0515 | 1.5608 | 0.6554 | 0.4208 | 0.8037 | 9.5m | standalone |
| ResidualMLP | 0.6041 | 2.1166 | 1.5960 | 0.6247 | 0.3849 | 0.7974 | - | base run from FRC ensemble |
| CrossAttention | 0.5931 | 2.1455 | 1.5895 | 0.6132 | 0.3680 | 0.7836 | 8.8m | standalone |
| WideDeep | 0.5707 | 2.1674 | 1.6246 | 0.6025 | 0.3551 | 0.8252 | 10.0m | standalone |
| FTTransformer | 0.5550 | 2.1911 | 1.6389 | - | - | 0.8223 | - | log summary only |
| TabNet | 0.5543 | 2.1695 | 1.6232 | - | - | 0.8163 | - | log summary only |
| TabTransformer | - | - | - | - | - | - | - | not run / not saved |

### Ensemble

| Ensemble | 구성 | Spearman | RMSE | MAE | Pearson | R² | NDCG@20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FRC weighted | FlatMLP + ResidualMLP + CrossAttention | 0.6280 | 2.0220 | 1.5296 | 0.6646 | 0.4415 | 0.8282 |
| Top3 weighted | FlatMLP + WideDeep + CrossAttention | 0.6179 | 2.0568 | 1.5430 | 0.6506 | 0.4221 | 0.8356 |

## Notes

- `exact slim + SMILES`의 `TabTransformer`는 사용자 지정 조기 종료 규칙으로 2개 fold만 실행되었습니다.
- `exact slim + strong context + SMILES`의 `TabNet`, `FTTransformer`는 이전 실행 로그 요약값만 남아 있어 `Pearson`, `R²`, `Time`은 `-`로 표시했습니다.
- `exact slim + strong context + SMILES`에서 현재 best ensemble은 `FlatMLP + ResidualMLP + CrossAttention` weighted 입니다.
