# Current Numeric-Only GroupCV Baseline

기준

- split: `drug GroupCV`
- input: current common table numeric-only
- source: `features + pair_features -> numeric columns only`

| Model | Spearman | RMSE | Pearson | R2 | Train Sp | Gap Sp | Time(min) | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| XGBoost | 0.5012 | 2.2246 | 0.5341 | 0.2662 | 0.9069 | 0.4057 | 21.0 | done |
| LightGBM | 0.4701 | 2.2663 | 0.4984 | 0.2382 | 0.8765 | 0.4064 | 8.1 | done |
| Cross-Attention | 0.3895 | 2.2669 | 0.4956 | 0.2319 | 0.7743 | 0.3848 | 0.8 | done |
| FlatMLP | 0.3599 | 2.3301 | 0.4466 | 0.1906 | 0.8201 | 0.4602 | 3.5 | done |
| ResidualMLP | 0.3565 | 2.3505 | 0.4266 | 0.1752 | 0.7148 | 0.3584 | 1.4 | done |
| CatBoost | - | - | - | - | - | - | - | running at capture time |

메모

- 현재 기준선은 "같은 입력에서 split만 `drug GroupCV`로 엄격하게 바꾼" 결과다.
- 이후 개선 실험은 이 표 대비 delta를 계산한다.
- CatBoost는 실행 완료 후 같은 형식으로 업데이트한다.
