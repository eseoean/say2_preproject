# Improving GroupCV

이 폴더는 `drug GroupCV` 일반화 성능 개선 실험을 위한 전용 작업 공간이다.

핵심 원칙

- 출발점은 현재 파이프라인의 공통 입력 방식이다.
- 즉, `features + pair_features`를 합친 뒤 numeric 컬럼만 뽑아 `X`를 만드는 현재 구조를 baseline으로 둔다.
- 이후 지침서의 아이디어를 참고해 입력 표현을 단계적으로 확장한다.
- 목표는 "완전히 다른 입력 체계로 갈아타기"가 아니라 "현재 입력에서 출발해 어떤 변환이 GroupCV를 개선하는지"를 확인하는 것이다.

실험 방향

- Baseline:
  - 현재 numeric-only common table
- Variant 1:
  - baseline numeric feature 유지
  - categorical/context feature를 추가 보존
- Variant 2:
  - numeric + categorical hybrid representation
  - role-aware encoding 적용
- Variant 3:
  - 모델 구조별 입력 분기 최적화

모델 범위

- 기존 사용 모델:
  - ResidualMLP
  - FlatMLP
  - TabNet
  - FTTransformer
  - Cross-Attention
- 추가 guideline 반영 모델:
  - TabTransformer
  - Wide&Deep

평가 원칙

- 기존에 사용하던 지표 유지:
  - Spearman
  - RMSE
  - Pearson
  - R2
  - Train Spearman
  - Gap Spearman
- 추가 지표:
  - MAE
  - NDCG@20
- GroupCV는 fold별 결과와 mean/std를 함께 저장한다.
- 현재 기본 실행 단위는 `3-fold drug GroupCV`다.

폴더 구조

- `results/`: 실험 결과 요약 및 모델별 산출물
- `scripts/`: 개선 실험용 러너와 유틸
- `notes/`: 실험 메모, 해석, 가설
