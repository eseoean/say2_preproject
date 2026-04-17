# Experiment Design

## Goal

현재 numeric-only 공통 입력 구조를 출발점으로 삼아, 지침서의 hybrid/context 아이디어를 점진적으로 반영했을 때 `drug GroupCV` 성능이 개선되는지 확인한다.

## Scope

이번 개선 실험의 모델 범위는 DL 계열로 둔다.

- 기존 모델:
  - ResidualMLP
  - FlatMLP
  - TabNet
  - FTTransformer
  - Cross-Attention
- 추가 모델:
  - TabTransformer
  - Wide&Deep

## Baseline

- 현재 파이프라인과 동일한 방식
- `features + pair_features` merge
- numeric columns only
- split은 `drug GroupCV`

## Progressive Variants

### Variant A

- baseline numeric feature 유지
- 지침서에서 강조한 categorical/context column만 추가
- categorical은 vocabulary 기반 정수 인코딩 후 embedding 처리

### Variant B

- Variant A + feature role 기반 분기
- numeric / categorical / auxiliary feature를 별도 처리 후 결합

### Variant C

- 모델별 입력 구조 최적화
- 예:
  - FTTransformer / TabTransformer: tokenized tabular input
  - Wide&Deep: wide numeric + deep categorical
  - Cross-Attention: sample branch와 context branch 분리

## Metrics

기존 지표 유지

- RMSE
- Spearman
- Pearson
- R2
- Train Spearman
- Gap Spearman

추가 지표

- MAE
- NDCG@20

GroupCV 결과는 fold별 raw metric과 mean/std를 모두 저장한다.

## Comparison Rule

- baseline과 variant는 같은 split, 같은 seed, 같은 label을 사용한다.
- delta는 항상 `variant - baseline` 기준으로 정리한다.
- RMSE/MAE는 낮을수록 좋고, Spearman/Pearson/R2/NDCG@20은 높을수록 좋다.
