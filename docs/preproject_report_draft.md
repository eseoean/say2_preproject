# 프리프로젝트 보고서 초안

작성일: 2026-04-10  
작성 기준: 현재 재현 완료 기준 Step 1~7 결과 반영, 프론트엔드는 진행 중  
저장소: `eseoean/say2_preproject`  
대시보드: `dashboard.html` 기준

---

## 01 프로젝트 개요

### 1.1 프로젝트명
유방암(BRCA) 약물 재창출 파이프라인 재현 및 후보 약물 발굴

### 1.2 프로젝트 배경
신약 개발은 긴 시간과 높은 비용이 요구되기 때문에, 기존 약물이나 임상 정보가 있는 물질을 새로운 적응증에 적용하는 약물 재창출 전략이 중요한 대안으로 활용되고 있다. 본 프로젝트는 팀원이 구축한 Drug Discovery Pipeline을 동일한 방법론으로 재현하고, 유방암 관련 공공 데이터와 머신러닝 기반 분석을 통해 재창출 후보 약물을 선별하는 것을 목표로 하였다.

### 1.3 프로젝트 목표
- 팀원의 프로토콜을 기준으로 동일한 분석 파이프라인을 재현한다.
- GDSC, DepMap, LINCS, ChEMBL, DrugBank, METABRIC, ADMET 데이터를 통합해 약물 반응 예측 모델을 구축한다.
- 여러 모델의 성능을 비교하고 앙상블을 통해 예측 안정성을 높인다.
- 외부 검증(METABRIC)과 ADMET 필터링을 통해 최종 후보 약물을 도출한다.

### 1.4 기대 효과
- 공공 바이오 데이터 기반 약물 재창출 워크플로우를 실습형으로 재현할 수 있다.
- ML/DL/Graph 모델 성능 비교 및 앙상블 전략의 효과를 확인할 수 있다.
- 향후 프론트엔드와 연결 가능한 결과 대시보드 구조를 확보할 수 있다.

---

## 02 프로젝트 팀 구성 및 역할

현재 팀 구성원은 총 4명이며, 아래 표는 보고서 초안 작성용 역할 정리본이다.  
이름은 최종 제출 전에 실제 팀원 이름으로 교체하면 된다.

| 구분 | 인원 | 담당 역할 | 현재 반영 상태 |
| --- | --- | --- | --- |
| 팀원 1 `[이름 입력]` | 1명 | 파이프라인 재현, AWS/Nextflow 실행, 모델 학습, 결과 정리 | 본 초안에 반영 |
| 팀원 2 `[이름 입력]` | 1명 | 프론트엔드 및 결과 시각화, 대시보드 연결 | 진행 중 |
| 팀원 3 `[이름 입력]` | 1명 | 데이터 조사 및 전처리 검토, 외부 검증 자료 정리 | 보고서에 세부 반영 가능 |
| 팀원 4 `[이름 입력]` | 1명 | 발표 자료 보조, 문서화, 결과 비교 및 QA | 보고서 후반 반영 가능 |

### 2.1 현재 기준 역할 요약
- 총 4인 팀으로 운영 중이다.
- 현재 재현 파이프라인은 Step 1~7까지 완료했다.
- 프론트엔드는 팀원이 작업 중이므로, 본 문서는 현재 대시보드와 산출물 기준의 중간 보고서 초안이다.

---

## 03 프로젝트 수행 절차 및 방법

### 3.1 전체 수행 절차

본 프로젝트는 다음 7단계로 진행하였다.

1. 환경 설정
2. 데이터 준비
3. Feature Engineering
4. 모델 학습
5. 앙상블
6. METABRIC 외부 검증
7. ADMET Gate 기반 최종 후보 선정

### 3.2 단계별 수행 방법

#### Step 1. 환경 설정
- Python 3.10 기반 `drug4` 환경을 구성하였다.
- Java 17, Nextflow, AWS CLI, ML/DL 라이브러리를 점검하였다.
- GitHub 저장소 및 AWS 계정 연결 상태를 확인하였다.

#### Step 2. 데이터 준비
- `curated_date/` 기반으로 재현용 데이터 경로를 구성하였다.
- GDSC는 `GDSC2`만 사용하였다.
- 생성 데이터는 다음과 같다.
  - `gdsc_ic50.parquet`: 13,388 rows / 52 cell lines / 295 drugs
  - `depmap_crispr_long`: 20,443,404 rows / 1,150 cell lines / 18,443 genes
  - `drug_features_catalog.parquet`: 243 / 295 matched (82.4%)
- LINCS는 빠른 재현 경로를 위해 reference의 `lincs_mcf7.parquet`를 사용하였다.

#### Step 3. Feature Engineering
- Nextflow + AWS Batch 기반으로 FE 파이프라인을 실행하였다.
- 총 실행 시간은 약 8분 7초였다.
- 주요 산출물:
  - `features.parquet`
  - `labels.parquet`
  - `pair_features_newfe.parquet`
  - `pair_lincs_features.parquet`
  - `pair_target_features.parquet`

#### Step 4. 모델 학습
- 5-fold CV 기준으로 총 15개 모델을 실행하였다.
- 모델 구성:
  - ML 8개
  - DL 5개
  - Graph 2개
- 앙상블 포함 기준:
  - Spearman ≥ 0.713
  - RMSE ≤ 1.385

#### Step 5. 앙상블
- 우선 6개 모델 기반 앙상블을 구성하였다.
- 이후 성능 저하가 매우 적은 경량형 3개 모델 앙상블도 별도로 구성하였다.
- 최종 운영 경로는 경량형 앙상블로 선택하였다.
  - 사용 모델: LightGBM + FlatMLP + Cross-Attention

#### Step 6. METABRIC 외부 검증
- Method A: 타깃 유전자 발현 확인
- Method B: 생존 분석 기반 유의성 검토
- Method C: Known Drug Precision(P@K) 평가
- 중복 약물은 `drug_name` 기준으로 제거하였다.

#### Step 7. ADMET Gate
- 총 22개 ADMET assay 기준으로 최종 후보 약물을 평가하였다.
- 후보를 `Approved`, `Candidate`, `Caution` 세 그룹으로 분류하였다.

### 3.3 사용 기술 및 환경

| 구분 | 내용 |
| --- | --- |
| 언어 | Python |
| 워크플로우 | Nextflow |
| 클라우드 | AWS Batch, S3 |
| 데이터 분석 | pandas, scikit-learn |
| ML | LightGBM, XGBoost, CatBoost |
| DL | PyTorch, TabNet, FT-Transformer |
| Graph | PyTorch Geometric |
| 검증 데이터 | METABRIC |
| 독성 평가 | ADMET benchmark task 22종 |

---

## 04 프로젝트 수행 경과

### 4.1 현재 진행 현황

| 단계 | 수행 내용 | 진행 상태 |
| --- | --- | --- |
| Step 1 | 환경 설정 | 완료 |
| Step 2 | 데이터 준비 | 완료 |
| Step 3 | Feature Engineering | 완료 |
| Step 4 | 모델 학습 | 완료 |
| Step 5 | 앙상블 | 완료 |
| Step 6 | METABRIC 외부 검증 | 완료 |
| Step 7 | ADMET Gate | 완료 |
| 프론트엔드 | UI 및 최종 시각화 | 진행 중 |

### 4.2 Step 4 모델 성능 요약

#### 4.2.1 성능 기준 통과 모델

| 구분 | 모델 | Spearman | RMSE | 비고 |
| --- | --- | --- | --- | --- |
| ML | CatBoost | 0.7997 | 1.3170 | 최고 ML 성능 |
| ML | LightGBM | 0.7913 | 1.3404 | 경량형 앙상블 포함 |
| ML | XGBoost | 0.7889 | 1.3427 | 6모델 앙상블 포함 |
| ML | Stacking Ridge | 0.7942 | 1.3265 | 메타 모델 |
| DL | FlatMLP | 0.7966 | 1.3327 | 최고 DL 성능 |
| DL | ResidualMLP | 0.7875 | 1.3765 | 6모델 앙상블 포함 |
| DL | Cross-Attention | 0.7854 | 1.3673 | 경량형 앙상블 포함 |

#### 4.2.2 참고 사항
- `TabNet`은 Spearman은 양호했지만 RMSE가 기준을 근소하게 초과하였다.
- `GraphSAGE`, `GAT`는 이번 재현에서는 회귀 성능이 낮아 최종 앙상블 후보에서는 제외하였다.
- `RSF`는 회귀 성능 대신 Step 6 생존 분석 보조 지표로 활용하였다.

### 4.3 Step 5 앙상블 결과

#### 4.3.1 6개 모델 앙상블
- Spearman: `0.8055 ± 0.0101`
- RMSE: `1.3008 ± 0.0227`
- Pearson: `0.8690`
- R²: `0.7544`

#### 4.3.2 경량형 3개 모델 앙상블
- Spearman: `0.8036 ± 0.0081`
- RMSE: `1.3033 ± 0.0158`
- Pearson: `0.8683`
- R²: `0.7535`

#### 4.3.3 해석
- 경량형 앙상블은 6개 모델 대비 성능 저하가 매우 작았다.
- Top15 약물 세트가 6개 모델 앙상블과 동일하였다.
- 운영 효율성과 재현 편의성을 고려해 경량형 앙상블을 최종 선택하였다.

### 4.4 Step 6 METABRIC 외부 검증 결과

경량형 기준과 6모델 기준 결과는 중복 제거 이후 거의 동일하였다.

| 항목 | 결과 |
| --- | --- |
| Target expressed | 27 / 28 |
| BRCA pathway relevant | 21 / 28 |
| Survival significant | 26 / 28 |
| RSF C-index | 0.8209 |
| RSF AUROC | 0.8462 |
| P@5 | 80.0% |
| P@10 | 90.0% |
| P@15 | 80.0% |
| P@20 | 65.0% |

### 4.5 Step 7 ADMET Gate 결과

#### 4.5.1 최종 분류 결과

| 분류 | 개수 |
| --- | --- |
| Approved | 8 |
| Candidate | 6 |
| Caution | 1 |

#### 4.5.2 경량형 기준 Top 5 후보

| 순위 | 약물명 | 분류 |
| --- | --- | --- |
| 1 | Romidepsin | Approved |
| 2 | Sepantronium bromide | Candidate |
| 3 | Staurosporine | Candidate |
| 4 | Dactinomycin | Approved |
| 5 | SN-38 | Candidate |

#### 4.5.3 해석
- 최종 후보는 총 15개 약물로 정리되었다.
- `Romidepsin`, `Sepantronium bromide`가 가장 높은 우선순위를 보였다.
- `Epirubicin`은 Ames 및 DILI 플래그로 인해 `Caution`으로 분류되었다.

### 4.6 결과 시각화 및 대시보드 현황
- 재현 결과 기준 메인 대시보드 작성 완료
- `dashboard.html`에서 Step 4~7 결과 비교 가능
- `dashboard_reference.html`에 팀원 참고 원본 보존
- 프론트엔드는 팀원이 추가 작업 중이며, 최종 결과물은 이후 통합 예정

---

## 05 자체 평가 의견

### 5.1 잘된 점
- 팀원 프로토콜을 기준으로 Step 1~7까지 전체 파이프라인 재현을 완료하였다.
- AWS Batch, Nextflow, ML/DL/Graph, METABRIC, ADMET까지 전 과정을 실제로 실행하였다.
- 6모델 앙상블과 경량형 앙상블을 모두 비교하여 성능과 운영 효율을 함께 고려하였다.
- 중복 약물 제거와 결과 비교 대시보드까지 반영하여 재현 결과를 문서화하기 쉬운 상태로 정리하였다.

### 5.2 아쉬운 점
- 프론트엔드가 아직 최종 완료 전이라 결과 시각화가 완전히 통합되지는 않았다.
- 팀원 실명, 세부 역할, 프론트엔드 최종 화면은 아직 보고서에 확정 반영되지 않았다.
- Graph 계열 모델은 이번 재현에서 기대한 수준의 회귀 성능을 확보하지 못했다.

### 5.3 개선 방향
- 프론트엔드 완료 후 현재 대시보드와 통합해 최종 결과 화면을 정리한다.
- 팀원별 실제 역할과 기여도를 반영하여 02장 내용을 최종 보정한다.
- 발표용 슬라이드에서는 경량형 앙상블 선택 이유와 Step 6/7 검증 근거를 중심으로 설명한다.
- 최종 보고서에는 대시보드 캡처, Top15 후보 표, Step 5 비교표를 이미지로 추가하면 전달력이 높아질 것이다.

### 5.4 최종 한 줄 평가
본 프로젝트는 팀원 프로토콜을 기반으로 약물 재창출 파이프라인을 실제로 재현하고, 외부 검증과 ADMET 필터링을 거쳐 유의미한 후보 약물을 도출했다는 점에서 실습 목적을 충실히 달성한 것으로 판단된다.

---

## 부록: 최종 제출 전 수정 필요 항목

- 팀원 4명의 실제 이름 입력
- 각 팀원의 실제 역할 비중 반영
- 프론트엔드 최종 화면 캡처 추가
- 발표용 문장 길이에 맞게 축약본 작성
- 필요 시 지도교수/과목명/제출일 표지 추가
