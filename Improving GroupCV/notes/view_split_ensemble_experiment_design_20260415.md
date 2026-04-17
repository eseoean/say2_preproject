# View-Split Ensemble 실험 설계

## 1. 목적

현재 `exact slim + strong context + SMILES` 조건에서 단일 모델 성능과 앙상블 성능은 충분히 좋아졌지만, ensemble diversity는 크게 벌어지지 않았다.

핵심 관찰:
- base model들의 prediction correlation은 다소 차이가 나더라도
- residual correlation이 여전히 높다
- 즉 모델들이 "다른 점수를 내는 것처럼 보여도, 어려운 샘플에서는 비슷하게 틀린다"

따라서 다음 단계의 목적은:

1. 입력 정보를 의미별 view로 분리하고
2. 각 view에 특화된 모델을 따로 학습한 뒤
3. OOF residual correlation이 낮은 조합으로 ensemble을 구성해
4. 현재보다 더 유효한 diversity를 얻는 것이다.

---

## 2. 기본 가설

현재 병목은 "모델 구조"보다 "공유 입력 view"에 더 가깝다.

즉,
- `FlatMLP`
- `ResidualMLP`
- `CrossAttention`
- `WideDeep`
- `LightGBM_DART`

같은 모델 조합을 바꿔도, 모두 거의 같은 full input을 보기 때문에 오류 패턴이 크게 달라지지 않는다.

반대로, 다음처럼 일부러 입력을 다르게 보면:

- 세포주 의존성만 보는 모델
- 약물 구조만 보는 모델
- 타깃/패스웨이만 보는 모델
- context만 보는 모델

각 모델이 포착하는 신호가 달라지고, ensemble에서 상호 보완 가능성이 높아질 수 있다.

---

## 3. 적용 대상 입력셋

1차 실험 기준 입력셋:

- `exact slim + strong context + SMILES`

이유:
- 현재 best 성능 조건
- 이미 numeric / strong context / SMILES bundle이 준비되어 있음
- 실험 결과 비교가 가장 쉽다

사용 가능한 입력 구성:
- slim numeric
- strong context categorical
- SMILES token ids

---

## 4. View 정의

현재 `exact slim`의 핵심 블록은 대략 아래처럼 나뉜다.

- `sample__crispr__*`
- `drug_morgan_*`
- `target_*`
- `lincs_*`
- `drug_desc_*`
- strong context 5개
- SMILES sequence

이걸 다음 4개 view로 나눈다.

### View A. Cell-only

의미:
- 세포주 민감도/취약성 신호

포함:
- `sample__crispr__*`

제외:
- Morgan
- target/pathway
- lincs
- drug descriptors
- strong context
- SMILES

목적:
- "이 세포주가 어떤 계열 약물에 전반적으로 취약한가"를 cell state만으로 학습

추천 모델:
- `ResidualMLP`
- `FlatMLP`

---

### View B. Drug-structure-only

의미:
- 약물 구조 자체의 일반화

포함:
- `drug_morgan_*`
- `drug_desc_*`
- `SMILES token ids`

제외:
- `sample__crispr__*`
- `target_*`
- `lincs_*`
- strong context

목적:
- unseen drug generalization의 핵심인 구조 기반 신호를 최대한 분리

추천 모델:
- `CrossAttention` 또는 `WideDeep`
- 보조로 `LightGBM_DART`

비고:
- 이 view는 GroupCV 개선의 핵심 후보

---

### View C. Target / Pathway / LINCS-only

의미:
- 기전 기반 요약 신호

포함:
- `target_*`
- `lincs_*`
- strong context 중
  - `PATHWAY_NAME_NORMALIZED`
  - `classification`
  - `stage3_resolution_status`

제외:
- `sample__crispr__*`
- `drug_morgan_*`
- `drug_desc_*`
- `SMILES`
- strong context 중 `TCGA_DESC`, `drug_bridge_strength`

목적:
- 구조 자체보다 "어떤 target / pathway를 치는 약물인가"를 별도 학습

추천 모델:
- `WideDeep`
- `FlatMLP`
- `LightGBM_DART`

---

### View D. Context-only

의미:
- cell tissue / drug annotation / bridge confidence 같은 보조 문맥

포함:
- `TCGA_DESC`
- `PATHWAY_NAME_NORMALIZED`
- `classification`
- `drug_bridge_strength`
- `stage3_resolution_status`

제외:
- 모든 numeric slim
- SMILES

목적:
- 단독 성능보다 "meta prior" 역할 확인

추천 모델:
- `WideDeep`
- `TabTransformer`

비고:
- 단독 성능은 낮을 가능성이 높지만, diversity source로 의미가 있을 수 있다

---

## 5. 추천 1차 실험 조합

모든 가능한 조합을 한 번에 다 돌리기보다, 1차는 아래 4개만 추천한다.

### 실험 1
- `Cell-only`
- 모델: `ResidualMLP`

### 실험 2
- `Drug-structure-only`
- 모델: `CrossAttention`

### 실험 3
- `Target/Pathway/LINCS-only`
- 모델: `WideDeep`

### 실험 4
- `Full best baseline`
- 입력: `exact slim + strong context + SMILES`
- 모델: `FlatMLP`

이 4개를 먼저 돌리는 이유:
- 서로 입력 view가 명확히 다르다
- 현재 best baseline과 직접 비교 가능하다
- diversity가 실제로 늘어나는지 보기 좋다

---

## 6. 1차 앙상블 후보

### Candidate A
- `FlatMLP(full)`
- `ResidualMLP(cell-only)`
- `CrossAttention(drug-only)`

의미:
- full / cell / drug 구조를 한 조합으로 묶음

### Candidate B
- `FlatMLP(full)`
- `CrossAttention(drug-only)`
- `WideDeep(target-pathway-only)`

의미:
- full / drug / mechanism view 조합

### Candidate C
- `ResidualMLP(cell-only)`
- `CrossAttention(drug-only)`
- `WideDeep(target-pathway-only)`

의미:
- 아예 full input을 빼고 view-diverse 조합만 테스트

---

## 7. 평가 기준

기본 평가는 기존과 동일:

- `Spearman`
- `RMSE`
- `MAE`
- `Pearson`
- `R²`
- `NDCG@20`

추가로 중요하게 볼 diversity 지표:

- pairwise prediction Pearson
- pairwise residual Pearson
- pairwise prediction Spearman
- pairwise residual Spearman
- mean absolute prediction gap

특히 핵심은 아래 두 개다.

1. `residual Pearson`
   - 낮을수록 좋음
   - ensemble이 같은 샘플에서 같이 틀리지 않는다는 뜻

2. `ensemble gain vs best base`
   - weighted ensemble이 best single model보다
   - `Spearman`을 얼마나 올리는지
   - `RMSE`를 얼마나 내리는지

---

## 8. 성공 기준

현재 strong-context+SMILES best인 `FRC ensemble` 기준:

- weighted `Spearman ~ 0.6280`
- weighted `RMSE ~ 2.0220`
- residual Pearson 평균 `~ 0.903`

view-split 실험의 성공 기준은:

### 최소 성공
- residual Pearson 평균이 `0.88 이하`
- weighted Spearman이 기존 best와 비슷하거나 근소 개선

### 명확한 성공
- residual Pearson 평균이 `0.86 이하`
- weighted Spearman `+0.01 이상`
또는
- weighted RMSE `-0.02 이상`

---

## 9. 구현 우선순위

### Phase 1. View matrix materialization

필요 작업:
- 기존 bundle에서 view별 입력 행렬 생성
- 예:
  - `X_view_cell.npy`
  - `X_view_drug.npy`
  - `X_view_mechanism.npy`
  - `context_codes_view.npy`
  - `smiles_token_ids.npy`

### Phase 2. Base model OOF

각 view별 base model OOF 생성:
- `ResidualMLP(cell-only)`
- `CrossAttention(drug-only)`
- `WideDeep(target-pathway-only)`
- `FlatMLP(full)`

### Phase 3. Ensemble and diversity

후보 조합 A/B/C에 대해:
- equal ensemble
- weighted ensemble
- diversity 계산

### Phase 4. Optional gating

1차 결과가 괜찮으면:
- `classification`
- `PATHWAY_NAME_NORMALIZED`
- `drug_bridge_strength`

기준으로 local weighting을 다르게 주는 conditional ensemble 실험

---

## 10. 왜 이 설계가 현재보다 나을 가능성이 있는가

현재는:
- 입력 view가 거의 동일
- 목적함수도 거의 동일
- 그래서 prediction은 조금 달라도 residual이 비슷함

이 설계는:
- 입력을 강제로 분리해 각 모델이 보는 세계를 다르게 만들고
- 그 차이를 ensemble에 활용하는 구조다

즉 핵심은:
- 모델을 더 "복잡하게" 만드는 것이 아니라
- 모델이 서로 다른 evidence를 보도록 만드는 것

---

## 11. 다음 구현 권장안

실제로 바로 구현할 때는 아래 순서를 추천한다.

1. `cell-only / drug-only / target-pathway-only / full` view matrix 생성
2. `ResidualMLP(cell-only)` 실행
3. `CrossAttention(drug-only)` 실행
4. `WideDeep(target-pathway-only)` 실행
5. 기존 `FlatMLP(full)` OOF 재사용
6. 3개 ensemble 후보 A/B/C 비교

가장 먼저 볼 후보는:

- `FlatMLP(full) + ResidualMLP(cell-only) + CrossAttention(drug-only)`

이 조합이 현재 문제인 `입력 공유로 인한 잔차 상관`을 가장 직접적으로 깨는 후보이기 때문이다.
