# exact slim numeric -> strong context + SMILES 재현 가이드

이 문서는 현재 워크스페이스에서 `exact slim numeric` 입력을 출발점으로 하여

- `exact slim + SMILES`
- `exact slim + strong context + SMILES`

입력을 다시 만들 수 있도록 정리한 재현용 가이드다.

대상 작업은 **입력셋 생성**이며, 모델 학습은 이 문서 마지막의 실행 스크립트를 이용한다.

## 1. 시작점

기준 시작 파일은 아래 두 개다.

- [features_slim_exact_repo.parquet](/Users/skku_aws2_18/pre_project/say2_preproject/Improving%20GroupCV/v3_input_reproduction/exact_repo_match/features_slim_exact_repo.parquet)
- [y_train_exact_repo.npy](/Users/skku_aws2_18/pre_project/say2_preproject/Improving%20GroupCV/v3_input_reproduction/exact_repo_match/y_train_exact_repo.npy)

이 `features_slim_exact_repo.parquet`는 이미 팀원 slim 파일과 값까지 동일한 로컬 복제본이다.

현재 기준 shape:

- rows: `6366`
- total columns: `5534`
- numeric columns used as base X: `5529`

기본 numeric slim 구성은 대략 아래처럼 나뉜다.

- `sample__crispr__*`: `4415`
- `drug_morgan_*`: `1088`
- `lincs_*`: `5`
- `target_*`: `10`
- `drug_desc_*`: `9`
- 기타 drug meta-like numeric: 소수

## 2. strong context가 무엇인지

전체 reconstructed categorical context는 11개를 만들 수 있지만, 실제 실험에서는 coverage와 신호가 안정적인 5개만 `strong context`로 사용했다.

strong context 5개:

- `TCGA_DESC`
- `PATHWAY_NAME_NORMALIZED`
- `classification`
- `drug_bridge_strength`
- `stage3_resolution_status`

이 컬럼들은 모두 row coverage가 `100%`다.

### 2.1 컬럼별 의미

`TCGA_DESC`

- 세포주가 속한 암종/조직 계열 정보
- cell line annotation에서 복원

`PATHWAY_NAME_NORMALIZED`

- 약물 annotation의 pathway 라벨
- 약물 단위로 부여됨

`classification`

- drug target mapping을 바탕으로 target token이 얼마나 gene-like 하게 해석되는지 요약한 라벨
- 예:
  - `all_tokens_gene_matched`
  - `mixed_gene_and_non_gene`
  - `mixed_gene_and_ambiguous`

`drug_bridge_strength`

- 약물에 대해 `SMILES`, `LINCS`, `target` 근거가 몇 개 확보되는지로 만든 신뢰도성 요약 컬럼
- `multi_source` 또는 `single_source`

`stage3_resolution_status`

- target 해석이 얼마나 깔끔하게 정리됐는지 상태를 나타내는 컬럼
- 예:
  - `resolved_or_cleaned`
  - `partial_gene_resolution_with_family_remaining`

## 3. strong context를 어떻게 복원하는지

strong context는 slim 안에 이미 수치로 들어 있는 것이 아니라, 기존 slim row의 key를 이용해 외부 메타 정보를 다시 붙여 만든다.

핵심 기준 key:

- `sample_id`
- `canonical_drug_id`

복원 함수:

- [run_groupcv_dl_progressive.py](/Users/skku_aws2_18/pre_project/say2_preproject/Improving%20GroupCV/scripts/run_groupcv_dl_progressive.py)
  - `build_reconstructed_context`

이 함수는 아래 소스를 사용한다.

- GDSC cellline annotation
- GDSC drug annotation
- drug feature catalog
- drug target mapping
- pair feature cache

복원 결과는 row 수를 줄이지 않는다.

- input rows: `6366`
- output rows: `6366`
- `match_rate = 1.0`

## 4. SMILES를 어떻게 처리하는지

SMILES는 `drug__canonical_smiles_raw`를 우선 사용하고, 없으면 `drug__smiles`로 보완한다.

### 4.1 DL용 SMILES

DL에서는 문자열을 문자 단위 token으로 바꾼다.

- 최대 길이: `256`
- 문자 vocab 생성
- 각 row를 `token id` 시퀀스로 변환

관련 함수:

- [run_exact_repo_slim_smiles_ab.py](/Users/skku_aws2_18/pre_project/say2_preproject/Improving%20GroupCV/scripts/run_exact_repo_slim_smiles_ab.py)
  - `build_smiles_tensor`

DL 모델은 이 token 시퀀스를 `SmilesCNNEncoder`로 처리해 `64`차원 표현으로 바꿔 쓴다.

### 4.2 ML용 SMILES

ML은 sequence branch를 직접 쓰지 않기 때문에, drug-level SMILES를 고정 숫자 벡터로 바꾼다.

변환 방식:

- character TF-IDF
- ngram range: `(2, 4)`
- 그 뒤 `TruncatedSVD`
- 최종 차원: `64`

관련 함수:

- [build_exact_repo_slim_strong_context_smiles_ml_matrix.py](/Users/skku_aws2_18/pre_project/say2_preproject/Improving%20GroupCV/scripts/build_exact_repo_slim_strong_context_smiles_ml_matrix.py)
  - `build_smiles_svd`

## 5. DL와 ML에서 입력이 어떻게 다른지

### 5.1 DL 입력

DL은 아래 3개 branch를 함께 사용한다.

1. exact slim numeric
2. strong context categorical code
3. SMILES token ids

즉 DL은 strong context를 one-hot으로 바꾸지 않고, **categorical embedding**으로 직접 넣는다.

### 5.2 ML 입력

ML은 아래처럼 모두 숫자 행렬로 변환해 한 번에 붙인다.

1. exact slim numeric `5529`
2. strong context one-hot `32`
3. SMILES SVD `64`

최종 ML 행렬:

- `5529 + 32 + 64 = 5625`

## 6. 이번에 만든 재현용 코드

새로 만든 메인 재현 스크립트:

- [materialize_exact_repo_slim_context_smiles_bundle.py](/Users/skku_aws2_18/pre_project/say2_preproject/Improving%20GroupCV/scripts/materialize_exact_repo_slim_context_smiles_bundle.py)

이 스크립트는 아래를 한 번에 만든다.

- reconstructed full context table
- strong-context-only table
- DL용 strong context codes / vocab
- DL용 SMILES token ids / vocab
- ML용 `exact slim + SMILES` 행렬
- ML용 `exact slim + strong context + SMILES` 행렬
- 전체 summary JSON

## 7. 실행 방법

```bash
/Users/skku_aws2_18/pre_project/.conda-envs/drug4/bin/python \
  "/Users/skku_aws2_18/pre_project/say2_preproject/Improving GroupCV/scripts/materialize_exact_repo_slim_context_smiles_bundle.py"
```

## 8. 생성 결과 위치

생성 위치:

- [exact_repo_match_context_smiles_bundle](/Users/skku_aws2_18/pre_project/say2_preproject/Improving%20GroupCV/v3_input_reproduction/exact_repo_match_context_smiles_bundle)

주요 출력:

- `reconstructed_context_full.parquet`
- `strong_context_only.parquet`
- `strong_context_codes.npy`
- `strong_context_vocab.json`
- `smiles_token_ids.npy`
- `smiles_vocab.json`
- `X_ml_exact_slim_smiles.npy`
- `X_ml_exact_slim_strong_context_smiles.npy`
- `context_smiles_bundle_summary.json`

## 9. 기존 학습 스크립트와의 연결

### exact slim + SMILES

- ML:
  - [run_exact_repo_slim_smiles_ml_groupcv.py](/Users/skku_aws2_18/pre_project/say2_preproject/Improving%20GroupCV/scripts/run_exact_repo_slim_smiles_ml_groupcv.py)
- DL:
  - [run_exact_repo_slim_smiles_all_dl.py](/Users/skku_aws2_18/pre_project/say2_preproject/Improving%20GroupCV/scripts/run_exact_repo_slim_smiles_all_dl.py)
  - `--context-mode none`

### exact slim + strong context + SMILES

- ML:
  - [run_exact_repo_slim_strong_context_smiles_ml_groupcv.py](/Users/skku_aws2_18/pre_project/say2_preproject/Improving%20GroupCV/scripts/run_exact_repo_slim_strong_context_smiles_ml_groupcv.py)
- DL:
  - [run_exact_repo_slim_smiles_all_dl.py](/Users/skku_aws2_18/pre_project/say2_preproject/Improving%20GroupCV/scripts/run_exact_repo_slim_smiles_all_dl.py)
  - `--context-mode strong`

## 10. 주의할 점

- 이 문서는 **입력 생성 재현** 기준이다.
- strong context는 fold별로 다시 계산하는 게 아니라, 현재 파이프라인 메타 테이블을 이용해 전 row에 대해 먼저 복원한다.
- ML 쪽 strong context는 one-hot, DL 쪽 strong context는 categorical embedding으로 처리한다.
- SMILES도 ML과 DL에서 표현 방식이 다르다.
  - ML: TF-IDF + SVD
  - DL: character token + CNN encoder

## 11. 한 줄 요약

이번 재현 흐름의 핵심은 `exact slim numeric`을 그대로 버리지 않고 유지한 채,

- 의미 있는 5개 strong context를 복원하고
- SMILES를 ML/DL 각각에 맞는 형태로 변환해
- 최종 입력을 확장하는 것

이다.
