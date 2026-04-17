# exact slim 기반 학습 실행 가이드

이 문서는 현재 워크스페이스에서 아래 세 가지 입력셋을 기준으로 모델 학습을 다시 실행하는 방법을 정리한 가이드다.

- `exact slim (numeric-only)`
- `exact slim + SMILES`
- `exact slim + strong context + SMILES`

전제:

- 입력 생성은 이미 [reproduce_exact_slim_to_strong_context_smiles.md](/Users/skku_aws2_18/pre_project/say2_preproject/Improving%20GroupCV/notes/reproduce_exact_slim_to_strong_context_smiles.md)에 정리되어 있다.
- 본 문서는 **학습 실행**과 **결과 파일 위치**를 중심으로 적는다.
- 모든 명령은 현재 로컬 환경 기준이며, Python은 `drug4` 환경을 사용한다.

## 1. 공통 준비

기본 실행 Python:

```bash
/Users/skku_aws2_18/pre_project/.conda-envs/drug4/bin/python
```

작업 루트:

```bash
cd "/Users/skku_aws2_18/pre_project/say2_preproject/Improving GroupCV"
```

## 2. 입력 생성 순서

### 2.1 exact slim 복제본 준비

```bash
/Users/skku_aws2_18/pre_project/.conda-envs/drug4/bin/python \
  "/Users/skku_aws2_18/pre_project/say2_preproject/Improving GroupCV/scripts/materialize_exact_repo_slim.py"
```

생성 위치:

- [exact_repo_match](/Users/skku_aws2_18/pre_project/say2_preproject/Improving%20GroupCV/v3_input_reproduction/exact_repo_match)

### 2.2 감사용 bundle 생성

이 단계는 강제는 아니지만, `strong context`, `SMILES`, `ML matrix`가 어떻게 만들어졌는지 중간 산출물을 보존하기 좋다.

```bash
/Users/skku_aws2_18/pre_project/.conda-envs/drug4/bin/python \
  "/Users/skku_aws2_18/pre_project/say2_preproject/Improving GroupCV/scripts/materialize_exact_repo_slim_context_smiles_bundle.py"
```

생성 위치:

- [exact_repo_match_context_smiles_bundle](/Users/skku_aws2_18/pre_project/say2_preproject/Improving%20GroupCV/v3_input_reproduction/exact_repo_match_context_smiles_bundle)

### 2.3 ML용 shared matrix 생성

`exact slim + SMILES`용:

```bash
/Users/skku_aws2_18/pre_project/.conda-envs/drug4/bin/python \
  "/Users/skku_aws2_18/pre_project/say2_preproject/Improving GroupCV/scripts/build_exact_repo_slim_smiles_ml_matrix.py"
```

생성 위치:

- [exact_repo_match_ml_smiles](/Users/skku_aws2_18/pre_project/say2_preproject/Improving%20GroupCV/v3_input_reproduction/exact_repo_match_ml_smiles)

`exact slim + strong context + SMILES`용:

```bash
/Users/skku_aws2_18/pre_project/.conda-envs/drug4/bin/python \
  "/Users/skku_aws2_18/pre_project/say2_preproject/Improving GroupCV/scripts/build_exact_repo_slim_strong_context_smiles_ml_matrix.py"
```

생성 위치:

- [exact_repo_match_ml_strong_context_smiles](/Users/skku_aws2_18/pre_project/say2_preproject/Improving%20GroupCV/v3_input_reproduction/exact_repo_match_ml_strong_context_smiles)

## 3. numeric-only 학습

### 3.1 DL

기본 DL runner:

- [run_exact_repo_slim_groupcv.py](/Users/skku_aws2_18/pre_project/say2_preproject/Improving%20GroupCV/scripts/run_exact_repo_slim_groupcv.py)

예시:

```bash
/Users/skku_aws2_18/pre_project/.conda-envs/drug4/bin/python \
  "/Users/skku_aws2_18/pre_project/say2_preproject/Improving GroupCV/scripts/run_exact_repo_slim_groupcv.py" \
  --models FlatMLP,ResidualMLP,FTTransformer,CrossAttention,TabNet,WideDeep,TabTransformer \
  --folds 3 \
  --output-stem exact_repo_slim_all_dl_repro_v1
```

### 3.2 ML

기본 ML runner:

- [run_exact_repo_slim_numeric_ml_groupcv.py](/Users/skku_aws2_18/pre_project/say2_preproject/Improving%20GroupCV/scripts/run_exact_repo_slim_numeric_ml_groupcv.py)

예시:

```bash
/Users/skku_aws2_18/pre_project/.conda-envs/drug4/bin/python \
  "/Users/skku_aws2_18/pre_project/say2_preproject/Improving GroupCV/scripts/run_exact_repo_slim_numeric_ml_groupcv.py" \
  --models LightGBM,LightGBM_DART,XGBoost,CatBoost,RandomForest,ExtraTrees \
  --folds 3 \
  --output-stem exact_repo_slim_numeric_ml_groupcv_repro_v1
```

### 3.3 Ensemble

numeric-only top3 ensemble:

- [run_exact_repo_slim_top3_ensemble.py](/Users/skku_aws2_18/pre_project/say2_preproject/Improving%20GroupCV/scripts/run_exact_repo_slim_top3_ensemble.py)

예시:

```bash
/Users/skku_aws2_18/pre_project/.conda-envs/drug4/bin/python \
  "/Users/skku_aws2_18/pre_project/say2_preproject/Improving GroupCV/scripts/run_exact_repo_slim_top3_ensemble.py" \
  --models WideDeep,CrossAttention,FlatMLP \
  --folds 3 \
  --output-stem exact_repo_slim_top3_ensemble_repro_v1
```

## 4. exact slim + SMILES 학습

### 4.1 DL

DL runner는 같은 스크립트를 쓰고 `--context-mode none`으로 실행한다.

- [run_exact_repo_slim_smiles_all_dl.py](/Users/skku_aws2_18/pre_project/say2_preproject/Improving%20GroupCV/scripts/run_exact_repo_slim_smiles_all_dl.py)

예시:

```bash
/Users/skku_aws2_18/pre_project/.conda-envs/drug4/bin/python \
  "/Users/skku_aws2_18/pre_project/say2_preproject/Improving GroupCV/scripts/run_exact_repo_slim_smiles_all_dl.py" \
  --models FlatMLP,WideDeep,CrossAttention,ResidualMLP,TabNet,FTTransformer,TabTransformer \
  --context-mode none \
  --folds 3 \
  --output-stem exact_repo_slim_smiles_all_dl_repro_v1 \
  --early-stop-model TabTransformer \
  --early-stop-after-folds 2 \
  --early-stop-spearman-threshold 0.57
```

### 4.2 ML

- [run_exact_repo_slim_smiles_ml_groupcv.py](/Users/skku_aws2_18/pre_project/say2_preproject/Improving%20GroupCV/scripts/run_exact_repo_slim_smiles_ml_groupcv.py)

예시:

```bash
/Users/skku_aws2_18/pre_project/.conda-envs/drug4/bin/python \
  "/Users/skku_aws2_18/pre_project/say2_preproject/Improving GroupCV/scripts/run_exact_repo_slim_smiles_ml_groupcv.py" \
  --models LightGBM,LightGBM_DART,XGBoost,CatBoost,RandomForest,ExtraTrees \
  --folds 3 \
  --output-stem exact_repo_slim_smiles_ml_groupcv_repro_v1
```

### 4.3 Ensemble

기본 FRC ensemble:

- [run_exact_repo_slim_smiles_custom_ensemble.py](/Users/skku_aws2_18/pre_project/say2_preproject/Improving%20GroupCV/scripts/run_exact_repo_slim_smiles_custom_ensemble.py)

예시:

```bash
/Users/skku_aws2_18/pre_project/.conda-envs/drug4/bin/python \
  "/Users/skku_aws2_18/pre_project/say2_preproject/Improving GroupCV/scripts/run_exact_repo_slim_smiles_custom_ensemble.py" \
  --models FlatMLP,ResidualMLP,CrossAttention \
  --context-mode none \
  --folds 3 \
  --output-stem exact_repo_slim_smiles_frc_ensemble_repro_v1
```

## 5. exact slim + strong context + SMILES 학습

### 5.1 DL

DL runner는 같은 스크립트를 쓰고 `--context-mode strong`으로 실행한다.

- [run_exact_repo_slim_smiles_all_dl.py](/Users/skku_aws2_18/pre_project/say2_preproject/Improving%20GroupCV/scripts/run_exact_repo_slim_smiles_all_dl.py)

예시:

```bash
/Users/skku_aws2_18/pre_project/.conda-envs/drug4/bin/python \
  "/Users/skku_aws2_18/pre_project/say2_preproject/Improving GroupCV/scripts/run_exact_repo_slim_smiles_all_dl.py" \
  --models FlatMLP,WideDeep,CrossAttention,ResidualMLP,TabNet,FTTransformer,TabTransformer \
  --context-mode strong \
  --folds 3 \
  --output-stem exact_repo_slim_strong_context_smiles_all_dl_repro_v1 \
  --early-stop-model TabTransformer \
  --early-stop-after-folds 2 \
  --early-stop-spearman-threshold 0.57
```

### 5.2 ML

- [run_exact_repo_slim_strong_context_smiles_ml_groupcv.py](/Users/skku_aws2_18/pre_project/say2_preproject/Improving%20GroupCV/scripts/run_exact_repo_slim_strong_context_smiles_ml_groupcv.py)

예시:

```bash
/Users/skku_aws2_18/pre_project/.conda-envs/drug4/bin/python \
  "/Users/skku_aws2_18/pre_project/say2_preproject/Improving GroupCV/scripts/run_exact_repo_slim_strong_context_smiles_ml_groupcv.py" \
  --models LightGBM,LightGBM_DART,XGBoost,CatBoost,RandomForest,ExtraTrees \
  --folds 3 \
  --output-stem exact_repo_slim_strong_context_smiles_ml_groupcv_repro_v1
```

### 5.3 Ensemble

`FlatMLP + ResidualMLP + CrossAttention`:

```bash
/Users/skku_aws2_18/pre_project/.conda-envs/drug4/bin/python \
  "/Users/skku_aws2_18/pre_project/say2_preproject/Improving GroupCV/scripts/run_exact_repo_slim_smiles_custom_ensemble.py" \
  --models FlatMLP,ResidualMLP,CrossAttention \
  --context-mode strong \
  --folds 3 \
  --output-stem exact_repo_slim_strong_context_smiles_frc_ensemble_repro_v1
```

`FlatMLP + WideDeep + CrossAttention`:

```bash
/Users/skku_aws2_18/pre_project/.conda-envs/drug4/bin/python \
  "/Users/skku_aws2_18/pre_project/say2_preproject/Improving GroupCV/scripts/run_exact_repo_slim_strong_context_smiles_top3_ensemble.py" \
  --models FlatMLP,WideDeep,CrossAttention \
  --folds 3 \
  --output-stem exact_repo_slim_strong_context_smiles_top3_ensemble_repro_v1
```

## 6. 결과 저장 위치

학습 결과는 기본적으로 아래에 저장된다.

- [results](/Users/skku_aws2_18/pre_project/say2_preproject/Improving%20GroupCV/results)

패턴:

- JSON 요약: `results/<output-stem>.json`
- OOF 예측: `results/<output-stem>_oof/`

예:

- `exact_repo_slim_smiles_ml_groupcv_repro_v1.json`
- `exact_repo_slim_smiles_ml_groupcv_repro_v1_oof/`

## 7. 추천 실행 순서

가장 실용적인 순서는 아래다.

1. `materialize_exact_repo_slim.py`
2. `materialize_exact_repo_slim_context_smiles_bundle.py`
3. `build_exact_repo_slim_smiles_ml_matrix.py`
4. `build_exact_repo_slim_strong_context_smiles_ml_matrix.py`
5. 원하는 입력셋에 대해 `ML` 또는 `DL` 단일 모델 학습
6. 마지막에 ensemble 실행

## 8. 지금 실험 기준 추천 조합

### 단일 모델

- DL: `FlatMLP`, `ResidualMLP`, `CrossAttention`
- ML: `CatBoost`, `LightGBM_DART`

### 앙상블

- `exact slim + SMILES`: `FlatMLP + ResidualMLP + CrossAttention`
- `exact slim + strong context + SMILES`: `FlatMLP + ResidualMLP + CrossAttention`

## 9. 한 줄 요약

입력 생성은 별도 빌더 스크립트로 만들고, 학습은 현재 `run_exact_repo_slim_*` 계열 스크립트에 `context-mode`, `models`, `output-stem`만 명시해서 바로 실행하면 된다.
