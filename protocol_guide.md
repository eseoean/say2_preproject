# Drug Discovery Pipeline 팀원 재현 프로토콜
## biso_myprotocol · say2-4team · v1.0
**작성일:** 2026-04-08 | **작성자:** say2-4team | **버전:** v1.0

---

## 참고 자료
- **메인 대시보드:** https://skkuaws0215.github.io/20260408_pre_project_biso_myprotocol/dashboard.html
- **GitHub 저장소:** https://github.com/skkuaws0215/20260408_pre_project_biso_myprotocol.git

---

## 이 문서의 목적

이 프로토콜은 biso_myprotocol의 Drug Discovery Pipeline을 팀원이 **자신의 저장소에서 동일한 방법론으로 재현**할 수 있도록 작성된 가이드입니다.

**시작점:** 본인이 보유한 전처리 완료 데이터 (curated_date/)
**목표:** 이 프로토콜을 따라 새 환경/저장소에서 동일한 방법론으로 실험을 재현하는 것

> 이 프로토콜은 재현 가이드입니다. 본인의 전처리 데이터와 저장소를 사용해서 각 단계를 직접 실행하세요.

---

## 전체 파이프라인 흐름

```
전처리 완료 데이터 (curated_date/)
        ↓
Step 1. 환경 설정 (~1~2시간)
        ↓
Step 2. 전처리 데이터 준비 (~1~2시간)
        ↓
Step 3. FE 실행 - AWS Batch + Nextflow (~3~6시간)
        ↓
Step 4. 모델 학습 (~4~8시간, GPU 권장)
        ↓
Step 5. 앙상블 Track 2 (~2~3시간)
        ↓
Step 6. METABRIC 외부 검증 A+B+C (~30~45분)
        ↓
Step 7. ADMET Gate ML 자동화 (~40분~1시간)
        ↓
결과 저장 + 대시보드 업데이트
```

**전체 소요 시간 (예상):** 약 12~24시간 (GPU 환경 기준)

---

## 재현 결과에 대한 기대치

이 프로토콜을 따르더라도 biso_myprotocol 수치와 완전히 동일한 결과가 나오지 않을 수 있습니다. 이는 정상입니다.

| 항목 | 차이가 생기는 이유 | 허용 여부 |
|---|---|---|
| FE 컬럼 수 | 데이터 버전·범위·전처리 조건 차이 | 정상 범위 허용 |
| 모델 Spearman·RMSE | FE 입력값 차이 + 랜덤 시드 | 방향성만 동일하면 OK |
| METABRIC 검증 지표 | FE 차이가 누적되어 반영 | 방향성만 동일하면 OK |
| 추천 약물 순위 | 모델 예측값 분포 차이 | 일부 순위 변동 가능 |

> 중요: 숫자가 다소 달라도 방법론이 동일하면 재현 성공입니다. biso_myprotocol 기준 수치는 비교 참고값입니다.

---

## 절대 규칙 (반드시 준수)

```
1. curated_date/         → 읽기 전용, 수정·삭제 절대 금지
2. curated_date/glue/    → 접근 자체 금지 (다른 팀원 영역)
3. Proxy 데이터 사용 시  → 즉시 멈추고 팀장에게 확인 요청
4. ADMET 외부 DB 추가 시 → 반드시 명시 후 진행
5. .aws/credentials      → Git 커밋 절대 금지
6. AWS 리소스 생성 시    → 태그 필수: Key=pre-batch-2-4-team, Value=YYYYMMDD_본인이름
```

---

## 커스터마이징 사항

암종이나 데이터가 달라지는 경우 아래 항목을 수정하세요.

### 암종 변경 시 수정 항목
| 항목 | 유방암(기준) | 변경 방법 |
|---|---|---|
| GDSC 필터 | TCGA_DESC == "BRCA" | 해당 암종 코드로 변경 |
| LINCS 세포주 | MCF7 | 해당 암종 세포주로 변경 |
| METABRIC 외부 검증 데이터 | METABRIC BRCA | 해당 암종 외부 검증 데이터로 교체 |
| 결과 저장 경로 | YYYYMMDD_biso | YYYYMMDD_본인이름 |

### 모델 추가/제거 시 주의사항
- 동일한 FE 산출물(features.parquet·labels.parquet)로 학습
- 동일한 5-fold CV 기준으로 평가
- Graph 계열 모델은 반드시 drug-split 적용
- 추가 모델도 앙상블 기준(Sp≥0.713 AND RMSE≤1.385) 동일 적용

### 데이터 버전 명시
- GDSC1+2 병합 또는 GDSC2만 사용 여부 명시
- LINCS 원본 추출 또는 기존 파일 사용 여부 명시
- 프로토콜과 다른 처리는 README·대시보드에 반드시 기재

---

## STEP 1. 환경 설정
**예상 소요 시간: 1~2시간**

### 1-1. GitHub 저장소 클론
```bash
# biso_myprotocol 참고 저장소
git clone https://github.com/skkuaws0215/20260408_pre_project_biso_myprotocol.git

# 본인 저장소 생성 후 클론
git clone https://github.com/[본인계정]/[본인저장소].git
cd [본인저장소]
```

### 1-2. Python 환경 설정
```bash
conda create -n drug4 python=3.10 -y
conda activate drug4
pip install pandas numpy scikit-learn lightgbm xgboost catboost \
    torch pyarrow boto3 awscli rdkit-pypi nextflow
```

### 1-3. Nextflow 설치
```bash
# Java 11+ 필요 (없으면 conda로 설치)
conda install -c conda-forge openjdk=11

# Nextflow 설치
curl -s https://get.nextflow.io | bash
chmod +x nextflow && sudo mv nextflow /usr/local/bin/
nextflow -version  # 정상 확인
```

### 1-4. AWS 설정
```bash
# 자격증명 설정
aws configure  # Account: 666803869796 확인

# 접근 확인
aws sts get-caller-identity
aws s3 ls s3://say2-4team/curated_date/

# 절대 접근 금지
# aws s3 ls s3://say2-4team/curated_date/glue/
```

### 1-5. AWS Batch 환경 세팅

**태그 설정 (모든 AWS 리소스에 필수 적용):**
```bash
TAG_KEY="pre-batch-2-4-team"
TAG_VAL="YYYYMMDD_본인이름"  # 예: 20260408_biso
```

**ECR 이미지 빌드 & 푸시:**
```bash
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
ECR_URI=${AWS_ACCOUNT}.dkr.ecr.ap-northeast-2.amazonaws.com/drug4-fe

aws ecr create-repository --repository-name drug4-fe \
  --region ap-northeast-2 \
  --tags Key=${TAG_KEY},Value=${TAG_VAL}

docker build -t drug4-fe --platform linux/amd64 .
docker tag drug4-fe:latest ${ECR_URI}:latest
docker push ${ECR_URI}:latest
```

**Compute Environment / Job Queue 확인:**
- 기존 팀 리소스(team4-fe-ce-cpu·team4-fe-queue-cpu) 재사용 가능한지 먼저 확인
- 없는 것만 새로 생성 (태그 반드시 적용)

### QC 체크
```
[ ] conda drug4 환경 생성 완료
[ ] Nextflow 정상 동작 확인
[ ] AWS Account 666803869796 확인
[ ] ECR 이미지 빌드·푸시 완료 (linux/amd64)
[ ] AWS Batch Compute Env·Job Queue 확인
[ ] GitHub 저장소 클론 완료
```

---

## STEP 2. 전처리 데이터 준비
**예상 소요 시간: 1~2시간 (LINCS 복사 포함)**

### 2-1. curated_date/ 파일 목록 확인 (읽기만)
```bash
# 전체 목록 확인
aws s3 ls s3://say2-4team/curated_date/ --human-readable

# 필수 6종 확인
REQUIRED=(
  "gdsc/gdsc2_basic_clean_*.parquet"
  "lincs/lincs_sig_info_*.parquet"
  "metabric/metabric_expression_*.parquet"
  "metabric/metabric_clinical_*.parquet"
  "drugbank/drug_master_*.parquet"
  "id_mapping*.parquet"
)
```

### 2-2. 업무폴더로 복사 (glue/ 제외 필수)
```bash
# curated_date/ → 업무폴더 복사 (원본 건드리지 않음)
aws s3 cp s3://say2-4team/curated_date/ \
  s3://say2-4team/[본인업무폴더]/data/ \
  --recursive \
  --exclude "glue/*"   # glue/ 절대 복사하지 말 것
```

### 2-3. LINCS MCF7 처리
biso_myprotocol에서는 두 가지 방식 중 선택:
- **방법 A:** oringinal_raw/lincs/ Level 5 GCTx(21GB)에서 직접 MCF7 추출
- **방법 B:** 팀 공유 lincs_mcf7.parquet 사용 (팀장 확인 후)

```bash
# 방법 A: 직접 추출 (권장)
# cmapPy 또는 h5py로 GCTx 파일에서 MCF7 시그니처 추출
python scripts/extract_lincs_mcf7.py \
  --input s3://say2-4team/oringinal_raw/lincs/GSE92742_*.gctx.gz \
  --cell_id MCF7 \
  --output s3://say2-4team/[본인업무폴더]/data/lincs_mcf7.parquet
```

### 2-4. GDSC 처리 결정
| 선택 | 설명 | 장단점 |
|---|---|---|
| GDSC2만 사용 | 품질 우수, 약물 295개 | 품질↑, 커버리지↓ |
| GDSC1+2 병합 | 약물 621개 | 커버리지↑, 품질↓ |

> biso_myprotocol은 GDSC2만 사용 (품질 우선 결정). 선택 이유를 README에 명시할 것.

### 2-5. config/data_paths.yaml 설정
```yaml
# 본인 업무폴더 경로로 수정
gdsc_ic50:     s3://say2-4team/[본인업무폴더]/data/gdsc/...
lincs:         s3://say2-4team/[본인업무폴더]/data/lincs_mcf7.parquet
metabric_expr: s3://say2-4team/[본인업무폴더]/data/metabric/...
metabric_clin: s3://say2-4team/[본인업무폴더]/data/metabric/...
output_dir:    s3://say2-4team/[본인업무폴더]/fe_output/
```

### QC 체크
```
[ ] curated_date/ 필수 데이터 확인 (glue/ 미접근)
[ ] 업무폴더로 복사 완료 (glue/ 제외 확인)
[ ] LINCS MCF7 처리 방식 결정 및 명시
[ ] GDSC 버전 선택 및 README 명시
[ ] config/data_paths.yaml 본인 경로로 수정
[ ] 팀4 원본 데이터(tmp_data) 복사 금지 확인
```

---

## STEP 3. FE 실행 (newfe_v8) — AWS Batch + Nextflow
**예상 소요 시간: 3~6시간 (LINCS 7.6GB 처리가 병목)**

### 3-1. 핵심 원칙 (반드시 준수)
```
✅ fit (imputation median, variance threshold) → TCGA 기준으로만 계산
✅ METABRIC → TCGA fit 기준값으로 transform만 적용 (fit 절대 금지)
✅ leakage 컬럼 제거 (샘플ID, 날짜, 결과 직접 포함)
✅ high-missing(>30%) 제거 · median imputation · variance filtering
✅ TCGA/METABRIC 컬럼 구조 반드시 일치
```

### 3-2. Drug Features 매칭
biso_myprotocol은 glue/ 접근 불가로 복합 매칭 사용:
```
1순위: ChEMBL (canonical_smiles 품질 우수)
2순위: DrugBank (미매칭 보완)
3순위: GDSC annotation fuzzy matching
4순위: PubChem API (외부 조회)
미매칭: SMILES=NA (all-zero fingerprint 처리)
```
매칭률 및 미매칭 목록 반드시 README·대시보드에 명시.

### 3-3. Nextflow config 설정
```bash
# nextflow.config 핵심 설정
process.executor = 'awsbatch'
process.queue    = 'team4-fe-queue-cpu'  # 또는 본인 Job Queue
aws.region       = 'ap-northeast-2'
workDir          = 's3://say2-4team/[본인업무폴더]/nextflow_work/'

# 태그 설정
aws.batch.jobDefinitionTags = ['pre-batch-2-4-team': 'YYYYMMDD_본인이름']
```

### 3-4. FE 실행
```bash
nextflow run nextflow/main.nf \
  -profile awsbatch \
  -c nextflow.config \
  --s3_input  s3://say2-4team/[본인업무폴더]/data/ \
  --s3_output s3://say2-4team/[본인업무폴더]/fe_output/ \
  --run_name  YYYYMMDD_newfe_v8_본인이름 \
  --config    config/data_paths.yaml \
  -resume \
  -with-report fe_report.html \
  -with-trace  fe_trace.txt
```

### 3-5. FE 완료 QC 기준
| 확인 항목 | biso 기준값 | 허용 범위 | 차이 원인 |
|---|---|---|---|
| 최종 컬럼 수 | 18,316 | ±500 | 데이터 버전·범위 차이 |
| TCGA 행 수 | 7,730 | GDSC 버전에 따라 다름 | GDSC1+2 vs GDSC2 |
| LINCS 매핑률 | 82.4% | ±5% | ID mapping 버전 차이 |
| Drug SMILES 매칭률 | 82.4% (243/295) | ±10% | 데이터 완성도 차이 |
| PASS/WARN/FAIL | PASS 위주 | WARN 1~2 허용 | 전처리 조건 차이 |

### QC 체크
```
[ ] FE 실행 완료 · 산출물 구조 확인
[ ] METABRIC에 TCGA fit transform만 적용 확인
[ ] 컬럼 수·행 수·매핑률 기록
[ ] 결측치 없음 확인 (median imputation 완료)
[ ] S3 출력 확인 (features.parquet · labels.parquet · preprocessing_stats.json)
[ ] biso 기준값과 편차 기록 및 README 명시
```

---

## STEP 4. 모델 학습
**예상 소요 시간: 4~8시간 (GPU: Apple M4 MPS 또는 CUDA 권장)**

### 4-1. 모델 구성

> 튜닝 현황: 이번 프로토콜의 모든 모델은 하이퍼파라미터 기본값(default) 사용. 튜닝 미적용.
> 개선 방향: Optuna·GridSearch로 Spearman +0.01~0.05 기대 가능 (Studying Project 예정).

| # | 모델 | 유형 | biso Spearman | biso RMSE | 비고 |
|---|---|---|---|---|---|
| 1 | LightGBM | ML | 0.7913 | 1.3438 | 앙상블 PASS |
| 2 | LightGBM DART | ML | 0.7848 | 1.4029 | RMSE FAIL |
| 3 | XGBoost | ML | 0.7895 | 1.3445 | 앙상블 PASS |
| 4 | CatBoost | ML | 0.8007 | 1.3172 | ML Best |
| 5 | RandomForest | ML | 0.6267 | 1.9747 | FAIL |
| 6 | ExtraTrees | ML | 0.6468 | 1.8704 | FAIL |
| 7 | Stacking (Ridge) | ML | 0.7981* | 1.3213* | 시간초과 제외 |
| 8 | RSF | ML | 0.6142 | - | METABRIC 전용 |
| 9 | ResidualMLP | DL | 0.7855 | 1.3776 | 앙상블 PASS |
| 10 | FlatMLP | DL | 0.7936 | 1.3429 | DL Best |
| 11 | TabNet | DL | 0.7780 | 1.3892 | RMSE FAIL |
| 12 | FT-Transformer | DL | 0.7625 | 1.4444 | FAIL |
| 13 | Cross-Attention | DL | 0.7852 | 1.3716 | Gap=0.024 최우수 |
| 14 | GraphSAGE | GNN | 0.3852 | 2.3189 | METABRIC 전용 |
| 15 | GAT | GNN | 0.0085 | 2.6608 | FAIL |

### 4-2. GPU 설정
```python
# Apple M4 MPS
device = torch.device("mps") if torch.backends.mps.is_available() \
         else torch.device("cpu")

# NVIDIA CUDA
device = torch.device("cuda") if torch.cuda.is_available() \
         else torch.device("cpu")
```

### 4-3. 실행 순서
```bash
# ML 8개 먼저 (CPU)
python ml/train_all_models.py \
  --fe_path    [FE_OUTPUT]/tcga/features.parquet \
  --label_path [FE_OUTPUT]/tcga/labels.parquet \
  --output_dir models/ \
  --cv_folds 5

# GraphSAGE (drug-split 필수)
python ml/train_graph.py \
  --split_strategy drug \  # 반드시 drug으로 설정
  --output_dir models/graph/

# GAT (drug-split 필수)
python ml/train_graph.py \
  --model gat \
  --split_strategy drug \
  --output_dir models/graph/
```

### 4-4. 병렬 실행 권장
- GPU 메모리 여유 시: 최대 2개 병렬 실행
- 메모리 부족 시: 1개씩 순차 실행

### 4-5. 앙상블 포함 기준
```
Spearman ≥ 0.713 AND RMSE ≤ 1.385 → 앙상블 포함
하나라도 FAIL → 앙상블 제외
Graph 계열 → P@20 ≥ 0.70 별도 기준 적용
RSF → C-index 기준 별도 판단 (METABRIC 전용)
GraphSAGE → P@20 우수 시 METABRIC Method C 전용
```

### 4-6. 과적합 기준
| Gap(Sp) | 판정 |
|---|---|
| < 0.05 | 과적합 없음 ✅ |
| 0.05~0.10 | 허용 범위 ⚠️ |
| 0.10~0.15 | 주의 ⚠️⚠️ |
| > 0.15 | 과적합 심각 ❌ |

> 피처 수 ~20,000개 환경에서 Gap 0.10 수준은 허용 범위.

### QC 체크
```
[ ] ML 8개 완료 · 결과값 기록
[ ] DL 5개 완료 · 결과값 기록
[ ] Graph 2개 완료 · drug-split 적용 확인
[ ] GraphSAGE·GAT drug-split 적용 확인
[ ] 앙상블 통과 모델 목록 확정
[ ] RSF·GraphSAGE METABRIC 전용 분류
[ ] 결측치·이상치·과적합 확인
[ ] 팀 기준값과 편차 기록
```

---

## STEP 5. 앙상블 (Track 2 — Spearman 가중 평균)
**예상 소요 시간: 2~3시간**

### 5-1. 앙상블 구조
```
[입력] features.parquet (전체 샘플)
        ↓
[6개 모델 병렬 예측]
각 모델이 동일 입력으로 IC50 예측값 생성
(OOF: Out-of-Fold 예측값 사용)
        ↓
[Spearman 기반 가중치 계산]
가중치 = 각 모델 Val Spearman / 전체 합계
        ↓
[가중 평균 → 최종 IC50 예측값]
약물별 앙상블 IC50 = Σ(모델 예측값 × 가중치)
        ↓
[Top 30 추출]
        ↓
[METABRIC 검증 → Top 15 선별]
```

### 5-2. 튜닝 현황
- 가중치 최적화 미적용 (Spearman 단순 가중 평균)
- 개선 방향: Optuna 기반 가중치 최적화 시 추가 성능 향상 가능

### 5-3. 실행
```bash
python ml/ensemble/ensemble_track2.py \
  --model_dir  models/ \
  --fe_path    [FE_OUTPUT]/tcga/features.parquet \
  --output     ensemble_results/
```

### 5-4. QC 기준
```
✅ 앙상블 Spearman > 개별 Best 모델 Spearman (핵심)
✅ Gap ≈ 0 (과적합 없음)
✅ Top 30 전부 Sensitivity 확인
✅ 동일 계열 약물 중복 여부 확인 (다양성)
✅ biso 기준: Spearman 0.8055, RMSE 1.3008, Gap 0.0004
```

### QC 체크
```
[ ] 앙상블 Spearman > 개별 Best 모델 확인
[ ] Gap 확인 (개별 모델보다 줄어야 정상)
[ ] Top 30 추출 완료
[ ] S3 저장 완료
```

---

## STEP 6. METABRIC 외부 검증 (A+B+C)
**예상 소요 시간: 30~45분**

### 6-1. 검증 방법

| 방법 | 입력 | 평가 지표 | biso 결과 |
|---|---|---|---|
| A — IC50 proxy | METABRIC FE + 앙상블 예측 | Spearman | 29/30 타겟 BRCA 발현 확인 |
| B — Survival binary | METABRIC OS/RFS + RSF 모델 | C-index · AUROC | C-index=0.821 |
| C — GraphSAGE P@20 | drug-drug 유사도 그래프 | P@20 | P@20=0.94 |

### 6-2. 실행
```bash
python ml/metabric_validation/validate_abc.py \
  --model_dir     models/ \
  --metabric_fe   [FE_OUTPUT]/metabric/ \
  --metabric_clin [본인업무폴더]/data/metabric/metabric_clinical_*.parquet \
  --output        metabric_results/
```

### 6-3. Top 30 → Top 15 선별 기준
- METABRIC 검증 점수 기반 선별
- Method A·B·C 종합 점수로 순위 산정
- 하위 15개 제거

### QC 체크
```
[ ] Method A·B·C 모두 완료
[ ] 방향성 확인 (Spearman 높을수록, AUROC 0.5 이상)
[ ] Top 15 선별 완료
[ ] S3 저장 완료
```

---

## STEP 7. ADMET Gate (ML 자동화 · 3단계 필터링)
**예상 소요 시간: 40분~1시간**

### 7-1. ADMET ML 자동화 개요
- 22개 ADMET assay 스크리닝
- Morgan Fingerprint (SMILES 기반) → 피처 생성
- Tanimoto 유사도 매칭으로 학습 데이터 구성
- 각 assay별 개별 모델 학습

### 7-2. 추가 외부 DB (반드시 명시)
| DB | 용도 | 목표 |
|---|---|---|
| PyTDC | 전체 성능 보강 | 85.8% → 90%+ |
| SuperCYP | CYP3A4 억제제/기질 분리 | 80% → 90%+ |
| NCATS ADME | 경계값 케이스 보강 | 오판 감소 |
| ChEMBL ADMET | 커버리지 확대 | - |

> 외부 DB 추가 시 반드시 README·대시보드에 명시.

### 7-3. CYP3A4 분리 모델 (중요)
CYP3A4는 억제제(inhibitor)와 기질(substrate)을 혼합 학습하면 오분류 발생.
반드시 분리하여 각각 학습:
```python
# 억제제 모델
model_cyp3a4_inhibitor = train(cyp3a4_inhibitor_data)

# 기질 모델
model_cyp3a4_substrate = train(cyp3a4_substrate_data)
```

### 7-4. 3단계 필터링 기준
```
Tier 1 Hard Fail → 즉시 탈락
  - hERG > 0.7
  - PAINS > 0
  - Lipinski 위반 > 2

Tier 2 Soft Flag → 검토 후 판단
  - hERG 0.5~0.7
  - DILI · Ames · CYP3A4 · PPB · Caco2

Tier 3 Context → 항암제 특성상 완화 적용
  - F(oral) · t_half · Carcinogenicity
```

### 7-5. 약물 분류 기준 (유방암 기준)
단순 FDA 승인 여부가 아닌 **유방암 적응증 기준**으로 분류:

| 카테고리 | 기준 |
|---|---|
| 유방암 현재 사용 | FDA 유방암 적응증 승인 |
| 유방암 적응증 확장/연구 중 | 임상시험 진행 중·타 암종 승인 |
| 유방암 미사용 | 신약 후보물질 |

> 암종이 다를 경우 해당 암종 기준으로 분류 기준 변경.

### 7-6. 실행
```bash
python admet/admet_ml_v2.py \
  --drug_list    ensemble_results/top15_drugs.csv \
  --external_db  pytdc supercyp ncats_adme chembl \
  --output       admet_results/
```

### QC 기준
| 항목 | biso 기준 | 목표 |
|---|---|---|
| 전체 정확도 | - | 85.8% 이상 |
| hERG | - | 100% |
| Ames | - | 80% 이상 |
| BBB | - | 93% 이상 |
| CYP3A4 | - | 80% 이상 |

### QC 체크
```
[ ] 22개 assay 스크리닝 완료
[ ] 외부 DB 추가 사용 시 명시 확인
[ ] CYP3A4 억제제/기질 분리 적용 확인
[ ] Tier 1 통과 약물 확인
[ ] 유방암 적응증 기준 분류 적용
[ ] 최종 약물 목록 S3 저장 완료
```

---

## 결과 저장 및 공유

```bash
BASE="s3://say2-4team/[본인업무폴더]"

# FE 결과
aws s3 sync fe_output/ ${BASE}/fe_output/

# 모델
aws s3 sync models/ ${BASE}/models/

# 앙상블
aws s3 sync ensemble_results/ ${BASE}/ensemble_results/

# METABRIC
aws s3 sync metabric_results/ ${BASE}/metabric_results/

# ADMET
aws s3 sync admet_results/ ${BASE}/admet_results/

# 대시보드 GitHub push
git add .
git commit -m "feat: [YYYYMMDD] 파이프라인 완료"
git push origin main
```

---

## 전체 체크리스트

```
환경
[ ] conda 환경 생성·패키지 설치 완료
[ ] AWS 자격증명 확인 (Account 666803869796)
[ ] GitHub 저장소 클론 완료
[ ] AWS 태그 설정 확인 (pre-batch-2-4-team)

데이터
[ ] curated_date/ 필수 데이터 확인 (glue/ 미접근)
[ ] 업무폴더로 복사 완료 (glue/ 제외)
[ ] GDSC 버전 선택 및 README 명시
[ ] LINCS 처리 방식 결정 및 명시
[ ] config/data_paths.yaml 본인 경로로 수정

FE
[ ] Nextflow AWS Batch 실행 완료
[ ] METABRIC에 TCGA fit transform만 적용 확인
[ ] 컬럼 수·행 수·매핑률 기록 및 명시
[ ] 결측치 없음 확인

모델
[ ] 15개 모델 학습 완료
[ ] GraphSAGE·GAT drug-split 적용 확인
[ ] 앙상블 통과 모델 목록 확정
[ ] RSF·GraphSAGE METABRIC 전용 분류 확인
[ ] 결측치·이상치·과적합 확인

앙상블
[ ] 앙상블 Spearman > 개별 Best 모델 확인
[ ] Top 30 추출 완료
[ ] Gap ≈ 0 확인

METABRIC
[ ] A+B+C 검증 완료·지표 기록
[ ] Top 15 선별 완료

ADMET
[ ] 외부 DB 명시 후 진행 확인
[ ] CYP3A4 분리 모델 적용 확인
[ ] 유방암 적응증 기준 분류 적용
[ ] 최종 약물 목록 확정

결과
[ ] S3 업로드 완료·경로 공유
[ ] 대시보드 GitHub push 완료
[ ] .aws/credentials Git 커밋 안 됨 확인
[ ] README에 커스터마이징 사항 명시
```

---

## 주의사항 요약

```
❌ curated_date/ 수정·삭제 금지
❌ curated_date/glue/ 접근 금지
❌ METABRIC으로 fit 금지 (transform만)
❌ GraphSAGE·GAT sample-split 금지
❌ .aws/credentials Git 커밋 금지
❌ 팀4 tmp_data 파일 직접 복사 금지 (방법론만 사용)
⚠️ Proxy 데이터 필요 시 팀장 확인 필수
⚠️ ADMET 외부 DB 추가 시 반드시 명시
⚠️ 커스터마이징 사항은 README·대시보드에 기록
```

---

## 문의

궁금한 점은 담당자(say2-4team)에게 문의하거나 GitHub Issues를 활용하세요.

- **메인 대시보드:** https://skkuaws0215.github.io/20260408_pre_project_biso_myprotocol/dashboard.html
- **GitHub:** https://github.com/skkuaws0215/20260408_pre_project_biso_myprotocol.git
