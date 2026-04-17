# KG/API Server MVP

`FRC (FlatMLP + ResidualMLP + CrossAttention)` 결과를 기준으로,
팀원 레포의 `Step 7+ KG/API 검증` 흐름을 로컬에서 재현하기 위한 경량 JSON API 서버입니다.

이 MVP는 Python 표준 라이브러리만 사용하므로 별도 패키지 설치 없이 바로 실행할 수 있습니다.

## 제공 엔드포인트

- `GET /health`
- `GET /api/drug/{name}`
- `GET /api/drug/{name}/targets`
- `GET /api/drug/{name}/pathways`
- `GET /api/drug/{name}/side_effects`
- `GET /api/drug/{name}/trials`
- `GET /api/pubmed?query=...&max_results=...`

## 응답 성격

- `drug / targets / pathways / side_effects`
  - 현재 로컬 `Step 6 / Step 7 / post-ADMET summary`를 바탕으로 응답합니다.
- `trials / pubmed`
  - 실제 외부 API를 먼저 시도합니다.
  - `trials` → ClinicalTrials.gov API v2
  - `pubmed` → NCBI E-utilities (PubMed)
- `side_effects`
  - `openFDA FAERS`를 먼저 시도합니다.
- 네트워크가 막히면 로컬 fallback 응답으로 내려갑니다.

## 캐시

- 외부 API 응답은 아래 폴더에 JSON으로 캐시됩니다.
  - `/Users/skku_aws2_18/pre_project/say2_preproject/kg_api_server/cache`
- 같은 질의를 반복할 때 응답 속도가 빨라지고, 외부 API 호출량도 줄어듭니다.

## 실행 순서

### 1. Seed catalog 생성

```bash
source /Users/skku_aws2_18/pre_project/say2_preproject/project_env.sh
/Users/skku_aws2_18/pre_project/.conda-envs/drug4/bin/python \
  /Users/skku_aws2_18/pre_project/say2_preproject/kg_api_server/build_seed_catalog.py
```

### 2. 서버 실행

```bash
source /Users/skku_aws2_18/pre_project/say2_preproject/project_env.sh
/Users/skku_aws2_18/pre_project/.conda-envs/drug4/bin/python \
  /Users/skku_aws2_18/pre_project/say2_preproject/kg_api_server/server.py \
  --host 127.0.0.1 \
  --port 8000
```

### 3. 상태 확인

```bash
curl http://127.0.0.1:8000/health
```

예시:

```bash
curl "http://127.0.0.1:8000/api/drug/Docetaxel"
curl "http://127.0.0.1:8000/api/drug/Docetaxel/trials"
curl "http://127.0.0.1:8000/api/pubmed?query=Docetaxel+breast+cancer&max_results=5"
```

### 4. Step 7+ 수집 실행

```bash
source /Users/skku_aws2_18/pre_project/say2_preproject/project_env.sh
/Users/skku_aws2_18/pre_project/.conda-envs/drug4/bin/python \
  /Users/skku_aws2_18/pre_project/say2_preproject/models/collect_frc_kg_api_data.py \
  --api-base http://127.0.0.1:8000 \
  --mode all
```

## 주요 파일

- seed catalog:
  - `/Users/skku_aws2_18/pre_project/say2_preproject/kg_api_server/data/frc_seed_catalog.json`
- server:
  - `/Users/skku_aws2_18/pre_project/say2_preproject/kg_api_server/server.py`
- collector:
  - `/Users/skku_aws2_18/pre_project/say2_preproject/models/collect_frc_kg_api_data.py`

## 다음 확장 포인트

- `side_effects` → openFDA FAERS 연동
- `targets/pathways` → Neo4j KG 질의 연동
- `drug/{name}` → PubChem / DrugBank 메타 확장
