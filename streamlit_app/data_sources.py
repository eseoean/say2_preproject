"""
Data source integrations for Drug Discovery Chat App.
Priority order: S3 > PubMed > FAERS > ClinicalTrials > ChEMBL > PubChem
"""
import json
import os
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

# ── S3 Config ──
S3_BASE = os.getenv("S3_BASE", "s3://say2-4team/20260409_eseo")
# Local results paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
ENSEMBLE_RESULTS = PROJECT_ROOT / "models" / "ensemble_results" / "ensemble_results.json"
TOP15_CSV = PROJECT_ROOT / "models" / "ensemble_results" / "top15_drugs.csv"
TOP30_CSV = PROJECT_ROOT / "models" / "ensemble_results" / "top30_drugs.csv"
ML_RESULTS = PROJECT_ROOT / "models" / "ml_results"
DL_RESULTS = PROJECT_ROOT / "models" / "dl_results" / "dl_results.json"
GRAPH_RESULTS = PROJECT_ROOT / "models" / "graph_results" / "graph_results.json"
METABRIC_RESULTS = PROJECT_ROOT / "models" / "metabric_results" / "step6_metabric_results.json"
ADMET_RESULTS_JSON = PROJECT_ROOT / "models" / "admet_results" / "step7_admet_results.json"
ADMET_RESULTS_CSV = PROJECT_ROOT / "models" / "admet_results" / "final_drug_candidates.csv"
TOP15_VALIDATED = PROJECT_ROOT / "models" / "metabric_results" / "top15_validated.csv"

# ── Drug classification (breast cancer indication) ──
DRUG_BRCA_CLASS = {
    "Dactinomycin": {"class": "유방암 미사용", "color": "novel", "reason": "FDA 승인: 윌름스 종양, 융모성 질환. 유방암 적응증 없음"},
    "Docetaxel": {"class": "유방암 현재 사용", "color": "current", "reason": "FDA 유방암 적응증 승인 (보조/전이성)"},
    "Vinblastine": {"class": "유방암 현재 사용", "color": "current", "reason": "FDA 유방암 적응증 포함"},
    "Staurosporine": {"class": "유방암 미사용", "color": "novel", "reason": "연구용 화합물, FDA 미승인"},
    "Bortezomib": {"class": "적응증 확장/연구 중", "color": "trial", "reason": "유방암 임상시험 진행 (NCT00025246)"},
    "Vinorelbine": {"class": "유방암 현재 사용", "color": "current", "reason": "전이성 유방암 표준요법"},
    "SN-38": {"class": "적응증 확장/연구 중", "color": "trial", "reason": "Sacituzumab govitecan (SN-38 접합체) FDA TNBC 승인"},
    "Dinaciclib": {"class": "적응증 확장/연구 중", "color": "trial", "reason": "CDK 억제제, 유방암 임상 1/2상"},
    "Paclitaxel": {"class": "유방암 현재 사용", "color": "current", "reason": "FDA 유방암 적응증 승인"},
    "Rapamycin": {"class": "적응증 확장/연구 중", "color": "trial", "reason": "유도체 Everolimus FDA HR+/HER2- 유방암 승인"},
    "Camptothecin": {"class": "유방암 미사용", "color": "novel", "reason": "연구용, Irinotecan/Topotecan 모 화합물"},
    "Luminespib": {"class": "적응증 확장/연구 중", "color": "trial", "reason": "HSP90 억제제, 유방암 2상 임상시험"},
    "Epirubicin": {"class": "유방암 현재 사용", "color": "current", "reason": "FDA 유방암 보조요법 승인"},
    "Romidepsin": {"class": "적응증 확장/연구 중", "color": "trial", "reason": "FDA CTCL/PTCL 승인, 유방암 TNBC 임상시험 (NCT02393794)"},
    "Sepantronium bromide": {"class": "유방암 미사용", "color": "novel", "reason": "연구용 Survivin(BIRC5) 억제제, 유방암 적응증 없음"},
}

# ── Data Source Status ──
DATA_SOURCE_STATUS = {
    "S3 Pipeline": {
        "status": "connected",
        "description": "파이프라인 결과 (약물 추천, 모델 결과, ADMET)",
    },
    "PubMed": {
        "status": "available",
        "description": "최신 논문 검색 (API 키 불필요)",
    },
    "FAERS (FDA)": {
        "status": "available",
        "description": "부작용 데이터 (open API)",
    },
    "ClinicalTrials.gov": {
        "status": "available",
        "description": "임상시험 정보 (API v2)",
    },
    "ChEMBL": {
        "status": "available",
        "description": "신약 후보물질 데이터",
    },
    "PubChem": {
        "status": "available",
        "description": "분자 구조, SMILES",
    },
    "HIRA": {
        "status": "api_key_required",
        "description": "약가/급여/비급여 (공공데이터포털 API 키 필요)",
    },
    "Bedrock LLM": {
        "status": "pending",
        "description": "자연어 대화 (연동 예정)",
    },
}


# ═══════════════════════════════════════════
# 1순위: S3 Pipeline Results
# ═══════════════════════════════════════════

def query_s3_drug_candidates():
    """Load drug candidates from ensemble results with breast cancer classification."""
    if ADMET_RESULTS_CSV.exists():
        df = pd.read_csv(ADMET_RESULTS_CSV)
        records = df.to_dict(orient="records")
        for r in records:
            name = r.get("drug_name", "")
            cls = DRUG_BRCA_CLASS.get(name, {})
            r["brca_class"] = cls.get("class", "미분류")
            r["brca_reason"] = cls.get("reason", "")
        return {"status": "ok", "data": records}

    if TOP15_CSV.exists():
        df = pd.read_csv(TOP15_CSV)
        return {"status": "ok", "data": df.to_dict(orient="records")}

    if ENSEMBLE_RESULTS.exists():
        with open(ENSEMBLE_RESULTS) as f:
            data = json.load(f)
        if "top15_drugs" in data:
            return {"status": "ok", "data": data["top15_drugs"]}
        if "top30_drugs" in data:
            return {"status": "ok", "data": data["top30_drugs"][:15]}

    return {
        "status": "pending",
        "message": "앙상블 학습이 진행 중입니다. 완료 후 약물 후보가 표시됩니다."
    }


def query_s3_model_results():
    """Load model training results from local files."""
    results = {"status": "ok", "data": {}}

    # ML results
    ml_known = [
        {"model": "CatBoost", "spearman": 0.8007, "rmse": 1.3172, "status": "통과"},
        {"model": "LightGBM", "spearman": 0.7913, "rmse": 1.3438, "status": "통과"},
        {"model": "XGBoost", "spearman": 0.7895, "rmse": 1.3445, "status": "통과"},
        {"model": "LightGBM-DART", "spearman": 0.7848, "rmse": 1.4029, "status": "실패 (RMSE)"},
        {"model": "Stacking", "spearman": 0.7981, "rmse": 1.3213, "status": "제외"},
        {"model": "ExtraTrees", "spearman": 0.6468, "rmse": 1.8704, "status": "실패"},
        {"model": "RandomForest", "spearman": 0.6267, "rmse": 1.9747, "status": "실패"},
        {"model": "RSF", "spearman": 0.6142, "rmse": 0.0, "status": "METABRIC 전용"},
    ]
    results["data"]["ML 모델 (8개)"] = ml_known

    # DL results
    if DL_RESULTS.exists():
        with open(DL_RESULTS) as f:
            dl_data = json.load(f)
        dl_models = []
        for m in dl_data:
            sp_pass = m["spearman_mean"] >= 0.713
            rm_pass = m["rmse_mean"] <= 1.385
            status = "통과" if (sp_pass and rm_pass) else "실패"
            dl_models.append({
                "model": m["model"],
                "spearman": m["spearman_mean"],
                "rmse": m["rmse_mean"],
                "status": status,
            })
        results["data"]["DL 모델 (5개)"] = dl_models
    else:
        results["data"]["DL 모델 (5개)"] = [
            {"model": "FlatMLP", "spearman": 0.7936, "rmse": 1.3429, "status": "통과"},
            {"model": "ResidualMLP", "spearman": 0.7855, "rmse": 1.3776, "status": "통과"},
            {"model": "Cross-Attention", "spearman": 0.7852, "rmse": 1.3716, "status": "통과"},
            {"model": "TabNet", "spearman": 0.7780, "rmse": 1.3892, "status": "실패 (RMSE)"},
            {"model": "FT-Transformer", "spearman": 0.7625, "rmse": 1.4444, "status": "실패 (RMSE)"},
        ]

    # Graph results
    if GRAPH_RESULTS.exists():
        with open(GRAPH_RESULTS) as f:
            gr_data = json.load(f)
        gr_models = []
        for m in gr_data:
            gr_models.append({
                "model": m["model"],
                "spearman": m["spearman_mean"],
                "rmse": m["rmse_mean"],
                "status": f"실패 (P@20={m.get('p_at_20_mean', 0):.2f})",
            })
        results["data"]["그래프 모델 (2개)"] = gr_models
    else:
        results["data"]["그래프 모델 (2개)"] = [
            {"model": "GraphSAGE", "spearman": 0.3852, "rmse": 2.3189, "status": "METABRIC P@20"},
            {"model": "GAT", "spearman": 0.0085, "rmse": 2.6608, "status": "실패"},
        ]

    # Ensemble
    if ENSEMBLE_RESULTS.exists():
        with open(ENSEMBLE_RESULTS) as f:
            ens = json.load(f)
        results["ensemble"] = {
            "spearman": ens["ensemble_metrics"]["spearman_mean"],
            "rmse": ens["ensemble_metrics"]["rmse_mean"],
        }

    return results


def query_s3_admet_results():
    """Load ADMET results with breast cancer classification."""
    if ADMET_RESULTS_CSV.exists():
        df = pd.read_csv(ADMET_RESULTS_CSV)
        text = "**ADMET 안전성 평가 결과 (22개 분석 항목)**\n\n"

        # Classification summary
        current = trial = novel = 0
        for _, row in df.iterrows():
            cls = DRUG_BRCA_CLASS.get(row["drug_name"], {}).get("class", "미분류")
            if cls == "유방암 현재 사용":
                current += 1
            elif cls == "적응증 확장/연구 중":
                trial += 1
            elif cls == "유방암 미사용":
                novel += 1

        text += f"**유방암 적응증 분류**: 현재 사용 {current} | 연구 중 {trial} | 미사용(신약 후보) {novel}\n\n"
        text += "| # | 약물 | 타겟 | 예측 IC50 | 안전성 | 종합 | 유방암 분류 | 우려사항 |\n"
        text += "|---|------|------|-----------|--------|------|------------|----------|\n"
        for _, row in df.iterrows():
            cls = DRUG_BRCA_CLASS.get(row["drug_name"], {}).get("class", "미분류")
            flags = row.get("flags", "[]")
            if "DILI" in str(flags):
                flag_str = "DILI"
            elif "Ames" in str(flags):
                flag_str = "Ames"
            else:
                flag_str = "-"
            text += (f"| {row['final_rank']} | {row['drug_name']} | {row['target']} | "
                    f"{row['pred_ic50']:.3f} | {row['safety_score']:.1f} | {row['combined_score']:.1f} | "
                    f"{cls} | {flag_str} |\n")

        text += "\n> 출처: TDC ADMET Benchmark (22개 분석) + DrugBank/ClinicalTrials.gov"
        return {"status": "ok", "data": text}

    return {
        "status": "pending",
        "message": "ADMET 필터링은 Step 7에서 진행 예정입니다."
    }


def query_s3_metabric_results():
    """Load METABRIC validation results."""
    if METABRIC_RESULTS.exists():
        with open(METABRIC_RESULTS) as f:
            data = json.load(f)

        ma = data.get("method_a", {})
        mb = data.get("method_b", {})
        mc = data.get("method_c", {})

        text = "**METABRIC 외부 검증 결과 (A+B+C)**\n\n"
        text += f"- **방법 A**: {ma.get('n_targets_expressed', 0)}/{ma.get('n_total', 0)} 타겟 발현, "
        text += f"{ma.get('n_brca_pathway', 0)}/{ma.get('n_total', 0)} BRCA 관련 경로\n"
        text += f"- **방법 B**: {mb.get('n_survival_significant', 0)}/{mb.get('n_total', 0)} 생존 유의 (p<0.05)\n"

        if mc.get("p_at_k"):
            for k, v in mc["p_at_k"].items():
                text += f"- **P@{k}**: {v:.1%}\n"

        text += f"\n> 검증 데이터: METABRIC (1,980 환자 × 20,603 유전자)"
        return {"status": "ok", "data": text}

    return {"status": "pending", "message": "METABRIC 검증은 Step 6에서 진행 예정입니다."}


def query_repurposing_candidates():
    """Return the 9 repurposing candidates (5 trial + 4 novel)."""
    if not ADMET_RESULTS_CSV.exists():
        return {"status": "pending", "message": "약물 재창출 후보 데이터가 아직 없습니다."}

    df = pd.read_csv(ADMET_RESULTS_CSV)
    candidates = []
    for _, row in df.iterrows():
        name = row["drug_name"]
        cls = DRUG_BRCA_CLASS.get(name, {})
        if cls.get("class") in ["적응증 확장/연구 중", "유방암 미사용"]:
            candidates.append({
                "rank": row["final_rank"],
                "name": name,
                "target": row["target"],
                "pred_ic50": row["pred_ic50"],
                "safety": row["safety_score"],
                "combined": row["combined_score"],
                "brca_class": cls["class"],
                "reason": cls["reason"],
            })

    text = f"**약물 재창출 후보 {len(candidates)}건**\n\n"
    text += "| # | 약물 | 타겟 | 예측 IC50 | 안전성 | 종합 | 유방암 분류 | 분류 근거 |\n"
    text += "|---|------|------|-----------|--------|------|------------|----------|\n"
    for c in candidates:
        text += (f"| {c['rank']} | **{c['name']}** | {c['target']} | "
                f"{c['pred_ic50']:.3f} | {c['safety']:.1f} | {c['combined']:.1f} | "
                f"{c['brca_class']} | {c['reason']} |\n")

    text += "\n> 상세 분석: [재창출 후보 9건 대시보드](repurposing_candidates.html)"
    return {"status": "ok", "data": text}


# ═══════════════════════════════════════════
# 1순위: PubMed API (Free, no API key)
# ═══════════════════════════════════════════

def query_pubmed(search_term: str, max_results: int = 5):
    """Search PubMed for articles."""
    try:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        search_url = (f"{base_url}/esearch.fcgi?"
                     f"db=pubmed&term={urllib.parse.quote(search_term)}"
                     f"&retmax={max_results}&sort=date&retmode=json")

        req = urllib.request.Request(search_url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            search_data = json.loads(resp.read().decode())

        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return {"status": "ok", "data": [], "message": "검색 결과가 없습니다."}

        ids_str = ",".join(id_list)
        fetch_url = (f"{base_url}/esummary.fcgi?"
                    f"db=pubmed&id={ids_str}&retmode=json")

        req = urllib.request.Request(fetch_url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            fetch_data = json.loads(resp.read().decode())

        articles = []
        result_data = fetch_data.get("result", {})
        for pmid in id_list:
            if pmid in result_data:
                article = result_data[pmid]
                authors = ", ".join(
                    a.get("name", "") for a in article.get("authors", [])[:3]
                )
                if len(article.get("authors", [])) > 3:
                    authors += " et al."
                articles.append({
                    "pmid": pmid,
                    "title": article.get("title", "N/A"),
                    "authors": authors,
                    "journal": article.get("source", "N/A"),
                    "pub_date": article.get("pubdate", "N/A"),
                })

        return {"status": "ok", "data": articles}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# ═══════════════════════════════════════════
# 2순위: FAERS (FDA Adverse Event Reporting)
# ═══════════════════════════════════════════

def query_faers(query: str, limit: int = 5):
    """Query FDA FAERS for adverse events."""
    try:
        drug_name = None
        for word in query.split():
            if len(word) > 3 and word.lower() not in [
                "부작용", "알려줘", "보여줘", "해줘", "이것", "약물"
            ]:
                drug_name = word
                break

        if not drug_name:
            drug_name = "docetaxel"

        url = (f"https://api.fda.gov/drug/event.json?"
              f"search=patient.drug.medicinalproduct:\"{urllib.parse.quote(drug_name)}\""
              f"&count=patient.reaction.reactionmeddrapt.exact&limit={limit}")

        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        results = data.get("results", [])
        if not results:
            return {"status": "ok", "data": f"'{drug_name}'에 대한 부작용 데이터가 없습니다."}

        text = f"**FAERS 부작용 보고** ('{drug_name}')\n\n"
        text += "| 부작용 | 보고 건수 |\n"
        text += "|--------|----------|\n"
        for r in results:
            text += f"| {r['term']} | {r['count']:,} |\n"
        text += f"\n> 출처: FDA FAERS (https://api.fda.gov)"

        return {"status": "ok", "data": text}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# ═══════════════════════════════════════════
# 3순위: ClinicalTrials.gov API v2
# ═══════════════════════════════════════════

def query_clinicaltrials(query: str, max_results: int = 5):
    """Search ClinicalTrials.gov."""
    try:
        terms = []
        for word in query.split():
            if len(word) > 2 and word.lower() not in [
                "임상", "시험", "알려줘", "보여줘", "현황"
            ]:
                terms.append(word)

        search_term = " ".join(terms) if terms else "breast cancer drug sensitivity"

        url = (f"https://clinicaltrials.gov/api/v2/studies?"
              f"query.term={urllib.parse.quote(search_term)}"
              f"&pageSize={max_results}&sort=LastUpdatePostDate:desc"
              f"&format=json")

        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        studies = data.get("studies", [])
        if not studies:
            return {"status": "ok", "data": f"'{search_term}' 관련 임상시험이 없습니다."}

        text = f"**ClinicalTrials.gov 검색 결과** ('{search_term}')\n\n"
        for i, study in enumerate(studies, 1):
            proto = study.get("protocolSection", {})
            id_mod = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            design_mod = proto.get("designModule", {})

            nct_id = id_mod.get("nctId", "N/A")
            title = id_mod.get("briefTitle", "N/A")
            status = status_mod.get("overallStatus", "N/A")
            phase = ", ".join(design_mod.get("phases", ["N/A"]))

            text += f"**{i}. {title}**\n"
            text += f"- NCT ID: [{nct_id}](https://clinicaltrials.gov/study/{nct_id})\n"
            text += f"- 상태: {status} | 단계: {phase}\n\n"

        return {"status": "ok", "data": text}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# ═══════════════════════════════════════════
# 5순위: ChEMBL API
# ═══════════════════════════════════════════

def query_chembl(query: str, limit: int = 5):
    """Search ChEMBL for compounds."""
    try:
        terms = []
        for word in query.split():
            if len(word) > 2 and word.lower() not in [
                "신약", "후보", "물질", "알려줘", "보여줘", "화합물"
            ]:
                terms.append(word)
        search_term = " ".join(terms) if terms else "breast cancer"

        url = (f"https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?"
              f"q={urllib.parse.quote(search_term)}&limit={limit}")

        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        molecules = data.get("molecules", [])
        if not molecules:
            return {"status": "ok", "data": f"'{search_term}' 관련 ChEMBL 화합물이 없습니다."}

        text = f"**ChEMBL 화합물 검색** ('{search_term}')\n\n"
        text += "| ChEMBL ID | 이름 | 유형 | 최대 임상 단계 |\n"
        text += "|-----------|------|------|---------------|\n"
        for m in molecules:
            chembl_id = m.get("molecule_chembl_id", "N/A")
            name = m.get("pref_name", "N/A") or "N/A"
            mol_type = m.get("molecule_type", "N/A")
            phase = m.get("max_phase", "N/A")
            text += f"| {chembl_id} | {name} | {mol_type} | {phase} |\n"

        text += f"\n> 출처: ChEMBL (https://www.ebi.ac.uk/chembl/)"
        return {"status": "ok", "data": text}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# ═══════════════════════════════════════════
# 5순위: PubChem API
# ═══════════════════════════════════════════

def query_pubchem(query: str):
    """Search PubChem for compound info."""
    try:
        terms = []
        for word in query.split():
            if len(word) > 2 and word.lower() not in [
                "구조", "분자", "알려줘", "보여줘", "smiles"
            ]:
                terms.append(word)
        compound_name = " ".join(terms) if terms else "docetaxel"

        url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
              f"{urllib.parse.quote(compound_name)}/JSON")

        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        compounds = data.get("PC_Compounds", [])
        if not compounds:
            return {"status": "ok", "data": f"'{compound_name}' PubChem 검색 결과 없음"}

        comp = compounds[0]
        cid = comp.get("id", {}).get("id", {}).get("cid", "N/A")

        props = {}
        for p in comp.get("props", []):
            label = p.get("urn", {}).get("label", "")
            name = p.get("urn", {}).get("name", "")
            val = p.get("value", {})
            value = val.get("sval", val.get("fval", val.get("ival", "")))
            if label and value:
                props[f"{label} ({name})" if name else label] = value

        text = f"**PubChem 화합물 정보** ('{compound_name}')\n\n"
        text += f"- CID: [{cid}](https://pubchem.ncbi.nlm.nih.gov/compound/{cid})\n"

        for key in ["IUPAC Name (Preferred)", "IUPAC Name (Systematic)",
                    "Molecular Formula", "Molecular Weight", "SMILES (Canonical)",
                    "InChI (Standard)"]:
            for pk, pv in props.items():
                if key.split(" (")[0] in pk:
                    text += f"- {pk}: `{pv}`\n"
                    break

        text += f"\n> 출처: PubChem (https://pubchem.ncbi.nlm.nih.gov)"
        return {"status": "ok", "data": text}

    except Exception as e:
        return {"status": "error", "message": str(e)}
