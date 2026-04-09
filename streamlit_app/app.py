#!/usr/bin/env python3
"""
Drug Discovery Pipeline - Streamlit Chat App
UI + Data source integration (Bedrock TBD)
"""
import streamlit as st
import json
import time
from pathlib import Path

from data_sources import (
    query_s3_drug_candidates,
    query_s3_model_results,
    query_s3_admet_results,
    query_s3_metabric_results,
    query_repurposing_candidates,
    query_pubmed,
    query_faers,
    query_clinicaltrials,
    query_chembl,
    query_pubchem,
    DATA_SOURCE_STATUS,
)

st.set_page_config(
    page_title="약물 재창출 어시스턴트",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (Beige warm tone) ──
st.markdown("""
<style>
    .stApp { background-color: #faf7f2; }
    .main .block-container { padding-top: 1rem; max-width: 1200px; }
    [data-testid="stSidebar"] { background-color: #f0e6d3; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #3d2c1e; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 { color: #5c3d2e; }

    /* Chat messages */
    [data-testid="stChatMessage"] { background-color: #fff8ef; border: 1px solid #e8d5b8; border-radius: 12px; }
    [data-testid="stChatMessage"]:nth-of-type(even) { background-color: #f5ebe0; }

    /* Header */
    h1, h2, h3 { color: #5c3d2e !important; }
    p, li, td, th { color: #3d2c1e; }

    /* Buttons */
    .stButton > button {
        background-color: #d4a574 !important; color: #3d2c1e !important;
        border: 1px solid #c4956a !important; border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background-color: #c4956a !important; color: #fff !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        background-color: #fff8ef !important; border: 2px solid #d4a574 !important;
        color: #3d2c1e !important; border-radius: 12px !important;
    }

    .source-badge {
        display: inline-block; padding: 2px 8px; border-radius: 4px;
        font-size: 0.7rem; font-weight: 600; margin-left: 4px;
    }
    .badge-ok { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .badge-pending { background: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
    .badge-na { background: #e2d5c1; color: #6b5b4f; border: 1px solid #d4c4ae; }

    .result-card {
        background: #fff8ef; border: 1px solid #e8d5b8; border-radius: 8px;
        padding: 16px; margin: 8px 0;
    }

    /* Tables */
    table { background-color: #fff8ef !important; }
    th { background-color: #f0e6d3 !important; color: #5c3d2e !important;
         border-bottom: 2px solid #d4a574 !important; }
    td { border-bottom: 1px solid #e8d5b8 !important; color: #3d2c1e !important; }

    /* Spinner */
    [data-testid="stSpinner"] { color: #8b6914 !important; }

    /* Sidebar buttons - each different color */
    [data-testid="stSidebar"] .stButton:nth-of-type(1) > button { background-color: #e8c4a0 !important; }
    [data-testid="stSidebar"] .stButton:nth-of-type(2) > button { background-color: #c9b8a8 !important; }
    [data-testid="stSidebar"] .stButton:nth-of-type(3) > button { background-color: #d4c4ae !important; }
    [data-testid="stSidebar"] .stButton:nth-of-type(4) > button { background-color: #b8d4c8 !important; }
    [data-testid="stSidebar"] .stButton:nth-of-type(5) > button { background-color: #d4b8b8 !important; }
    [data-testid="stSidebar"] .stButton:nth-of-type(6) > button { background-color: #b8c4d4 !important; }
    [data-testid="stSidebar"] .stButton:nth-of-type(7) > button { background-color: #c8c4b8 !important; }
    [data-testid="stSidebar"] .stButton:nth-of-type(8) > button { background-color: #c4b8d4 !important; }
    [data-testid="stSidebar"] .stButton:nth-of-type(9) > button { background-color: #d4c8b8 !important; }

    /* Caption */
    [data-testid="stSidebar"] .stCaption p { color: #8b7355 !important; font-size: 0.75rem; }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ──
if "messages" not in st.session_state:
    st.session_state.messages = []
if "follow_up" not in st.session_state:
    st.session_state.follow_up = None


def classify_query(query: str) -> str:
    """Classify user query into a category for routing."""
    q = query.lower()
    if any(k in q for k in ["재창출", "repurposing", "후보 9", "신약 후보"]):
        return "repurposing"
    if any(k in q for k in ["약물 후보", "추천", "drug candidate", "top 15", "top15", "top 30"]):
        return "drug_candidates"
    if any(k in q for k in ["모델 결과", "model result", "성능", "spearman", "rmse", "앙상블", "ensemble"]):
        return "model_results"
    if any(k in q for k in ["metabric", "외부 검증", "메타브릭"]):
        return "metabric"
    if any(k in q for k in ["부작용", "adverse", "side effect", "faers", "sider"]):
        return "adverse_events"
    if any(k in q for k in ["논문", "연구", "pubmed", "paper", "research", "최신"]):
        return "pubmed"
    if any(k in q for k in ["약가", "보험", "급여", "hira", "가격"]):
        return "hira"
    if any(k in q for k in ["임상", "clinical trial", "시험"]):
        return "clinical_trials"
    if any(k in q for k in ["admet", "독성", "안전", "toxicity", "safety"]):
        return "admet"
    if any(k in q for k in ["신약", "후보물질", "chembl", "화합물"]):
        return "chembl"
    if any(k in q for k in ["구조", "분자", "pubchem", "smiles", "structure"]):
        return "pubchem"
    if any(k in q for k in ["파이프라인", "진행", "현황", "status", "pipeline"]):
        return "pipeline_status"
    return "general"


def get_follow_up_questions(category: str) -> list:
    """Return follow-up question suggestions based on category."""
    follow_ups = {
        "drug_candidates": [
            "재창출 후보 9건 보여줘",
            "ADMET 결과 알려줘",
            "관련 임상시험 알려줘",
        ],
        "repurposing": [
            "ADMET 안전성 결과 알려줘",
            "관련 임상시험 검색해줘",
            "관련 최신 논문 검색해줘",
        ],
        "model_results": [
            "앙상블 결과 보여줘",
            "약물 후보 추천해줘",
            "METABRIC 검증 결과 알려줘",
        ],
        "metabric": [
            "약물 후보 추천해줘",
            "ADMET 결과 알려줘",
            "재창출 후보 9건 보여줘",
        ],
        "adverse_events": [
            "재창출 후보 9건 보여줘",
            "관련 논문 검색해줘",
            "ADMET 프로파일 보여줘",
        ],
        "pubmed": [
            "약물 후보 목록 보여줘",
            "임상시험 현황 알려줘",
            "ADMET 결과 알려줘",
        ],
        "clinical_trials": [
            "재창출 후보 9건 보여줘",
            "최신 논문 검색해줘",
            "부작용 데이터 알려줘",
        ],
        "admet": [
            "재창출 후보 9건 보여줘",
            "약물 후보 목록 보여줘",
            "관련 논문 검색해줘",
        ],
        "pipeline_status": [
            "모델 결과 보여줘",
            "재창출 후보 9건 보여줘",
            "ADMET 결과 알려줘",
        ],
    }
    return follow_ups.get(category, ["재창출 후보 9건 보여줘", "모델 결과 보여줘", "최신 논문 검색해줘"])


def process_query(query: str) -> tuple:
    """Process user query and return (response_text, category)."""
    category = classify_query(query)

    if category == "repurposing":
        result = query_repurposing_candidates()
        if result["status"] == "ok":
            text = result["data"]
        else:
            text = f"재창출 후보 데이터 조회 오류: {result['message']}"

    elif category == "drug_candidates":
        result = query_s3_drug_candidates()
        if result["status"] == "ok":
            drugs = result["data"]
            text = f"**최종 약물 후보 {len(drugs)}건** (ADMET 필터링 + 유방암 적응증 분류)\n\n"
            text += "| # | 약물 | 예측 IC50 | 안전성 | 유방암 분류 |\n"
            text += "|---|------|-----------|--------|------------|\n"
            for d in drugs:
                cls = d.get('brca_class', d.get('category', '-'))
                text += (f"| {d.get('final_rank', d.get('rank', '-'))} | {d.get('drug_name', d.get('drug_id', '-'))} | "
                        f"{d.get('pred_ic50', d.get('mean_pred_ic50', 0)):.3f} | "
                        f"{d.get('safety_score', '-')} | {cls} |\n")
            text += f"\n> 유방암 현재 사용: 6 | 적응증 확장/연구 중: 5 | 유방암 미사용(신약 후보): 4"
        else:
            text = f"아직 약물 후보 데이터가 준비되지 않았습니다.\n\n상태: {result['message']}"

    elif category == "model_results":
        result = query_s3_model_results()
        if result["status"] == "ok":
            text = "**모델 학습 결과 요약**\n\n"
            for section, models in result["data"].items():
                text += f"\n### {section}\n"
                text += "| 모델 | 검증 Spearman | RMSE | 상태 |\n"
                text += "|------|-------------|------|------|\n"
                for m in models:
                    text += f"| {m['model']} | {m['spearman']:.4f} | {m['rmse']:.4f} | {m['status']} |\n"

            if "ensemble" in result:
                e = result["ensemble"]
                text += f"\n### 앙상블\n"
                text += f"- Spearman: {e['spearman']:.4f}\n"
                text += f"- RMSE: {e['rmse']:.4f}\n"
                text += f"- 방법: Spearman 가중 평균 (6개 모델)\n"
        else:
            text = f"모델 결과 조회 중 오류: {result['message']}"

    elif category == "metabric":
        result = query_s3_metabric_results()
        if result["status"] == "ok":
            text = result["data"]
        else:
            text = f"METABRIC 검증 결과 조회 오류: {result['message']}"

    elif category == "admet":
        result = query_s3_admet_results()
        text = result.get("message", "ADMET 결과를 조회합니다...")
        if result["status"] == "ok":
            text = result["data"]

    elif category == "pubmed":
        drug_terms = []
        for word in query.split():
            if len(word) > 2 and word not in ["최신", "논문", "검색", "연구", "알려줘", "보여줘", "해줘"]:
                drug_terms.append(word)
        search_term = " ".join(drug_terms) if drug_terms else "breast cancer drug sensitivity GDSC"
        result = query_pubmed(search_term)
        if result["status"] == "ok":
            articles = result["data"]
            text = f"**PubMed 검색 결과** ('{search_term}')\n\n"
            for i, a in enumerate(articles, 1):
                text += f"**{i}. {a['title']}**\n"
                text += f"- 저자: {a['authors']}\n"
                text += f"- 저널: {a['journal']} ({a['pub_date']})\n"
                text += f"- PMID: [{a['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{a['pmid']}/)\n\n"
        else:
            text = f"PubMed 검색 오류: {result['message']}"

    elif category == "adverse_events":
        result = query_faers(query)
        if result["status"] == "ok":
            text = result["data"]
        else:
            text = f"부작용 데이터 조회 오류: {result['message']}"

    elif category == "clinical_trials":
        result = query_clinicaltrials(query)
        if result["status"] == "ok":
            text = result["data"]
        else:
            text = f"임상시험 데이터 조회 오류: {result['message']}"

    elif category == "chembl":
        result = query_chembl(query)
        if result["status"] == "ok":
            text = result["data"]
        else:
            text = f"ChEMBL 조회 오류: {result['message']}"

    elif category == "pubchem":
        result = query_pubchem(query)
        if result["status"] == "ok":
            text = result["data"]
        else:
            text = f"PubChem 조회 오류: {result['message']}"

    elif category == "hira":
        text = ("HIRA API 연동에는 **공공데이터포털 API 키**가 필요합니다.\n\n"
                "API 키를 제공해 주시면 연동을 진행하겠습니다.\n"
                "(https://www.data.go.kr)")

    elif category == "pipeline_status":
        text = ("**파이프라인 진행 현황 (7/7 완료)**\n\n"
                "| 단계 | 내용 | 상태 |\n"
                "|------|------|------|\n"
                "| 1 | 환경 설정 | ✅ 완료 |\n"
                "| 2 | 데이터 준비 | ✅ 완료 |\n"
                "| 3 | 특성 공학 | ✅ 완료 |\n"
                "| 4 | 모델 학습 (15개) | ✅ 완료 |\n"
                "| 5 | 앙상블 Track2 | ✅ 완료 (Sp=0.8055) |\n"
                "| 6 | METABRIC 외부 검증 | ✅ 완료 (P@15=93.3%) |\n"
                "| 7 | ADMET 게이트 | ✅ 완료 (15개 최종 후보) |\n\n"
                "> **7/7 단계 완료 (100%)**\n\n"
                "**최종 결과**: 15개 약물 후보 (유방암 현재 사용 6 | 연구 중 5 | 신약 후보 4)")

    else:
        text = ("질문을 이해했습니다. 현재 Bedrock 연동 전이라 자연어 대화는 제한적입니다.\n\n"
                "아래 빠른 질문 버튼을 사용해 보세요:\n"
                "- 약물 후보 추천\n"
                "- 재창출 후보 9건\n"
                "- 모델 결과 조회\n"
                "- ADMET 안전성 결과\n"
                "- 파이프라인 현황")

    return text, category


# ── Sidebar ──
with st.sidebar:
    st.markdown("### 💊 약물 재창출 어시스턴트")
    st.markdown("---")

    st.markdown("#### 빠른 질문")
    quick_questions = [
        ("약물 후보 추천해줘", "drug"),
        ("재창출 후보 9건 보여줘", "repurpose"),
        ("모델 결과 보여줘", "model"),
        ("ADMET 결과 알려줘", "admet"),
        ("METABRIC 검증 결과", "metabric"),
        ("최신 논문 검색해줘", "paper"),
        ("부작용 알려줘", "adverse"),
        ("임상시험 알려줘", "trial"),
        ("파이프라인 현황", "status"),
    ]

    for q_text, q_key in quick_questions:
        if st.button(q_text, key=f"quick_{q_key}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q_text})
            response, cat = process_query(q_text)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.follow_up = get_follow_up_questions(cat)
            st.rerun()

    st.markdown("---")
    st.markdown("#### 데이터 소스 현황")
    for source, info in DATA_SOURCE_STATUS.items():
        badge_class = {
            "connected": "badge-ok",
            "available": "badge-ok",
            "pending": "badge-pending",
            "api_key_required": "badge-na",
        }.get(info["status"], "badge-na")
        status_text = {
            "connected": "연동 완료",
            "available": "사용 가능",
            "pending": "예정",
            "api_key_required": "API 키 필요",
        }.get(info["status"], info["status"])
        st.markdown(
            f"**{source}** <span class='source-badge {badge_class}'>{status_text}</span>",
            unsafe_allow_html=True,
        )
        st.caption(info["description"])

    st.markdown("---")
    st.caption("Bedrock 연동: 별도 지시 예정")
    st.caption(f"Version: 2.0.0 | 2026-04-08")


# ── Main Chat Area ──
st.markdown("## 약물 재창출 파이프라인 채팅")

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Follow-up buttons
if st.session_state.follow_up and st.session_state.messages:
    cols = st.columns(len(st.session_state.follow_up))
    for i, fq in enumerate(st.session_state.follow_up):
        with cols[i]:
            if st.button(fq, key=f"followup_{i}_{len(st.session_state.messages)}"):
                st.session_state.messages.append({"role": "user", "content": fq})
                response, cat = process_query(fq)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.follow_up = get_follow_up_questions(cat)
                st.rerun()

# Chat input
if prompt := st.chat_input("질문을 입력하세요 (예: 재창출 후보 9건 보여줘)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("데이터 조회 중..."):
            response, category = process_query(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.follow_up = get_follow_up_questions(category)

    st.rerun()
