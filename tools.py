"""
FortuneAI Tools - 노트북 방식으로 단순화된 도구 모음
"""

from langchain_core.tools import tool
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.prompts import PromptTemplate
from langchain_teddynote.tools.tavily import TavilySearch
from langchain.tools import DuckDuckGoSearchResults
from langchain_google_genai import ChatGoogleGenerativeAI

# 사주 계산 모듈 import
from saju_calculator import SajuCalculator, format_saju_analysis
from reranker import create_saju_compression_retriever


# =============================================================================
# 1. 사주 계산 도구 (Manse Tool)
# =============================================================================
        
@tool
def calculate_saju_tool(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int = 0,
    is_male: bool = True,
    is_leap_month: bool = False
) -> str:
    """
    대한민국 출생자 기준, 생년월일·시간·성별을 입력받아 사주팔자 해석을 반환합니다.
    윤달 출생자의 경우 is_leap_month=True로 설정하세요.
    """
    chart = SajuCalculator().calculate_saju(
        year=year,
        month=month,
        day=day,
        hour=hour,
        minute=minute,
        is_male=is_male,
        is_leap_month=is_leap_month
    )
    return format_saju_analysis(chart, SajuCalculator())

# =============================================================================
# 2. RAG 검색 도구 (Retriever Tool)
# =============================================================================

def create_retriever_tool_for_saju():
    """사주 관련 RAG 검색 도구 생성"""
    pdf_retriever = create_saju_compression_retriever()
    return create_retriever_tool(
        pdf_retriever,
        "pdf_retriever",
        "A tool for searching information related to Saju (Four Pillars of Destiny)",
        document_prompt=PromptTemplate.from_template(
        "<document><context>{page_content}</context><metadata><source>{source}</source></metadata></document>"
    ),
    )

# 전역으로 생성하여 재사용
retriever_tool = create_retriever_tool_for_saju()

# =============================================================================
# 3. 웹 검색 도구들 (Web Tools)
# =============================================================================

tavily_tool = TavilySearch(
    max_results=5,
    include_domains=["namu.wiki", "wikipedia.org"]
)

duck_tool = DuckDuckGoSearchResults(
    max_results=5,
)

web_tools = [tavily_tool, duck_tool]

# =============================================================================
# 4. 일반 QA 도구 (General QA Tool)
# =============================================================================
        
@tool
def general_qa_tool(state):
    """
    일반적인 질문이나 상식적인 내용에 대해 사용자의 사주에 대한 정보를 포함해 답변합니다. 
    """
    google_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    messages = state.get("messages", [])
    # 메시지 없을 때 대비
    query = messages[-1].content if messages else ""
    birth_info = state.get("birth_info")
    saju_result = state.get("saju_result")
    # 전체 메시지 content를 \n으로 연결
    context = "\n".join([m.content for m in messages])

    prompt = f"""아래는 사용자의 질문과 사주 정보입니다.
        질문: {query}
        사주 정보: {birth_info}
        사주 해석: {saju_result}
        대화 기록: {context}

        위 정보를 반드시 참고해서, 사주 특성을 녹여서 친절하고 존댓말로 답변해 주세요.
        맥락에 맞는 답변만 해야합니다.
        """
    return google_llm.invoke(prompt)

# =============================================================================
# 도구 그룹화 (노트북 방식과 동일)
# =============================================================================

# 각 에이전트별 도구 그룹
manse_tools = [calculate_saju_tool]
search_tools = [retriever_tool] + web_tools
general_qa_tools = [general_qa_tool]

# 전체 도구 목록
all_tools = {
    'manse': manse_tools,
    'search': search_tools,
    'web': web_tools,
    'general_qa': general_qa_tools
}

# 직접 import 가능한 도구들
__all__ = [
    'parse_birth_info_tool',
    'calculate_saju_tool',
    'retriever_tool', 
    'tavily_tool',
    'duck_tool',
    'web_tools',
    'general_qa_tool',
    'manse_tools',
    'search_tools',
    'general_qa_tools',
    'all_tools'
] 