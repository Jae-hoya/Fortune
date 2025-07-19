"""
FortuneAI Tools - 노트북 방식으로 단순화된 도구 모음
"""

from langchain_core.tools import tool
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.prompts import PromptTemplate
from langchain_teddynote.tools.tavily import TavilySearch
from langchain.tools import DuckDuckGoSearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
import re
from typing import Dict, Any, List, Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# 사주 계산 모듈 import
from saju_calculator import SajuCalculator, format_saju_analysis
from reranker import create_saju_compression_retriever

# =============================================================================
# 0. Supervisor 도구들
# =============================================================================

class BirthInfoParsed(BaseModel):
    """파싱된 출생 정보"""
    year: int = Field(description="출생 연도 (4자리)")
    month: int = Field(description="출생 월 (1-12)")
    day: int = Field(description="출생 일 (1-31)")
    hour: int = Field(description="출생 시간 (0-23)")
    minute: int = Field(description="출생 분 (0-59)")
    is_male: bool = Field(description="남성 여부 (True: 남성, False: 여성)")
    is_leap_month: bool = Field(description="윤달 여부")


class SupervisorDecision(BaseModel):
    action: Literal["ROUTE", "DIRECT", "BIRTH_INFO_REQUEST", "FINISH"] = Field(
        description="수행할 액션 유형. 'ROUTE'는 전문 에이전트로 위임, 'DIRECT'는 직접 짧은 답변, 'BIRTH_INFO_REQUEST'는 생일 정보 요청, 'FINISH'는 최종 답변 제공."
    )
    next: Literal["SajuExpert", "Search", "GeneralAnswer", "FINISH", "Supervisor"] = Field(
        description="다음으로 라우팅할 에이전트 노드의 이름 (SajuExpert, Search, GeneralAnswer) 또는 'FINISH' (대화 종료) 또는 'Supervisor' (Supervisor가 재추론)."
    )
    request: str = Field(
        description="다음 에이전트에게 전달할 구체적인 작업 명령 (action이 'ROUTE'일 때만 사용) 또는 '명령 없음'."
    )
    final_answer: str | None = Field(
        default=None,
        description="최종 사용자에게 전달할 답변 (action이 'DIRECT', 'BIRTH_INFO_REQUEST', 'FINISH'일 때 사용).",
    )
    reason: str = Field(description="이 결정을 내린 간략한 이유.")
    birth_info: Dict[str, Any] | None = Field(
        default=None, description="업데이트되거나 파싱된 출생 정보 (딕셔너리 형태)."
    )
    query_type: Literal["saju", "general", "concept", "unknown"] | None = Field(
        default=None, description="사용자 질의의 유형 (예: 'saju', 'general', 'concept')."
    )


@tool
def parse_birth_info_tool(user_input: str) -> dict:
    """LLM을 사용해서 사용자 입력에서 출생 정보를 파싱합니다.
    
    Args:
        user_input: 사용자가 입력한 텍스트 (예: "1995년 8월 26일 오전 10시 반 남자")
        
    Returns:
        dict: 파싱된 출생 정보 (BirthInfo 형식) 또는 빈 딕셔너리
    """
    if not user_input or len(user_input.strip()) < 5:
        return {}
    
    try:
        # LLM 설정
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # 출력 파서 설정
        parser = JsonOutputParser(pydantic_object=BirthInfoParsed)
        
        # 프롬프트 템플릿
        prompt = ChatPromptTemplate.from_template(
                """
                다음 텍스트에서 출생 정보를 추출해주세요.

                텍스트: {input}

                ## 추출 규칙:
                1. 연도: 4자리 연도로 변환 (예: 95년 → 1995년, 05년 → 2005년)
                2. 시간: 24시간 형식 (오전/오후 고려, 새벽=0-6시, 밤=18-23시)
                3. 분: 명시되지 않으면 0, "반"이면 30분
                4. 성별: 남자/남성/남 → True, 여자/여성/여 → False
                5. 윤달: "윤"이 언급되면 True, 아니면 False
                6. 만약 시각 정보가 없으면 00시 00분으로 설정
                
                ## 출력 형식
                {format_instructions}
                """
                )
        
        # 체인 생성
        chain = prompt | llm | parser
        
        # 실행
        result = chain.invoke({
            "input": user_input,
            "format_instructions": parser.get_format_instructions()
        })
        
        
        # BirthInfo 형식으로 변환
        return {
            "year": result["year"],
            "month": result["month"],
            "day": result["day"],
            "hour": result["hour"],
            "minute": result["minute"],
            "is_male": result["is_male"],
            "is_leap_month": result["is_leap_month"],
        }
        
    except Exception as e:
        print(f"출생정보 파싱 중 오류: {e}")
        return {}

@tool
def make_supervisor_decision(decision: SupervisorDecision) -> str:
    """주어진 SupervisorDecision 객체를 바탕으로 다음 단계를 결정하고, 시스템의 상태를 업데이트하도록 지시합니다.
    이 도구를 호출하면 LangGraph의 라우팅이 시작됩니다.
    decision: SupervisorDecision 모델에 정의된 결정을 JSON 형식으로 제공합니다.
    """
    print(f"\n[Tool Called] make_supervisor_decision with decision:\n{decision.model_dump_json(indent=2)}")
    return decision.model_dump_json()

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
            '{{"context": "{page_content}", "metadata": {{"source": "{source}"}}'
        ),
    )

# 전역으로 생성하여 재사용
saju_retriever_tool = create_retriever_tool_for_saju()

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
def general_qa_tool(query: str) -> str:
    """
    일반적인 질문이나 상식적인 내용에 대해 답변합니다. 사주와 관련 없는 모든 질문에 사용할 수 있습니다.
    """
    google_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    return google_llm.invoke(query).content

# =============================================================================
# 도구 그룹화 (노트북 방식과 동일)
# =============================================================================

# 각 에이전트별 도구 그룹
supervisor_tools = [parse_birth_info_tool, make_supervisor_decision]  # Supervisor 전용 도구
saju_tools = [calculate_saju_tool]
search_tools = [saju_retriever_tool] + web_tools
general_qa_tools = [general_qa_tool]

# 전체 도구 목록
all_tools = {
    'supervisor': supervisor_tools,
    'saju': saju_tools,
    'search': search_tools,
    'web': web_tools,
    'general_qa': general_qa_tools
}

# 직접 import 가능한 도구들
__all__ = [
    'parse_birth_info_tool',
    'calculate_saju_tool',
    'saju_retriever_tool', 
    'tavily_tool',
    'duck_tool',
    'general_qa_tool',
    'supervisor_tools',
    'saju_tools',
    'search_tools',
    'general_qa_tools',
    'all_tools'
] 