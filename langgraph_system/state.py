"""
사주 시스템에 특화된 상태 관리
"""

import operator
from typing import Sequence, Annotated, Dict, List, Any, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage

# 사주 관련 정보 구조
class BirthInfo(TypedDict):
    """출생 정보"""
    year: int
    month: int
    day: int
    hour: int
    minute: int
    is_male: bool
    is_leap_month: bool

class SajuResult(TypedDict):
    """사주 계산 결과"""
    year_pillar: str      # 년주 (예: "을해")
    month_pillar: str     # 월주 (예: "갑신")
    day_pillar: str       # 일주 (예: "기축")
    hour_pillar: str      # 시주 (예: "기사")
    day_master: str       # 일간 (예: "기")
    age: int              # 현재 나이
    korean_age: int       # 한국식 나이
    current_datetime: str # 계산 기준 시점
    
    # 추가 분석 결과
    element_strength: Optional[Dict[str, int]]  # 오행 강약
    ten_gods: Optional[Dict[str, List[str]]]    # 십신 분석
    great_fortunes: Optional[List[Dict[str, Any]]]  # 대운
    yearly_fortunes: Optional[List[Dict[str, Any]]]  # 세운 (연운)
    useful_gods: Optional[List[str]]  # 용신 (유용한 신)
    taboo_gods: Optional[List[str]]   # 기신 (피해야 할 신)

# 핵심 AgentState
class AgentState(TypedDict):
    # 기본 LangGraph 요구사항
    question: str
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    final_answer: Optional[str]
    
    # 세션 관리
    session_id: str
    session_start_time: str
    current_time: str
    
    # 사주 시스템 핵심 정보
    birth_info: Optional[BirthInfo]     # 출생 정보
    saju_result: Optional[SajuResult]   # 사주 계산 결과
    query_type: str                     # "saju", "tarot", "general" 등
    
    # 에이전트 간 데이터 공유
    retrieved_docs: List[Dict[str, Any]]  # RAG 검색 결과
    web_search_results: List[Dict[str, Any]]  # 웹 검색 결과
