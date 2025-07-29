"""
사주 시스템에 특화된 상태 관리
"""

import operator
from typing import Sequence, Annotated, Dict, List, Any, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# 사주 관련 정보 구조
class BirthInfo(TypedDict):
    """출생 정보"""
    year: Annotated[int, "태어난 년도 (양력)"]
    month: Annotated[int, "태어난 월 (1-12)"]
    day: Annotated[int, "태어난 일 (1-31)"]
    hour: Annotated[int, "태어난 시간 (0-23)"]
    minute: Annotated[int, "태어난 분 (0-59)"]
    is_male: Annotated[bool, "성별 (True: 남성, False: 여성)"]
    is_leap_month: Annotated[bool, "윤달 여부"]

class SajuInfo(TypedDict):
    """사주 계산 결과"""
    year_pillar: Annotated[str, "년주 - 년간년지 (예: '을해')"]
    month_pillar: Annotated[str, "월주 - 월간월지 (예: '갑신')"]
    day_pillar: Annotated[str, "일주 - 일간일지 (예: '기축')"]
    hour_pillar: Annotated[str, "시주 - 시간시지 (예: '기사')"]
    day_master: Annotated[str, "일간 - 자신을 나타내는 천간 (예: '기')"]
    age: Annotated[int, "현재 만나이"]
    korean_age: Annotated[int, "한국식 나이 (세는나이)"]
    
    # 추가 분석 결과
    element_strength: Annotated[Optional[Dict[str, int]], "오행별 강약 분석 (목화토금수)"]
    ten_gods: Annotated[Optional[Dict[str, List[str]]], "십신 분석 (비견, 겁재, 식신 등)"]
    great_fortunes: Annotated[Optional[List[Dict[str, Any]]], "대운 - 10년 단위 운세"]
    yearly_fortunes: Annotated[Optional[List[Dict[str, Any]]], "세운 - 연도별 운세"]
    useful_gods: Annotated[Optional[List[str]], "용신 - 도움이 되는 오행/십신"]
    taboo_gods: Annotated[Optional[List[str]], "기신 - 피해야 할 오행/십신"]


# 핵심 AgentState
class AgentState(TypedDict):
    # 기본 LangGraph 요구사항
    question: Annotated[str, "사용자의 질문 또는 요청"]
    messages: Annotated[Sequence[BaseMessage], add_messages, "대화 메시지 목록 (자동 중복 제거 및 메시지 관리)"]
    next: Annotated[str, "다음에 실행할 노드명"]
    final_answer: Annotated[Optional[str], "최종 답변 결과"]
    
    # 세션 관리
    session_id: Annotated[str, "세션 고유 식별자"]
    session_start_time: Annotated[str, "세션 시작 시간"]
    current_time: Annotated[str, "현재 시간"]
    
    # 사주 시스템 핵심 정보
    birth_info: Annotated[Optional[BirthInfo], "사용자 출생 정보 (년월일시분, 성별, 윤달여부)"]
    saju_info: Annotated[Optional[SajuInfo], "사주 계산 및 분석 결과"]
    query_type: Annotated[str, "질문 유형 (saju/tarot/general)"]
    
    # 에이전트 간 데이터 공유
    saju_analysis: Annotated[Optional[str], "AI 사주 전문가의 종합 해석 결과"]
    retrieved_docs: Annotated[List[Dict[str, Any]], "RAG 시스템에서 검색된 문서들"]
    web_search_results: Annotated[List[Dict[str, Any]], "웹 검색 결과"]
    request: Annotated[Optional[str], "에이전트 간 전달하는 요청사항 (다음 행동)"]