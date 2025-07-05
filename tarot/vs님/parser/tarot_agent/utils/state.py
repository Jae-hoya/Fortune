"""

타로 그래프 상태 정의

"""

from typing import Annotated, List, Dict, Any, Optional, Literal

from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage

from langgraph.graph.message import add_messages

class TarotState(TypedDict):
    """최적화된 타로 상태"""
    # 기본 메시지 관리
    messages: Annotated[List[BaseMessage], add_messages]
    # 사용자 의도 (핵심!)
    user_intent: Literal["card_info", "spread_info", "consultation", "general", "simple_card", "unknown"]
    user_input: str
    # 상담 전용 데이터 (consultation일 때만 사용)
    consultation_data: Optional[Dict[str, Any]]
    # Supervisor 관련 필드
    supervisor_decision: Optional[Dict[str, Any]]
    # 라우팅 관련 (새로 추가)
    routing_decision: Optional[str]
    target_handler: Optional[str]
    needs_llm: Optional[bool]
    # 세션 메모리 (새로 추가)
    session_memory: Optional[Dict[str, Any]]
    conversation_memory: Optional[Dict[str, Any]]
    # 시간 맥락 정보 (새로 추가)
    temporal_context: Optional[Dict[str, Any]]
    search_timestamp: Optional[str]
    # 웹 검색 관련 필드 (새로 추가)
    search_results: Optional[Dict[str, Any]]
    search_decision: Optional[Dict[str, Any]]
