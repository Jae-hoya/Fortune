"""
간단한 상태 관리 - 메시지 기반 에이전트
"""

import operator
from typing import Sequence, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage

# 새로운 AgentState 정의 (멀티턴 채팅 지원)
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]  # 메시지
    next: str  # 다음으로 라우팅할 에이전트
    session_start_time: str  # 세션 시작 시간 (고정)
    current_time: str  # 현재 쿼리 시간 (매번 갱신)
    session_id: str  # 세션 ID (고정)
