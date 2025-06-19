"""
간단한 상태 관리 - 메시지 기반 에이전트
"""

from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage

class SajuState(TypedDict):
    """메시지 기반 에이전트 상태"""
    
    # 메시지 히스토리 (핵심)
    messages: List[BaseMessage]
    
    # 워크플로 제어
    next: Optional[str]  # 다음 에이전트 이름
    
    # 최종 결과
    final_response: Optional[str]
    response_generated: bool 