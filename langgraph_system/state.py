"""
LangGraph 상태 관리
Supervisor와 Worker 간 데이터 공유를 위한 상태 정의
"""

from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime

class SupervisorState(TypedDict):
    """Supervisor 기반 사주 에이전트의 상태 관리"""
    
    # 사용자 입력
    user_query: str
    user_id: Optional[str]
    timestamp: datetime
    
    # 의도 분석 결과
    intent_analysis: Dict[str, Any]
    confidence_score: float
    question_type: str  # "saju_calculation", "fortune_consultation", "general_search"
    
    # 워커 관리
    assigned_workers: List[str]  # ["saju", "rag", "web"]
    completed_workers: List[str]
    worker_results: Dict[str, Any]
    
    # 사주 관련 정보
    birth_info: Optional[Dict[str, Any]]
    saju_chart: Optional[Dict[str, Any]]
    
    # 검색 결과
    rag_results: Optional[List[Dict[str, Any]]]
    web_results: Optional[List[Dict[str, Any]]]
    
    # 응답 생성
    final_response: str
    response_sources: List[str]  # 응답에 사용된 소스들
    
    # 제어 플래그
    need_more_work: bool
    max_iterations: int
    current_iteration: int
    
    # 메타데이터
    processing_time: Optional[float]
    tokens_used: Optional[int]
    error_messages: List[str]

class WorkerResult(TypedDict):
    """워커 실행 결과 표준 형식"""
    worker_name: str
    success: bool
    result: Any
    error_message: Optional[str]
    execution_time: float
    tokens_used: Optional[int]

class IntentAnalysis(TypedDict):
    """의도 분석 결과 구조"""
    primary_intent: str
    secondary_intents: List[str]
    confidence_scores: Dict[str, float]
    extracted_entities: Dict[str, Any]
    requires_birth_info: bool
    complexity_level: str  # "simple", "medium", "complex" 