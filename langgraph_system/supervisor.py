"""
Supervisor 노드 구현
사용자 질문 분석 및 워커 할당을 담당하는 중앙 제어 노드
"""

import re
import time
from datetime import datetime
from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .state import SupervisorState, IntentAnalysis

class SupervisorNode:
    """Supervisor 노드 - 중앙 제어 및 워커 관리"""
    
    def __init__(self, llm_model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0.1)
        self.max_iterations = 5
    
    def __call__(self, state: SupervisorState) -> SupervisorState:
        """Supervisor 메인 로직 실행"""
        start_time = time.time()
        
        try:
            # 초기화 (처음 실행시)
            if not state.get("timestamp"):
                state = self._initialize_state(state)
            
            # 의도 분석 (아직 안 했으면)
            if not state.get("intent_analysis"):
                state = self._analyze_intent(state)
            
            # 워커 할당 결정 (아직 안 했으면)
            if not state.get("assigned_workers"):
                state = self._assign_workers(state)
            
            # 완료 여부 확인
            state = self._check_completion(state)
            
            # 처리 시간 기록
            state["processing_time"] = time.time() - start_time
            
            return state
            
        except Exception as e:
            # 오류 메시지 초기화
            if "error_messages" not in state:
                state["error_messages"] = []
            state["error_messages"].append(f"Supervisor 오류: {str(e)}")
            state["need_more_work"] = False
            return state
    
    def _initialize_state(self, state: SupervisorState) -> SupervisorState:
        """상태 초기화"""
        return {
            **state,
            "timestamp": datetime.now(),
            "current_iteration": 0,
            "max_iterations": self.max_iterations,
            "completed_workers": [],
            "worker_results": {},
            "error_messages": [],
            "response_sources": [],
            "need_more_work": True,
            "confidence_score": 0.0
        }
    
    def _analyze_intent(self, state: SupervisorState) -> SupervisorState:
        """사용자 의도 분석 및 생년월일시 추출"""
        user_query = state["user_query"]
        
        # 통합 분석 프롬프트
        system_prompt = """당신은 사주/운세 상담 시스템의 분석 전문가입니다.
사용자 질문을 분석하여 다음을 수행해주세요:

1. 질문 유형 분류:
   - saju_calculation: 사주팔자 계산 요청
   - fortune_consultation: 운세 상담/해석 요청  
   - general_search: 일반 정보 검색

2. 생년월일시 정보 추출 (있는 경우):
   - 년, 월, 일, 시, 분
   - 성별 (남성/여성)

응답 형식 (JSON):
{
    "question_type": "분류 결과",
    "confidence": 0.0-1.0,
    "birth_info": {
        "year": 1995,
        "month": 8, 
        "day": 26,
        "hour": 10,
        "minute": 15,
        "is_male": true
    } 또는 null,
    "complexity": "simple/medium/complex"
}"""

        try:
            # LLM 통합 분석
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"분석할 질문: {user_query}")
            ]
            
            response = self.llm.invoke(messages)
            
            # JSON 파싱
            import json
            result = json.loads(response.content)
            
            # 의도 분석 결과 구성
            intent_analysis = {
                "primary_intent": result["question_type"],
                "confidence_scores": {result["question_type"]: result["confidence"]},
                "complexity_level": result.get("complexity", "medium"),
                "requires_birth_info": result["birth_info"] is not None
            }
            
            return {
                **state,
                "intent_analysis": intent_analysis,
                "question_type": result["question_type"],
                "confidence_score": result["confidence"],
                "birth_info": result["birth_info"]
            }
            
        except Exception as e:
            # 실패시 기본 분석
            return self._fallback_analysis(state, user_query)
    
    def _assign_workers(self, state: SupervisorState) -> SupervisorState:
        """워커 할당 결정"""
        question_type = state["question_type"]
        intent_analysis = state["intent_analysis"]
        
        # 기본 워커 할당 로직
        assigned_workers = []
        
        if question_type == "saju_calculation":
            assigned_workers.append("saju")
            # 사주 계산 후 해석이 필요한 경우
            if intent_analysis.get("complexity_level") in ["medium", "complex"]:
                assigned_workers.append("rag")
        
        elif question_type == "fortune_consultation":
            assigned_workers.extend(["rag", "web"])
            # 생년월일시가 있으면 사주도 계산
            if state.get("birth_info"):
                assigned_workers.insert(0, "saju")
        
        elif question_type == "general_search":
            assigned_workers.append("web")
            # 사주 관련 일반 질문이면 RAG도 추가
            if self._is_fortune_related(state["user_query"]):
                assigned_workers.append("rag")
        
        else:
            # 불확실한 경우 모든 워커 활용
            assigned_workers = ["rag", "web"]
            if state.get("birth_info"):
                assigned_workers.insert(0, "saju")
        
        # 중복 제거 및 순서 유지
        assigned_workers = list(dict.fromkeys(assigned_workers))
        
        return {
            **state,
            "assigned_workers": assigned_workers
        }
    
    def _check_completion(self, state: SupervisorState) -> SupervisorState:
        """작업 완료 여부 확인"""
        assigned_workers = state.get("assigned_workers", [])
        completed_workers = state.get("completed_workers", [])
        current_iteration = state.get("current_iteration", 0)
        max_iterations = state.get("max_iterations", 5)
        
        # 모든 할당된 워커가 완료되었는지 확인
        all_workers_completed = all(
            worker in completed_workers for worker in assigned_workers
        )
        
        # 최대 반복 횟수 초과 확인
        max_iterations_reached = current_iteration >= max_iterations
        
        # 추가 작업 필요성 판단
        need_more_work = not (all_workers_completed or max_iterations_reached)
        
        # 결과 품질 확인 (필요시 추가 워커 할당)
        if all_workers_completed and not max_iterations_reached:
            need_more_work = self._assess_result_quality(state)
        
        return {
            **state,
            "need_more_work": need_more_work,
            "current_iteration": current_iteration + 1
        }
    
    def _is_fortune_related(self, query: str) -> bool:
        """사주/운세 관련 질문인지 판단"""
        fortune_keywords = [
            "사주", "운세", "팔자", "명리", "오행", "십신", "대운", "신살",
            "궁합", "택일", "개명", "풍수", "관상", "손금", "타로"
        ]
        
        return any(keyword in query for keyword in fortune_keywords)
    
    def _assess_result_quality(self, state: SupervisorState) -> bool:
        """결과 품질 평가 및 추가 작업 필요성 판단"""
        worker_results = state.get("worker_results", {})
        
        # 결과가 부족한 경우 추가 작업 필요
        if not worker_results:
            return True
        
        # 사주 계산 결과가 있지만 해석이 부족한 경우
        if "saju" in worker_results and "rag" not in worker_results:
            if state["question_type"] == "fortune_consultation":
                # RAG 워커 추가 할당
                current_assigned = state.get("assigned_workers", [])
                if "rag" not in current_assigned:
                    state["assigned_workers"] = current_assigned + ["rag"]
                    return True
        
        # 웹 검색 결과가 부족한 경우
        web_result = worker_results.get("web", {})
        if isinstance(web_result, dict) and not web_result.get("success", False):
            return False  # 웹 검색 실패는 추가 작업으로 해결 안됨
        
        return False  # 추가 작업 불필요
    
    def _fallback_analysis(self, state: SupervisorState, user_query: str) -> SupervisorState:
        """LLM 실패시 백업 분석"""
        # 간단한 키워드 기반 분석
        if any(keyword in user_query for keyword in ["사주", "팔자", "봐주세요"]):
            question_type = "saju_calculation"
            confidence = 0.7
        elif any(keyword in user_query for keyword in ["운세", "운", "궁합"]):
            question_type = "fortune_consultation"
            confidence = 0.6
        else:
            question_type = "general_search"
            confidence = 0.5
        
        # 기본 의도 분석 결과
        intent_analysis = {
            "primary_intent": question_type,
            "confidence_scores": {question_type: confidence},
            "complexity_level": "medium",
            "requires_birth_info": False
        }
        
        return {
            **state,
            "intent_analysis": intent_analysis,
            "question_type": question_type,
            "confidence_score": confidence,
            "birth_info": None
        }

# 노드 함수로 래핑
def supervisor_node(state: SupervisorState) -> SupervisorState:
    """Supervisor 노드 실행 함수"""
    supervisor = SupervisorNode()
    return supervisor(state) 