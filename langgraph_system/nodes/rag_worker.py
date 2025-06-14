"""
RAG 검색 워커 노드
기존 RAG 도구를 활용한 사주/운세 전문 검색 워커
"""

import time
from typing import Dict, Any, List

from ..state import SupervisorState, WorkerResult
from tools import ToolManager

class RAGWorker:
    """RAG 검색 전문 워커"""
    
    def __init__(self):
        self.worker_name = "rag"
        # RAG 도구 초기화
        self.tool_manager = ToolManager(enable_rag=True, enable_web=False, enable_calendar=False)
        self.tool_manager.initialize()
        self.rag_tools = {tool.name: tool for tool in self.tool_manager._get_rag_tools()}
    
    def __call__(self, state: SupervisorState) -> SupervisorState:
        """RAG 워커 실행"""
        # 할당된 워커가 아니면 스킵
        if self.worker_name not in state.get("assigned_workers", []):
            return state
        
        # 이미 완료된 워커면 스킵
        if self.worker_name in state.get("completed_workers", []):
            return state
        
        start_time = time.time()
        
        try:
            # RAG 검색 실행
            result = self._search_fortune_knowledge(state, start_time)
            
            # 상태 업데이트
            return self._update_state(state, result)
            
        except Exception as e:
            # 오류 처리
            result = WorkerResult(
                worker_name=self.worker_name,
                success=False,
                result=None,
                error_message=f"RAG 검색 오류: {str(e)}",
                execution_time=time.time() - start_time,
                tokens_used=0
            )
            return self._update_state(state, result)
    
    def _search_fortune_knowledge(self, state: SupervisorState, start_time: float) -> WorkerResult:
        """RAG 기반 사주/운세 지식 검색"""
        try:
            user_query = state["user_query"]
            
            # 검색 쿼리 최적화
            search_query = self._optimize_search_query(state)
            
            # RAG 검색 실행 - smart_search_saju 도구 사용
            rag_tool = self.rag_tools.get("smart_search_saju")
            if rag_tool:
                rag_result = rag_tool.invoke({"query": search_query})
            else:
                # 대체 도구 사용
                rag_tool = self.rag_tools.get("search_saju_knowledge")
                rag_result = rag_tool.invoke({"query": search_query}) if rag_tool else "RAG 도구를 찾을 수 없습니다."
            
            # 결과 구조화
            structured_result = {
                "search_query": search_query,
                "raw_result": rag_result,
                "relevant_passages": self._extract_relevant_passages(rag_result),
                "knowledge_type": self._classify_knowledge_type(rag_result),
                "confidence": self._calculate_confidence(rag_result)
            }
            
            return WorkerResult(
                worker_name=self.worker_name,
                success=True,
                result=structured_result,
                error_message=None,
                execution_time=time.time() - start_time,
                tokens_used=self._estimate_tokens(rag_result)
            )
            
        except Exception as e:
            return WorkerResult(
                worker_name=self.worker_name,
                success=False,
                result=None,
                error_message=f"RAG 검색 실패: {str(e)}",
                execution_time=time.time() - start_time,
                tokens_used=0
            )
    
    def _optimize_search_query(self, state: SupervisorState) -> str:
        """검색 쿼리 최적화"""
        user_query = state["user_query"]
        question_type = state.get("question_type", "")
        birth_info = state.get("birth_info")
        saju_chart = state.get("saju_chart")
        
        # 기본 쿼리
        search_query = user_query
        
        # 사주 정보가 있으면 추가
        if saju_chart:
            day_master = saju_chart.get("saju_chart", {}).get("day_master", "")
            if day_master:
                search_query += f" {day_master}일간"
        
        # 질문 유형에 따른 키워드 추가
        if question_type == "fortune_consultation":
            if "연애" in user_query or "사랑" in user_query:
                search_query += " 연애운 사랑운"
            elif "재물" in user_query or "돈" in user_query:
                search_query += " 재물운 금전운"
            elif "직업" in user_query or "일" in user_query:
                search_query += " 직업운 사업운"
            elif "건강" in user_query:
                search_query += " 건강운"
        
        return search_query
    
    def _extract_relevant_passages(self, rag_result: str) -> List[str]:
        """관련 구절 추출"""
        if not rag_result:
            return []
        
        # 간단한 문장 분할 (실제로는 더 정교하게 구현)
        sentences = rag_result.split('.')
        relevant = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        return relevant[:5]  # 상위 5개 구절
    
    def _classify_knowledge_type(self, rag_result: str) -> str:
        """지식 유형 분류"""
        if not rag_result:
            return "unknown"
        
        if any(keyword in rag_result for keyword in ["오행", "십신", "대운"]):
            return "saju_theory"
        elif any(keyword in rag_result for keyword in ["운세", "궁합", "택일"]):
            return "fortune_consultation"
        elif any(keyword in rag_result for keyword in ["해석", "의미", "특징"]):
            return "interpretation"
        else:
            return "general"
    
    def _calculate_confidence(self, rag_result: str) -> float:
        """검색 결과 신뢰도 계산"""
        if not rag_result:
            return 0.0
        
        # 간단한 신뢰도 계산 (길이 기반)
        length_score = min(len(rag_result) / 500, 1.0)  # 500자 기준
        
        # 전문 용어 포함 여부
        professional_terms = ["사주", "팔자", "오행", "십신", "대운", "명리"]
        term_score = sum(1 for term in professional_terms if term in rag_result) / len(professional_terms)
        
        return (length_score * 0.6 + term_score * 0.4)
    
    def _estimate_tokens(self, text: str) -> int:
        """토큰 수 추정"""
        if not text:
            return 0
        # 대략적인 토큰 수 추정 (한국어 기준)
        return len(text) // 3
    
    def _update_state(self, state: SupervisorState, result: WorkerResult) -> SupervisorState:
        """상태 업데이트"""
        # 워커 결과 저장
        worker_results = state.get("worker_results", {})
        worker_results[self.worker_name] = result
        
        # 완료된 워커 목록에 추가
        completed_workers = state.get("completed_workers", [])
        if self.worker_name not in completed_workers:
            completed_workers.append(self.worker_name)
        
        # RAG 결과 저장 (성공한 경우)
        rag_results = None
        if result["success"] and result["result"]:
            rag_results = [result["result"]]
        
        return {
            **state,
            "worker_results": worker_results,
            "completed_workers": completed_workers,
            "rag_results": rag_results
        }

# 노드 함수로 래핑
def rag_worker_node(state: SupervisorState) -> SupervisorState:
    """RAG 워커 노드 실행 함수"""
    worker = RAGWorker()
    return worker(state) 