"""
웹 검색 워커 노드
기존 웹 검색 도구를 활용한 최신 정보 검색 워커
"""

import time
from typing import Dict, Any, List

from ..state import SupervisorState, WorkerResult
from tools import ToolManager

class WebWorker:
    """웹 검색 전문 워커"""
    
    def __init__(self):
        self.worker_name = "web"
        # 웹 검색 도구 초기화
        self.tool_manager = ToolManager(enable_rag=False, enable_web=True, enable_calendar=False)
        self.tool_manager.initialize()
        self.web_tools = {tool.name: tool for tool in self.tool_manager._get_web_tools()}
    
    def __call__(self, state: SupervisorState) -> SupervisorState:
        """웹 워커 실행"""
        # 할당된 워커가 아니면 스킵
        if self.worker_name not in state.get("assigned_workers", []):
            return state
        
        # 이미 완료된 워커면 스킵
        if self.worker_name in state.get("completed_workers", []):
            return state
        
        start_time = time.time()
        
        try:
            # 웹 검색 실행
            result = self._search_web_information(state, start_time)
            
            # 상태 업데이트
            return self._update_state(state, result)
            
        except Exception as e:
            # 오류 처리
            result = WorkerResult(
                worker_name=self.worker_name,
                success=False,
                result=None,
                error_message=f"웹 검색 오류: {str(e)}",
                execution_time=time.time() - start_time,
                tokens_used=0
            )
            return self._update_state(state, result)
    
    def _search_web_information(self, state: SupervisorState, start_time: float) -> WorkerResult:
        """웹 기반 최신 정보 검색"""
        try:
            user_query = state["user_query"]
            
            # 검색 쿼리 최적화
            search_query = self._optimize_web_query(state)
            
            # 웹 검색 실행 - search_web_saju 도구 사용
            web_tool = self.web_tools.get("search_web_saju")
            if web_tool:
                web_result = web_tool.invoke({"query": search_query})
            else:
                web_result = "웹 검색 도구를 찾을 수 없습니다."
            
            # 결과 구조화
            structured_result = {
                "search_query": search_query,
                "raw_result": web_result,
                "filtered_content": self._filter_relevant_content(web_result),
                "source_urls": self._extract_source_urls(web_result),
                "information_type": self._classify_information_type(web_result),
                "freshness": self._assess_information_freshness(web_result),
                "reliability": self._assess_reliability(web_result)
            }
            
            return WorkerResult(
                worker_name=self.worker_name,
                success=True,
                result=structured_result,
                error_message=None,
                execution_time=time.time() - start_time,
                tokens_used=self._estimate_tokens(web_result)
            )
            
        except Exception as e:
            return WorkerResult(
                worker_name=self.worker_name,
                success=False,
                result=None,
                error_message=f"웹 검색 실패: {str(e)}",
                execution_time=time.time() - start_time,
                tokens_used=0
            )
    
    def _optimize_web_query(self, state: SupervisorState) -> str:
        """웹 검색 쿼리 최적화"""
        user_query = state["user_query"]
        question_type = state.get("question_type", "")
        
        # 기본 쿼리
        search_query = user_query
        
        # 질문 유형에 따른 최적화
        if question_type == "fortune_consultation":
            # 운세 관련 검색에 연도 추가
            current_year = "2024"
            if current_year not in search_query:
                search_query += f" {current_year}년"
            
            # 신뢰할 만한 사이트 우선
            search_query += " site:sajuplus.com OR site:fortuneteller.co.kr OR site:myfortune.co.kr"
        
        elif question_type == "general_search":
            # 일반 검색은 최신 정보 우선
            search_query += " 최신 정보"
        
        return search_query
    
    def _filter_relevant_content(self, web_result: str) -> List[str]:
        """관련 콘텐츠 필터링"""
        if not web_result:
            return []
        
        # 간단한 필터링 (실제로는 더 정교하게 구현)
        lines = web_result.split('\n')
        relevant_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) > 30:  # 너무 짧은 라인 제외
                # 광고성 내용 제외
                if not any(ad_word in line for ad_word in ["광고", "쿠폰", "할인", "이벤트"]):
                    relevant_lines.append(line)
        
        return relevant_lines[:10]  # 상위 10개 라인
    
    def _extract_source_urls(self, web_result: str) -> List[str]:
        """소스 URL 추출"""
        import re
        
        if not web_result:
            return []
        
        # URL 패턴 매칭
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, web_result)
        
        # 중복 제거 및 신뢰할 만한 도메인 우선
        trusted_domains = ["sajuplus.com", "fortuneteller.co.kr", "myfortune.co.kr", "naver.com", "daum.net"]
        trusted_urls = [url for url in urls if any(domain in url for domain in trusted_domains)]
        other_urls = [url for url in urls if not any(domain in url for domain in trusted_domains)]
        
        return list(dict.fromkeys(trusted_urls + other_urls))[:5]  # 중복 제거 후 상위 5개
    
    def _classify_information_type(self, web_result: str) -> str:
        """정보 유형 분류"""
        if not web_result:
            return "unknown"
        
        if any(keyword in web_result for keyword in ["2024년", "올해", "최신"]):
            return "current_fortune"
        elif any(keyword in web_result for keyword in ["뉴스", "기사", "보도"]):
            return "news"
        elif any(keyword in web_result for keyword in ["블로그", "후기", "경험"]):
            return "personal_experience"
        elif any(keyword in web_result for keyword in ["전문가", "명리학", "이론"]):
            return "expert_opinion"
        else:
            return "general_information"
    
    def _assess_information_freshness(self, web_result: str) -> float:
        """정보 신선도 평가"""
        if not web_result:
            return 0.0
        
        # 날짜 관련 키워드로 신선도 평가
        fresh_keywords = ["2024", "올해", "최근", "최신", "오늘", "이번"]
        old_keywords = ["2022", "2021", "작년", "예전"]
        
        fresh_score = sum(1 for keyword in fresh_keywords if keyword in web_result)
        old_score = sum(1 for keyword in old_keywords if keyword in web_result)
        
        if fresh_score + old_score == 0:
            return 0.5  # 중립
        
        return fresh_score / (fresh_score + old_score)
    
    def _assess_reliability(self, web_result: str) -> float:
        """정보 신뢰도 평가"""
        if not web_result:
            return 0.0
        
        # 신뢰할 만한 소스 키워드
        reliable_keywords = ["전문가", "명리학", "연구", "분석", "통계"]
        unreliable_keywords = ["카더라", "소문", "추측", "개인적"]
        
        reliable_score = sum(1 for keyword in reliable_keywords if keyword in web_result)
        unreliable_score = sum(1 for keyword in unreliable_keywords if keyword in web_result)
        
        # 기본 신뢰도 0.6에서 시작
        base_reliability = 0.6
        
        if reliable_score > 0:
            base_reliability += 0.2
        if unreliable_score > 0:
            base_reliability -= 0.3
        
        return max(0.0, min(1.0, base_reliability))
    
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
        
        # 웹 검색 결과 저장 (성공한 경우)
        web_results = None
        if result["success"] and result["result"]:
            web_results = [result["result"]]
        
        return {
            **state,
            "worker_results": worker_results,
            "completed_workers": completed_workers,
            "web_results": web_results
        }

# 노드 함수로 래핑
def web_worker_node(state: SupervisorState) -> SupervisorState:
    """웹 워커 노드 실행 함수"""
    worker = WebWorker()
    return worker(state) 