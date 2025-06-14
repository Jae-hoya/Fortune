"""
사주 계산 워커 노드
기존 사주 계산기와 도구들을 활용한 전문 워커
"""

import time
from typing import Dict, Any
from datetime import datetime

from ..state import SupervisorState, WorkerResult
from saju_calculator import SajuCalculator, format_saju_analysis

class SajuWorker:
    """사주 계산 전문 워커"""
    
    def __init__(self):
        self.calculator = SajuCalculator()
        self.worker_name = "saju"
    
    def __call__(self, state: SupervisorState) -> SupervisorState:
        """사주 워커 실행"""
        # 할당된 워커가 아니면 스킵
        if self.worker_name not in state.get("assigned_workers", []):
            return state
        
        # 이미 완료된 워커면 스킵
        if self.worker_name in state.get("completed_workers", []):
            return state
        
        start_time = time.time()
        
        try:
            # 생년월일시 정보 확인 (Supervisor에서 이미 추출됨)
            birth_info = state.get("birth_info")
            
            if not birth_info:
                # 생년월일시 정보가 없으면 실패
                result = WorkerResult(
                    worker_name=self.worker_name,
                    success=False,
                    result=None,
                    error_message="생년월일시 정보가 필요합니다.",
                    execution_time=time.time() - start_time,
                    tokens_used=0
                )
            else:
                # 사주 계산 실행
                result = self._calculate_saju(birth_info, start_time)
            
            # 상태 업데이트
            return self._update_state(state, result)
            
        except Exception as e:
            # 오류 처리
            result = WorkerResult(
                worker_name=self.worker_name,
                success=False,
                result=None,
                error_message=f"사주 계산 오류: {str(e)}",
                execution_time=time.time() - start_time,
                tokens_used=0
            )
            return self._update_state(state, result)
    

    def _calculate_saju(self, birth_info: Dict[str, Any], start_time: float) -> WorkerResult:
        """사주 계산 실행"""
        try:
            # 사주 계산
            saju_chart = self.calculator.auto_calculate_saju(
                year=birth_info["year"],
                month=birth_info["month"],
                day=birth_info["day"],
                hour=birth_info["hour"],
                minute=birth_info.get("minute", 0),
                is_male=birth_info.get("is_male", True)
            )
            
            # 포맷팅된 분석 결과 생성
            formatted_analysis = format_saju_analysis(saju_chart, self.calculator)
            
            # 구조화된 결과 생성
            structured_result = {
                "saju_chart": {
                    "year_pillar": str(saju_chart.year_pillar),
                    "month_pillar": str(saju_chart.month_pillar),
                    "day_pillar": str(saju_chart.day_pillar),
                    "hour_pillar": str(saju_chart.hour_pillar),
                    "day_master": saju_chart.get_day_master()
                },
                "analysis": {
                    "elements": self.calculator.get_element_strength_with_season(saju_chart),
                    "ten_gods": self.calculator.get_ten_gods_summary(saju_chart),
                    "day_master_strength": self.calculator.analyze_day_master_strength(saju_chart),
                    "great_fortune": self.calculator.calculate_great_fortune_improved(saju_chart)
                },
                "formatted_text": formatted_analysis,
                "birth_info": birth_info
            }
            
            return WorkerResult(
                worker_name=self.worker_name,
                success=True,
                result=structured_result,
                error_message=None,
                execution_time=time.time() - start_time,
                tokens_used=None  # 사주 계산은 LLM 미사용
            )
            
        except Exception as e:
            return WorkerResult(
                worker_name=self.worker_name,
                success=False,
                result=None,
                error_message=f"사주 계산 실패: {str(e)}",
                execution_time=time.time() - start_time,
                tokens_used=0
            )
    
    def _update_state(self, state: SupervisorState, result: WorkerResult) -> SupervisorState:
        """상태 업데이트"""
        # 워커 결과 저장
        worker_results = state.get("worker_results", {})
        worker_results[self.worker_name] = result
        
        # 완료된 워커 목록에 추가
        completed_workers = state.get("completed_workers", [])
        if self.worker_name not in completed_workers:
            completed_workers.append(self.worker_name)
        
        # 사주 차트 정보 저장 (성공한 경우)
        saju_chart = None
        if result["success"] and result["result"]:
            saju_chart = result["result"]
        
        return {
            **state,
            "worker_results": worker_results,
            "completed_workers": completed_workers,
            "saju_chart": saju_chart
        }

# 노드 함수로 래핑
def saju_worker_node(state: SupervisorState) -> SupervisorState:
    """사주 워커 노드 실행 함수"""
    worker = SajuWorker()
    return worker(state) 