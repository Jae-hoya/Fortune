"""
LangGraph 기반 사주 시스템 그래프 구성
Supervisor 패턴으로 동적 워크플로우 구현
"""

from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import SupervisorState
from .supervisor import supervisor_node
from .nodes import (
    saju_worker_node,
    rag_worker_node, 
    web_worker_node,
    response_generator_node
)

def should_continue(state: SupervisorState) -> Literal["continue", "generate_response"]:
    """
    Supervisor가 더 많은 작업이 필요한지 결정
    """
    # 모든 할당된 워커가 완료되었는지 확인
    assigned_workers = set(state.get("assigned_workers", []))
    completed_workers = set(state.get("completed_workers", []))
    
    # 아직 완료되지 않은 워커가 있으면 계속
    if assigned_workers - completed_workers:
        return "continue"
    
    # Supervisor가 추가 작업이 필요하다고 판단하면 계속
    if state.get("need_more_work", False):
        return "continue"
    
    # 모든 작업 완료시 응답 생성
    return "generate_response"

def route_to_worker(state: SupervisorState) -> Literal["saju_worker", "rag_worker", "web_worker", "response_generator"]:
    """
    다음에 실행할 워커 결정
    """
    assigned_workers = state.get("assigned_workers", [])
    completed_workers = state.get("completed_workers", [])
    
    # 완료되지 않은 워커 찾기
    pending_workers = [w for w in assigned_workers if w not in completed_workers]
    
    if not pending_workers:
        # 모든 워커 완료시 응답 생성
        return "response_generator"
    
    # 우선순위에 따라 워커 선택
    # 1. 사주 계산 (기본 정보)
    # 2. RAG 검색 (전문 지식)  
    # 3. 웹 검색 (최신 정보)
    
    if "saju" in pending_workers:
        return "saju_worker"
    elif "rag" in pending_workers:
        return "rag_worker"
    elif "web" in pending_workers:
        return "web_worker"
    else:
        return "response_generator"

def create_fortune_graph() -> StateGraph:
    """
    사주 시스템 그래프 생성
    """
    # 그래프 초기화
    workflow = StateGraph(SupervisorState)
    
    # 노드 추가
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("saju_worker", saju_worker_node)
    workflow.add_node("rag_worker", rag_worker_node)
    workflow.add_node("web_worker", web_worker_node)
    workflow.add_node("response_generator", response_generator_node)
    
    # 시작점 설정
    workflow.set_entry_point("supervisor")
    
    # Supervisor에서 조건부 라우팅 - 통합된 라우팅 함수 사용
    def supervisor_router(state: SupervisorState) -> Literal["saju_worker", "rag_worker", "web_worker", "response_generator"]:
        """Supervisor 통합 라우터"""
        # 먼저 작업 완료 여부 확인
        assigned_workers = set(state.get("assigned_workers", []))
        completed_workers = set(state.get("completed_workers", []))
        
        # 아직 완료되지 않은 워커가 있으면 워커 실행
        if assigned_workers - completed_workers:
            return route_to_worker(state)
        
        # Supervisor가 추가 작업이 필요하다고 판단하면 워커 실행
        if state.get("need_more_work", False):
            return route_to_worker(state)
        
        # 모든 작업 완료시 응답 생성
        return "response_generator"
    
    workflow.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "saju_worker": "saju_worker",
            "rag_worker": "rag_worker", 
            "web_worker": "web_worker",
            "response_generator": "response_generator"
        }
    )
    
    # 각 워커에서 Supervisor로 돌아가기
    workflow.add_edge("saju_worker", "supervisor")
    workflow.add_edge("rag_worker", "supervisor")
    workflow.add_edge("web_worker", "supervisor")
    
    # 응답 생성 후 종료
    workflow.add_edge("response_generator", END)
    
    return workflow

def create_fortune_app():
    """
    실행 가능한 사주 시스템 앱 생성
    """
    # 그래프 생성
    workflow = create_fortune_graph()
    
    # 메모리 체크포인터 추가 (대화 상태 유지)
    memory = MemorySaver()
    
    # 앱 컴파일
    app = workflow.compile(checkpointer=memory)
    
    return app

# 간단한 실행 함수
def run_fortune_query(query: str, thread_id: str = "default") -> str:
    """
    사주 질의 실행
    
    Args:
        query: 사용자 질문
        thread_id: 대화 스레드 ID (세션 관리용)
    
    Returns:
        최종 응답 문자열
    """
    app = create_fortune_app()
    
    # 초기 상태 설정
    initial_state = {
        "user_query": query,
        "intent_analysis": {},
        "question_type": "",
        "assigned_workers": [],
        "completed_workers": [],
        "worker_results": {},
        "birth_info": None,
        "need_more_work": False,
        "final_response": "",
        "response_generated": False
    }
    
    # 그래프 실행
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # 스트리밍 실행
        final_state = None
        for state in app.stream(initial_state, config):
            final_state = state
                    # 간단한 진행 상태만 표시 (verbose 모드가 아닐 때)
        if not hasattr(run_fortune_query, '_verbose_mode'):
            print(f"🔄 진행 상태: {list(state.keys())}")
            
            # 상세 디버깅 정보
            for node_name, node_state in state.items():
                if isinstance(node_state, dict):
                    if "assigned_workers" in node_state:
                        print(f"   📋 {node_name} - 할당된 워커: {node_state.get('assigned_workers', [])}")
                    if "completed_workers" in node_state:
                        print(f"   ✅ {node_name} - 완료된 워커: {node_state.get('completed_workers', [])}")
                    if "need_more_work" in node_state:
                        print(f"   🔧 {node_name} - 추가 작업 필요: {node_state.get('need_more_work', False)}")
                    if "question_type" in node_state:
                        print(f"   🎯 {node_name} - 질문 유형: {node_state.get('question_type', '')}")
                    if "birth_info" in node_state and node_state["birth_info"]:
                        print(f"   📅 {node_name} - 생년월일시: 추출됨")
        
        # 최종 응답 반환
        if final_state:
            # 모든 노드 상태 확인
            for node_name, node_state in final_state.items():
                if node_name == "response_generator" and isinstance(node_state, dict):
                    response = node_state.get("final_response", "")
                    if response:
                        return response
            
            # response_generator가 없으면 다른 방법으로 응답 찾기
            if "final_response" in final_state:
                return final_state["final_response"]
            
            return f"응답 생성 실패. 상태: {list(final_state.keys())}"
        else:
            return "시스템 오류가 발생했습니다."
            
    except Exception as e:
        return f"실행 중 오류 발생: {str(e)}"

# 디버깅용 상세 실행 함수
def run_fortune_query_debug(query: str, thread_id: str = "debug", verbose: bool = True) -> dict:
    """
    디버깅용 상세 실행 함수
    모든 중간 상태를 반환
    
    Args:
        query: 사용자 질문
        thread_id: 스레드 ID
        verbose: 상세 로깅 여부
    """
    app = create_fortune_app()
    
    initial_state = {
        "user_query": query,
        "intent_analysis": {},
        "question_type": "",
        "assigned_workers": [],
        "completed_workers": [],
        "worker_results": {},
        "birth_info": None,
        "need_more_work": False,
        "final_response": "",
        "response_generated": False
    }
    
    config = {"configurable": {"thread_id": thread_id}}
    
    if verbose:
        print("🔍 " + "=" * 60)
        print(f"🎯 디버그 모드 시작: '{query}'")
        print("🔍 " + "=" * 60)
    
    try:
        states = []
        step_count = 0
        
        for state in app.stream(initial_state, config):
            step_count += 1
            states.append(state)
            
            if verbose:
                _print_detailed_step_info(step_count, state)
        
        if verbose:
            print("\n🎉 " + "=" * 60)
            print("✅ 실행 완료!")
            print("🎉 " + "=" * 60)
        
        return {
            "success": True,
            "final_response": states[-1].get("response_generator", {}).get("final_response", "") if states else "",
            "all_states": states,
            "execution_summary": {
                "total_steps": len(states),
                "workers_used": list(set().union(*[s.get("completed_workers", []) for s in states])),
                "question_type": states[-1].get("supervisor", {}).get("question_type", "") if states else "",
                "execution_path": [list(s.keys())[0] for s in states]
            }
        }
        
    except Exception as e:
        if verbose:
            print(f"\n❌ 오류 발생: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "final_response": f"실행 중 오류 발생: {str(e)}"
        }

def _print_detailed_step_info(step: int, state: dict):
    """상세 단계 정보 출력"""
    node_name = list(state.keys())[0]
    node_state = state[node_name]
    
    print(f"\n📍 Step {step}: {_get_node_emoji(node_name)} {node_name.upper()}")
    print("-" * 50)
    
    if isinstance(node_state, dict):
        # 기본 정보
        if "question_type" in node_state:
            print(f"   🎯 질문 유형: {node_state['question_type']}")
        
        if "birth_info" in node_state and node_state["birth_info"]:
            birth = node_state["birth_info"]
            print(f"   📅 생년월일시: {birth.get('year', '')}년 {birth.get('month', '')}월 {birth.get('day', '')}일 {birth.get('hour', '')}시")
        
        # 워커 관련 정보
        if "assigned_workers" in node_state:
            assigned = node_state.get("assigned_workers", [])
            print(f"   📋 할당된 워커: {assigned if assigned else '없음'}")
        
        if "completed_workers" in node_state:
            completed = node_state.get("completed_workers", [])
            print(f"   ✅ 완료된 워커: {completed if completed else '없음'}")
        
        if "need_more_work" in node_state:
            print(f"   🔧 추가 작업 필요: {'예' if node_state['need_more_work'] else '아니오'}")
        
        # 워커 결과 정보
        if "worker_results" in node_state:
            results = node_state.get("worker_results", {})
            if results:
                print(f"   📊 워커 결과:")
                for worker, result in results.items():
                    success = result.get("success", False) if isinstance(result, dict) else False
                    print(f"      • {worker}: {'✅ 성공' if success else '❌ 실패'}")
        
        # 실행 시간 정보
        if "processing_time" in node_state:
            print(f"   ⏱️  처리 시간: {node_state['processing_time']:.2f}초")
        
        # 오류 정보
        if "error_messages" in node_state and node_state["error_messages"]:
            print(f"   ⚠️  오류: {node_state['error_messages']}")
        
        # 최종 응답 정보
        if "final_response" in node_state and node_state["final_response"]:
            response_len = len(node_state["final_response"])
            print(f"   📝 응답 생성: {response_len}자")

def _get_node_emoji(node_name: str) -> str:
    """노드별 이모지 반환"""
    emoji_map = {
        "supervisor": "🧠",
        "saju_worker": "🔮", 
        "rag_worker": "📚",
        "web_worker": "🌐",
        "response_generator": "📝"
    }
    return emoji_map.get(node_name, "🔧")

# 간단한 verbose 실행 함수
def run_fortune_query_verbose(query: str, thread_id: str = "verbose") -> str:
    """
    상세 로깅과 함께 사주 질의 실행 (기본 로깅 없이)
    """
    app = create_fortune_app()
    
    initial_state = {
        "user_query": query,
        "intent_analysis": {},
        "question_type": "",
        "assigned_workers": [],
        "completed_workers": [],
        "worker_results": {},
        "birth_info": None,
        "need_more_work": False,
        "final_response": "",
        "response_generated": False
    }
    
    config = {"configurable": {"thread_id": thread_id}}
    
    print("🔍 " + "=" * 60)
    print(f"🎯 상세 모드 시작: '{query}'")
    print("🔍 " + "=" * 60)
    
    try:
        step_count = 0
        final_response = ""
        
        for state in app.stream(initial_state, config):
            step_count += 1
            _print_detailed_step_info(step_count, state)
            
            # 최종 응답 추출
            for node_name, node_state in state.items():
                if node_name == "response_generator" and isinstance(node_state, dict):
                    response = node_state.get("final_response", "")
                    if response:
                        final_response = response
        
        print("\n🎉 " + "=" * 60)
        print("✅ 실행 완료!")
        print("🎉 " + "=" * 60)
        
        return final_response if final_response else "응답 생성 실패"
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        return f"실행 중 오류 발생: {str(e)}" 