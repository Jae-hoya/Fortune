"""
LangGraph 워크플로 그래프 생성
"""

from langgraph.graph import StateGraph, END
from .state import SajuState
from .nodes import get_node_manager

def create_workflow():
    """워크플로 그래프 생성 및 반환"""
    
    # 싱글톤 NodeManager 인스턴스 가져오기
    node_manager = get_node_manager()
    
    # StateGraph 생성
    workflow = StateGraph(SajuState)
    
    # 노드 추가
    workflow.add_node("supervisor", node_manager.create_supervisor_node())
    workflow.add_node("SajuAgent", node_manager.create_saju_node())
    workflow.add_node("RagAgent", node_manager.create_rag_node())
    workflow.add_node("WebAgent", node_manager.create_web_node())
    workflow.add_node("result_generator", node_manager.create_result_generator_node())
    
    # 라우팅 조건부 함수 정의
    def should_continue(state):
        # supervisor의 next 결정에 따라 라우팅
        next_agent = state.get("next")
        
        if next_agent == "FINISH":
            return "result_generator"
        elif next_agent in ["SajuAgent", "RagAgent", "WebAgent"]:
            return next_agent
        else:
            # 기본값: 결과 생성으로
            return "result_generator"
    
    # 엣지 설정
    workflow.set_entry_point("supervisor")
    
    # supervisor에서 조건부 라우팅
    workflow.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "SajuAgent": "SajuAgent",
            "RagAgent": "RagAgent", 
            "WebAgent": "WebAgent",
            "result_generator": "result_generator"
        }
    )
    
    # 각 에이전트에서 supervisor로 돌아가기
    workflow.add_edge("SajuAgent", "supervisor")
    workflow.add_edge("RagAgent", "supervisor")  
    workflow.add_edge("WebAgent", "supervisor")
    
    # 결과 생성 후 종료
    workflow.add_edge("result_generator", END)
    
    # 워크플로 컴파일
    app = workflow.compile()
    
    return app 