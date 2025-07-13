"""
LangGraph 워크플로 그래프 생성 - Jupyter Notebook 구조 적용
"""

from langchain_core.messages import BaseMessage

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

# AgentState를 state.py에서 import
from state import AgentState

# NodeManager 사용
from nodes import get_node_manager
from agents import members


def create_workflow():
    """워크플로 그래프 생성 및 반환"""
    
    # 메인 그래프 생성
    workflow = StateGraph(AgentState)

    # NodeManager 인스턴스 가져오기
    node_manager = get_node_manager()
    
    # 노드 생성
    supervisor_node = node_manager.supervisor_agent_node
    saju_expert_agent_node = node_manager.saju_expert_agent_node
    search_agent_node = node_manager.search_agent_node
    general_answer_agent_node = node_manager.general_answer_agent_node
    
    # 그래프에 노드 추가
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("SajuExpert", saju_expert_agent_node)
    workflow.add_node("Search", search_agent_node)
    workflow.add_node("GeneralAnswer", general_answer_agent_node)
    
    # 각 에이전트 실행 후 Supervisor로 돌아가도록
    for member in members:
        workflow.add_edge(member, "Supervisor")
    
    # 조건부 엣지 추가
    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    
    def get_next(state):
        return state["next"]
    
    # Supervisor 노드에서 조건부 엣지 추가
    workflow.add_conditional_edges("Supervisor", get_next, conditional_map)
    
    workflow.add_edge(START, "Supervisor")
    
    # 그래프 컴파일
    app = workflow.compile(checkpointer=MemorySaver())
    
    return app 