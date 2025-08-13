from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
from nodes import get_node_manager
from nodes import members


def create_workflow():
    """워크플로 그래프 생성 및 반환"""
    
    # 메인 그래프 생성
    workflow = StateGraph(AgentState)

    # NodeManager 인스턴스 가져오기
    node_manager = get_node_manager()
    
    # 노드 생성
    supervisor_agent = node_manager.supervisor_agent_node
    manse_tool_agent_node = node_manager.create_manse_tool_agent_node()
    search_agent_node = node_manager.search_agent_node
    general_qa_agent_node = node_manager.general_qa_agent_node
    
    # 그래프에 노드 추가
    workflow.add_node("search", search_agent_node)
    workflow.add_node("manse", manse_tool_agent_node)
    workflow.add_node("general_qa", general_qa_agent_node)
    workflow.add_node("supervisor", supervisor_agent)

    workflow.add_edge("search", END)
    workflow.add_edge("manse", END)
    workflow.add_edge("general_qa", END)

    conditional_map = {k: k for k in members}
    def get_next(state):
        return state["next"]
    workflow.add_conditional_edges("supervisor", get_next, conditional_map)
    workflow.add_edge(START, "supervisor")
    
    return workflow.compile(checkpointer=MemorySaver())