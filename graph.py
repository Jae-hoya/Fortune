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
    general_qa_agent_node = node_manager.create_general_qa_agent_node()
    
    # 그래프에 노드 추가
    workflow.add_node("search", search_agent_node)
    workflow.add_node("manse", manse_tool_agent_node)
    workflow.add_node("general_qa", general_qa_agent_node)
    workflow.add_node("Supervisor", supervisor_agent)

    for member in members:
        workflow.add_edge(member, "Supervisor")
    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    
    def get_next(state):
        return state["next"]
   
    workflow.add_conditional_edges("Supervisor", get_next, conditional_map)
    workflow.add_edge(START, "Supervisor")
    
    return workflow.compile(checkpointer=MemorySaver())