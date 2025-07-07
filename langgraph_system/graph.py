"""
LangGraph 워크플로 그래프 생성 - Jupyter Notebook 구조 적용
"""

from langchain_core.messages import BaseMessage

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

# AgentState를 state.py에서 import
from .state import AgentState

# NodeManager 사용
from .nodes import get_node_manager
from .agents import members

def create_saju_expert_subgraph():
    """SajuExpert 서브그래프 생성 (Manse -> Retriever)"""
    # NodeManager 인스턴스 가져오기
    node_manager = get_node_manager()
    
    # 1. Manse -> Retriever Sub-graph 생성
    saju_expert_workflow = StateGraph(AgentState)
    
    # Sub-graph에 노드 추가 (NodeManager에서 생성)
    # manse_tool_agent_node = node_manager.create_manse_tool_agent_node()
    saju_expert_agent_node = node_manager.saju_expert_agent_node
    retriever_agent_node = node_manager.retriever_agent_node
    # retriever_tool_agent_node = node_manager.create_retriever_tool_agent_node()
    
    saju_expert_workflow.add_node("saju_calculator", saju_expert_agent_node)
    saju_expert_workflow.add_node("retriever", retriever_agent_node)
    
    # Sub-graph 엣지 연결
    saju_expert_workflow.add_edge(START, "saju_calculator")
    saju_expert_workflow.add_edge("saju_calculator", "retriever")
    saju_expert_workflow.add_edge("retriever", END)
    
    # Sub-graph를 컴파일하여 Runnable로 만듭니다.
    return saju_expert_workflow.compile()

def create_workflow():
    """워크플로 그래프 생성 및 반환 (notebook 구조 적용)"""
    
    # 메인 그래프 생성
    workflow = StateGraph(AgentState)

    # NodeManager 인스턴스 가져오기
    node_manager = get_node_manager()
    
    # SajuExpert 서브그래프 생성
    saju_expert_graph = create_saju_expert_subgraph()
    
    # 노드들 생성 (NodeManager 사용)
    # web_tool_agent_node = node_manager.create_web_tool_agent_node()
    # general_qa_agent_node = node_manager.create_general_qa_agent_node()
    # supervisor_agent = node_manager.agent_manager.create_supervisor_agent(tools=[])
    supervisor_node = node_manager.supervisor_agent_node
    web_search_agent_node = node_manager.web_search_agent_node
    general_answer_agent_node = node_manager.general_answer_agent_node
    
    
    # 그래프에 노드 추가: ManseTool과 RetrieverTool을 SajuExpert로 대체
    workflow.add_node("SajuExpert", saju_expert_graph)
    workflow.add_node("WebSearch", web_search_agent_node)
    workflow.add_node("GeneralAnswer", general_answer_agent_node)
    # workflow.add_node("Supervisor", supervisor_agent)
    workflow.add_node("Supervisor", supervisor_node)
    
    # 각 에이전트 실행 후 Supervisor로 돌아가도록 수정
    for member in members:
        workflow.add_edge(member, "Supervisor")
    
    # 조건부 엣지 추가 - Supervisor에서 각 에이전트로 라우팅만
    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    
    def get_next(state):
        return state["next"]
    
    # Supervisor 노드에서 조건부 엣지 추가
    workflow.add_conditional_edges("Supervisor", get_next, conditional_map)
    
    # 시작점 - QueryExpansion 제거하고 바로 Supervisor로 시작
    workflow.add_edge(START, "Supervisor")
    
    # 그래프 컴파일
    app = workflow.compile(checkpointer=MemorySaver())
    
    return app 