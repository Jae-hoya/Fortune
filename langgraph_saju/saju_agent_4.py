# ... 기존 코드 ...
# --- 5. LangGraph 그래프 구성 ---

def saju_chat_node(state):
    last_msg = state["messages"][-1].content
    # 사주풀이 결과와 추가 질문을 바탕으로 대화형 답변 생성 (예시)
    response = "(사주 대화 결과: 추가 질문에 대한 답변)"
    return {"messages": state["messages"] + [HumanMessage(content=response, name="SajuChat")]}

def create_workflow_graph():
    """워크플로우 그래프 생성 (병렬 구조: manse → saju_chat, manse → retriever)"""
    # SajuExpert Sub-graph 생성 (병렬)
    saju_expert_workflow = StateGraph(AgentState)
    saju_expert_workflow.add_node("manse", manse_tool_agent_node)
    saju_expert_workflow.add_node("saju_chat", saju_chat_node)
    saju_expert_workflow.add_node("retriever", retriever_tool_agent_node)

    saju_expert_workflow.add_edge(START, "manse")
    saju_expert_workflow.add_edge("manse", ["saju_chat", "retriever"])
    saju_expert_workflow.add_edge("saju_chat", END)
    saju_expert_workflow.add_edge("retriever", END)
    saju_expert_graph = saju_expert_workflow.compile(MemorySaver())

    # ... 이하 기존 create_workflow_graph 코드 동일 ... 