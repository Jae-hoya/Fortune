"""
노드 함수들 - NodeManager 클래스로 노드 생성 및 관리
"""

import functools
from langchain_core.messages import HumanMessage

from .agents import AgentManager

class NodeManager:
    """노드 생성 및 관리 클래스"""
    
    def __init__(self):
        # 에이전트 관리자 초기화 (단순화)
        self.agent_manager = AgentManager()
    
    # === 새로운 노드들 (notebook 구조 지원) ===
    def _agent_node(self, state, agent, name):
        """지정한 agent와 name을 사용하여 agent 노드를 생성하는 헬퍼 함수"""
        # agent 호출
        agent_response = agent.invoke(state)
        # agent의 마지막 메시지를 HumanMessage로 변환하여 반환
        return {
            "messages": [
                HumanMessage(content=agent_response["messages"][-1].content, name=name)
            ]
        }

    # === Notebook 스타일 노드 생성 메서드들 ===
    def create_manse_tool_agent_node(self):
        """Manse Tool Agent 노드 생성"""
        manse_tool_agent = self.agent_manager.create_manse_tool_agent()
        return functools.partial(self._agent_node, agent=manse_tool_agent, name="ManseTool")

    def create_retriever_tool_agent_node(self):
        """Retriever Tool Agent 노드 생성"""
        retriever_tool_agent = self.agent_manager.create_retriever_tool_agent()
        return functools.partial(self._agent_node, agent=retriever_tool_agent, name="RetrieverTool")

    def create_web_tool_agent_node(self):
        """Web Tool Agent 노드 생성"""
        web_tool_agent = self.agent_manager.create_web_tool_agent()
        return functools.partial(self._agent_node, agent=web_tool_agent, name="WebTool")

    def create_general_qa_agent_node(self):
        """General QA Agent 노드 생성"""
        general_qa_agent = self.agent_manager.create_general_qa_agent()
        return functools.partial(self._agent_node, agent=general_qa_agent, name="GeneralQA")

# 전역 NodeManager 인스턴스 (싱글톤 패턴)
_node_manager = None

def get_node_manager():
    """싱글톤 NodeManager 인스턴스 반환"""
    global _node_manager
    if _node_manager is None:
        _node_manager = NodeManager()
    return _node_manager
