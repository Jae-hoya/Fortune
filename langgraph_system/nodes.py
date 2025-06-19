"""
노드 함수들 - NodeManager 클래스로 노드 생성 및 관리
"""

import re
import functools
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, Any, List

from .state import SajuState
from .agents import AgentManager
from tools import ToolManager

class NodeManager:
    """노드 생성 및 관리 클래스"""
    
    def __init__(self):
        # 도구 관리자 초기화
        self.tool_manager = ToolManager(
            enable_rag=True,
            enable_web=True, 
            enable_calendar=True
        )
        # 에이전트 관리자 초기화
        self.agent_manager = AgentManager()
    
    def create_saju_node(self):
        """사주 계산 노드 생성"""
        calendar_tools = self.tool_manager.calendar_tools
        
        def saju_node(state: SajuState):
            saju_agent = self.agent_manager.create_saju_agent(calendar_tools)
            result = saju_agent.invoke(state)
            return {
                "messages": [result["messages"][-1]],
                "sender": "SajuAgent"
            }
        
        return saju_node
    
    def create_rag_node(self):
        """RAG 검색 노드 생성"""
        rag_tools = self.tool_manager.rag_tools
        
        def rag_node(state: SajuState):
            rag_agent = self.agent_manager.create_rag_agent(rag_tools)
            result = rag_agent.invoke(state)
            return {
                "messages": [result["messages"][-1]],
                "sender": "RagAgent"
            }
        
        return rag_node
    
    def create_web_node(self):
        """웹 검색 노드 생성"""
        web_tools = self.tool_manager.web_tools
        
        def web_node(state: SajuState):
            web_agent = self.agent_manager.create_web_agent(web_tools)
            result = web_agent.invoke(state)
            return {
                "messages": [result["messages"][-1]],
                "sender": "WebAgent"
            }
        
        return web_node
    
    def create_supervisor_node(self):
        """Supervisor 노드 생성"""
        
        def supervisor_node(state: SajuState):
            supervisor = self.agent_manager.create_supervisor_agent(tools=[])
            result = supervisor(state)
            
            # 구조화된 출력에서 next 값 추출
            if hasattr(result, 'next'):
                next_agent = result.next
            else:
                # 후폐방안: 딕셔너리 형태인 경우
                next_agent = result.get('next', 'FINISH')
            
            return {"next": next_agent}
        
        return supervisor_node
    
    def create_result_generator_node(self):
        """결과 생성 노드 생성"""
        all_tools = self.tool_manager.get_all_tools()
        
        def result_generator_node(state: SajuState):
            response_agent = self.agent_manager.create_response_generator_agent(all_tools)
            result = response_agent.invoke(state)
            return {
                "messages": [result["messages"][-1]],
                "final_response": result["messages"][-1].content,
                "sender": "ResultGenerator"
            }
        
        return result_generator_node

# 전역 NodeManager 인스턴스 (싱글톤 패턴)
_node_manager = None

def get_node_manager():
    """싱글톤 NodeManager 인스턴스 반환"""
    global _node_manager
    if _node_manager is None:
        _node_manager = NodeManager()
    return _node_manager
