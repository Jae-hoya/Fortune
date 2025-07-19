"""
노드 함수들 - NodeManager 클래스로 노드 생성 및 관리
"""
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Literal, Optional
import functools
import operator
from typing import Sequence, Annotated, Dict, List, Any, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
import re
import json

from agents import AgentManager


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
    
    def supervisor_agent_node(self, state):
        """Supervisor React Agent 노드"""
        print("🔧 Supervisor 노드 실행")

        input_state = {
            "question": state.get("question", ""),
            "current_time": state.get("current_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "session_id": state.get("session_id", "unknown"),
            "session_start_time": state.get("session_start_time", "unknown"),
            "birth_info": state.get("birth_info"),
            "saju_result": state.get("saju_result"),
            "query_type": state.get("query_type", "unknown"),
            "retrieved_docs": state.get("retrieved_docs", []),
            "web_search_results": state.get("web_search_results", []),
        }
        
        supervisor_agent = self.agent_manager.create_supervisor_agent(input_state)
        
        response = supervisor_agent.invoke({
            "messages": state.get("messages", [HumanMessage(content=state.get("question", ""))]),
        })
        
        # 응답에서 라우팅 정보 추출
        output = json.loads(response["output"]) if isinstance(response["output"], str) else response["output"]

        updated_state = state.copy()

        if output.get("action"):
            pass
        if output.get("next"):
            updated_state["next"] = output.get("next")
        if output.get("message"):
            updated_state["messages"].append(AIMessage(content=output.get("message")))
        if output.get("final_answer"):
            updated_state["final_answer"] = output.get("final_answer")
        if output.get("reason"):
            pass
        if output.get("birth_info"):
            updated_state["birth_info"] = output.get("birth_info")
        if output.get("query_type"):
            updated_state["query_type"] = output.get("query_type")

        return updated_state

    def saju_expert_agent_node(self, state):
        """Saju Expert Agent 노드"""
        print("🔧 Saju Expert 노드 실행")
        
        current_time = state.get("current_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        session_id = state.get("session_id", "unknown")
        session_start_time = state.get("session_start_time", "unknown")
        messages = state.get("messages", [])

        supervisor_command = state.get("messages")[-1].content

        year = state.get("birth_info", {}).get("year")
        month = state.get("birth_info", {}).get("month")
        day = state.get("birth_info", {}).get("day")
        hour = state.get("birth_info", {}).get("hour")
        minute = state.get("birth_info", {}).get("minute")
        gender = "남자" if state.get("birth_info", {}).get("is_male") else "여자"
        is_leap_month = state.get("birth_info", {}).get("is_leap_month")

        saju_result = state.get("saju_result", "")

        saju_expert_agent = self.agent_manager.create_saju_expert_agent()

        response = saju_expert_agent.invoke({
            "current_time": current_time,
            "session_id": session_id,
            "session_start_time": session_start_time,
            "supervisor_command": supervisor_command,
            "year": year,
            "month": month,
            "day": day,
            "hour": hour,
            "minute": minute,
            "gender": gender,
            "is_leap_month": is_leap_month,
            "saju_result": saju_result,
            "messages": messages,
        })
       
        output = json.loads(response["output"]) if isinstance(response["output"], str) else response["output"]

        updated_state = state.copy()
        updated_state["saju_result"] = output
        updated_state["messages"].append(AIMessage(content=output.get("saju_analysis")))

        return updated_state

    def search_agent_node(self, state):
        """Search Agent 노드 (RAG + 웹검색 통합)"""
        print("🔧 Search 노드 실행")

        current_time = state.get("current_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        session_id = state.get("session_id", "unknown")
        session_start_time = state.get("session_start_time", "unknown")
        messages = state.get("messages", [])
        question = state.get("question", "")
        supervisor_command = state.get("messages")[-1].content
        saju_result = state.get("saju_result", "")

        search_agent = self.agent_manager.create_search_agent()

        response = search_agent.invoke({
            "current_time": current_time,
            "session_id": session_id,
            "session_start_time": session_start_time,
            "supervisor_command": supervisor_command,
            "question": question,
            "saju_result": saju_result,
            "messages": messages,
        })

        output = json.loads(response["output"]) if isinstance(response["output"], str) else response["output"]

        updated_state = state.copy()
        updated_state["retrieved_docs"] = output.get("retrieved_docs", [])
        updated_state["web_search_results"] = output.get("web_search_results", [])
        updated_state["messages"].append(AIMessage(content=output.get("generated_result")))

        return updated_state

    def general_answer_agent_node(self, state):
        """General Answer Agent 노드"""
        print("🔧 General Answer 노드 실행")

        current_time = state.get("current_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        session_id = state.get("session_id", "unknown")
        session_start_time = state.get("session_start_time", "unknown")
        messages = state.get("messages", [])
        question = state.get("question", "")
        supervisor_command = state.get("messages")[-1].content

        general_answer_agent = self.agent_manager.create_general_answer_agent()

        response = general_answer_agent.invoke({
            "current_time": current_time,
            "session_id": session_id,
            "session_start_time": session_start_time,
            "supervisor_command": supervisor_command,
            "question": question,
            "messages": messages,
        })

        output = json.loads(response["output"]) if isinstance(response["output"], str) else response["output"]

        updated_state = state.copy()
        updated_state["general_answer"] = output.get("general_answer")
        updated_state["messages"].append(AIMessage(content=output.get("general_answer")))

        return updated_state


# 전역 NodeManager 인스턴스 (싱글톤 패턴)
_node_manager = None

def get_node_manager():
    """싱글톤 NodeManager 인스턴스 반환"""
    global _node_manager
    if _node_manager is None:
        _node_manager = NodeManager()
    return _node_manager 