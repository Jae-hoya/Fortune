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
from prompts import PromptManager
from tools import calculate_saju_tool

class NodeManager:
    """노드 생성 및 관리 클래스"""
    
    def __init__(self):
        # 에이전트 관리자 초기화 (단순화)
        self.agent_manager = AgentManager()
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    
    # === 새로운 노드들 (notebook 구조 지원) ===
    def _agent_node(self, state, agent, name):
        """지정한 agent와 name을 사용하여 agent 노드를 생성하는 헬퍼 함수"""
        # agent 호출
        agent_response = agent.invoke(state)
        state["messages"] = state.get("messages", []) + [HumanMessage(content=agent_response["messages"][-1].content, name=name)]
        return state

    def supervisor_agent_node(self, state):
        """Supervisor Agent 노드 생성"""
        members = ["search", "manse", "general_qa"]
        options_for_next = ["FINISH"] + members

        class RouteResponse(BaseModel):
            next: Literal[*options_for_next]

        now = self.agent_manager.now
        supervisor_prompt = PromptManager().supervisor_prompt()

        supervisor_chain = (
            supervisor_prompt.partial(options=str(options_for_next), members=", ".join(members), now=now)
            | self.llm.with_structured_output(RouteResponse)
        )

        route_response = supervisor_chain.invoke(state)
        return {"next": route_response.next}

    def manse_agent_node(self, state):
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
    
    def manse_agent_node(self, state):
        """Manse Tool Agent 노드 생성"""
        user_input = state["question"]
        birth_info = parse_birth_info_with_llm(user_input, self.llm)
        state["birth_info"] = birth_info
        saju_result = calculate_saju_tool(birth_info)
        state["saju_result"] = saju_result
        prompt = f"""
        아래는 사용자의 사주 정보와 계산 결과입니다.
        - 입력: {user_input}
        - 사주 계산 결과: {json.dumps(saju_result, ensure_ascii=False, indent=2)}
        위 정보를 바탕으로, 사용자가 이해하기 쉽게 사주풀이 결과를 자연어로 설명해 주세요.
        """
        llm_response = self.llm.invoke(prompt)
        state["messages"].append(HumanMessage(content=llm_response.content, name="ManseLLM"))
        return state
    
    def search_agent_node(self, state):
        user_input = state.get("question") or (state["messages"][0].content if state.get("messages") else "")
        if any(k in user_input for k in ["자료", "문서", "pdf", "검색", "출처"]):
            return self.create_retriever_tool_agent_node()(state)
        else:
            return self.create_web_tool_agent_node()(state)
    


# 전역 NodeManager 인스턴스 (싱글톤 패턴)
_node_manager = None

def get_node_manager():
    """싱글톤 NodeManager 인스턴스 반환"""
    global _node_manager
    if _node_manager is None:
        _node_manager = NodeManager()
    return _node_manager 


def parse_birth_info_with_llm(user_input, llm):
    prompt = f"""
아래 문장에서 출생 정보를 추출해서 JSON 형태로 반환하세요.
필드: year, month, day, hour, minute, is_male, is_leap_month
예시 입력: "1996년 12월 13일 남자, 10시 30분 출생"
예시 출력: {{"year": 1996, "month": 12, "day": 13, "hour": 10, "minute": 30, "is_male": true, "is_leap_month": false}}

입력: {user_input}
"""
    result = llm.invoke(prompt)
    try:
        birth_info = json.loads(result.content)
        return birth_info
    except Exception as e:
        print("파싱 오류:", e)
        return None
    