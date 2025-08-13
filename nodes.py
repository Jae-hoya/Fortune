from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Literal
import functools
import json

from agents import AgentManager
from prompts import PromptManager
from tools import calculate_saju_tool

members = ["search", "manse", "general_qa"]
options_for_next = ["FINISH"] + members

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

        class RouteResponse(BaseModel):
            next: Literal[*options_for_next]

        llm = ChatOpenAI(model="gpt-4.1", temperature=0)
        user_input = state.get("question") or (state["messages"][0].content if state.get("messages") else "")

        birth_info = parse_birth_info_with_llm(user_input, self.llm)
        state["birth_info"] = birth_info

        now = self.agent_manager.now
        supervisor_prompt = PromptManager().supervisor_prompt()

        supervisor_chain = (
            supervisor_prompt.partial(
                options=str(options_for_next), 
                members=", ".join(members), 
                now=now,
                birth_info=state.get('birth_info', 'None'),
                saju_result=state.get('saju_result', 'None'),
            )
            | llm.with_structured_output(RouteResponse)
        )

        route_response = supervisor_chain.invoke(state)

        if route_response.next == "manse":
            if birth_info and all(key in birth_info and birth_info[key] is not None for key in ["year", "month", "day"]):
                # 생년월일 정보가 있으면 manse로 진행
                return {"next": "manse"}
            else:
                # 생년월일 정보가 없으면 사용자에게 안내하고 general_qa로 유도
                state["messages"].append(
                    AIMessage(
                        content=(
                            "사주 분석을 위해서는 생년월일, 태어난 시간, 성별 정보가 필요합니다.\n"
                            "예: **1990년 3월 5일 오후 3시, 남자**\n\n"
                            "생년월일 정보를 알려주시면 정밀한 운세를 안내해 드릴게요 🙂"
                        ),
                        name="Supervisor",
                    )
                )
                return {"next": "general_qa"}
            
        return {"next": route_response.next}

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
    
    def general_qa_agent_node(self, state):
        general_qa_agent = self.agent_manager.create_general_qa_agent()
        agent_response = general_qa_agent.invoke({
            "birth_info": state.get("birth_info"),
            "saju_result": state.get("saju_result"),
            "messages": state.get("messages", []),
        })
        state["messages"].append(
            HumanMessage(content=agent_response["messages"][-1].content, name="GeneralQA")
        )
        return state
    
    def search_agent_node(self, state):
        user_input = state.get("question") or (state["messages"][0].content if state.get("messages") else "")
        category = self._classify_search_llm(user_input)
        if category == "retriever":
            return self.create_retriever_tool_agent_node()(state)
        else:
            return self.create_web_tool_agent_node()(state)
        
    def _classify_search_llm(self, user_input):
        prompt = """
        - 사주에 대한 자세한 설명이 필요하면 retriever
        - 특별한 내부 언급이 없거나, 일반적/공개 정보/공식/인터넷/최신/정의/설명/이론/근거/출처 등은 web
        - 십신분석의 개념, 사주개념, 또는 사주 오행의 개념적 질문이 들어오면, web
        둘 중 가장 적합한 카테고리( retriever / web )만 답변하세요.
        

        질문: "{user_input}"
        정답:
        """.format(user_input=user_input)
        result = self.llm.invoke(prompt)
        
        return result.content.strip().lower()
    


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

    만약 출생 정보가 명확하지 않거나 부족하면 null을 반환하세요.
    year, month, day는 필수이고, hour, minute, is_male, is_leap_month는 선택사항입니다.
    is_male은 true(남자), false(여자)로 설정하세요.
    is_leap_month는 윤달인 경우에만 true로 설정하세요.

    입력: {user_input}
    """
    result = llm.invoke(prompt)
    try:
        # JSON 문자열에서 불필요한 문자 제거
        content = result.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        birth_info = json.loads(content)
        
        # 필수 필드 확인
        if not birth_info or not all(key in birth_info and birth_info[key] is not None for key in ["year", "month", "day"]):
            return None
            
        return birth_info
    except Exception as e:
        print("파싱 오류:", e)
        return None
    