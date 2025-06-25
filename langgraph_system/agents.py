"""
에이전트 생성 및 관리
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, load_prompt
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
from typing import Literal
from datetime import datetime

# 단순화된 tools import (노트북 방식)
from tools import (
    calculate_saju_tool, 
    saju_retriever_tool, 
    web_tools, 
    general_qa_tool,
    manse_tools,
    retriever_tools,
    general_qa_tools
)

# 멤버 Agent 목록 정의 (notebook 구조에 맞게 변경)
members = ["SajuExpert", "WebTool", "GeneralQA"]

# 다음 작업자 선택 옵션 목록 정의
options_for_next = ["FINISH"] + members

# 작업자 선택 응답 모델 정의: 다음 작업자를 선택하거나 작업 완료를 나타냄
class RouteResponse(BaseModel):
    next: Literal[*options_for_next]

class AgentManager:
    """에이전트 생성 및 관리 클래스"""
    
    def __init__(self):
        # 기본 LLM 설정
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    
    def create_supervisor_agent(self, tools=None):
        """Supervisor 에이전트 생성 (다른 에이전트들과 동일한 패턴)"""
        # 시스템 프롬프트 정의
        system_prompt = (
            "You are a supervisor tasked with orchestrating a multi-step workflow with the following specialized agents: {members}.\n"
            "Current time: {current_datetime}\n\n"
            "The tools are:\n"
            "- SajuExpert: This agent is an expert in Saju (Four Pillars of Destiny). It handles everything from calculating the Saju from birth information to providing detailed interpretations and analysis. Use this for any query that involves birth dates/times or asks for a Saju reading.\n"
            "- WebTool: For answering general or conceptual questions about Saju, or handling everyday/non-specialized queries, by searching the web. Use this if the user is asking what Saju is, but not for a personal reading.\n"
            "- GeneralQA: For answering general questions that are NOT related to Saju at all (e.g., programming, science, general knowledge, etc.).\n\n"
            "Your job is to route the user's request to the most appropriate tool:\n"
            "   - If the user input contains birth information or asks for a fortune reading, call SajuExpert.\n"
            "   - If the input is a general or conceptual question about Saju, call WebTool.\n"
            "   - If the input is completely unrelated to Saju, call GeneralQA.\n"
            "After the agent has finished, respond with FINISH if the task is complete."
        )

        # ChatPromptTemplate 생성
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next? "
                "Or should we FINISH? Select one of: {options}",
            ),
        ])
        
        # Supervisor 함수 정의
        def supervisor_agent_func(state):
            # 현재 시간 가져오기
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 프롬프트와 LLM을 결합하여 체인 구성
            supervisor_chain = (
                prompt.partial(
                    options=str(options_for_next),
                    members=", ".join(members),
                    current_datetime=current_datetime
                ) | self.llm.with_structured_output(RouteResponse)
            )
            
            # Agent 호출하고 'next' 키로 딕셔너리 반환
            route_response = supervisor_chain.invoke(state)
            return {"next": route_response.next}
        
        return supervisor_agent_func

    def create_manse_tool_agent(self):
        """만세력 계산 에이전트 생성 (노트북 방식으로 단순화)"""
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        return create_react_agent(llm, manse_tools)

    def create_retriever_tool_agent(self):
        """RAG 검색 에이전트 생성 (노트북 방식으로 단순화)"""
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        
        # 프롬프트 로드 시도
        try:
            base_prompt = load_prompt("prompt/saju-rag-promt.yaml")
            saju_prompt = ChatPromptTemplate.from_messages([
                ("system", base_prompt.template),
                MessagesPlaceholder("messages"),
            ])
            return create_react_agent(llm, retriever_tools, prompt=saju_prompt)
        except Exception as e:
            print(f"프롬프트 로드 실패: {e}")
            # 기본 프롬프트 사용
            return create_react_agent(llm, retriever_tools)

    def create_web_tool_agent(self):
        """웹 검색 에이전트 생성 (노트북 방식으로 단순화)"""
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        
        prompt = """
사주 또는 사주 오행의 개념적 질문이나, 일상 질문이 들어오면, web search를 통해 답합니다.
"""
        return create_react_agent(llm, tools=web_tools, prompt=prompt)

    def create_general_qa_agent(self):
        """일반 QA 에이전트 생성 (노트북 방식으로 단순화)"""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        prompt = "일반적인 질문이나 상식적인 내용에 대해 답변합니다."
        return create_react_agent(llm, tools=general_qa_tools, prompt=prompt)


