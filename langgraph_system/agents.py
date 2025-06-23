"""
에이전트 생성 및 관리
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
from typing import Literal
from datetime import datetime

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
    
    def create_supervisor_agent(self, tools):
        """Supervisor ReAct 에이전트 생성 (notebook 구조에 맞게 수정)"""
        # 시스템 프롬프트 정의: 작업자 간의 대화를 관리하는 감독자 역할
        system_prompt = (
            "Today's date is {current_datetime}.\n"
            "You are a supervisor tasked with orchestrating a multi-step workflow with the following specialized agents: {members}.\n"
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
        
        # Supervisor Agent 생성 (구조화된 출력 사용)
        def supervisor_agent(state):
            # 현재 시간 포맷팅
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 프롬프트와 LLM을 결합하여 체인 구성
            supervisor_chain = (
                prompt.partial(
                    options=str(options_for_next),
                    members=", ".join(members),
                    current_datetime=current_time
                ) | self.llm.with_structured_output(RouteResponse)
            )
            
            # Agent 호출하고 'next' 키로 딕셔너리 반환
            route_response = supervisor_chain.invoke(state)
            return {"next": route_response.next}
        
        return supervisor_agent

    def create_manse_agent(self, tools):
        """만세력 계산 ReAct 에이전트 생성 (notebook의 manse_tool_agent에 해당)"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 사주 전문가입니다. 생년월일시를 받아 사주팔자를 계산하세요. 반드시 도구를 사용하여 정확한 계산을 수행하세요."),
            MessagesPlaceholder(variable_name="messages")
        ])
        return create_react_agent(self.llm, tools=tools, prompt=prompt)

    def create_retriever_agent(self, tools):
        """RAG 검색 ReAct 에이전트 생성 (notebook의 retriever_tool_agent에 해당)"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 사주 해석 전문가입니다. 사주 관련 지식을 검색하고 해석하세요. 반드시 도구를 사용하여 정확한 정보를 찾으세요."),
            MessagesPlaceholder(variable_name="messages")
        ])
        return create_react_agent(self.llm, tools=tools, prompt=prompt)

    def create_web_agent(self, tools):
        """웹 검색 ReAct 에이전트 생성 (notebook의 web_tool_agent에 해당)"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "사주 또는 사주 오행의 개념적 질문이나, 일상 질문이 들어오면, web search를 통해 답합니다."),
            MessagesPlaceholder(variable_name="messages")
        ])
        return create_react_agent(self.llm, tools=tools, prompt=prompt)
    
    def create_general_qa_agent(self, tools):
        """일반 QA ReAct 에이전트 생성 (notebook의 general_qa_agent에 해당)"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "일반적인 질문이나 상식적인 내용에 대해 답변합니다."),
            MessagesPlaceholder(variable_name="messages")
        ])
        return create_react_agent(self.llm, tools=tools, prompt=prompt)

    # 기존 메서드들 유지 (호환성을 위해)
    def create_saju_agent(self, tools):
        """사주 계산 ReAct 에이전트 생성 (기존 호환성 유지)"""
        return self.create_manse_agent(tools)

    def create_rag_agent(self, tools):
        """RAG 검색 ReAct 에이전트 생성 (기존 호환성 유지)"""
        return self.create_retriever_agent(tools)
    
    def create_response_generator_agent(self, tools):
        """응답 생성 ReAct 에이전트 생성"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 최종 응답 생성 전문가입니다. 
다른 에이전트들이 수집한 정보를 종합하여 사용자에게 완성도 높은 최종 답변을 제공하세요.

- 여러 에이전트의 결과를 통합
- 일관성 있고 이해하기 쉬운 응답 생성
- 필요시 추가 검색을 통해 정보 보완"""),
            MessagesPlaceholder(variable_name="messages")
        ])
        return create_react_agent(self.llm, tools=tools, prompt=prompt)
