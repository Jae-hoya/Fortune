"""
에이전트 생성 및 관리
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
from typing import Literal

# 멤버 Agent 목록 정의
members = ["SajuAgent", "RagAgent", "WebAgent"]

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
        """Supervisor ReAct 에이전트 생성"""
        # 시스템 프롬프트 정의: 작업자 간의 대화를 관리하는 감독자 역할
        system_prompt = (
            "당신은 다음과 같은 전문 에이전트들로 구성된 다단계 워크플로를 관리하는 감독자입니다: {members}.\n"
            "각 에이전트의 역할은 다음과 같습니다:\n"
            "- SajuAgent: 사용자 입력에서 생년월일시 정보를 추출하고 사주팔자(년주, 월주, 일주, 시주)를 계산합니다.\n"
            "- RagAgent: 계산된 사주 정보를 바탕으로 심층적인 사주 해석과 명리학적 분석을 제공합니다.\n"
            "- WebAgent: 사주와 관련된 일반적이거나 개념적인 질문, 또는 일상적인 질문에 웹 검색을 통해 답변합니다.\n\n"

            "당신의 임무는 다음과 같습니다:\n"
            "1. 사용자 요청을 가장 적절한 에이전트로 라우팅:\n"
            "   - 사용자 입력에 생년월일시 정보가 포함되어 있으면, 반드시 SajuAgent를 먼저 호출하세요.\n"
            "   - 중요: **SajuAgent 사용 후에는 반드시 RagAgent를 호출하여 계산된 사주 결과를 해석해야 합니다. 이 단계를 건너뛰지 마세요.**\n"
            "   - 순수하게 일반적이거나 개념적인 사주 질문으로 계산이 필요하지 않다면, RagAgent를 직접 호출하세요.\n"
            "   - 사주와 완전히 무관하거나 최신 정보 검색이 필요한 경우, WebAgent를 사용하세요.\n"
            "   - 사용자 입력이 너무 짧거나 애매하거나 의미가 불분명한 경우(예: 'new', 'test', 'hello' 등), 바로 FINISH를 선택하세요.\n"
            "2. SajuAgent 단독 사용 후 절대 워크플로를 종료하지 마세요. FINISH를 고려하기 전에 반드시 결과를 RagAgent에 전달하세요.\n"
            "3. 사주와 완전히 무관한 질문은 WebAgent를 직접 사용하세요.\n"
            "4. 모든 필요한 단계가 완료되면 FINISH로 응답하세요.\n"
            "5. 사용자 입력이 의미가 없거나 처리할 수 없는 경우 즉시 FINISH로 응답하세요.\n"
            "반드시 작업에 가장 논리적인 다음 에이전트를 결정하고 이 에이전트 순서를 엄격히 따라야 합니다."
        )

        # ChatPromptTemplate 생성
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "위 대화를 바탕으로 다음에 어떤 에이전트가 작업해야 할까요? "
                "또는 작업을 완료해야 할까요? 다음 중에서 선택하세요: {options}",
            ),
        ]).partial(options=str(options_for_next), members=", ".join(members))
        
        # Supervisor Agent 생성 (구조화된 출력 사용)
        def supervisor_agent(state):
            supervisor_chain = prompt | self.llm.with_structured_output(RouteResponse)
            return supervisor_chain.invoke(state)
        
        return supervisor_agent
    
    def create_saju_agent(self, tools):
        """사주 계산 ReAct 에이전트 생성"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 사주 전문가입니다. 생년월일시를 받아 사주팔자를 계산하고 해석하세요. 반드시 도구를 사용하여 정확한 계산을 수행하세요."),
            MessagesPlaceholder(variable_name="messages")
        ])
        return create_react_agent(self.llm, tools=tools, prompt=prompt)

    def create_rag_agent(self, tools):
        """RAG 검색 ReAct 에이전트 생성"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 사주 해석 전문가입니다. 사주 관련 지식을 검색하고 해석하세요. 반드시 도구를 사용하여 정확한 정보를 찾으세요."),
            MessagesPlaceholder(variable_name="messages")
        ])
        return create_react_agent(self.llm, tools=tools, prompt=prompt)

    def create_web_agent(self, tools):
        """웹 검색 ReAct 에이전트 생성"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 웹 검색 전문가입니다. 최신 정보를 검색하세요. 반드시 도구를 사용하여 정확한 정보를 찾으세요."),
            MessagesPlaceholder(variable_name="messages")
        ])
        return create_react_agent(self.llm, tools=tools, prompt=prompt)
    
    def create_general_agent(self, tools):
        """범용 ReAct 에이전트 생성"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 사주 전문 상담사입니다. 주어진 도구들을 활용하여 사용자의 질문에 정확하고 전문적으로 답변하세요."),
            MessagesPlaceholder(variable_name="messages")
        ])
        return create_react_agent(self.llm, tools=tools, prompt=prompt)
 
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
