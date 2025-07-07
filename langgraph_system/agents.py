"""
에이전트 생성 및 관리
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, load_prompt
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any
from datetime import datetime
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, SystemMessage
from prompts import PromptManager, SupervisorResponse
from langchain_core.output_parsers import JsonOutputParser

# 단순화된 tools import (노트북 방식)
from tools import (
    calculate_saju_tool, 
    saju_retriever_tool, 
    web_tools, 
    general_qa_tool,
    saju_tools,
    retriever_tools,
    general_qa_tools,
    supervisor_tools
)

# 멤버 Agent 목록 정의 (notebook 구조에 맞게 변경)
members = ["SajuExpert", "WebTool", "GeneralQA"]


class AgentManager:
    """에이전트 생성 및 관리 클래스"""
    
    def __init__(self):
        # 기본 LLM 설정
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    
    def create_supervisor_agent(self):
        """
        Tool Calling Agent를 생성합니다.
        State 정보를 동적으로 프롬프트에 주입합니다.
        """
        llm = ChatOpenAI(temperature=0, model="gpt-4.1-mini")
        
        # Tool Calling Agent용 프롬프트 템플릿
        prompt = PromptManager().supervisor_system_prompt()
        
        # Tool Calling Agent 생성
        agent = create_tool_calling_agent(llm, supervisor_tools, prompt)
        
        # AgentExecutor 생성
        agent_executor = AgentExecutor(
            agent=agent,
            tools=supervisor_tools,
            verbose=True,
            max_iterations=3,
            early_stopping_method="generate"
        )
        
        return agent_executor
    
    def create_saju_expert_agent(self):
        """사주 전문 에이전트 생성"""
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        prompt = PromptManager().saju_expert_system_prompt()

        agent = create_tool_calling_agent(llm, saju_tools, prompt)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=saju_tools,
            verbose=True,
            max_iterations=3,
            early_stopping_method="generate"
        )
        return agent_executor
    
    def create_retriever_agent(self):
        """RAG 검색 에이전트 생성"""
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        prompt = PromptManager().retriever_system_prompt()

        agent = create_tool_calling_agent(llm, retriever_tools, prompt)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=retriever_tools,
            verbose=True,
            max_iterations=3,
            early_stopping_method="generate"
        )
        return agent_executor
    
    def create_web_search_agent(self):
        """Web Search Agent 생성"""
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        prompt = PromptManager().web_search_system_prompt()
        
        agent = create_tool_calling_agent(llm, web_tools, prompt)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=web_tools,
            verbose=True,
            max_iterations=3,
            early_stopping_method="generate"
        )

        return agent_executor
    
    def create_general_answer_agent(self):
        """General Answer Agent 생성"""
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        prompt = PromptManager().general_answer_system_prompt()
        
        agent = create_tool_calling_agent(llm, general_qa_tools, prompt)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=general_qa_tools,
            verbose=True,
            max_iterations=3,
            early_stopping_method="generate"
        )
        
        return agent_executor

    # def create_manse_tool_agent(self):
    #     """만세력 계산 에이전트 생성 (노트북 방식으로 단순화)"""
    #     llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    #     return create_react_agent(llm, manse_tools)

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


