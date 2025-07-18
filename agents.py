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
    retriever_tool,
    web_tools,
    manse_tools,
    search_tools,
    general_qa_tools,
    supervisor_tools
)

# 멤버 Agent 목록 정의 (notebook 구조에 맞게 변경)
members = ["SajuExpert", "Search", "GeneralAnswer"]


class AgentManager:
    """에이전트 생성 및 관리 클래스"""
    
    def __init__(self):
        # 기본 LLM 설정
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        self.now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def create_supervisor_agent(self):
        """
        Supervisor Agent를 생성합니다.
        State 정보를 동적으로 프롬프트에 주입합니다.
        """
        llm = ChatOpenAI(temperature=0, model="gpt-4.1-mini")
        
        # Agent용 프롬프트 템플릿
        prompt = PromptManager().supervisor_system_prompt()
        
        # Tool Calling Agent 생성
        agent = create_tool_calling_agent(
            llm=llm, 
            tools=supervisor_tools, 
            prompt=prompt
        )

        agent_executor = AgentExecutor(
            agent=agent,
            tools=supervisor_tools,
            verbose=True,
            max_iterations=3,
            early_stopping_method="generate"
        )
        
        return agent_executor
    

    def create_manse_tool_agent(self):
        """만세력 계산 에이전트 생성 (노트북 방식으로 단순화)"""
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        manse_tool_prompt = PromptManager().manse_tool_prompt()


        return create_react_agent(llm, manse_tools, prompt=manse_tool_prompt).with_config({"tags": ["final_answer_agent"]})

    def create_retriever_tool_agent(self):
        """RAG 검색 에이전트 생성 (노트북 방식으로 단순화)"""

        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        retriever_tool_prompt = PromptManager().retriever_tool_prompt()
        retriever_tools = [retriever_tool]
        
        saju_prompt = ChatPromptTemplate.from_messages([
            ("system", f"Today is {self.now}"),
            ("system", retriever_tool_prompt),
            MessagesPlaceholder("messages"),
        ])
        return create_react_agent(llm, retriever_tools, prompt=saju_prompt).with_config({"tags": ["final_answer_agent"]})


    def create_web_tool_agent(self):
        """웹 검색 에이전트 생성 (노트북 방식으로 단순화)"""
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        web_search_prompt = "사주 또는 사주 오행의 개념적 질문이나, 일상 질문이 들어오면, web search를 통해 답합니다."

        return create_react_agent(llm, tools=web_tools, prompt=web_search_prompt).with_config({"tags": ["final_answer_agent"]})

    def create_general_qa_agent(self):
        """일반 QA 에이전트 생성 (노트북 방식으로 단순화)"""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        general_qa_prompt = """
        일반적인 질문을 나의 사주와 관련하여 답변합니다.
        만약 state에 birth_info 또는 saju_result가 포함되어 있다면, 그 정보를 참고해서 답변에 반영하세요.
        birth_info: {birth_info}
        saju_result: {saju_result}
        """
        return create_react_agent(llm, tools=general_qa_tools, prompt=general_qa_prompt).with_config({"tags": ["final_answer_agent"]}) 