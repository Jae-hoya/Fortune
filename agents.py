from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from datetime import datetime

from Fortune.prompts import PromptManager
from Fortune.tools import (
    retriever_tool,
    web_tools,
    manse_tools,
    general_qa_tools,
)


class AgentManager:
    """에이전트 생성 및 관리 클래스"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        self.now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")   

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
    