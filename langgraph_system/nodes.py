"""
노드 함수들 - NodeManager 클래스로 노드 생성 및 관리
"""

import re
import functools
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.prompts import PromptTemplate, load_prompt, ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from typing import Dict, Any, List

from .state import SajuState
from .agents import AgentManager
from tools import ToolManager

# 새로운 imports (notebook 구조 지원용)
from saju_calculator import calculate_saju_tool
from reranker import create_saju_compression_retriever
from langchain_teddynote.tools.tavily import TavilySearch
from langchain.tools import DuckDuckGoSearchResults, tool
from langchain_google_genai import ChatGoogleGenerativeAI

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
    
    # === 기존 노드들 (호환성 유지) ===
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

    def create_manse_tool_agent(self):
        """만세력 계산 에이전트 생성"""
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        tools = [calculate_saju_tool]
        return create_react_agent(llm, tools)

    def create_retriever_tool_agent(self):
        """RAG 검색 에이전트 생성"""
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        
        # PDF 문서를 기반으로 검색 도구 생성
        pdf_retriever = create_saju_compression_retriever()
        retriever_tool = create_retriever_tool(
            pdf_retriever,
            "pdf_retriever",
            "A tool for searching information related to Saju (Four Pillars of Destiny)",
            document_prompt=PromptTemplate.from_template(
                "<document><context>{page_content}</context><metadata><source>{source}</source><page>{page}</page></metadata></document>"
            ),
        )
        
        retriever_tools = [retriever_tool]
        
        # 프롬프트 로드
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
        """웹 검색 에이전트 생성"""
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        
        # 웹 검색 도구들 설정
        tavily_tool = TavilySearch(
            max_results=5,
            include_domains=["namu.wiki", "wikipedia.org"]
        )
        
        duck_tool = DuckDuckGoSearchResults(
            max_results=5,
        )
        
        web_tools = [tavily_tool, duck_tool]
        
        prompt = """
사주 또는 사주 오행의 개념적 질문이나, 일상 질문이 들어오면, web search를 통해 답합니다.
"""
        
        return create_react_agent(llm, tools=web_tools, prompt=prompt)

    def create_general_qa_agent(self):
        """일반 QA 에이전트 생성"""
        @tool
        def general_qa_tool(query: str) -> str:
            """
            일반적인 질문이나 상식적인 내용에 대해 답변합니다. 사주와 관련 없는 모든 질문에 사용할 수 있습니다.
            """
            google_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
            return google_llm.invoke(query).content

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        general_qa_tools = [general_qa_tool]
        
        prompt = "일반적인 질문이나 상식적인 내용에 대해 답변합니다."
        
        return create_react_agent(llm, tools=general_qa_tools, prompt=prompt)

    # === Notebook 스타일 노드 생성 메서드들 ===
    def create_manse_tool_agent_node(self):
        """Manse Tool Agent 노드 생성"""
        manse_tool_agent = self.create_manse_tool_agent()
        return functools.partial(self._agent_node, agent=manse_tool_agent, name="ManseTool")

    def create_retriever_tool_agent_node(self):
        """Retriever Tool Agent 노드 생성"""
        retriever_tool_agent = self.create_retriever_tool_agent()
        return functools.partial(self._agent_node, agent=retriever_tool_agent, name="RetrieverTool")

    def create_web_tool_agent_node(self):
        """Web Tool Agent 노드 생성"""
        web_tool_agent = self.create_web_tool_agent()
        return functools.partial(self._agent_node, agent=web_tool_agent, name="WebTool")

    def create_general_qa_agent_node(self):
        """General QA Agent 노드 생성"""
        general_qa_agent = self.create_general_qa_agent()
        return functools.partial(self._agent_node, agent=general_qa_agent, name="GeneralQA")

# 전역 NodeManager 인스턴스 (싱글톤 패턴)
_node_manager = None

def get_node_manager():
    """싱글톤 NodeManager 인스턴스 반환"""
    global _node_manager
    if _node_manager is None:
        _node_manager = NodeManager()
    return _node_manager
