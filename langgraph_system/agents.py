"""
에이전트 생성 및 관리
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, load_prompt
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
from typing import Literal
from datetime import datetime

# 새로운 imports (notebook 구조 지원용)
from saju_calculator import calculate_saju_tool
from reranker import create_saju_compression_retriever
from langchain_core.tools.retriever import create_retriever_tool
from langchain_teddynote.tools.tavily import TavilySearch
from langchain.tools import DuckDuckGoSearchResults, tool
from langchain_google_genai import ChatGoogleGenerativeAI

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
