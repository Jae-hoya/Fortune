# http://xn--vj1b09xs1b16ct2c.com/share/calendar/?selboxDirect=&y=1995&m=8&d=16
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
사주 에이전틱 RAG 시스템
LangGraph를 사용한 멀티 에이전트 사주 분석 시스템
"""

import os
import functools
import operator
from datetime import datetime
from typing import Sequence, Annotated, Literal
from typing_extensions import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, load_prompt
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from langgraph.prebuilt import create_react_agent
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

from langchain_teddynote.tools.tavily import TavilySearch
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_teddynote.messages import random_uuid, stream_graph, invoke_graph


from datetime import datetime
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 환경 변수 로드
load_dotenv()

class RouteResponse(BaseModel):
    """라우팅 응답 모델"""
    next: Literal["ManseTool", "RetrieverTool", "WebTool", "GeneralQA", "FINISH"]

class AgentState(TypedDict):
    """에이전트 상태 정의"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

def load_retriever_and_chain():
    """PDF Retriever 및 Chain 로드"""
    try:
        from pdf_retriever_saju import pdf_rag_chain, compression_retriever
        pdf_retriever = compression_retriever()
        pdf_chain = pdf_rag_chain()
        return pdf_retriever, pdf_chain
    except ImportError:
        print("pdf_retriever_saju 모듈을 찾을 수 없습니다.")
        return None, None

def load_manse_tool():
    """만세력 계산 도구 로드"""
    try:
        from manse_7 import calculate_saju_tool
        return calculate_saju_tool
    except ImportError:
        print("manse_7 모듈을 찾을 수 없습니다.")
        return None

def load_query_expansion_agent():
    """쿼리 확장 에이전트 로드"""
    try:
        from query_expansion_agent import get_query_expansion_node, get_query_expansion_agent
        query_expansion_node = get_query_expansion_node()
        query_expansion_agent = get_query_expansion_agent()
        return query_expansion_node, query_expansion_agent
    except ImportError:
        print("query_expansion_agent 모듈을 찾을 수 없습니다.")
        return None, None

def create_manse_tool_agent():
    """만세력 도구 에이전트 생성"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    calculate_saju_tool = load_manse_tool()
    if calculate_saju_tool:
        tools = [calculate_saju_tool]
        return create_react_agent(llm, tools)
    return None

def create_retriever_tool_agent():
    """검색 도구 에이전트 생성"""
    pdf_retriever, _ = load_retriever_and_chain()
    if not pdf_retriever:
        return None
    
    # PDF 문서를 기반으로 검색 도구 생성
    retriever_tool = create_retriever_tool(
        pdf_retriever,
        "pdf_retriever",
        "A tool for searching information related to Saju (Four Pillars of Destiny)",
        document_prompt=PromptTemplate.from_template(
            "<document><context>{page_content}</context><metadata><source>{source}</source><page>{page}</page></metadata></document>"
        ),
    )
    
    retriever_tools = [retriever_tool]
    
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    
    try:
        base_prompt = load_prompt("prompt/saju-rag-promt_2.yaml")
        # base_prompt = load_prompt("prompt/saju-rag-promt_korea2.yaml")
        saju_prompt = ChatPromptTemplate.from_messages([
            ("system", base_prompt.template),
            MessagesPlaceholder("messages"),
        ])
        return create_react_agent(llm, retriever_tools, prompt=saju_prompt)
    except FileNotFoundError:
        # 프롬프트 파일이 없으면 기본 프롬프트 사용
        return create_react_agent(llm, retriever_tools)

def create_web_tool_agent():
    """웹 검색 도구 에이전트 생성"""
    tavily_tool = TavilySearch(
        max_results=5,
        include_domains=["namu.wiki", "wikipedia.org"]
    )
    
    duck_tool = DuckDuckGoSearchResults(max_results=5)
    web_tools = [tavily_tool, duck_tool]
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = """
    사주 또는 사주 오행의 개념적 질문이나, 일상 질문이 들어오면, web search를 통해 답합니다.
    """
    
    return create_react_agent(llm, tools=web_tools, prompt=prompt)

def create_general_qa_tool():
    """일반 QA 도구 생성"""
    @tool
    def general_qa_tool(query: str) -> str:
        """
        일반적인 질문이나 상식적인 내용에 대해 답변합니다. 사주와 관련 없는 모든 질문에 사용할 수 있습니다.
        """
        google_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        return google_llm.invoke(query)
    
    return general_qa_tool

def create_general_qa_agent():
    """일반 QA 에이전트 생성"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    general_qa_tools = [create_general_qa_tool()]
    
    prompt = "일반적인 질문이나 상식적인 내용에 대해 답변합니다."
    
    return create_react_agent(llm, tools=general_qa_tools, prompt=prompt)

def agent_node(state, agent, name):
    """에이전트 노드 함수"""
    agent_response = agent.invoke(state)
    return {
        "messages": [
            HumanMessage(content=agent_response["messages"][-1].content, name=name)
        ]
    }

def create_supervisor_agent():
    """감독자 에이전트 생성"""
    members = ["ManseTool", "RetrieverTool", "WebTool", "GeneralQA"]
    options_for_next = ["FINISH"] + members
    
    system_prompt = (
        "Today is {now}.\n"
        "You are a supervisor tasked with orchestrating a multi-step workflow with the following specialized agents: {members}.\n"
        "Your primary responsibility is to route user requests to the appropriate agent and ensure the conversation flows logically. "
        "The user's Saju (Four Pillars) information, once calculated, is maintained in the conversation history.\n\n"
        "AGENT ROLES:\n"
        "- ManseTool: Use this ONLY when the user provides their birth information (date and time) for the first time. This tool calculates the user's Saju and adds it to the conversation history.\n"
        "- RetrieverTool: This is the primary tool for interpreting Saju. Use it for ANY questions related to the user's fortune, destiny, or Saju analysis (e.g., 'what about my wealth luck?', 'explain my Day Pillar', 'how is my health outlook?'). It uses the Saju information from the conversation history. This tool should also be used for any follow-up questions regarding the Saju reading.\n"
        "- WebTool: Use this for general, conceptual questions about Saju theory or history that are not specific to the user's own Saju (e.g., 'what is the history of the Four Pillars?'). Also use it for any non-Saju related questions that require web search.\n"
        "- GeneralQA: Use this for general knowledge questions that are completely unrelated to Saju.\n\n"
        "ROUTING RULES:\n"
        "1.  Initial Request: If the message contains birth details, route to ManseTool.\n"
        "2.  **Mandatory Follow-up**: After ManseTool runs, you MUST route to RetrieverTool to provide the initial interpretation.\n"
        "3.  **Subsequent Questions**: For ANY follow-up questions about the user's Saju (luck, health, career, relationships, etc.), ALWAYS route to RetrieverTool. Do not re-use ManseTool if the Saju is already calculated.\n"
        "4.  If the Saju information is already in the chat history, do not call ManseTool again. Use RetrieverTool directly for any Saju-related inquiries.\n"
        "5.  When the conversation is complete and the user's questions are answered, you can respond with FINISH."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation history and the user's latest request, who should act next? "
            "Select one of: {options}",
        ),
    ]).partial(options=str(options_for_next), members=", ".join(members), now=now)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    supervisor_chain = prompt | llm.with_structured_output(RouteResponse)
    
    def supervisor_agent(state):
        return supervisor_chain.invoke(state)
        
    return supervisor_agent

def create_workflow_graph():
    """워크플로우 그래프 생성"""
    # 에이전트들 생성
    manse_tool_agent = create_manse_tool_agent()
    retriever_tool_agent = create_retriever_tool_agent()
    web_tool_agent = create_web_tool_agent()
    general_qa_agent = create_general_qa_agent()
    query_expansion_node, _ = load_query_expansion_agent()
    supervisor_agent = create_supervisor_agent()
    
    if not all([manse_tool_agent, retriever_tool_agent, web_tool_agent, general_qa_agent, query_expansion_node]):
        print("일부 에이전트 생성에 실패했습니다. 필요한 모듈들을 확인해주세요.")
        return None
    
    # 노드 생성
    manse_tool_agent_node = functools.partial(agent_node, agent=manse_tool_agent, name="ManseTool")
    retriever_tool_agent_node = functools.partial(agent_node, agent=retriever_tool_agent, name="RetrieverTool")
    web_tool_agent_node = functools.partial(agent_node, agent=web_tool_agent, name="WebTool")
    general_qa_agent_node = functools.partial(agent_node, agent=general_qa_agent, name="GeneralQA")
    
    # 그래프 생성
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("ManseTool", manse_tool_agent_node)
    workflow.add_node("QueryExpansion", query_expansion_node)
    workflow.add_node("RetrieverTool", retriever_tool_agent_node)
    workflow.add_node("WebTool", web_tool_agent_node)
    workflow.add_node("GeneralQA", general_qa_agent_node)
    workflow.add_node("Supervisor", supervisor_agent)
    
    # 멤버 노드 > Supervisor 노드로 엣지 추가
    members = ["ManseTool", "RetrieverTool", "WebTool", "GeneralQA"]
    for member in members:
        workflow.add_edge(member, "Supervisor")
    
    # 조건부 엣지 추가
    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    
    def get_next(state):
        return state["next"]
    
    # Supervisor 노드에서 조건부 엣지 추가
    workflow.add_conditional_edges("Supervisor", get_next, conditional_map)
    
    # 시작점
    workflow.add_edge(START, "QueryExpansion")
    workflow.add_edge("QueryExpansion", "Supervisor")
    
    # 그래프 컴파일
    return workflow.compile(checkpointer=MemorySaver())

def run_saju_analysis(question, thread_id, use_stream=True):
    """사주 분석 실행"""
    graph = create_workflow_graph()
    if not graph:
        return "그래프 생성에 실패했습니다."
    
    # config 설정
    config = RunnableConfig(recursion_limit=20, configurable={"thread_id": thread_id})
    
    # 입력 준비
    inputs = {
        "messages": [HumanMessage(content=question)]
    }
    
    # 실행
    if use_stream:
        return stream_graph(graph, inputs, config)
    else:
        return invoke_graph(graph, inputs, config)

def main():
    """메인 함수"""
    print("사주 에이전틱 RAG 시스템을 시작합니다... ")
    print("생년월일, 태이난 시각, 성별을 입력해 주세요.")
    print("윤달에 태어나신 경우, 윤달이라고 작성해주세요.")
    
    # 예시 질문들
    example_questions = [
        "1996년 12월 13일 남자, 10시 30분 출생 운세봐줘.",
        "대운과 세운, 조심해야 할것들 알려줘",
        "금전운알려줘",
        "정관이 뭐야? 상세히 설명해줘",
        "사주의 개념에 대해서 알려줘"
    ]
    
    print("\n사용 가능한 예시 질문:")
    for i, question in enumerate(example_questions, 1):
        print(f"{i}. {question}")
    
    # 대화 세션을 위한 thread_id 생성
    #thread_id = random_uuid()
    thread_id = random_uuid()
    print(f"\n새로운 대화 세션을 시작합니다. (ID: {thread_id})")
    print("\n질문을 입력하세요 (종료하려면 'quit' 입력):")
    
    while True:
        user_input = input("\n질문: ").strip()
        
        if user_input.lower() in ['quit', 'exit', '종료']:
            print("시스템을 종료합니다.")
            break
        
        if not user_input:
            continue
        
        try:
            print("\n분석을 시작합니다...")
            # 대화의 연속성을 위해 동일한 thread_id를 전달
            result = run_saju_analysis(user_input, thread_id=thread_id, use_stream=True)
            print("\n분석 완료!")
            
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main() 