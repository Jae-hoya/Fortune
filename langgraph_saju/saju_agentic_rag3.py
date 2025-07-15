#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
사주 에이전틱 RAG 시스템 (대화 맥락 기억 버전)
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
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
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
    next: Literal["ManseTool", "RetrieverTool", "WebTool", "GeneralQA", "FINISH"]

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

def load_retriever_and_chain():
    try:
        from pdf_retriever_saju import pdf_rag_chain, compression_retriever
        pdf_retriever = compression_retriever()
        pdf_chain = pdf_rag_chain()
        return pdf_retriever, pdf_chain
    except ImportError:
        print("pdf_retriever_saju 모듈을 찾을 수 없습니다.")
        return None, None

def load_manse_tool():
    try:
        from manse_7 import calculate_saju_tool
        return calculate_saju_tool
    except ImportError:
        print("manse_7 모듈을 찾을 수 없습니다.")
        return None

def load_query_expansion_agent():
    try:
        from query_expansion_agent import get_query_expansion_node, get_query_expansion_agent
        query_expansion_node = get_query_expansion_node()
        query_expansion_agent = get_query_expansion_agent()
        return query_expansion_node, query_expansion_agent
    except ImportError:
        print("query_expansion_agent 모듈을 찾을 수 없습니다.")
        return None, None

def create_manse_tool_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    calculate_saju_tool = load_manse_tool()
    if calculate_saju_tool:
        tools = [calculate_saju_tool]
        return create_react_agent(llm, tools)
    return None

def create_retriever_tool_agent():
    pdf_retriever, _ = load_retriever_and_chain()
    if not pdf_retriever:
        return None
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
        saju_prompt = ChatPromptTemplate.from_messages([
            ("system", base_prompt.template),
            MessagesPlaceholder("messages"),
        ])
        return create_react_agent(llm, retriever_tools, prompt=saju_prompt)
    except FileNotFoundError:
        return create_react_agent(llm, retriever_tools)

def create_web_tool_agent():
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
    @tool
    def general_qa_tool(query: str) -> str:
        """
        일반적인 질문이나 상식적인 내용에 대해 답변합니다. 사주와 관련 없는 모든 질문에 사용할 수 있습니다.
        """
        google_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        return google_llm.invoke(query)
    return general_qa_tool

def create_general_qa_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    general_qa_tools = [create_general_qa_tool()]
    prompt = "일반적인 질문이나 상식적인 내용에 대해 답변합니다."
    return create_react_agent(llm, tools=general_qa_tools, prompt=prompt)

def agent_node(state, agent, name):
    agent_response = agent.invoke(state)
    # 마지막 메시지 추출
    last_msg = agent_response["messages"][-1]
    return {
        "messages": [HumanMessage(content=last_msg.content, name=name)]
    }

def create_supervisor_agent():
    """감독자 에이전트 생성"""
    members = ["ManseTool", "RetrieverTool", "WebTool", "GeneralQA"]
    options_for_next = ["FINISH"] + members
    
    system_prompt = (
        "Today is {now}.\n"
        "You are a supervisor orchestrating a workflow with these tools: {members}.\n"
        "- ManseTool: Extracts and calculates Saju details from birth info.\n"
        "- RetrieverTool: Interprets and explains Saju results from ManseTool, or answers any Saju-related follow-up question.\n"
        "- WebTool: For Saju general knowledge/history or external info not related to user's calculated Saju.\n"
        "- GeneralQA: For questions unrelated to Saju.\n\n"

        "You must strictly follow this sequence rule:\n"
        "1. **RetrieverTool must never be called first.** Before using RetrieverTool, you must always call ManseTool to obtain Saju calculation results.\n"
        "2. **After ManseTool is called, you must always call RetrieverTool next** to interpret the results. Never skip this step.\n"
        "3. Only after both steps above are complete may you consider WebTool or GeneralQA, if the user's question requires it.\n"
        "4. Never finish after ManseTool alone or after RetrieverTool without first going through ManseTool.\n"
        "5. Respond with FINISH only after all necessary tools have been executed in order.\n"
        "If the user's input is not Saju-related, call GeneralQA.\n"
    )


    # ChatPromptTemplate 생성
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next? "
                "Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options_for_next), members=", ".join(members), now=now)
    
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    
        
    def supervisor_agent(state):
        # 프롬프트와 LLM을 결합하여 체인 구성
        supervisor_chain = prompt | llm.with_structured_output(RouteResponse)
        # Agent 호출
        return supervisor_chain.invoke(state)
        
    return supervisor_agent

def create_workflow_graph():
    manse_tool_agent = create_manse_tool_agent()
    retriever_tool_agent = create_retriever_tool_agent()
    web_tool_agent = create_web_tool_agent()
    general_qa_agent = create_general_qa_agent()
    query_expansion_node, _ = load_query_expansion_agent()
    supervisor_agent = create_supervisor_agent()
    if not all([manse_tool_agent, retriever_tool_agent, web_tool_agent, general_qa_agent, query_expansion_node]):
        print("일부 에이전트 생성에 실패했습니다. 필요한 모듈들을 확인해주세요.")
        return None
    manse_tool_agent_node = functools.partial(agent_node, agent=manse_tool_agent, name="ManseTool")
    retriever_tool_agent_node = functools.partial(agent_node, agent=retriever_tool_agent, name="RetrieverTool")
    web_tool_agent_node = functools.partial(agent_node, agent=web_tool_agent, name="WebTool")
    general_qa_agent_node = functools.partial(agent_node, agent=general_qa_agent, name="GeneralQA")
    workflow = StateGraph(AgentState)
    workflow.add_node("ManseTool", manse_tool_agent_node)
    workflow.add_node("QueryExpansion", query_expansion_node)
    workflow.add_node("RetrieverTool", retriever_tool_agent_node)
    workflow.add_node("WebTool", web_tool_agent_node)
    workflow.add_node("GeneralQA", general_qa_agent_node)
    workflow.add_node("Supervisor", supervisor_agent)
    members = ["ManseTool", "RetrieverTool", "WebTool", "GeneralQA"]
    for member in members:
        workflow.add_edge(member, "Supervisor")
    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    def get_next(state):
        return state["next"]
    workflow.add_conditional_edges("Supervisor", get_next, conditional_map)
    workflow.add_edge(START, "QueryExpansion")
    workflow.add_edge("QueryExpansion", "Supervisor")
    return workflow.compile(checkpointer=MemorySaver())

def run_saju_analysis(messages, thread_id=None, use_stream=True):
    graph = create_workflow_graph()
    if not graph:
        return "그래프 생성에 실패했습니다."
    if thread_id is None:
        thread_id = random_uuid()
    config = RunnableConfig(recursion_limit=20, configurable={"thread_id": thread_id})
    inputs = {"messages": messages}
    if use_stream:
        return stream_graph(graph, inputs, config)
    else:
        return invoke_graph(graph, inputs, config)

def main():
    print("사주 에이전틱 RAG 시스템 (대화 맥락 기억 버전)을 시작합니다... ")
    print("생년월일, 태이난 시각, 성별을 입력해 주세요.")
    print("윤달에 태어나신 경우, 윤달이라고 작성해주세요.")
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
    print("\n질문을 입력하세요 (종료하려면 'quit' 입력):")
    chat_history = []
    thread_id = random_uuid()
    while True:
        user_input = input("\n질문: ").strip()
        if user_input.lower() in ['quit', 'exit', '종료']:
            print("시스템을 종료합니다.")
            break
        if not user_input:
            continue
        chat_history.append(HumanMessage(content=user_input))
        try:
            print("\n분석을 시작합니다...")
            result = run_saju_analysis(chat_history, thread_id=thread_id, use_stream=True)
            print("\n분석 완료!")
            # 답변 메시지 추출 및 기록 (AIMessage로 저장)
            if hasattr(result, '__iter__') and not isinstance(result, str):
                # stream_graph의 경우 generator이므로, 마지막 메시지 추출
                last_ai_msg = None
                for msg in result:
                    if hasattr(msg, 'content'):
                        last_ai_msg = msg
                if last_ai_msg:
                    chat_history.append(AIMessage(content=last_ai_msg.content))
            # invoke_graph의 경우는 별도 처리 필요
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main() 