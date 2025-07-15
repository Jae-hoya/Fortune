# 병렬구조

import functools
import operator
from datetime import datetime
from typing import Sequence, Annotated, Literal, Optional, Dict, List, Any
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, load_prompt
from langchain_core.runnables import RunnableConfig
from langchain_core.tools.retriever import create_retriever_tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_teddynote.messages import stream_graph
from langchain_teddynote.tools.tavily import TavilySearch
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
import uuid
import asyncio
import sys
import json

# --- 로컬 모듈 임포트 ---
from manse_8 import calculate_saju_tool
from pdf_retriever_saju import pdf_rag_chain, compression_retriever
from query_expansion_agent import get_query_expansion_node, get_query_expansion_agent

# --- 환경 변수 로드 ---
load_dotenv()

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# --- LLM 및 기본 설정 ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- 1. Retriever 및 관련 도구 설정 ---
pdf_retriever = compression_retriever()
pdf_chain = pdf_rag_chain()

# retriever_tool = create_retriever_tool(
#     pdf_retriever,
#     "pdf_retriever",
#     "A tool for searching information related to Saju (Four Pillars of Destiny)",
#     document_prompt=PromptTemplate.from_template(
#         "<document><context>{page_content}</context><metadata><source>{source}</source><page>{page}</page></metadata></document>"
#     ),
# )
retriever_tool = create_retriever_tool(
    pdf_retriever,
    "pdf_retriever",
    "A tool for searching information related to Saju (Four Pillars of Destiny)",
    document_prompt=PromptTemplate.from_template(
        "<document><context>{page_content}</context><metadata><source>{source}</source></metadata></document>"
    ),
)
retriever_tools = [retriever_tool]

# --- 2. Agent 생성 ---

# Manse Tool Agent
manse_tools = [calculate_saju_tool]
manse_tool_agent = create_react_agent(llm, manse_tools)

# Retriever Tool Agent
base_prompt = load_prompt("prompt/saju-rag-promt_2.yaml")
saju_prompt = ChatPromptTemplate.from_messages([
    ("system", f"Today is {now}"),
    ("system", base_prompt.template),
    MessagesPlaceholder("messages"),
])

retriever_tool_agent = create_react_agent(llm, retriever_tools, prompt=saju_prompt).with_config({"tags": ["final_answer_agent"]})

# Web Search Tool Agent
tavily_tool = TavilySearch(max_results=2, include_domains=["namu.wiki", "wikipedia.org"])
duck_tool = DuckDuckGoSearchResults(max_results=2)
web_search_tools = [tavily_tool, duck_tool]
web_search_prompt = "사주 또는 사주 오행의 개념적 질문이나, 일상 질문이 들어오면, web search를 통해 답합니다."
web_tool_agent = create_react_agent(llm, tools=web_search_tools, prompt=web_search_prompt).with_config({"tags": ["final_answer_agent"]})

# General QA Tool Agent
@tool
def general_qa_tool(query: str) -> str:
    """
    일반적인 질문이나 상식적인 내용에 대해 답변합니다. 사주와 관련 없는 모든 질문에 사용할 수 있습니다.
    """
    google_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    return google_llm.invoke(query)

general_qa_tools = [general_qa_tool]
general_qa_prompt = "일반적인 질문이나 상식적인 내용에 대해 답변합니다."
general_qa_agent = create_react_agent(llm, tools=general_qa_tools, prompt=general_qa_prompt).with_config({"tags": ["final_answer_agent"]})

# Query Expansion Agent
query_expansion_node = get_query_expansion_node()

# --- 사주 정보 구조 정의 ---
class BirthInfo(TypedDict):
    year: int
    month: int
    day: int
    hour: int
    minute: int
    is_male: bool
    is_leap_month: bool

class SajuResult(TypedDict):
    year_pillar: str
    month_pillar: str
    day_pillar: str
    hour_pillar: str
    day_master: str
    age: int
    korean_age: int
    element_strength: Optional[Dict[str, int]]
    ten_gods: Optional[Dict[str, List[str]]]
    great_fortunes: Optional[List[Dict[str, Any]]]
    yearly_fortunes: Optional[List[Dict[str, Any]]]
    useful_gods: Optional[List[str]]
    taboo_gods: Optional[List[str]]
    saju_analysis: Optional[str]

# --- 3. Agent 상태 및 노드 정의 ---

class AgentState(TypedDict):
    # 기본 LangGraph 요구사항
    question: str
    messages: Annotated[List[BaseMessage], operator.add]
    next: str
    final_answer: Optional[str]
    # 세션 관리
    session_id: str
    session_start_time: str
    current_time: str
    # 사주 시스템 핵심 정보
    birth_info: Optional[BirthInfo]
    saju_result: Optional[SajuResult]
    query_type: str  # "saju", "tarot", "general" 등
    # 에이전트 간 데이터 공유
    retrieved_docs: List[Dict[str, Any]]
    web_search_results: List[Dict[str, Any]]

# agent_node 함수 리팩토링: state 전체를 반환하도록 수정

def agent_node(state, agent, name):
    agent_response = agent.invoke(state)
    # 기존 메시지에 새 메시지 추가
    state["messages"] = state.get("messages", []) + [HumanMessage(content=agent_response["messages"][-1].content, name=name)]
    return state

def parse_birth_info_with_llm(user_input, llm):
    prompt = f"""
아래 문장에서 출생 정보를 추출해서 JSON 형태로 반환하세요.
필드: year, month, day, hour, minute, is_male, is_leap_month
예시 입력: "1996년 12월 13일 남자, 10시 30분 출생"
예시 출력: {{"year": 1996, "month": 12, "day": 13, "hour": 10, "minute": 30, "is_male": true, "is_leap_month": false}}

입력: {user_input}
"""
    result = llm.invoke(prompt)
    # LLM이 반환한 JSON 문자열을 파싱
    try:
        birth_info = json.loads(result.content)
        return birth_info
    except Exception as e:
        print("파싱 오류:", e)
        return None

from manse_8 import calculate_saju_tool

def manse_agent_node(state):
    user_input = state["question"]
    birth_info = parse_birth_info_with_llm(user_input, llm)
    state["birth_info"] = birth_info
    saju_result = calculate_saju_tool(birth_info)
    state["saju_result"] = saju_result

    # LLM에게 자연어 해석 요청
    prompt = f"""
    아래는 사용자의 사주 정보와 계산 결과입니다.
    - 입력: {user_input}
    - 사주 계산 결과: {json.dumps(saju_result, ensure_ascii=False, indent=2)}
    위 정보를 바탕으로, 사용자가 이해하기 쉽게 사주풀이 결과를 자연어로 설명해 주세요.
    """
    llm_response = llm.invoke(prompt)
    state["messages"].append(HumanMessage(content=llm_response.content, name="ManseLLM"))
    return state

manse_tool_agent_node = functools.partial(agent_node, agent=manse_tool_agent, name="ManseTool")
retriever_tool_agent_node = functools.partial(agent_node, agent=retriever_tool_agent, name="RetrieverTool")
web_tool_agent_node = functools.partial(agent_node, agent=web_tool_agent, name="WebTool")
general_qa_agent_node = functools.partial(agent_node, agent=general_qa_agent, name="GeneralQA")

# --- search 노드: retrievertool과 web_search tool을 내부에서 분기 ---
def search_agent_node(state):
    user_input = state.get("question") or (state["messages"][0].content if state.get("messages") else "")
    # 간단한 키워드 분기(추후 LLM 분기로 개선 가능)
    if any(k in user_input for k in ["자료", "문서", "pdf", "검색", "출처"]):
        return retriever_tool_agent_node(state)
    else:
        return web_tool_agent_node(state)

# --- 4. Supervisor Agent 정의 ---
members = ["search", "manse", "general_qa"]
options_for_next = ["FINISH"] + members

class RouteResponse(BaseModel):
    next: Literal[*options_for_next]

supervisor_system_prompt = (
    f"오늘 날짜는 {{now}}입니다.\n"
    "당신은 다음과 같은 전문 에이전트들을 조율하는 Supervisor입니다: {members}.\n"
    "각 에이전트의 역할은 다음과 같습니다:\n"
    "- search: 웹검색 및 문서/DB 검색(내부에서 자동 분기)\n"
    "- manse: 생년월일/시간 등 사주풀이, 운세 해석, 상세 분석 담당\n"
    "- general_qa: 사주와 무관한 일반 상식, 과학, 프로그래밍 등 모든 질문에 답변\n"
    "입력에 따라 가장 적합한 에이전트로 라우팅하세요."
)

supervisor_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", supervisor_system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "위 대화를 참고하여, 다음 중 누가 다음 행동을 해야 하는지 선택하세요: {options}"),
    ]
)

def supervisor_agent(state):
    supervisor_chain = (
        supervisor_prompt.partial(options=str(options_for_next), members=", ".join(members), now=now)
        | llm.with_structured_output(RouteResponse)
    )
    route_response = supervisor_chain.invoke(state)
    return {"next": route_response.next}

@tool
def saju_chat_tool(state):
    """
    사주에 대한 대화형 챗봇 답변을 생성합니다. 간단하게 답변해주세요.
    """
    # state에서 messages를 안전하게 가져오기
    messages = state.get("messages", [])
    response = "(사주 대화 결과: 추가 질문에 대한 답변)"
    # 기존 메시지 리스트에 새 메시지를 추가
    return {"messages": messages + [HumanMessage(content=response, name="SajuChat")]}

saju_chat_tools = [saju_chat_tool]
saju_chat_prompt = "사주 대화 결과를 참고하여 추가 질문에 대한 답변을 생성합니다. 예를들어, 내일의 운세, 내년의 운세등을 대화형식으로 답변합니다."
saju_chat_agent = create_react_agent(llm, saju_chat_tools, prompt=saju_chat_prompt)
saju_chat_node = functools.partial(agent_node, agent=saju_chat_agent, name="SajuChat")


# LLM 분기 프롬프트
branch_prompt = ChatPromptTemplate.from_messages([
    ("system", "다음 중 어디로 가야 할지 판단하세요: 'saju_chat' 또는 'retriever'.\n"
               "사주풀이에 추가 설명이 필요하면 'saju_chat', 생년월일을 입력받으면 'retriever'를 선택하세요.\n"
               "반드시 하나만 골라주세요.\n"),
    ("user", "질문: {user_input}\n"
             "manse 결과: {manse_result}\n"
             "")
])

def llm_branch_decision(state):
    user_input = ""
    manse_result = ""
    # state에서 user_input, manse_result 추출 (구조에 따라 조정)
    if state.get("messages"):
        # messages가 리스트인지 문자열인지 확인
        if isinstance(state["messages"], list) and len(state["messages"]) > 0:
            # 첫 번째 메시지가 HumanMessage 객체인 경우
            if hasattr(state["messages"][0], 'content'):
                user_input = state["messages"][0].content
            else:
                user_input = str(state["messages"][0])
            
            # 마지막 메시지가 AIMessage 객체인 경우
            if hasattr(state["messages"][-1], 'content'):
                manse_result = state["messages"][-1].content
            else:
                manse_result = str(state["messages"][-1])
        else:
            # messages가 문자열인 경우
            user_input = str(state["messages"])
            manse_result = ""
    
    prompt = branch_prompt.format(user_input=user_input, manse_result=manse_result)
    # LLM 호출
    result = llm.invoke(prompt)
    # 결과에서 'saju_chat' 또는 'retriever'만 추출
    if "retriever" in result.content.lower():
        return "retriever"
    return "saju_chat"

# --- 5. LangGraph 그래프 구성 ---
def create_workflow_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("search", search_agent_node)
    workflow.add_node("manse", manse_tool_agent_node)
    workflow.add_node("general_qa", general_qa_agent_node)
    workflow.add_node("Supervisor", supervisor_agent)

    for member in members:
        workflow.add_edge(member, "Supervisor")

    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END

    def get_next(state):
        return state["next"]

    workflow.add_conditional_edges("Supervisor", get_next, conditional_map)
    workflow.add_edge(START, "Supervisor")

    return workflow.compile(checkpointer=MemorySaver())

from langchain_teddynote.messages import random_uuid, stream_graph, invoke_graph, stream_graph_v2
from langchain_core.messages import AIMessage

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
    print("사주 에이전틱 RAG 시스템 (병렬 구조 버전)을 시작합니다... ")
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