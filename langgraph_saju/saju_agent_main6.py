# 1. 표준/외부 라이브러리 임포트
import functools
import operator
from datetime import datetime
from typing import Sequence, Annotated, Literal, Optional, Dict, List, Any
from typing_extensions import TypedDict
import uuid
import asyncio
import sys
import json
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, load_prompt
from langchain_core.runnables import RunnableConfig
from langchain_core.tools.retriever import create_retriever_tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_teddynote.messages import stream_graph, random_uuid, invoke_graph
from langchain_teddynote.tools.tavily import TavilySearch
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

# 2. 환경 변수 및 상수
load_dotenv()
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 3. 로컬 모듈 임포트
from manse_8 import calculate_saju_tool
from pdf_retriever_saju import pdf_rag_chain, compression_retriever
from query_expansion_agent import get_query_expansion_node, get_query_expansion_agent

# 4. 데이터 구조 정의
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

class AgentState(TypedDict):
    question: str
    messages: Annotated[List[BaseMessage], operator.add]
    next: str
    final_answer: Optional[str]
    session_id: str
    session_start_time: str
    current_time: str
    birth_info: Optional[BirthInfo]
    saju_result: Optional[SajuResult]
    query_type: str
    retrieved_docs: List[Dict[str, Any]]
    web_search_results: List[Dict[str, Any]]

# 5. 도구 및 에이전트 생성
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
pdf_retriever = compression_retriever()
pdf_chain = pdf_rag_chain()
retriever_tool = create_retriever_tool(
    pdf_retriever,
    "pdf_retriever",
    "A tool for searching information related to Saju (Four Pillars of Destiny)",
    document_prompt=PromptTemplate.from_template(
        "<document><context>{page_content}</context><metadata><source>{source}</source></metadata></document>"
    ),
)

# manse tool
manse_tools = [calculate_saju_tool]
manse_tool_prompt = """
사주 계산 결과를 바탕으로, 사용자에게 친절하고 이해하기 쉬운 자연어로 사주풀이 결과를 설명해 주세요. 
십신 분석에 대한 용어 및 특징 풀이가 필요합니다.
설명에는 반드시 다음 항목들을 포함해 주세요: 대운, 세운, 건강운, 재물운, 금전운, 직업운, 성공운.
사용자가 특정 항목만 물어보거나, 추가적인 운을 질문한 경우에는 해당 항목만 중심적으로 답변해 주세요.
각 항목은 구체적인 근거(오행, 십성, 용신, 기운의 균형 등의 설명)와 함께, 긍정적이고 조언을 담은 존댓말로 전달해 주세요.
오행에서, 부족한 부분을 채우기 위해서 어떤 것을 해야 하는지도 안내해 주세요.
예언이나 단정적인 표현 대신, 경향·조언·주의점 중심으로 안내해 주세요. 
불안감을 줄 수 있는 부정적인 표현("불행하다", "위험하다", "나쁘다" 등)은 사용하지 마시고, 사용자가 삶에 도움이 될 수 있는 방향으로 해석해 주세요.
항목별로 비슷한 문장이 반복되지 않도록 주의해 주시고, 구체적으로 설명해주세요.
답변 마지막에는 "더 궁금하신 점이 있으시면 언제든 질문해 주세요."와 같은 마무리 멘트를 넣어주세요.
"""

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
manse_tool_agent = create_react_agent(llm, manse_tools, prompt=manse_tool_prompt).with_config({"tags": ["final_answer_agent"]})

# retriever tool
retriever_tools = [retriever_tool]
base_prompt = load_prompt("prompt/saju-rag-promt_2.yaml")
saju_prompt = ChatPromptTemplate.from_messages([
    ("system", f"Today is {now}"),
    ("system", base_prompt.template),
    MessagesPlaceholder("messages"),
])
retriever_tool_agent = create_react_agent(llm, retriever_tools, prompt=saju_prompt).with_config({"tags": ["final_answer_agent"]})

# web search tool
tavily_tool = TavilySearch(max_results=2, include_domains=["namu.wiki", "wikipedia.org"])
duck_tool = DuckDuckGoSearchResults(max_results=2)
web_search_tools = [tavily_tool, duck_tool]
web_search_prompt = "사주 또는 사주 오행의 개념적 질문이나, 일상 질문이 들어오면, web search를 통해 답합니다."
web_tool_agent = create_react_agent(llm, tools=web_search_tools, prompt=web_search_prompt).with_config({"tags": ["final_answer_agent"]})

@tool
def general_qa_tool(query: str) -> str:
    """
    일반적인 질문이나 상식적인 내용에 대해 답변합니다. 사주와 관련 없는 모든 질문에 사용할 수 있습니다.
    """
    google_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    return google_llm.invoke(query)

general_qa_tools = [general_qa_tool]
general_qa_prompt = """
일반적인 질문을 나의 사주와 관련하여 답변합니다.
만약 state에 birth_info 또는 saju_result가 포함되어 있다면, 그 정보를 참고해서 답변에 반영하세요.
birth_info: {birth_info}
saju_result: {saju_result}
"""
general_qa_agent = create_react_agent(llm, tools=general_qa_tools, prompt=general_qa_prompt).with_config({"tags": ["final_answer_agent"]})
query_expansion_node = get_query_expansion_node()

# 6. 핵심 함수(노드, 파싱, 라우팅 등)
def agent_node(state, agent, name):
    agent_response = agent.invoke(state)
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
    try:
        birth_info = json.loads(result.content)
        return birth_info
    except Exception as e:
        print("파싱 오류:", e)
        return None

def manse_agent_node(state):
    user_input = state["question"]
    birth_info = parse_birth_info_with_llm(user_input, llm)
    state["birth_info"] = birth_info
    saju_result = calculate_saju_tool(birth_info)
    state["saju_result"] = saju_result
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

def search_agent_node(state):
    user_input = state.get("question") or (state["messages"][0].content if state.get("messages") else "")
    if any(k in user_input for k in ["자료", "문서", "pdf", "검색", "출처"]):
        return retriever_tool_agent_node(state)
    else:
        return web_tool_agent_node(state)

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
    "무한루프에 빠지지 않도록 주의해주세요."
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

# 7. LangGraph 워크플로우 생성
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

# 8. 실행(main) 함수
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
            if hasattr(result, '__iter__') and not isinstance(result, str):
                last_ai_msg = None
                for msg in result:
                    if hasattr(msg, 'content'):
                        last_ai_msg = msg
                if last_ai_msg:
                    chat_history.append(AIMessage(content=last_ai_msg.content))
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main() 

# 메모리 저장할때, 요약해서 저장. 
# 최종 출력전에 요약을 해서, state에 저장을 하고 출력은 하지 않음.

# 12시 30분