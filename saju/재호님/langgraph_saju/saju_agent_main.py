import functools
import operator
from datetime import datetime
from typing import Sequence, Annotated, Literal
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

# --- 로컬 모듈 임포트 ---
# (실제 환경에 맞게 경로를 확인하거나 수정해야 할 수 있습니다.)
from manse_7 import calculate_saju_tool
from pdf_retriever_saju import pdf_rag_chain, compression_retriever
from query_expansion_agent import get_query_expansion_node, get_query_expansion_agent

# --- 환경 변수 로드 ---
load_dotenv()

# --- LLM 및 기본 설정 ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- 1. Retriever 및 관련 도구 설정 ---
pdf_retriever = compression_retriever()
pdf_chain = pdf_rag_chain()

retriever_tool = create_retriever_tool(
    pdf_retriever,
    "pdf_retriever",
    "A tool for searching information related to Saju (Four Pillars of Destiny)",
    document_prompt=PromptTemplate.from_template(
        "<document><context>{page_content}</context><metadata><source>{source}</source><page>{page}</page></metadata></document>"
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
    ("system", base_prompt.template),
    MessagesPlaceholder("messages"),
])
retriever_tool_agent = create_react_agent(llm, retriever_tools, prompt=saju_prompt).with_config({"tags": ["final_answer_agent"]})

# Web Search Tool Agent
tavily_tool = TavilySearch(max_results=5, include_domains=["namu.wiki", "wikipedia.org"])
duck_tool = DuckDuckGoSearchResults(max_results=5)
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

# --- 3. Agent 상태 및 노드 정의 ---

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

def agent_node(state, agent, name):
    agent_response = agent.invoke(state)
    return {"messages": [HumanMessage(content=agent_response["messages"][-1].content, name=name)]}

manse_tool_agent_node = functools.partial(agent_node, agent=manse_tool_agent, name="ManseTool")
retriever_tool_agent_node = functools.partial(agent_node, agent=retriever_tool_agent, name="RetrieverTool")
web_tool_agent_node = functools.partial(agent_node, agent=web_tool_agent, name="WebTool")
general_qa_agent_node = functools.partial(agent_node, agent=general_qa_agent, name="GeneralQA")

# --- 4. Supervisor Agent 정의 ---
members = ["SajuExpert", "WebTool", "GeneralQA"]
options_for_next = ["FINISH"] + members

class RouteResponse(BaseModel):
    next: Literal[*options_for_next]

supervisor_system_prompt = (
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

supervisor_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", supervisor_system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}"),
    ]
)

def supervisor_agent(state):
    supervisor_chain = (
        supervisor_prompt.partial(options=str(options_for_next), members=", ".join(members))
        | llm.with_structured_output(RouteResponse)
    )
    route_response = supervisor_chain.invoke(state)
    return {"next": route_response.next}

# --- 5. LangGraph 그래프 구성 ---

# SajuExpert Sub-graph 생성
saju_expert_workflow = StateGraph(AgentState)
saju_expert_workflow.add_node("manse", manse_tool_agent_node)
saju_expert_workflow.add_node("retriever", retriever_tool_agent_node)
saju_expert_workflow.add_edge(START, "manse")
saju_expert_workflow.add_edge("manse", "retriever")
saju_expert_workflow.add_edge("retriever", END)
saju_expert_graph = saju_expert_workflow.compile()

# 메인 그래프 생성
workflow = StateGraph(AgentState)
workflow.add_node("SajuExpert", saju_expert_graph)
workflow.add_node("QueryExpansion", query_expansion_node)
workflow.add_node("WebTool", web_tool_agent_node)
workflow.add_node("GeneralQA", general_qa_agent_node)
workflow.add_node("Supervisor", supervisor_agent)

for member in members:
    workflow.add_edge(member, "Supervisor")

conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END

def get_next(state):
    return state["next"]

workflow.add_conditional_edges("Supervisor", get_next, conditional_map)
workflow.add_edge(START, "QueryExpansion")
workflow.add_edge("QueryExpansion", "Supervisor")

graph = workflow.compile(checkpointer=MemorySaver())

# --- 6. 스크립트 실행 (스트리밍 로직 수정) ---
async def main():
    """메인 실행 함수"""
    thread_id = str(uuid.uuid4())
    config = RunnableConfig(recursion_limit=10, configurable={"thread_id": thread_id})
    print(f"사주 상담을 시작합니다. (세션 ID: {thread_id})")
    print("종료하시려면 'exit'를 입력하세요.")

    while True:
        user_input = await asyncio.to_thread(input, "질문: ")
        if user_input.lower() == 'exit':
            print("상담을 종료합니다.")
            break

        inputs = {"messages": [HumanMessage(content=user_input)]}
        
        print("\n답변:")
        
        async for event in graph.astream_events(inputs, config, version="v1"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                # 태그가 'final_answer_agent'인 에이전트의 출력만 스트리밍합니다.
                if "final_answer_agent" in event.get("tags", []):
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        sys.stdout.write(chunk.content)
                        sys.stdout.flush()

        print("\n" + "-" * 50)


if __name__ == "__main__":
    # 윈도우에서 asyncio.run() 사용 시 발생하는 이벤트 루프 에러 방지
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main()) 