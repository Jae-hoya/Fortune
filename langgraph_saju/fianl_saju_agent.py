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
사주(四柱) 계산 결과를 바탕으로, 사용자에게 친절하고 이해하기 쉬운 자연어로 사주풀이 결과를 설명해 주세요.
답변은 다음의 **항목을 모두 포함**하거나, 사용자가 특정 항목만 질문한 경우에는 해당 항목을 중심적으로 구체적으로 안내해 주세요.
사주 계산 결과를 바탕으로, 사용자에게 친절하고 이해하기 쉬운 자연어로 사주풀이 결과를 설명해 주세요.

**답변 맨 앞에, 아래와 같이 ‘사주 전체 요약’(3~5줄)을 먼저 제시해 주세요.**
- 사주 전반의 흐름, 가장 큰 특징, 기운의 방향성, 전체적인 조언 등을 종합적으로 안내해 주세요.
- 중요한 변화 시기, 주목할 만한 운, 가장 강한/부족한 에너지 등도 요약에 포함해 주세요.

이후 다음의 항목을 모두 포함하거나, 사용자가 특정 항목만 질문한 경우 해당 항목만 집중적으로 안내해 주세요.

1. 십신 분석 (비견, 겁재, 식신, 상관, 편재, 정재, 편관, 정관, 편인, 정인 등)
2. 오행 분석 (목, 화, 토, 금, 수)
3. 오행 보완법 (실생활에서 적용할 구체적 방법)
4. 대운
5. 세운
6. 건강운
7. 재물운
8. 금전운
9. 직업운
10. 성공운
11. 애정운 (연애, 결혼, 인연, 대인관계)
12. 학업운/시험운
13. 가족운/자녀운
14. 이동운/변화운
15. 사회운/인복/대인관계
16. 사업운/창업운
17. 명예운/승진운
18. 기타 특수운(필요 시: 소송, 법률, 여행, 복권, 투자 등)

- 상황에 따라서 궁합운을 추가해주세요

**답변 마지막에는, 전체 내용을 간단히 정리하거나 종합적 조언(1~3줄)과 마무리 멘트(“더 궁금하신 점이 있으시면 언제든 질문해 주세요.” 등)를 꼭 넣어주세요.**

- **각 항목은 반드시 구체적인 근거(오행, 십신, 용신, 기운의 균형 등)를 들어 설명하고**, 조언을 담은 존댓말로 전달해 주세요.
- **긍정적이고 조언을 담은 존댓말**로 전달해 주세요. 하지만 주의할점이 있다면 그 내용또한 전달해 주세요.
- 예언이나 단정적인 표현 대신, 경향·조언·주의점 중심으로 안내해 주세요.
- 불안감을 줄 수 있는 부정적 표현("불행하다", "위험하다", "나쁘다" 등)은 사용하지 마세요.
- 항목별로 비슷한 문장이 반복되지 않도록 주의하고, 각 항목마다 어휘와 문장 구조를 다양하게 사용해 주세요.

---

**[예시]**

**사주 전체 요약**  
올해는 새로운 기운이 강하게 들어오는 시기입니다. 대인관계와 직업적 기회가 풍부하며, 금전과 애정운도 전반적으로 긍정적인 흐름을 보입니다. 다만, 건강과 감정 관리에는 꾸준한 관심이 필요합니다.

**십신 분석**  
겁재(2개): 경쟁심이 강하며, 대인관계에서 주도적입니다. 가끔은 융통성이 필요할 수 있습니다.  
정관(1개): 책임감이 있고 규칙을 중시하지만, 때로는 융통성을 보완하면 더 좋습니다.

**오행 분석**  
목(木)이 강하고 수(水)가 부족합니다. 목이 강해 추진력과 성장성이 뛰어나지만, 감정 조절이나 유연함이 부족할 수 있습니다.

**오행 보완법**  
수(水)가 부족하다면 파란색 계열 옷, 물과 관련된 활동(수영, 산책), 해조류·생선 등 수의 기운을 돋우는 음식을 추천합니다.

**대운**  
2025~2034년(38~47세) 대운에는 식신과 편관의 기운이 두드러집니다.  
이 시기에는 새로운 프로젝트나 사업을 시작하면 성과를 내기 쉽고, 직장에서 중요한 역할을 맡게 될 가능성이 높습니다.  
다만, 이 시기에는 경쟁이 심해질 수 있으니, 감정 조절과 꾸준한 자기계발이 중요합니다.

2035~2044년(48~57세) 대운에는 재물운과 인복이 강해집니다.  
금전적 기회가 많아지는 한편, 가족·친구와의 관계도 깊어질 수 있습니다.  
그러나 투자와 소비의 균형에 신경 써야 안정적인 성과를 유지할 수 있습니다.

**세운**  
2024년: 정관이 강해져서 직장 내 평가나 승진 운이 들어옵니다.  
새로운 업무 기회를 잘 활용하면 성장의 발판이 마련될 수 있습니다.

2025년: 재물운이 상승하여 뜻밖의 수입이나 투자 기회가 생길 수 있습니다.  
하지만 무리한 지출이나 충동구매에는 주의해야 합니다.

2026년: 건강운이 약간 약해질 수 있어 꾸준한 운동과 규칙적인 생활습관이 필요합니다.  
특히 소화기·순환기 관리에 신경을 쓰면 도움이 됩니다.

**애정운**  
올해는 새로운 인연이 생기기 쉬운 시기입니다. 열린 마음으로 주변 사람들과 소통하면 좋은 결과가 있을 것입니다.

**종합 해석**  
전반적으로 긍정적 변화가 기대되는 시기이며, 적극적으로 도전하신다면 좋은 결과를 얻을 수 있습니다. 궁금하신 점이 있으시면 언제든 질문해 주세요.

...
---
"""



llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
manse_tool_agent = create_react_agent(llm, manse_tools, prompt=manse_tool_prompt).with_config({"tags": ["final_answer_agent"]})

# retriever tool
retriever_tools = [retriever_tool]
base_prompt = load_prompt("prompt/saju-rag-promt_2.yaml")
saju_prompt = ChatPromptTemplate.from_messages([
    ("system", f"Today is {now}, 사주에 대해서 자세한 설명이 필요하면 retriever를 사용해 답합니다."),
    ("system", base_prompt.template),
    MessagesPlaceholder("messages"),
])
retriever_tool_agent = create_react_agent(llm, retriever_tools, prompt=saju_prompt).with_config({"tags": ["final_answer_agent"]})

# web search tool
tavily_tool = TavilySearch(max_results=2, include_domains=["namu.wiki", "wikipedia.org"])
duck_tool = DuckDuckGoSearchResults(max_results=2)
web_search_tools = [tavily_tool, duck_tool]
web_search_prompt = "십신분석의 개념, 사주개념, 또는 사주 오행의 개념적 질문이 들어오면, web search를 통해 답합니다."
web_tool_agent = create_react_agent(llm, tools=web_search_tools, prompt=web_search_prompt).with_config({"tags": ["final_answer_agent"]})

@tool
def general_qa_tool(state):
    """
    state에서 query, birth_info, saju_result를 추출해 프롬프트를 생성합니다.
    """
    google_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    messages = state.get("messages", [])
    # 메시지 없을 때 대비
    query = messages[-1].content if messages else ""
    birth_info = state.get("birth_info")
    saju_result = state.get("saju_result")
    # 전체 메시지 content를 \n으로 연결
    context = "\n".join([m.content for m in messages])

    prompt = f"""아래는 사용자의 질문과 사주 정보입니다.
        질문: {query}
        사주 정보: {birth_info}
        사주 해석: {saju_result}
        대화 기록: {context}

        위 정보를 반드시 참고해서, 사주 특성을 녹여서 친절하고 존댓말로 답변해 주세요.
        맥락에 맞는 답변만 해야합니다.
        """
    return google_llm.invoke(prompt)


general_qa_tools = [general_qa_tool]

general_qa_prompt = """
아래는 사용자가 일반적인 질문을 한 경우입니다.

#####
'- 반드시 지금까지의 모든 대화 기록(messages), 즉 이전 질문과 답변을 충분히 참고해서, 사용자의 맥락에 어긋나지 않는 답변을 제공해 주세요.'


- 후속 질문이 들어오면, 앞서 안내했던 사주 해석, 조언, 분석 결과와 중복되지 않도록 새로운 정보나 추가 설명, 보충 조언을 중심으로 답변해 주세요.
- 질문 흐름과 맥락을 반영하여, 자연스럽게 이어지는 상담/설명/추천이 되도록 해 주세요.

사주 정보(birth_info, saju_result 등)가 포함되어 있다면,
답변에 자연스럽게 그 정보를 녹여서, **실제 대화하듯, 친근하고 진심 어린 존댓말**로 상담해 주세요.

- 답변은 핵심만 간결하게, 한 번에 3~5문장 이내로 해주세요.
- 질문이 사주와 무관하다면, 우선 일반적인 정보/상식/지식 답변을 먼저 제공해 주세요.
- 답변 마지막에는 "참고로, 사주 정보에 따르면 ~~~도 도움/영향이 될 수 있습니다."처럼 사주와 연결된 맞춤 조언을 추가하거나,
- 또는 "추가로 궁금하신 점 있으시면 언제든 물어봐 주세요!" 등 친근한 마무리 멘트도 넣어주세요.
- 사주 정보가 없는 경우에는, 일반 QA 스타일로 답변해 주세요.

만약 사용자의 사주 정보(십신, 오행, 격국 등 명식 데이터)가 이미 파악된 상태라면,
반드시 해당 사주 정보(십신, 오행 등)를 반영하여, 사용자의 명식 특성(오행의 강/약, 십신 중 두드러지는 기운 등)에 맞춘
맞춤형 조언, 추천, 상담을 제공하세요.

답변 중 십신, 오행 등 사주 전문 용어(예: 편재, 식신, 화(火) 등)는 그대로 사용하셔도 되나,
반드시 그 뜻을 쉽고 명확하게 설명해 주세요. 이해를 돕기 위해, 필요하다면 일상적인 예시나 비유도 활용해 주세요.
처음 듣는 용어라도 부담 없이 이해할 수 있게, 항상 친절하고 쉽게 안내해 주세요.


예시:
- "당신은 화(火) 기운이 강해서 활동적이고 에너지가 넘치는 스타일입니다. 오늘은 가벼운 운동이나 매운 음식을 추천드려요."
- "식신이 두드러져서 창의적이고 새로운 것을 배우는 데 강점이 있습니다. 공부법 중에서도 다양한 시도를 해보는 방식이 잘 맞을 수 있습니다."
- "편재는 재물과 기회를 잡는 힘을 의미합니다. 오늘은 새로운 사람을 만나보는 것도 좋은 운이 있으니, 모임에 참여해 보세요."

아래는 예시입니다.

---
1. 최근 운세 변화 질문 예시
Q: 최근 운세가 달라진 것 같은데, 올해는 어떤 변화가 있을까요?
A:
올해는 새로운 기회가 많아지는 시기입니다. 적극적으로 도전하면 좋은 결과를 기대할 수 있습니다.
사주에서 금(金) 기운이 두드러지는 시기라, 실용적인 계획과 꾸준한 준비가 특히 도움이 됩니다.
혹시 구체적으로 궁금한 부분이 있으시면 언제든 말씀해 주세요!

2. 성격(십신/오행 기반) 질문 예시
Q: 저는 어떤 성격인가요?
A:
기본적으로 책임감이 강하고, 남을 배려하는 면이 두드러집니다.
사주에서 식신이 두드러져, 창의적이면서도 실용적인 성향이 많으세요.
참고로, 새로운 아이디어를 실제로 실행에 옮기는 능력도 뛰어납니다.
혹시 더 궁금한 점 있으면 언제든 말씀해 주세요!

3. 생활 조언 질문 (연애/직장 등)
Q: 연애운이 궁금합니다.
A:
올해는 인간관계에서 긍정적인 변화가 기대됩니다.
사주에서 수(水) 기운이 강해져, 상대방과의 대화와 감정 교류가 평소보다 더 중요할 수 있습니다.
진솔하게 마음을 표현해 보시면 좋은 만남으로 이어질 가능성이 높아요.
추가로 궁금하신 점 있으시면 언제든 말씀해 주세요!

Q: 직장에서 좋은 평가를 받으려면 어떻게 해야 할까요?
A:
책임감을 가지고 꾸준히 노력하시는 게 중요합니다.
사주에 따르면 정관이 강해, 원칙을 잘 지키고 신뢰를 주는 사람이십니다.
동료들과의 협업을 의식적으로 신경 쓰면 더 좋은 결과로 이어질 수 있습니다.
더 궁금한 점 있으시면 언제든 질문해 주세요!

Q: 다이어트에 좋은 음식이 뭔가요?
A:  
다이어트에는 단백질이 풍부한 음식(닭가슴살, 두부, 생선), 신선한 채소, 견과류, 충분한 수분 섭취가 도움이 됩니다.  
참고로, 사주에서 토(土) 기운이 약하신 경우, 고구마나 콩류, 노란색·갈색 음식이 몸의 균형에 더 도움이 될 수 있습니다.  
추가로 궁금한 점 있으시면 언제든 질문해 주세요!
---
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

        만약 출생 정보가 명확하지 않거나 부족하면 null을 반환하세요.
        year, month, day는 필수이고, hour, minute, is_male, is_leap_month는 선택사항입니다.
        is_male은 true(남자), false(여자)로 설정하세요.
        is_leap_month는 윤달인 경우에만 true로 설정하세요.

        입력: {user_input}
        """
    result = llm.invoke(prompt)
    try:
        # JSON 문자열에서 불필요한 문자 제거
        content = result.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        birth_info = json.loads(content)
        
        # 필수 필드 확인
        if not birth_info or not all(key in birth_info and birth_info[key] is not None for key in ["year", "month", "day"]):
            return None
            
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

manse_tool_agent_node = functools.partial(agent_node, agent=manse_tool_agent, name="manse")
retriever_tool_agent_node = functools.partial(agent_node, agent=retriever_tool_agent, name="retriever")
web_tool_agent_node = functools.partial(agent_node, agent=web_tool_agent, name="web")
# general_qa_agent_node = functools.partial(agent_node, agent=general_qa_agent, name="general_qa")

#  general_qa agent 수정
def general_qa_agent_node(state):
    agent_response = general_qa_agent.invoke({
        "birth_info": state.get("birth_info"),
        "saju_result": state.get("saju_result"),
        "messages": state.get("messages", []),
    })
    state["messages"].append(
        HumanMessage(content=agent_response["messages"][-1].content, name="GeneralQA")
    )
    return state


def classify_search_llm(user_input, llm):
    prompt = """
    - 'retriever': 사주에 대한 자세한 설명이 필요하면 retriever(retriever_tool_agent_node)
    - 'web': 특별한 내부 언급이 없거나, 일반적/공개 정보/공식/인터넷/최신/정의/설명/이론/근거/출처 등은 web(web_tool_agent_node)
    - 'web': 십신분석의 개념, 사주개념, 또는 사주 오행의 개념적 질문이 들어오면, web(web_tool_agent_node)
    둘 중 가장 적합한 카테고리( retriever / web )만 답변하세요.
    

    질문: "{user_input}"
    정답:
    """.format(user_input=user_input)
    result = llm.invoke(prompt)
    
    return result.content.strip().lower()

def search_agent_node(state):
    user_input = state.get("question") or (state["messages"][0].content if state.get("messages") else "")
    category = classify_search_llm(user_input, llm)
    if category == "retriever":
        return retriever_tool_agent_node(state)
    else:
        return web_tool_agent_node(state)


members = ["search", "manse", "general_qa"]
options_for_next = ["FINISH"] + members

class RouteResponse(BaseModel):
    next: Literal[*options_for_next]

supervisor_system_prompt = (
    f"오늘 날짜는 {{now}}입니다.\n"
    "당신은 아래 전문 에이전트들을 조율하는 Supervisor입니다: {members}.\n"
    "입력(사용자 질문)에 따라, 가장 적합한 에이전트로 분기하세요.\n\n"
    "각 에이전트의 역할과 분기 기준은 아래와 같습니다:\n\n"

    "1. **search**: "
    "용어, 개념, 정의, 이론, 분류, 공식, 자료, 논문, 출처 등 **정보성 질문**에만 사용합니다.\n"
    "예시: '불속성이 뭐야?', '오행 각각의 의미', '정관 정의', '십신 종류', '사주에서 겁재는?', '오행 설명' 등.\n"
    "※ **운세 풀이/해석/미래/금전운/재물운 등 사주 해석, 개인 운세 질문은 절대 search로 분기하지 않습니다.**\n"
    "※ 사주에 대한 개념 설명이 필요하면 retriever, 십신/오행 등 사주 용어·이론 설명은 web 사용.\n"
    "※ search는 오직 정보성(정의·분석·이론·자료) 질문에서만 답변을 생성합니다.\n"
    "※ 일상, 고민, 메뉴, 추천, 잡담 등에는 절대 사용하지 않습니다.\n\n"

    "2. **manse**: "
    "생년월일 2개를 입력받고, 궁합운을 물어본다면 manse로 분기합니다."
    "두 사람의 운세를 비교하는 질문이라면 manse로 분기합니다."
    "**이미 state에 출생 정보(birth_info, saju_result 등)가 저장되어 있으면,  "
    "질문에 새로운 출생 정보가 명확하게 포함되지 않는 한 manse로 분기하지 마세요.**  "
    "(즉, 운세/해석/미래/궁합/시련 등 모든 질문은 기존 출생 정보가 있을 때 반드시 general_qa로 분기합니다."
    "특정 운세에 대해서 자세히 알려달라고 하면 다시 manse로 분기합니다."
    "※ **사주 개념·용어·이론 설명, 일상 메뉴, 잡담, 선택, 고민 등은 manse로 보내지 않습니다.**\n"
    "※ 용어/개념/정의/설명/이론 질문(예: '겁재가 뭐야?', '오행 설명', '십신 의미' 등)은 절대 manse에서 처리하지 않습니다.\n\n"

    "3. **general_qa**: "
    "일반 상식, 생활 정보, 건강, 공부, 영어, 주식, 투자, 고민 상담, 일상 메뉴, 잡담, 선택, 추천 등은 모두 general_qa가 담당합니다.\n"
    "특히, 생년월일이 없이 사주에 대한 질문또한 general_qa로 분기합니다.\n"
    "예시:"
    "1. 최근 운세 변화 질문 예시 \n"
    "2. 성격(십신/오행 기반) 질문 예시\n"
    "3. 생활 조언 질문 (연애/직장 등)\n"
    "※ 사주 풀이가 끝난 뒤, 후속 일상/상담/선택 등 대화는 manse에서 general_qa로 자연스럽게 이어지도록 해주세요.\n"
    "※ 운세/대운/궁합/변화 등 사주성 질문이라도, **사주 정보(생년월일 등)가 없으면 반드시 general_qa 또는 search로 분기**\n\n"
    
    "[분기 기준 예시]\n"
    "- '겁재가 뭐야?', '오행이 뭔가요?', '정관 정의 알려줘', '화속성이 뭐야?', '수기운 설명해줘' → **search**\n"
    "- '1995년생 3월 28일 11시 30분 남자', '1997년 9월 1일 11시 30분 여자' → **manse**\n"
    "- '오늘 뭐 먹지?', '공부법 알려줘', '영어회화 공부법', '기분전환 메뉴 추천' → **general_qa**\n"
    "- '올해 운세는?', '결혼 언제쯤?', '대운이 언제 바뀌어?', '궁합은 어때?' → **general_qa**\n\n"
    
)

supervisor_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """현재 사용자의 사주 정보:
            - birth_info: {birth_info}
            - saju_result: {saju_result}
            최근 맥락에 맞는 대화만 출력해야합니다.
            """
        ),
        ("system", supervisor_system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
              """
              위 입력 및 대화를 참고하여, 다음 중 누가 다음 행동을 해야 하는지 선택하세요: {options}            
              """
              ),
    ]
)

# 7. LangGraph 워크플로우 생성
def supervisor_agent(state):
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    user_input = state.get("question") or (state["messages"][0].content if state.get("messages") else "")
    
    # 1️⃣ birth_info 추출
    birth_info = parse_birth_info_with_llm(user_input, llm)
    state["birth_info"] = birth_info

    # 2️⃣ LLM에게 분기 예측 시도
    supervisor_chain = (
        supervisor_prompt.partial(
            options=str(options_for_next), 
            members=", ".join(members), 
            now=now,
            birth_info=state.get('birth_info', 'None'),
            saju_result=state.get('saju_result', 'None'),
        )
        | llm.with_structured_output(RouteResponse)
    )
    route_response = supervisor_chain.invoke(state)
    return {"next": route_response.next}


# 워크플로우 생성

def create_workflow_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("search", search_agent_node)
    workflow.add_node("manse", manse_tool_agent_node)
    workflow.add_node("general_qa", general_qa_agent_node)
    workflow.add_node("supervisor", supervisor_agent)
    
    # 각 노드에서 직접 END로 이동 (finish)
    workflow.add_edge("search", END)
    workflow.add_edge("manse", END)
    workflow.add_edge("general_qa", END)
    
    # supervisor에서 분기 결정
    conditional_map = {k: k for k in members}
    
    def get_next(state):
        return state["next"]
    workflow.add_conditional_edges("supervisor", get_next, conditional_map)
    
    # 시작점에서 supervisor로
    workflow.add_edge(START, "supervisor")
    
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
        "궁합운이 2개의 생년월일과 함께, 궁합운을 봐달라고 하세요!"
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

