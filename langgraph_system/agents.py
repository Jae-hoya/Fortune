"""
에이전트 생성 및 관리
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, load_prompt
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
from typing import Literal, Optional, Dict, Any
from datetime import datetime

# 단순화된 tools import (노트북 방식)
from tools import (
    calculate_saju_tool, 
    saju_retriever_tool, 
    web_tools, 
    general_qa_tool,
    manse_tools,
    retriever_tools,
    general_qa_tools
)

# 멤버 Agent 목록 정의 (notebook 구조에 맞게 변경)
members = ["SajuExpert", "WebTool", "GeneralQA"]

# Supervisor의 모든 행동 옵션 정의 (확장된 역할 포함)
supervisor_options = [
    "FINISH",
    "SajuExpert", 
    "WebTool", 
    "GeneralQA",
    "BIRTH_INFO_REQUEST",
    "DIRECT_ANSWER", 
    "FINAL_ANSWER"
]

# 기존 호환성을 위한 별칭
options_for_next = supervisor_options

# Supervisor 응답 모델 정의
class RouteResponse(BaseModel):
    action: Literal[*supervisor_options]
    message: Optional[str] = None  # DIRECT_ANSWER, BIRTH_INFO_REQUEST, FINAL_ANSWER 시 사용

class AgentManager:
    """에이전트 생성 및 관리 클래스"""
    
    def __init__(self):
        # 기본 LLM 설정
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    
    def create_supervisor_agent(self, tools=None):
        """Supervisor 에이전트 생성 (다른 에이전트들과 동일한 패턴)"""
        # 시스템 프롬프트 정의
        supervisor_system_prompt = (
            "당신은 사주 전문 AI 시스템의 Supervisor입니다. 현재 시간: {current_time}\n"
            "세션 ID: {session_id}, 세션 시작: {session_start_time}\n\n"
            
            "=== 현재 상태 정보 ===\n"
            "출생 정보: {birth_info_status}\n"
            "사주 계산 결과: {saju_result_status}\n"
            "질의 유형: {query_type}\n"
            "검색된 문서: {retrieved_docs_count}개\n"
            "웹 검색 결과: {web_results_count}개\n\n"
            
            "=== 사용 가능한 에이전트 ===\n"
            "- SajuExpert: 사주팔자 계산 및 운세 해석 (manse → retriever 순차 실행)\n"
            "- WebTool: 사주 개념, 일반 지식 웹 검색\n"
            "- GeneralQA: 사주와 무관한 일반 질문 답변\n\n"
            
            "=== 당신의 역할 ===\n"
            "1. **출생 정보 관리**:\n"
            "   - 사용자 입력에서 생년월일시 정보 추출 및 BirthInfo에 저장\n"
            "   - 사주 관련 질문인데 출생 정보가 없으면 재질문\n"
            "   - 출생 정보 형식: 'YYYY년 MM월 DD일 HH시 MM분, 성별(남/여), 윤달 여부'\n\n"
            
            "2. **라우팅 결정**: 사용자 질의 분석 후 적절한 에이전트 선택\n\n"
            
            "3. **최종 답변 생성**: 에이전트들의 결과를 종합하여 완성된 답변 제공\n\n"
            
            "4. **직접 응답**: 간단한 질문은 에이전트 없이 바로 답변\n\n"
            
            "=== 출생 정보 처리 기준 ===\n"
            "다음과 같은 패턴에서 출생 정보 추출:\n"
            "- '1995년 8월 26일 10시 15분 남자'\n"
            "- '1990년 3월 5일 오후 2시 30분 여자'\n"
            "- '1988년 윤 4월 12일 새벽 3시 남성'\n"
            "- '92년 12월 1일 밤 11시 여성'\n\n"
            
            "출생 정보가 불완전한 경우 재질문:\n"
            "- 연도만 있고 월일 없음 → '정확한 월일을 알려주세요'\n"
            "- 월일만 있고 연도 없음 → '태어난 연도를 알려주세요'\n"
            "- 시간 정보 없음 → '태어난 시간을 알려주세요 (예: 오전 10시 30분)'\n"
            
            "=== 라우팅 기준 ===\n"
            "다음 경우 SajuExpert 호출:\n"
            "- 완전한 출생 정보가 있고 사주 계산 요청\n"
            "- 기존 사주 결과 기반 추가 해석 요청 (건강운, 재물운, 애정운 등)\n"
            "- 대운, 세운, 용신 등 고급 분석 요청\n"
            "- 미래 운세 질문 (내일, 다음달, 내년 등)\n\n"
            
            "다음 경우 출생 정보 재질문:\n"
            "- 사주 관련 질문인데 출생 정보가 없거나 불완전함\n"
            "- '사주 봐주세요', '운세 알려주세요' 등의 요청\n\n"
            
            "다음 경우 WebTool 호출:\n"
            "- 사주 개념, 용어 설명 요청\n"
            "- 오행, 십신 등 이론적 질문\n"
            "- 일반적인 운세, 점술 관련 질문\n\n"
            
            "다음 경우 GeneralQA 호출:\n"
            "- 사주와 무관한 모든 질문\n\n"
            
            "다음 경우 직접 응답:\n"
            "- 간단한 인사 ('안녕하세요', '감사합니다')\n"
            "- 출생 정보 재질문\n"
            "- 이전 대화 확인 ('아까 말한 것', '방금 전 결과')\n"
            "- 단순 확인 ('네', '알겠습니다')\n\n"
            
            "=== 응답 양식 지침 ===\n"
            "**DIRECT_ANSWER 시 message 필드에는 다음과 같이 작성하세요:**\n"
            "- 자연스럽고 친근한 톤으로 답변\n"
            "- 불필요한 구조화된 텍스트 없이 대화체로 응답\n"
            "- 사용자가 바로 이해할 수 있는 명확하고 간단한 문장\n"
            "- 예: '안녕하세요! 사주나 운세에 관해 궁금한 것이 있으시면 언제든 말씀해주세요.'\n"
            "- 예: '네, 알겠습니다. 추가로 궁금한 점이 있으시면 언제든 물어보세요.'\n\n"
            
            "=== 최종 답변 생성 기준 ===\n"
            "- 에이전트 실행 후 사용자 질문이 완전히 해결되었다면 종합적인 최종 답변 생성\n"
            "- 여러 에이전트 결과가 있다면 일관성 있게 통합\n"
            "- 사주 결과가 있다면 구체적인 사주 정보 포함\n"
            "- 추가 질문 유도나 후속 서비스 안내 포함\n\n"
            
            "=== 컨텍스트 활용 ===\n"
            "- 이미 사주가 계산되었다면 재계산 없이 기존 결과 활용\n"
            "- 이전 대화 맥락을 고려한 연속성 있는 응답\n"
            "- 사용자의 질문 패턴과 관심사 파악하여 맞춤형 서비스\n"
        )

        # ChatPromptTemplate 생성
        prompt = ChatPromptTemplate.from_messages([
            ("system", supervisor_system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "위 대화를 분석하여 다음 중 가장 적절한 행동을 선택하세요: {options}\n\n"
                "선택 기준:\n"
                "- SajuExpert/WebTool/GeneralQA: 해당 에이전트로 라우팅\n"
                "- BIRTH_INFO_REQUEST: 사주 관련 질문인데 출생 정보가 부족한 경우\n"
                "- DIRECT_ANSWER: 간단한 인사나 확인 질문 등 직접 답변 가능한 경우\n"
                "- FINAL_ANSWER: 에이전트 실행 후 최종 답변 생성이 필요한 경우\n"
                "- FINISH: 사용자 질문이 완전히 해결되어 작업 완료하는 경우\n\n"
                "현재 상태 정보를 반드시 고려하여 결정하세요."
            ),
        ])
        
        # Supervisor 함수 정의
        def supervisor_agent_func(state):
            # State에서 정보 추출
            current_time = state.get("current_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            session_id = state.get("session_id", "unknown")
            session_start_time = state.get("session_start_time", "unknown")
            
            # State 상태 분석
            birth_info = state.get("birth_info")
            saju_result = state.get("saju_result")
            query_type = state.get("query_type", "unknown")
            retrieved_docs = state.get("retrieved_docs", [])
            web_search_results = state.get("web_search_results", [])
            
            # 상태 정보를 문자열로 변환
            birth_info_status = True if birth_info else False
            saju_result_status = True if saju_result else False
            retrieved_docs_count = len(retrieved_docs)
            web_results_count = len(web_search_results)
            
            # 프롬프트와 LLM을 결합하여 체인 구성
            supervisor_chain = (
                prompt.partial(
                    current_time=current_time,
                    session_id=session_id,
                    session_start_time=session_start_time,
                    birth_info_status=birth_info_status,
                    saju_result_status=saju_result_status,
                    query_type=query_type,
                    retrieved_docs_count=retrieved_docs_count,
                    web_results_count=web_results_count,
                    options=str(options_for_next)
                ) | self.llm.with_structured_output(RouteResponse)
            )
            
            # Agent 호출하고 응답 반환
            response = supervisor_chain.invoke(state)
            
            # 응답 형태에 따라 적절한 반환값 생성
            if response.action in members:
                return {"next": response.action}
            elif response.action == "FINISH":
                return {"next": "FINISH"}
            elif response.action == "DIRECT_ANSWER":
                # 직접 답변의 경우 메시지를 messages에 추가하고 FINISH
                from langchain_core.messages import AIMessage
                return {
                    "messages": [AIMessage(content=response.message or "안녕하세요!")],
                    "next": "FINISH"
                }
            else:
                # BIRTH_INFO_REQUEST, FINAL_ANSWER 등
                from langchain_core.messages import AIMessage
                return {
                    "messages": [AIMessage(content=response.message or "처리 중입니다...")],
                    "next": "FINISH",
                    "supervisor_action": response.action
                }
        
        return supervisor_agent_func

    def create_manse_tool_agent(self):
        """만세력 계산 에이전트 생성 (노트북 방식으로 단순화)"""
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        return create_react_agent(llm, manse_tools)

    def create_retriever_tool_agent(self):
        """RAG 검색 에이전트 생성 (노트북 방식으로 단순화)"""
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        
        # 프롬프트 로드 시도
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
        """웹 검색 에이전트 생성 (노트북 방식으로 단순화)"""
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        
        prompt = """
사주 또는 사주 오행의 개념적 질문이나, 일상 질문이 들어오면, web search를 통해 답합니다.
"""
        return create_react_agent(llm, tools=web_tools, prompt=prompt)

    def create_general_qa_agent(self):
        """일반 QA 에이전트 생성 (노트북 방식으로 단순화)"""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        prompt = "일반적인 질문이나 상식적인 내용에 대해 답변합니다."
        return create_react_agent(llm, tools=general_qa_tools, prompt=prompt)


