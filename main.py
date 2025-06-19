"""
FortuneAI Agent System
확장 가능한 사주 상담 에이전트 시스템
"""

import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 환경 변수 로드
load_dotenv()

# 모델 import
from models import get_openai_llm, get_gemini_llm

# 통합 도구 시스템 import
from tools import ToolManager

class FortuneAgentSystem:
    """확장 가능한 FortuneAI Agent 시스템"""
    
    def __init__(self, use_openai: bool = False, enable_web: bool = False, enable_calendar: bool = False):
        """
        Agent 시스템 초기화
        
        Args:
            use_openai: OpenAI 모델 사용 여부 (기본값: False, Gemini 사용)
            enable_web: 웹 검색 도구 활성화 여부
            enable_calendar: 만세력 도구 활성화 여부
        """
        self.use_openai = use_openai
        self.enable_web = enable_web
        self.enable_calendar = enable_calendar
        
        self.llm = None
        self.tool_manager = None
        self.tools = []
        self.agent_executor = None
        self.agent_with_chat_history = None
        self.session_store = {}
        
        self.setup_system()
    
    def setup_system(self):
        """전체 시스템 설정"""
        self.setup_llm()
        self.setup_tools()
        self.setup_agent()
    
    def setup_llm(self):
        """LLM 모델 설정"""
        if self.use_openai:
            print("🤖 OpenAI 모델 초기화 중...")
            self.llm = get_openai_llm("gpt-4.1-mini")
        else:
            print("🤖 Gemini 모델 초기화 중...")
            self.llm = get_gemini_llm("gemini-2.0-flash")
        
        print("✅ LLM 모델 초기화 완료!")
    
    def setup_tools(self):
        """도구들 설정"""
        # 통합 도구 관리자 초기화
        self.tool_manager = ToolManager(
            enable_rag=True,  # RAG는 항상 활성화
            enable_web=self.enable_web,
            enable_calendar=self.enable_calendar
        )
        
        # 도구들 가져오기
        self.tools = self.tool_manager.get_tools()
    
    def setup_agent(self):
        """Agent 설정"""
        print("🤖 Agent 설정 중...")
        
        # 현재 날짜 정보 가져오기
        current_date = datetime.now()
        today_info = f"{current_date.year}년 {current_date.month}월 {current_date.day}일 ({current_date.strftime('%A')})"
        korean_weekdays = {
            'Monday': '월요일',
            'Tuesday': '화요일', 
            'Wednesday': '수요일',
            'Thursday': '목요일',
            'Friday': '금요일',
            'Saturday': '토요일',
            'Sunday': '일요일'
        }
        today_korean = f"{current_date.year}년 {current_date.month}월 {current_date.day}일 ({korean_weekdays[current_date.strftime('%A')]})"
        
        # 도구 목록 생성
        tool_descriptions = []
        for i, tool in enumerate(self.tools, 1):
            tool_descriptions.append(f"{i}. {tool.name}: {tool.description}")
        
        tools_text = "\n".join(tool_descriptions)
        
        # Agent 프롬프트 정의
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""당신은 전문적인 사주명리학 상담사 AI입니다. 사주팔자, 오행, 십신, 대운, 세운 등 전통 명리학 지식을 바탕으로 정확하고 전문적인 상담을 제공하세요.

현재 날짜: {today_korean}

사용 가능한 도구:
{tools_text}

🔮 사주 상담 전문 가이드라인:

**1. 생년월일시 정보 처리:**
- 사용자가 생년월일시를 제공하면 반드시 analyze_birth_info 도구를 사용하세요
- 년주(年柱), 월주(月柱), 일주(日柱), 시주(時柱)의 천간지지를 파악하세요
- 일간(日干)을 중심으로 한 오행 분석을 수행하세요

**2. 사주 전문 용어 활용:**
- 십신(十神): 비견, 겁재, 식신, 상관, 편재, 정재, 편관, 정관, 편인, 정인
- 오행(五行): 목(木), 화(火), 토(土), 금(金), 수(水)의 상생상극 관계
- 십이운성: 장생, 목욕, 관대, 건록, 제왕, 쇠, 병, 사, 묘, 절, 태, 양
- 신살(神殺): 천을귀인, 역마, 도화, 공망 등

**3. 운세 분석 방법:**
- 현재 대운(大運)과 세운(歲運) 분석
- 일간과 다른 간지의 합충형해 관계 파악
- 용신(用神)과 기신(忌神) 구분
- 계절과 시간대에 따른 오행 강약 판단

**4. 상담 진행 방식:**
- 먼저 사주팔자 기본 구조를 파악한 후 운세 해석
- 구체적인 사주 용어와 근거를 제시하며 설명
- 일반적인 조언보다는 사주 분석에 기반한 맞춤 조언 제공
- 오행 균형과 십신 배치를 고려한 성격 및 운세 분석

**5. 질문별 대응:**
- 직업운: 일간의 십신 배치와 관성(官星), 인성(印星) 분석
- 재물운: 재성(財星)의 강약과 일간과의 관계 분석  
- 연애운: 관성과 재성의 배치, 도화살 유무 확인
- 건강운: 일간의 강약과 상극 오행 분석
- 학업운: 인성과 식상(食傷)의 배치 분석

**6. 도구 활용 가이드 (필수):**
- **모든 사주 관련 질문에 반드시 도구를 사용하세요**
- 1순위: 사주 계산 도구들 - 생년월일시 제공시 필수 사용
  * get_comprehensive_saju_analysis: 종합 사주 분석 (가장 많이 사용)
  * calculate_saju_chart: 기본 사주팔자만 계산
  * analyze_five_elements: 오행 강약 분석
  * analyze_ten_gods: 십신 분석
  * calculate_great_fortune: 대운 계산
  * 주의: 모든 도구는 생년월일시 문자열에서 성별을 자동 파싱합니다 ('남성', '여성', '남', '여' 키워드 인식)
- 2순위: 지능형 검색 도구 (smart_search_saju) - 질문 유형 자동 분석 후 최적 검색
- 3순위: 전문 RAG 검색 (search_saju_knowledge) - 사주 전문 지식 직접 검색
- 4순위: 웹 검색 도구 (search_web_saju) - 일반 정보나 보완 자료 검색
- **절대 도구 없이 기본 지식만으로 답변하지 마세요**

**7. 지능형 검색 시스템:**
- 사주 분석 질문 → RAG 우선 검색 (정확성 보장)
- 일반적인 질문 → 웹 검색 우선 (최신 정보)
- RAG 결과 부족시 → 자동으로 웹 검색 보완
- 시스템이 자동으로 최적의 검색 전략을 선택합니다

**8. 출처 표시 가이드:**
- 웹 검색 결과를 활용할 때는 반드시 출처를 명시하세요
- 형식: "참고: [제목] (출처: URL)"
- RAG 검색 결과는 문서명으로 표시
- 여러 출처를 참고한 경우 모든 출처를 나열하세요

**9. 답변 작성 순서 (반드시 준수):**
1. 생년월일시 정보가 있으면 get_comprehensive_saju_analysis로 종합 사주 분석 우선 실행
2. 특정 분야만 필요하면 해당 전문 도구 사용 (오행, 십신, 대운 등)
3. 사주 지식이 필요하면 smart_search_saju로 지능형 검색 실행
4. 검색 전략과 결과를 확인
5. 필요시 추가 도구 사용 (웹 검색 등)
6. 계산 결과와 검색 결과를 종합하여 답변 작성
7. 반드시 출처 명시 (문서명 또는 웹 출처)

**9. 주의사항:**
- 도구 사용 없이는 절대 답변하지 마세요
- 단순한 일반론이 아닌 구체적이고 근거 있는 분석 제공
- 사주 용어를 사용하되 이해하기 쉽게 설명 병행
- 부정적인 내용도 건설적인 조언과 함께 제시

대화 시작 시 생년월일시를 정확히 확인하고 사주 분석부터 시작하세요."""),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        # Agent 생성
        agent = create_tool_calling_agent(self.llm, self.tools, agent_prompt)
        
        # AgentExecutor 생성
        self.agent_executor = AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=False,  # 중간 과정 출력 표시
            max_iterations=3,
            early_stopping_method="generate"
        )
        
        # 대화 기록을 포함한 Agent 생성
        self.agent_with_chat_history = RunnableWithMessageHistory(
            self.agent_executor,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        
        print("✅ Agent 설정 완료!")
    
    def get_session_history(self, session_id: str):
        """세션별 대화 기록을 관리하는 함수"""
        if session_id not in self.session_store:
            self.session_store[session_id] = ChatMessageHistory()
        return self.session_store[session_id]
    
    def chat(self, message: str, session_id: str = None) -> str:
        """사용자와 대화하는 메인 함수"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        try:
            response = self.agent_with_chat_history.invoke(
                {"input": message},
                config={"configurable": {"session_id": session_id}}
            )
            return response['output']
        except Exception as e:
            return f"죄송합니다. 오류가 발생했습니다: {str(e)}"
    
    def interactive_chat(self):
        """대화형 채팅 시스템"""
        session_id = str(uuid.uuid4())
        
        print("=== FortuneAI Agent 대화형 상담 ===")
        print("💬 언제든지 질문해주세요!")
        print("💡 종료하려면 'quit', 'exit', 'q' 중 하나를 입력하세요.")
        print("🔄 새 대화를 시작하려면 'new'를 입력하세요.")
        print(f"🛠️ 활성화된 도구: {len(self.tools)}개")
        print("-" * 60)
        
        # 첫 인사
        welcome_message = self.chat("안녕하세요", session_id)
        print(f"🔮 AI 상담사: {welcome_message}")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\n🙋 사용자: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n🌟 상담을 종료합니다. 좋은 하루 되세요! 🌟")
                    break
                    
                if user_input.lower() == 'new':
                    session_id = str(uuid.uuid4())
                    print(f"\n🔄 새로운 대화를 시작합니다. (세션 ID: {session_id[:8]}...)")
                    welcome_message = self.chat("안녕하세요", session_id)
                    print(f"🔮 AI 상담사: {welcome_message}")
                    print("-" * 60)
                    continue
                
                # AI 응답 생성
                response = self.chat(user_input, session_id)
                print(f"\n🔮 AI 상담사: {response}")
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n\n🌟 상담을 종료합니다. 좋은 하루 되세요! 🌟")
                break
            except Exception as e:
                print(f"\n❌ 오류가 발생했습니다: {e}")
                print("다시 질문해 주세요.")
    
    def get_system_info(self):
        """시스템 정보 출력"""
        tool_info = self.tool_manager.get_tool_info()
        
        print("\n=== FortuneAI Agent 시스템 정보 ===")
        print(f"🤖 LLM 모델: {'OpenAI GPT-4.1-mini' if self.use_openai else 'Google Gemini 2.0 Flash'}")
        print(f"🛠️ 활성화된 도구 수: {tool_info['total_tools']}개")
        print(f"🔧 RAG 도구: {'활성화' if tool_info['rag_enabled'] else '비활성화'}")
        print(f"🌐 웹 검색: {'활성화' if tool_info['web_enabled'] else '비활성화'}")
        print(f"🧮 사주 계산: {'활성화' if tool_info['calendar_enabled'] else '비활성화'}")
        print("=" * 40)

def main():
    """메인 실행 함수"""
    print("🌟 FortuneAI Agent 시스템 시작 🌟\n")
    
    # Agent 시스템 초기화 (웹 검색 도구 활성화)
    # 확장하려면: FortuneAgentSystem(use_openai=True, enable_web=True, enable_calendar=True)
    agent_system = FortuneAgentSystem(
        use_openai=False,      # Gemini 사용
        enable_web=True,       # 웹 검색 활성화
        enable_calendar=True   # 사주 계산 도구 활성화
    )
    
    # 시스템 정보 출력
    agent_system.get_system_info()
    
    # 대화형 채팅 시작
    agent_system.interactive_chat()

if __name__ == "__main__":
    main() 