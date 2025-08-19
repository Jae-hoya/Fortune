# -*- coding: utf-8 -*-

"""

LangGraph 표준: 그래프 노드 함수 모듈

모든 *_node, *_handler 함수 정의 (주석, 타입, 프롬프트, 내부 로직 100% 보존)

"""

import json

from .state import TarotState

# =================================================================
# 공통 상수 정의 - 일관성 유지를 위해
# =================================================================

# 새로운 상담 요청 트리거 (띄어쓰기 변형 포함)
NEW_CONSULTATION_TRIGGERS = [
    "새로 봐줘", "새로봐줘", "새로 봐주", "새로봐주",
    "새로 상담", "새 상담", "다시 봐줘", "다시봐줘", "처음부터"
]

# 일반 타로 상담 트리거 (기존 맥락 이어받기)
TAROT_CONSULTATION_TRIGGERS = [
    "타로 봐줘", "타로봐줘", "타로 상담", "점 봐줘", "운세 봐줘"
]

# 모든 상담 트리거 (분류용)
ALL_CONSULTATION_TRIGGERS = NEW_CONSULTATION_TRIGGERS + TAROT_CONSULTATION_TRIGGERS

# 간단한 카드 뽑기 트리거
SIMPLE_CARD_TRIGGERS = [
    "카드 한장", "카드 뽑", "카드뽑", "간단히", "가볍게", "알아볼까",
    "뽑아서", "한장만", "뽑아줘", "뽑아주", "뽑아"
]

# 단순 응답 (원래 질문 찾기에서 제외할 항목)
SIMPLE_RESPONSES = [
    "네", "좋아", "그래", "응", "해줘", "부탁해", "yes", "예",  # 긍정
    "아니", "아니요", "싫어", "안해", "no", "됐어"  # 부정
]

from .helpers import (
    get_last_user_input, is_simple_followup, determine_consultation_handler, determine_target_handler, performance_monitor

)

from .analysis import *

from .timing import *

from .translation import *

from .helpers import *

from .tools import search_tarot_spreads, search_tarot_cards

from langchain_core.messages import HumanMessage, AIMessage

from langchain_openai import ChatOpenAI

from langgraph.prebuilt import ToolNode

# =================================================================

# 최적화된 핵심 노드들

# =================================================================

def state_classifier_node(state: TarotState) -> TarotState:
   """🆕 상태 기반 빠른 분류 - LLM 호출 최소화"""
   # Step 1: 명확한 상태는 바로 분류 (LLM 없이)
   consultation_data = state.get("consultation_data", {})
   status = consultation_data.get("status", "") if consultation_data else ""
   print(f"🔍 State Classifier: status='{status}'")
   # 상담 진행 중이면 바로 라우팅
   if status in ["spread_selection", "card_selection", "summary_shown"]:
       handler = determine_consultation_handler(status)
       print(f"🚀 Fast Track: CONSULTATION_ACTIVE -> {handler}")
       return {
           "routing_decision": "CONSULTATION_ACTIVE",
           "target_handler": handler,
           "needs_llm": False
       }
   # 상담 완료 후 추가 질문 판단
   if status == "completed":
       user_input = get_last_user_input(state)
       
       # 🔧 새로운 상담 요청인지 확인
       is_new_consultation = any(trigger in user_input.lower() for trigger in NEW_CONSULTATION_TRIGGERS)
       
       if is_new_consultation:
           print(f"🚀 Fast Track: NEW_CONSULTATION_REQUEST")
           return {
               "routing_decision": "NEW_SESSION",
               "needs_llm": False
           }
       else:
           # 🔧 "새로 봐줘"가 아닌 모든 경우는 꼬리 질문으로 처리
           print(f"🚀 Fast Track: FOLLOWUP_QUESTION (completed 상태)")
           return {
               "routing_decision": "FOLLOWUP_QUESTION", 
               "target_handler": "context_reference_handler",
               "needs_llm": False
           }
   # Step 2: 애매한 경우만 LLM 사용
   print(f"🧠 Complex Analysis: NEW_SESSION")
   return {
       "routing_decision": "NEW_SESSION",
       "needs_llm": True
   }
def card_info_handler(state: TarotState) -> TarotState:
   """카드 정보 핸들러 - 기존 RAG 기능 완전 통합"""
   user_input = state["user_input"]
   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
   llm_with_tools = llm.bind_tools([search_tarot_cards])
   prompt = f"""
   사용자가 타로 카드에 대해 질문했습니다: "{user_input}"
   search_tarot_cards 도구를 사용해서 관련 카드 정보를 검색하고, 
   친근하고 이해하기 쉽게 설명해주세요.
   마지막에 "다른 카드나 타로 상담이 필요하시면 언제든 말씀해주세요!"라고 덧붙여주세요.
   🔮 타로 상담사 톤으로 답변하세요.
   """
   try:
       response = llm_with_tools.invoke([HumanMessage(content=prompt)], {"metadata": {"final_response": "yes", "handler": "card_info_handler"}})
       return {"messages": [response]}
   except Exception as e:
       fallback_msg = f"🔮 카드 정보를 찾는 중 문제가 생겼어요. 다시 질문해주시면 더 정확히 답변드릴게요!\n\n다른 궁금한 점이나 고민이 있으시면 언제든 말씀해주세요!"
       return {"messages": [AIMessage(content=fallback_msg)]}
def spread_info_handler(state: TarotState) -> TarotState:
   """스프레드 정보 핸들러 - 기존 RAG 기능 완전 통합"""
   user_input = state["user_input"]
   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
   llm_with_tools = llm.bind_tools([search_tarot_spreads])
   prompt = f"""
   사용자가 타로 스프레드에 대해 질문했습니다: "{user_input}"
   search_tarot_spreads 도구를 사용해서 관련 스프레드 정보를 검색하고,
   스프레드의 특징, 사용법, 언제 사용하면 좋은지 등을 친근하게 설명해주세요.
          마지막에 "카드 한 장으로 간단한 조언을 원하시면 '네'를, **여러 장으로 깊은 상담**을 원하시면 '타로 봐줘'라고 말씀해주세요!"라고 덧붙여주세요.
   🔮 타로 상담사 톤으로 답변하세요.
   """
   try:
       response = llm_with_tools.invoke([HumanMessage(content=prompt)], {"metadata": {"final_response": "yes", "handler": "spread_info_handler"}})
       return {"messages": [response]}
   except Exception as e:
       fallback_msg = f"🔮 스프레드 정보를 찾는 중 문제가 생겼어요. 다시 질문해주시면 더 정확히 답변드릴게요!\n\n이 스프레드로 상담받고 싶으시거나 다른 고민이 있으시면 언제든 말씀해주세요!"
       return {"messages": [AIMessage(content=fallback_msg)]}
def simple_card_handler(state: TarotState) -> TarotState:
   """🆕 간단한 카드 한 장 뽑기 핸들러"""
   user_input = state["user_input"]
   # 🆕 대화 히스토리에서 원래 질문 찾기
   original_question = ""
   messages = state.get("messages", [])
   # 최근 사용자 메시지들을 역순으로 검색 (현재 "네" 제외)
   for msg in reversed(messages):
       if isinstance(msg, HumanMessage) and msg.content.strip().lower() not in ["네", "좋아", "그래", "응", "해줘", "부탁해", "yes", "예"]:
           original_question = msg.content.strip()
           break
   # 실제 질문이 있으면 그것을 사용, 없으면 현재 입력 사용
   question_for_interpretation = original_question if original_question else user_input
   print(f"🎯 원래 질문: '{original_question}' | 해석용 질문: '{question_for_interpretation}'")
   # 랜덤으로 카드 한 장 선택
   import random
   card_number = random.randint(1, 78)
   orientation = random.choice(["정방향", "역방향"])
   # 카드 정보 가져오기
   selected_card = select_cards_randomly_but_keep_positions([card_number], 1)[0]
   card_name = selected_card['name']
   card_info_kr = translate_card_info(card_name, orientation)
   
   # 카드 이름 추출 (딕셔너리인 경우 처리)
   if isinstance(card_info_kr, dict):
       card_name_kr = card_info_kr.get('name', card_name)
       if orientation == "역방향":
           card_name_kr = f"{card_name_kr} (역방향)"
   else:
       card_name_kr = str(card_info_kr)
   # 카드 검색으로 의미 가져오기
   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
   llm_with_tools = llm.bind_tools([search_tarot_cards])
   search_prompt = f"{card_name} {orientation} meaning"
   try:
       # 카드 의미 검색
       search_response = llm_with_tools.invoke([HumanMessage(content=f"search_tarot_cards('{search_prompt}')")])
       # 간단한 해석 생성
       interpretation_prompt = f"""
       사용자가 간단한 질문을 했습니다: "{question_for_interpretation}"
       뽑힌 카드: {card_name_kr}
       **중요: 반드시 사용자의 구체적인 질문에 카드로 답변해주세요!**
       
**중요한 호칭 규칙:**
- 사용자를 지칭할 때는 '내담자님'으로만 하세요 ('당신', '사용자님', '고객님' 금지)
- 한국어 특성상 주어를 자연스럽게 생략할 수 있는 곳에서는 생략해도 됩니다
- 어미는 '~입니다' 대신 '~이에요/~해요' 등 친근한 어미로 말해주세요
       **답변 구조:**
       1. **반드시 첫 문장은: "{card_name_kr} 카드가 뽑혔네요!"로 시작**
       2. 뽑힌 카드 간단 소개 (1줄)
       3. **사용자 질문에 대한 카드의 직접적인 답변** (2-3줄) - 가장 중요!
       4. 카드가 제시하는 간단한 조언이나 방향성 (1-2줄)
       **예시 (만약 "짬뽕? 짜장?" 질문이라면):**
       - "운명의 수레바퀴가 말하길, 지금은 변화를 받아들일 때라고 하네요. 매운 짬뽕으로 가세요!"
       - "이 카드는 새로운 도전을 의미해요. 평소와 다른 선택을 해보는 건 어떨까요?"
       **사용자의 질문이 선택 관련이면:** 카드가 어떤 선택을 제시하는지
       **사용자의 질문이 상황 관련이면:** 카드가 그 상황을 어떻게 보는지
       **사용자의 질문이 감정 관련이면:** 카드가 그 감정에 어떤 메시지를 주는지
       
               **마지막에 반드시 다음 안내를 추가해주세요:**
        
        "💫 **다음 중 원하시는 것을 선택해주세요:**
        
        🔮 **더 깊은 상담**: '타로 봐줘' (여러 장으로 자세한 분석)
        
        🆕 **새로운 질문**: '새로 봐줘' (완전히 다른 주제로)"
       
       🔮 친근하고 실용적인 톤으로, 사용자 질문과 카드를 확실히 연결해서 답변하세요.
       """
       interpretation_response = llm.invoke([HumanMessage(content=interpretation_prompt)], {"metadata": {"final_response": "yes", "handler": "simple_card_handler"}})
       # 카드 정보를 해석에 자연스럽게 포함 (별도 표시 제거)
       final_message = interpretation_response.content
       return {"messages": [AIMessage(content=final_message)]}
   except Exception as e:
       # 간단한 폴백 응답
       fallback_msg = f"""🃏 {card_name_kr} 카드가 뽑혔네요!
       
카드 해석 중 문제가 생겼지만, 이 카드는 분명 당신의 질문 "{user_input}"에 대한 답을 가지고 있을 거예요. 다시 시도해주시거나, 더 자세한 상담이 필요하시면 언제든 말씀해주세요!"""
       
       return {"messages": [AIMessage(content=fallback_msg)]}

def consultation_handler(state: TarotState) -> TarotState:
    """리팩토링된 상담 핸들러 - 새로운 노드들을 순차 실행"""
    print("🔧 기존 consultation_handler 호출 -> 리팩토링된 노드들로 처리")
    # 🔧 핵심 수정: user_input을 state에 설정
    user_input = get_last_user_input(state)
    state["user_input"] = user_input
    print(f"🔧 사용자 입력 설정: '{user_input}'")
    # 🆕 단순한 "타로 봐줘"만 있는 경우 처리 개선
    simple_triggers = ["타로 봐줘", "타로봐줘", "타로 상담", "점 봐줘", "운세 봐줘"]
    # 새로운 상담 요청인지 확인 (공통 상수 사용)
    is_new_consultation_request = any(trigger in user_input.lower() for trigger in NEW_CONSULTATION_TRIGGERS)
    
    if is_new_consultation_request:
        print(f"🆕 새로운 상담 요청 감지: '{user_input}' - 이전 히스토리 무시")
        return {
            "messages": [AIMessage(content="""🔮 

어떤 고민이나 궁금한 점이 있으신가요? 편하게 이야기해주세요. 

💫 **예시:**
- 연애, 진로, 인간관계, 건강 등 어떤 주제든 괜찮아요
- "요즘 직장에서 스트레스가 많아서..."
- "새로운 연애를 시작할지 고민이에요"
- "이직을 해야 할지 모르겠어요"

무엇이든 편하게 이야기해주시면, 가장 적합한 타로 스프레드로 답을 찾아드릴게요! ✨""")],
            "consultation_data": {
                "status": "waiting_for_concern"
            }
        }
    
    # 🔧 대화 맥락 추출 - 사용자의 원래 질문 찾기 (새로운 상담이 아닌 경우에만)
    conversation_context = ""
    original_user_question = ""
    messages = state.get("messages", [])
    
    # 최근 사용자 질문 찾기 (타로 봐줘 이전의 질문)
    user_messages = []
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_messages.append(msg.content.strip())
            if len(user_messages) >= 3:  # 최근 3개까지만
                break
    
    # "타로 봐줘" 이전의 실제 질문 찾기
    for msg_content in user_messages[1:]:  # 첫 번째(현재)는 "타로 봐줘"이므로 제외
                 # 🔧 타로 트리거와 단순 응답 모두 제외 (공통 상수 사용)
         excluded_responses = NEW_CONSULTATION_TRIGGERS + SIMPLE_RESPONSES
         if msg_content.lower().strip() not in [resp.lower() for resp in excluded_responses]:
            original_user_question = msg_content
            print(f"🎯 원래 사용자 질문 발견: '{original_user_question}'")
            break
    
    # AI 응답에서도 맥락 추출
    if len(messages) >= 2:
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                conversation_context = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                break
    
    # LLM으로 고민이 포함되어 있는지 동적 판단 (대화 맥락 포함)
    has_specific_concern = check_if_has_specific_concern(user_input, conversation_context)
    # 🔧 대화 히스토리에서 최근 고민이 있는지 확인
    recent_concern = None
    if user_input.strip() in simple_triggers and not has_specific_concern:
        # 최근 3개의 사용자 메시지에서 고민 찾기
        user_messages = []
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_messages.append(msg.content)
                if len(user_messages) >= 3:
                    break
        # 최근 메시지들에서 고민이 있는지 확인
        for recent_msg in user_messages:
            if check_if_has_specific_concern(recent_msg):
                recent_concern = recent_msg
                print(f"🔧 대화 히스토리에서 고민 발견: '{recent_concern}'")
                break
    # 🔧 "타로 봐줘" 트리거는 이전 대화에서 고민을 더 넓게 찾기
    if user_input.strip() in simple_triggers and not has_specific_concern and not recent_concern:
        # 🆕 더 넓은 범위에서 고민 찾기 (최근 5개 메시지)
        extended_concern = None
        user_messages = []
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_messages.append(msg.content)
                if len(user_messages) >= 5:  # 범위 확장
                    break
        
        # 더 관대한 기준으로 고민 찾기
        for recent_msg in user_messages:
            # 🔧 단순 감정 표현도 고민으로 인정 ("고민있어", "힘들어" 등)
            if len(recent_msg.strip()) > 3 and not recent_msg.strip() in simple_triggers:
                extended_concern = recent_msg
                print(f"🔧 확장 검색에서 고민 발견: '{extended_concern}'")
                break
        
        if extended_concern:
            print(f"🔧 확장된 고민으로 상담 시작: '{extended_concern}'")
            state["user_input"] = extended_concern  # 확장된 고민으로 교체
        else:
            print("🔧 단순 트리거 감지 - 고민 문의")
            return {
                "messages": [AIMessage(content="""🔮 
어떤 고민이나 궁금한 점이 있으신가요? 편하게 말씀해주세요.

예를 들어:

• 연애나 인간관계 고민

• 진로나 직업 관련 고민  

• 현재 상황에 대한 조언

• 미래에 대한 궁금증

• 중요한 결정을 앞둔 상황

무엇이든 편하게 이야기해주시면, 가장 적합한 타로 스프레드로 답을 찾아드릴게요! ✨""")],
                "consultation_data": {
                    "status": "waiting_for_concern"
                }
            }
    # 🔧 원래 사용자 질문이 있으면 그것을 사용해서 상담 시작
    if original_user_question:
        print(f"🔧 원래 사용자 질문으로 상담 시작: '{original_user_question}'")
        state["user_input"] = original_user_question  # 원래 질문으로 교체
    # 🔧 최근 고민이 있으면 그것을 사용해서 상담 시작
    elif recent_concern:
        print(f"🔧 최근 고민으로 상담 시작: '{recent_concern}'")
        state["user_input"] = recent_concern  # 최근 고민으로 교체
    # Phase 1 리팩토링: 웹 검색 제거 후 2개 노드 순차 실행
    try:
        # 1. 감정 분석
        result1 = emotion_analyzer_node(state)
        state.update(result1)
        # 2. 스프레드 추천 (웹 검색 관련 코드 제거됨)
        result4 = spread_recommender_node(state)
        state.update(result4)
        print("✅ 실제 상담 성공적으로 완료")
        return state
    except Exception as e:
        print(f"❌ 실제 상담 오류: {e}")
        # 기본 에러 처리
        return {
            "messages": [AIMessage(content="🔮 상담 처리 중 문제가 발생했습니다. 다시 시도해주세요.")],
            "consultation_data": {
                "status": "error"
            }
        }

def general_handler(state: TarotState) -> TarotState:
   """일반 질문 핸들러 - 날짜 질문 특별 처리 """
   user_input = state["user_input"]
   # 🔧 LLM 기반 날짜 질문 감지 (하드코딩 제거)
   def is_date_question(text: str) -> bool:
       """LLM으로 날짜/시간 관련 질문인지 판단"""
       try:
           llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
           prompt = f"""
           다음 사용자 입력이 **순수하게 현재 날짜나 시간 정보 자체**를 알고 싶어하는 질문인지 판단해주세요:
           "{text}"
           **True (순수 날짜/시간 질문)**: 
           - 오늘 몇일?, 지금 몇시?, 오늘 날짜 알려줘, 현재 시간?, 며칠이야?, 몇시야?
           - 단순히 날짜나 시간 정보만 원하는 경우
           **False (다른 목적이 있는 질문)**:
           - 오늘부터 투자해도 돼?, 지금 당장 들어가도 돼?, 현재 상황에서 투자할까?
           - 언제 투자해야해?, 언제 만나?, 언제가 좋을까?, 타이밍이 언제?, 시기가 언제?
           - "오늘/지금/현재" 등이 포함되어도 투자/조언/결정을 묻는 질문
           **핵심 판단 기준**: 사용자가 **날짜/시간 정보 자체**를 원하는가? 아니면 **그 시점과 관련된 조언/결정**을 원하는가?
           JSON 형식으로 답변:
           {{"is_date_question": true/false, "reasoning": "판단 근거"}}
           """
           response = llm.invoke([HumanMessage(content=prompt)])
           import json
           result = json.loads(response.content.strip())
           return result.get("is_date_question", False)
       except Exception as e:
           # 🔧 LLM 실패시 False 반환 (하드코딩 완전 제거)
           print(f"⚠️ 날짜 질문 판단 실패: {e}")
           return False  # 확실하지 않으면 일반 질문으로 처리
   if is_date_question(user_input):
       # 시간 맥락 설정
       state = ensure_temporal_context(state)
       current_context = state.get("temporal_context", {})
       current_date = current_context.get("current_date", "날짜 정보를 가져올 수 없습니다")
       weekday = current_context.get("weekday_kr", "")
       season = current_context.get("season", "")
       date_response = f"""🔮
오늘은 **{current_date} {weekday}**입니다. 

현재 {season}철이네요! ✨

매일매일이 새로운 가능성으로 가득 차 있으니, 오늘도 좋은 하루 되시길 바랍니다!

타로 상담을 받고 싶으시거나 다른 궁금한 점이 있으시면 언제든 말씀해주세요! 🔮"""
       
       return {"messages": [AIMessage(content=date_response)]}

   # 일반 질문 처리
   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

   # 🆕 supervisor 분석 결과 활용 - 감정 상태 기반 응답
   supervisor_decision = state.get("supervisor_decision", {})
   confidence = supervisor_decision.get("confidence", "medium")
   reasoning = supervisor_decision.get("reasoning", "")
   emotion_type = supervisor_decision.get("emotion_type", "general")
   support_needed = supervisor_decision.get("support_needed", "low")
   
   print(f"🎯 Supervisor 분석 활용: emotion_type={emotion_type}, support_needed={support_needed}")
   print(f"🎯 Reasoning: {reasoning}")
   
   if emotion_type == "greeting":
       # 간단한 인사 - 고민 유도
       prompt = f"""
       사용자가 간단한 인사를 했습니다: "{user_input}"
       타로 상담사로서 친근하게 인사를 받고 고민을 말해달라고 유도해주세요.
       
       **응답 규칙:**
       - 2-3줄 정도의 따뜻한 인사
       - 친근하고 따뜻한 톤
       - 이모지 적절히 사용
       - 자연스럽게 고민이나 궁금한 점을 말해달라고 유도
       
       **중요한 호칭 규칙:**
       - 사용자를 지칭할 때는 '내담자님'으로만 하세요
       - 어미는 '~이에요/~해요' 등 친근한 어미 사용
       """
   elif emotion_type == "emotional_concern":
       # 감정적 고민 - 공감과 지원 중심
       prompt = f"""
       사용자가 감정적 고민을 표현했습니다: "{user_input}"
       타로 상담사로서 공감하고 지원하는 답변을 해주세요.
       
       **응답 규칙:**
       - 먼저 사용자의 감정을 인정하고 공감
       - 따뜻하고 위로가 되는 톤
       - 구체적인 고민을 더 말해달라고 자연스럽게 유도
       - 타로 상담 제안은 하지 말고, 먼저 더 자세한 이야기를 들어보겠다는 의지 표현

       **중요한 호칭 규칙:**
       - 사용자를 지칭할 때는 '내담자님'으로만 하세요
       - 어미는 '~이에요/~해요' 등 친근한 어미 사용
       """
   elif emotion_type == "specific_concern":
       # 구체적 고민 - 바로 타로 상담 제안
       prompt = f"""
       사용자가 구체적인 고민을 표현했습니다: "{user_input}"
       타로 상담사로서 공감하고 도움이 되는 답변을 해주세요.
       
       **응답 규칙:**
       - 사용자의 고민에 공감하고 이해한다는 메시지
       - 따뜻하고 전문적인 톤
       - 마지막에 반드시 다음 문구를 포함: "🔮 **카드 한 장으로 간단한 조언**을 원하시면 '네'를, **여러 장으로 깊은 상담**을 원하시면 '타로 봐줘'라고 말씀해주세요!"

       **중요한 호칭 규칙:**
       - 사용자를 지칭할 때는 '내담자님'으로만 하세요
       - 어미는 '~이에요/~해요' 등 친근한 어미 사용
       """
   elif emotion_type == "consultation_inquiry":
       # 상담 관련 문의 - 타로 제안 없이 간단한 답변
       prompt = f"""
       사용자가 상담 관련 질문을 했습니다: "{user_input}"
       타로 상담사로서 친근하고 도움이 되는 답변을 해주세요.
       
       **응답 규칙:**
       - 상담 가능하다는 긍정적인 답변
       - 친근하고 따뜻한 톤 유지
       - 2-3문장 정도의 간단한 응답
       - 타로 상담 제안은 하지 말고, 자연스럽게 고민을 말해달라고 유도

       **중요한 호칭 규칙:**
       - 사용자를 지칭할 때는 '내담자님'으로만 하세요
       - 어미는 '~이에요/~해요' 등 친근한 어미 사용
       """
   elif emotion_type == "gratitude":
       # 감사 표현 - 타로 제안 없이 따뜻한 답변
       prompt = f"""
       사용자가 감사 인사를 했습니다: "{user_input}"
       타로 상담사로서 따뜻하게 받아주세요.
       
       **응답 규칙:**
       - 감사 인사에 대한 따뜻한 답변
       - 친근하고 감사한 톤 유지
       - 2-3문장 정도의 간단한 응답
       - 타로 상담 제안은 하지 말고, 도움이 되었다면 기쁘다는 메시지
       - 필요시 언제든 찾아오라는 자연스러운 마무리

       **중요한 호칭 규칙:**
       - 사용자를 지칭할 때는 '내담자님'으로만 하세요
       - 어미는 '~이에요/~해요' 등 친근한 어미 사용
       """
   else:
       # 일반적인 대화나 질문
       prompt = f"""
       사용자 입력: "{user_input}"
       타로 상담사로서 친근하고 도움이 되는 답변을 해주세요.
       
       **응답 규칙:**
       - 사용자의 입력에 적절하게 반응
       - 친근하고 따뜻한 톤 유지
       - 입력이 매우 짧으면 응답도 간결하게, 길면 더 자세하게
       - 마지막에 반드시 다음 문구를 포함: "🔮 **카드 한 장으로 간단한 조언**을 원하시면 '네'를, **여러 장으로 깊은 상담**을 원하시면 '타로 봐줘'라고 말씀해주세요!"

       **중요한 호칭 규칙:**
       - 사용자를 지칭할 때는 '내담자님'으로만 하세요
       - 어미는 '~이에요/~해요' 등 친근한 어미 사용
       """
   
   try:
       response = llm.invoke([HumanMessage(content=prompt)], {"metadata": {"final_response": "yes", "handler": "general_handler"}})
       return {"messages": [AIMessage(content=response.content)]}

   except Exception as e:
       fallback_msg = "🔮 질문에 답변드리는 중 문제가 생겼어요. 다시 질문해주시면 더 정확히 답변드릴게요!\n\n🔮 **카드 한 장으로 간단한 조언**을 원하시면 '네'라고 답해주세요. 다른 궁금한 점이 있으시면 언제든 말씀해주세요!"
       return {"messages": [AIMessage(content=fallback_msg)]}

def unknown_handler(state: TarotState) -> TarotState:
    """
알 수 없는 입력 핸들러"""
    return {
        "messages": [AIMessage(content="""

🔮\n\n어떤 도움이 필요하신가요? 

- 타로 카드 의미가 궁금하시거나

- 스프레드 정보를 알고 싶으시거나

- 타로 상담을 받고 싶으시면 언제든 말씀해주세요!

편안하게 대화해요 😊""")]
    }

def consultation_flow_handler(state: TarotState) -> TarotState:
   """상담 진행 중 처리 - 안전성 강화"""
   # 안전성 체크
   if not state:
       print("❌ state가 None입니다")
       return {"messages": [AIMessage(content="🔮 상담을 시작하겠습니다. 어떤 고민이 있으신가요?")]}
   consultation_data = state.get("consultation_data", {})
   if not consultation_data:
       consultation_data = {}
   status = consultation_data.get("status", "")
   user_input = state.get("user_input", "")
   print(f"🔧 상담 흐름 처리: status={status}, user_input='{user_input}'")
   if status == "waiting_for_concern":
       # 🆕 고민을 받은 후 실제 상담 시작
       print("🔧 고민 접수 - 실제 상담 시작")
       # 원래 consultation_handler의 로직 실행 (고민 체크 제외)
       return start_actual_consultation(state)
   elif status == "spread_selection":
       if any(num in user_input for num in ["1", "2", "3"]):
           return consultation_continue_handler(state)
       else:
           return {"messages": [AIMessage(content="1, 2, 3 중에서 선택해주세요.")]}
   elif status == "card_selection":
       if any(char.isdigit() or char == ',' for char in user_input):
           return consultation_summary_handler(state)
       else:
           return {"messages": [AIMessage(content="카드 번호를 입력해주세요. (예: 7, 23, 45)")]}
   elif status == "summary_shown":
       user_input_lower = user_input.lower()
       if any(keyword in user_input_lower for keyword in ["상세해석", "상세 해석"]):
           return consultation_individual_handler(state)
       elif any(keyword in user_input_lower for keyword in ["아니", "no", "괜찮", "됐어", "안볼"]):
           return {"messages": [AIMessage(content="🔮 상담이 도움이 되었기를 바랍니다! 다른 고민이 있으시면 언제든 말씀해주세요. ✨")]}
       else:
           return {"messages": [AIMessage(content="자세한 해석을 보고 싶으시면 '상세 해석'이라고 말씀해주세요!")]}
   elif status == "completed":
       # 🆕 상세 해석 완료 후 처리 - triggerplan.md 핵심 개선사항
       print(f"🔧 상담 완료 후 처리: user_input='{user_input}'")
       trigger_result = simple_trigger_check(user_input)
       print(f"🎯 트리거 결과: {trigger_result}")
       if trigger_result == "new_consultation":
           print("🔧 새 상담 시작 트리거 감지")
           return consultation_handler(state)
       elif trigger_result == "individual_reading":
           # 이미 완료된 상태에서 상세 해석 재요청 - 안내 메시지
                       return {"messages": [AIMessage(content="이미 상세 해석을 모두 보여드렸습니다. 새로운 고민이 있으시면 '새로 봐줘'라고 말씀해주세요!")]}
       else:
           # context_reference - 추가 질문으로 처리
           print("🔧 추가 질문으로 처리")
           return context_reference_handler(state)
   else:
       # 새로운 상담 시작
       print("🔧 새로운 상담 시작")
       return consultation_handler(state)
def consultation_continue_handler(state: TarotState) -> TarotState:
    """
상담 계속 진행 핸들러 - 스프레드 선택 후"""
    
    consultation_data = state.get("consultation_data", {})
    if not consultation_data or consultation_data.get("status") != "spread_selection":
        return {"messages": [AIMessage(content="상담 정보가 없습니다. 새로운 고민을 말씀해주세요.")]}
    # 마지막 사용자 메시지에서 스프레드 선택 추출
    user_input = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_input = msg.content.strip()
            break
    # 🔧 기본 스프레드 이름 감지 우선 처리
    default_spreads = get_default_spreads()
    selected_default_spread = None
    
    # 기본 스프레드 이름 매칭
    spread_name_mapping = {
        "세 장": "THE THREE CARD SPREAD",
        "세장": "THE THREE CARD SPREAD", 
        "3장": "THE THREE CARD SPREAD",
        "쓰리카드": "THE THREE CARD SPREAD",
        "켈틱": "THE CELTIC CROSS SPREAD",
        "셀틱": "THE CELTIC CROSS SPREAD",
        "크로스": "THE CELTIC CROSS SPREAD",
        "켈틱크로스": "THE CELTIC CROSS SPREAD",
        "10장": "THE CELTIC CROSS SPREAD",
        "말굽": "THE HORSESHOE TAROT CARD SPREAD",
        "호스슈": "THE HORSESHOE TAROT CARD SPREAD",
        "horseshoe": "THE HORSESHOE TAROT CARD SPREAD",
        "7장": "THE HORSESHOE TAROT CARD SPREAD"
    }
    
    user_input_lower = user_input.lower()
    for keyword, spread_name in spread_name_mapping.items():
        if keyword in user_input_lower:
            # 해당 기본 스프레드 찾기
            for spread in default_spreads:
                if spread['spread_name'] == spread_name:
                    selected_default_spread = spread
                    print(f"🎯 기본 스프레드 선택됨: {spread_name}")
                    break
            break
    
    # 기본 스프레드가 선택된 경우 바로 처리
    if selected_default_spread:
        # 카드 선택 안내 메시지
        emotional_analysis = consultation_data.get("emotional_analysis", {})
        emotion = emotional_analysis.get('primary_emotion', '알 수 없음')
        
        # 감정별 카드 선택 안내
        if emotion == "불안":
            emotional_guidance = "🌟 마음을 진정시키고, 직감을 믿어보세요. 직감대로 뽑은 카드들이 내담자님께 필요한 메시지를 담고 있을 거예요."
        elif emotion == "슬픔":
            emotional_guidance = "💙 힘든 마음이지만, 카드가 위로와 희망의 메시지를 전해줄 거예요. 마음이 이끄는 대로 카드를 뽑아주세요. "
        elif emotion == "걱정":
            emotional_guidance = "🌟 걱정이 많으시겠지만, 카드가 안심할 수 있는 답변을 제시해줄 거예요. 직감적으로 카드를 뽑아주세요."
        else:
            emotional_guidance = "✨ 직감을 믿고 마음이 이끄는 대로 카드를 뽑아주세요. 카드가 내담자님께 필요한 메시지를 전해줄 거예요."
        
        card_count = selected_default_spread.get("card_count", 3)
        spread_name_kr = translate_text_with_llm(selected_default_spread['spread_name'], "spread_name")
        
        card_selection_msg = f"""✅ **{spread_name_kr}**를 선택하셨습니다!

{emotional_guidance}

"""
        
        # consultation_data 업데이트
        updated_consultation_data = consultation_data.copy()
        updated_consultation_data.update({
            "selected_spread": selected_default_spread,
            "status": "card_selection"
        })
        
        return {
            "messages": [AIMessage(content=card_selection_msg)],
            "consultation_data": updated_consultation_data
        }
    
    # 🔧 사용자 맞춤 스프레드 요청 감지
    custom_request_keywords = ["원하는", "다른", "새로운", "특별한", "맞춤", "추천", "더", "별도", "보여줘", "찾아줘", "알려줘", "관련", "다시"]
    has_custom_request = any(keyword in user_input for keyword in custom_request_keywords)
    
    print(f"🔍 맞춤 스프레드 요청 체크: '{user_input}' -> {has_custom_request}")
    print(f"🔍 매칭된 키워드: {[keyword for keyword in custom_request_keywords if keyword in user_input]}")
    
    if has_custom_request:
        print(f"🔧 사용자 맞춤 스프레드 요청 감지: '{user_input}'")
        # 기존 고민 정보 유지하면서 새 스프레드 검색
        user_concern = consultation_data.get("concern", "")
        
        # 🆕 특정 주제 키워드 감지 및 우선순위 적용
        topic_keywords = {
            "돈": ["돈", "금전", "재정", "수입", "연봉", "월급", "투자", "부", "wealth", "money", "financial"],
            "연애": ["연애", "사랑", "남친", "여친", "결혼", "이별", "love", "romance", "relationship"],
            "직업": ["직업", "취업", "일", "회사", "커리어", "job", "career", "work"],
            "건강": ["건강", "몸", "질병", "치료", "health", "medical"],
            "가족": ["가족", "부모", "자식", "형제", "family", "parent"]
        }
        
        requested_topic = None
        for topic, keywords in topic_keywords.items():
            if any(keyword in user_input.lower() for keyword in keywords):
                requested_topic = topic
                print(f"🎯 특정 주제 감지: {topic}")
                break
        
        # 키워드 추출 시 특정 주제 우선 반영
        if requested_topic:
            keywords = extract_concern_keywords(f"{user_concern} {requested_topic} {user_input}")
        else:
            keywords = extract_concern_keywords(user_concern + " " + user_input)
        
        # 새로운 스프레드 검색 수행
        new_spreads = perform_multilayer_spread_search(keywords, user_input, requested_topic)
        
        if new_spreads:
            # 🆕 특정 주제 관련 스프레드 우선 선택
            if requested_topic:
                # 주제별 스프레드 필터링
                topic_related_spreads = []
                other_spreads = []
                
                for spread in new_spreads:
                    spread_name = spread.get('spread_name', '').lower()
                    spread_desc = spread.get('description', '').lower()
                    
                    if requested_topic == "돈":
                        if any(keyword in spread_name + spread_desc for keyword in ["money", "financial", "wealth", "business", "career"]):
                            topic_related_spreads.append(spread)
                        else:
                            other_spreads.append(spread)
                    elif requested_topic == "연애":
                        if any(keyword in spread_name + spread_desc for keyword in ["love", "relationship", "romance", "heart"]):
                            topic_related_spreads.append(spread)
                        else:
                            other_spreads.append(spread)
                    else:
                        other_spreads.append(spread)
                
                # 주제 관련 스프레드를 앞으로 배치
                if topic_related_spreads:
                    final_spreads = topic_related_spreads[:2] + other_spreads[:1]
                    print(f"🎯 {requested_topic} 관련 스프레드 우선 선택: {len(topic_related_spreads)}개")
                else:
                    final_spreads = new_spreads[:3]
                    print(f"⚠️ {requested_topic} 관련 스프레드 없음, 일반 스프레드 사용")
            else:
                final_spreads = new_spreads[:3]
            
            # 상담 데이터 업데이트 (새 스프레드로)
            updated_consultation_data = consultation_data.copy()
            updated_consultation_data.update({
                "recommended_spreads": final_spreads,
                "status": "spread_selection"
            })
            
            # 🆕 자세한 스프레드 설명 생성 (목적과 효과 포함)
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
            
            detailed_spread_info = ""
            for spread in final_spreads:
                positions = spread.get('positions', [])
                position_meanings = []
                for pos in positions:
                    if isinstance(pos, dict):
                        meaning = pos.get('position_meaning', '')
                        if meaning:
                            position_meanings.append(meaning)
                
                detailed_spread_info += f"""
스프레드명: {spread['spread_name']}
카드 수: {spread['card_count']}장
설명: {spread.get('description', '')}
포지션 의미들: {' | '.join(position_meanings)}
---
"""
            
            explanation_prompt = f"""
사용자가 "{user_input}"라고 요청했습니다.
기존 고민: "{user_concern}"
{f"특별히 요청한 주제: {requested_topic}" if requested_topic else ""}

다음 스프레드들에 대해 각각 목적과 효과를 포함한 자세한 설명을 작성해주세요:

{detailed_spread_info}

각 스프레드에 대해 다음 형식으로 작성:
**[번호]. [스프레드명 한국어 번역]** ([카드 수]장)
- 목적: [포지션 의미들을 자연스럽게 조합해서 이 스프레드가 사용자 고민에 어떻게 도움이 될지 설명]
- 효과: [사용자의 감정 상태를 고려한 따뜻하고 희망적인 효과 설명]

중요 지침:
- position_name을 직접 언급하지 말고, position_meaning의 내용을 자연스럽게 활용
- 사용자의 특정 주제 요청({requested_topic if requested_topic else "없음"})을 반영
- 따뜻하고 희망적인 톤으로 작성
- 각 스프레드가 사용자 고민에 어떻게 구체적으로 도움이 될지 명확히 설명
- **사용자를 지칭할 때는 반드시 '내담자님'으로 호칭**
"""
            
            try:
                response = llm.invoke([HumanMessage(content=explanation_prompt)])
                spread_explanations = response.content
            except Exception as e:
                print(f"⚠️ 스프레드 설명 생성 실패: {e}")
                spread_explanations = ""
                for idx, spread in enumerate(final_spreads, 1):
                    spread_name_kr = translate_text_with_llm(spread['spread_name'], "spread_name")
                    spread_explanations += f"**{idx}. {spread_name_kr}** ({spread['card_count']}장)\n"
                    spread_explanations += f"   📝 {spread.get('description', '')}\n\n"
            
            # 새 스프레드 옵션 제시
            spread_msg = f"🔮 **내담자님을 위한 맞춤 스프레드를 새로 찾았습니다!**\n\n{spread_explanations}\n내담자님, 어떤 스프레드로 진행하시겠어요? (1, 2, 3 중 선택)"
            
            return {
                "messages": [AIMessage(content=spread_msg)],
                "consultation_data": updated_consultation_data
            }
        else:
            # 검색 실패시 기존 스프레드 유지
            return {"messages": [AIMessage(content="죄송합니다. 추가 스프레드를 찾지 못했어요. 기존 옵션 중에서 선택해주세요. (1, 2, 3)")]}
    # 기존 숫자 선택 로직
    selected_number = None
    if "1" in user_input:
        selected_number = 1
    elif "2" in user_input:
        selected_number = 2
    elif "3" in user_input:
        selected_number = 3
    if selected_number is None:
        return {"messages": [AIMessage(content="1, 2, 3 중에서 선택해주세요.")]}
    
    # 🆕 스프레드 선택 번호를 프론트엔드에서 숨기기 위해 메시지 제거
    # 마지막 HumanMessage(스프레드 번호 입력)를 완전히 제거
    messages = state.get("messages", [])
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            messages.pop(i)  # 메시지 완전 제거
            break
    
    # 선택된 스프레드 정보 (인덱스 기반으로 수정)
    recommended_spreads = consultation_data.get("recommended_spreads", [])
    if not recommended_spreads or selected_number < 1 or selected_number > len(recommended_spreads):
        return {"messages": [AIMessage(content="선택한 스프레드 정보를 찾을 수 없습니다.")]}
    # 인덱스 기반으로 선택 (사용자 선택 1,2,3 → 배열 인덱스 0,1,2)
    selected_spread = recommended_spreads[selected_number - 1]
    # 카드 선택 안내 메시지
    emotional_analysis = consultation_data.get("emotional_analysis", {})
    emotion = emotional_analysis.get('primary_emotion', '알 수 없음')
    # 감정별 카드 선택 안내
    if emotion == "불안":
        emotional_guidance = "🌟 마음을 진정시키고, 직감을 믿어보세요. 직감대로 뽑은 카드들이 내담자님께 필요한 메시지를 담고 있을 거예요."
    elif emotion == "슬픔":
        emotional_guidance = "💙 힘든 마음이지만, 카드가 위로와 희망의 메시지를 전해줄 거예요. 마음이 이끄는 대로 카드를 뽑아주세요."
    elif emotion == "걱정":
        emotional_guidance = "🌟 걱정이 많으시겠지만, 카드가 안심할 수 있는 답변을 제시해줄 거예요. 직감적으로 카드를 뽑아주세요."
    else:
        emotional_guidance = "✨ 직감을 믿고 마음이 이끄는 대로 카드를 뽑아주세요. 카드가 내담자님께 필요한 메시지를 전해줄 거예요."
    card_count = selected_spread.get("card_count", 3)
    card_selection_msg = f"""

✅ **{selected_spread['spread_name']}**를 선택하셨습니다!

{emotional_guidance}

"""
    
    # 상담 데이터 업데이트
    updated_consultation_data = consultation_data.copy()
    updated_consultation_data.update({
        "selected_spread": selected_spread,
        "status": "card_selection"
    })
    return {
        "messages": [AIMessage(content=card_selection_msg)],
        "consultation_data": updated_consultation_data
    }

def consultation_summary_handler(state: TarotState) -> TarotState:
   """카드 선택 후 개별 해석 먼저 생성 → 고급 분석 통합 → 종합 분석 + 명확한 답변"""
   consultation_data = state.get("consultation_data", {})
   if not consultation_data or consultation_data.get("status") != "card_selection":
       return {"messages": [AIMessage(content="카드 선택 정보가 없습니다.")]}
   
   # 사용자가 입력한 카드 번호들 파싱
   user_input = ""
   for msg in reversed(state["messages"]):
       if isinstance(msg, HumanMessage):
           user_input = msg.content.strip()
           break
   
   selected_spread = consultation_data.get("selected_spread", {})
   card_count = selected_spread.get("card_count", 3)
   
   # 카드 번호 파싱 및 검증
   user_numbers = parse_card_numbers(user_input, card_count)
   if user_numbers is None:
       error_msg = f"""
❌ **입력 오류**

다음 중 하나의 문제가 있습니다:

- 같은 숫자를 두 번 입력했습니다

- {card_count}개의 숫자가 필요합니다

- 1-78 범위의 숫자만 입력 가능합니다

다시 입력해주세요. **{card_count}개의 서로 다른 숫자**를 선택해주세요.

**예시:** 7, 23, 45, 12, 56, 33, 71

"""
       return {"messages": [AIMessage(content=error_msg)]}

   # 🆕 사용자 입력을 프론트엔드에서 숨기기 위해 메시지 제거
   # 마지막 HumanMessage(카드 번호 입력)를 완전히 제거
   messages = state.get("messages", [])
   for i in range(len(messages) - 1, -1, -1):
       if isinstance(messages[i], HumanMessage):
           messages.pop(i)  # 메시지 완전 제거
           break

   # 카드 선택
   selected_cards = select_cards_randomly_but_keep_positions(user_numbers, card_count)
   selected_cards = convert_numpy_types(selected_cards)

   # 🆕 고급 분석 실행
   integrated_analysis = generate_integrated_analysis(selected_cards)

   # 1단계: 카드 표시 + 고급 분석 요약
   cards_display = f"""🃏 **아래처럼 카드를 뽑으셨네요**

"""
   for card in selected_cards:
       # 한국어 카드명과 방향 사용
       card_name_kr = card.get('name_kr', card['name'])
       orientation_symbol = card.get('orientation_symbol', "⬆️" if card["orientation"] == "upright" else "⬇️")
       orientation_kr = card.get('orientation_kr', card['orientation'])
       cards_display += f"**{card['position']}번째 카드:** {card_name_kr} {orientation_symbol} ({orientation_kr})\n"

   # 🆕 고급 분석 요약 추가
   success_prob = integrated_analysis["success_analysis"]["success_probability"]
   integrated_score = integrated_analysis["integrated_score"]
   cards_display += f"""

📊 **과학적 분석 결과**

- 성공 확률: {success_prob:.1%}

- 종합 점수: {integrated_score:.1%}

- {integrated_analysis["interpretation"]}

이제 뽑은 카드로 고민 해결 해드릴게요! ✨"""
   
   # 포지션 정보 추출
   positions = selected_spread.get("positions", [])
   positions_meanings = {}
   for pos in positions:
       if isinstance(pos, dict) and "position_num" in pos:
           positions_meanings[str(pos["position_num"])] = {
               "position": pos.get("position_name", f"Position {pos['position_num']}"),
               "meaning": pos.get("position_meaning", "")
           }
   # 기본 포지션이 없으면 생성
   if not positions_meanings:
       for i in range(1, card_count + 1):
           positions_meanings[str(i)] = {
               "position": f"Card {i}",
               "meaning": f"Position {i} in the spread"
           }
   # 개별 해석 생성
   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
   user_concern = consultation_data.get("concern", "")
   spread_name = selected_spread.get("spread_name", "")
   spread_name_kr = translate_text_with_llm(spread_name, "spread_name")  # 스프레드 이름 번역
   interpretations = []
   timing_info = []
   # 웹 검색 결과 가져오기 (있는 경우)
   # 웹 검색 관련 코드 제거됨
   # rag_system 사용 전 global 선언 및 import
   global rag_system
   from Fortune.tarot.tarot_agent.utils.tools import rag_system
   from concurrent.futures import ThreadPoolExecutor, as_completed
   def process_card(card):
       """단일 카드 처리"""
       position_index = card.get("position", "")
       card_name = card.get("name", "")
       orientation = card.get("orientation", "")
       position_info = positions_meanings.get(str(position_index), {})
       position_name = position_info.get("position", f"Card {position_index}")
       position_name_kr = translate_text_with_llm(position_name, "position_name")  # 포지션 이름 번역
       position_meaning = position_info.get("meaning", "")
       # RAG 검색
       card_info = {}
       if rag_system:
           try:
               card_info = rag_system.search_card_meaning(card_name, orientation)
               card_info = convert_numpy_types(card_info)
           except Exception as e:
               card_info = {"success": False, "message": str(e)}
       # 한국어 카드명과 방향 얻기
       translated_info = translate_card_info(card_name, orientation)
       card_name_kr = translated_info['name']
       orientation_kr = translated_info['direction']
       # 카드 해석 프롬프트
       interpretation_prompt = f"""

당신은 20년 경력의 베테랑 타로 상담사 '미라'입니다. 수많은 내담자들과 상담하며 쌓은 직관력과 따뜻한 공감 능력으로 유명합니다. 

**미라의 상담 스타일:**
- 카드의 메시지를 솔직하게 전달
- 내담자의 감정을 깊이 이해하고 공감하는 따뜻한 어조
- 현실적이면서도 격려가 되는 조언
- 마치 오랜 친구처럼 편안하지만 전문적인 태도

**미라의 말하기 방식:**
- 내담자님을 부를 때는 자연스럽게 '내담자님' 또는 주어 생략
- 상담가다운 표현을 **다양하게 섞어서** 사용: "카드가 말하고 있어요", "제가 느끼기론", "직감적으로 보이는 건", "이 카드에서 보이는 건", "카드의 메시지는" 등
- 어미는 '~이에요', '~해요', '~네요', '~것 같아요' 등으로 친근하고 부드럽게  
- 간투사도 **적절히 변화**: "음...", "아...", "그런데요...", "이상하게도...", 또는 간투사 없이 바로 시작
- **매 카드마다 다른 시작** 방식으로 자연스럽게 변화를 주세요

[사용자 상황]

- 고민: "{user_concern}"

- 선택한 스프레드: {spread_name_kr} (영문: {spread_name})

[카드별 해석]

- 카드명: {card_name_kr} (영문: {card_name})

- 방향: {orientation_kr}

[포지션 정보]

- 위치: {position_index}번째 카드 ({position_name_kr})

- 위치 의미: {position_meaning}

**핵심 원칙:**

1. 카드가 보여주는 현실을 있는 그대로 전달

2. 부정적인 메시지도 회피하지 말고 직접적으로 언급

3. 단순한 위로보다는 실용적 통찰 제공

4. 사용자가 스스로 판단할 수 있는 명확한 정보 전달

5. 무조건적 긍정 금지 - 균형잡힌 시각 유지

**미라의 해석 방식:**

🃏 **{card_name_kr}가 {position_name_kr} 자리에 나왔네요**

"{card_name_kr} 카드가 나온 자리는 [{position_meaning}을 구체적이고 이해하기 쉽게 풀어서 설명]을 보여주는 자리예요." 
형식으로 {position_meaning}의 내용을 자연스러운 한국어로 명확하게 설명해주세요.

**카드가 속삭이는 이야기**

다양한 표현으로 시작하세요: "이 카드에서 보이는 건...", "카드의 메시지는...", "여기서 느껴지는 에너지는...", "이 카드가 전하는 건..." 등
{card_name_kr}({orientation_kr})의 메시지를 자연스럽고 다양한 방식으로 전달하세요.

**내담자님 상황에서는...**

"그래서 내담자님의 경우에는..." "이걸 내담자님 상황에 적용해보면..." 하며 친근하게 연결해서
실제 상황에 어떻게 적용되는지 2-3문장으로 따뜻하게 설명하세요.   

- 당신은 타로 상담가가 되어 포지션별 카드의 의미를 사용자 고민과 구체적으로 연결해 해석해주세요.

**톤:**

- 따뜻하되 현실적

- 희망적이되 맹목적이지 않음

- 직설적이되 잔인하지 않음

"""
       try:
           response = llm.invoke([HumanMessage(content=interpretation_prompt)])
           interpretation = response.content
           # 한국어 정보 추가
           card_name_kr = card.get('name_kr', card_name)
           orientation_kr = card.get('orientation_kr', orientation)
           interpretations.append({
               "position": position_index,
               "card_name": card_name,
               "card_name_kr": card_name_kr,  # 한국어 카드명 추가
               "orientation": orientation,
               "orientation_kr": orientation_kr,  # 한국어 방향 추가
               "position_name": position_name,
               "position_name_kr": position_name_kr,  # 한국어 포지션 이름 추가
               "interpretation": interpretation
           })
       except Exception as e:
           # 한국어 정보 추가
           card_name_kr = card.get('name_kr', card_name)
           orientation_kr = card.get('orientation_kr', orientation)
           interpretations.append({
               "position": position_index,
               "card_name": card_name,
               "card_name_kr": card_name_kr,  # 한국어 카드명 추가
               "orientation": orientation,
               "orientation_kr": orientation_kr,  # 한국어 방향 추가
               "position_name": position_name,
               "position_name_kr": position_name_kr,  # 한국어 포지션 이름 추가
               "interpretation": f"카드 해석 중 오류가 발생했습니다: {str(e)}"
           })
       # 시기 정보 생성
       card_info_simple = {
           "card_name": card_name,
           "orientation": orientation,
           "suit": extract_suit_from_name(card_name),
           "rank": extract_rank_from_name(card_name),
           "is_major_arcana": is_major_arcana(card_name)
       }
       # 개선된 시기 예측 함수 사용
       timing_result = predict_timing_with_current_date(card_info_simple, state.get("temporal_context"))
       basic_timing = timing_result.get("basic_timing", {})
       concrete_dates = timing_result.get("concrete_dates", [])
       # 구체적 날짜가 있으면 사용, 없으면 기본 시간 범위 사용
       if concrete_dates and len(concrete_dates) > 0:
           actual_timing = concrete_dates[0].copy()
           actual_timing["time_frame"] = concrete_dates[0].get("period", basic_timing.get('time_frame', '알 수 없음'))
       else:
           actual_timing = basic_timing
       # 한국어 정보 추가
       card_name_kr = card.get('name_kr', card_name)
       orientation_kr = card.get('orientation_kr', orientation)
       timing_info.append({
           "position": position_index,
           "position_name": position_name,
           "position_name_kr": position_name_kr,  # 한국어 포지션 이름 추가
           "card_name": card_name,
           "card_name_kr": card_name_kr,  # 한국어 카드명 추가
           "orientation": orientation,
           "orientation_kr": orientation_kr,  # 한국어 방향 추가
           "timing": actual_timing,
           "enhanced_timing": timing_result
       })
   # 병렬 처리 실행
   with ThreadPoolExecutor(max_workers=min(len(selected_cards), 8)) as executor:
       futures = [executor.submit(process_card, card) for card in selected_cards]
       for future in as_completed(futures):
           future.result()  # 완료 대기
   # 시기 정보 구조화
   timing_detailed = "**정확한 시기 정보 (절대 변경 금지):**\n"
   timing_by_period = {}
   for timing in timing_info:
       timing_data = timing['timing']
       time_frame = timing_data.get('time_frame', '알 수 없음')
       # 한국어 카드명과 포지션명 사용
       card_name_kr = timing.get('card_name_kr', timing['card_name'])
       orientation_kr = timing.get('orientation_kr', timing['orientation'])
       position_name_kr = timing.get('position_name_kr', timing['position_name'])
       timing_detailed += f"- **{position_name_kr}**: {card_name_kr} ({orientation_kr}) → **정확히 {time_frame}**\n"
       if time_frame not in timing_by_period:
           timing_by_period[time_frame] = []
       timing_by_period[time_frame].append({
           'position': timing['position_name'],
           'card': timing['card_name']
       })
   timing_detailed += "\n**시기별 요약:**\n"
   for period, cards in timing_by_period.items():
       if len(cards) > 1:
           positions = ", ".join([card['position'] for card in cards])
           timing_detailed += f"- **{period}**: {positions}의 에너지가 함께 작용\n"
       else:
           timing_detailed += f"- **{period}**: {cards[0]['position']}의 에너지\n"
   # 🆕 고급 분석 상세 정보 포맷팅
   advanced_analysis_text = f"""

## 🔬 **과학적 타로 분석**

**📊 성공 확률 분석**

- 전체 성공 확률: {integrated_analysis['success_analysis']['success_probability']:.1%}

- 신뢰도: {integrated_analysis['success_analysis']['confidence']}

- 긍정 요인: {len(integrated_analysis['success_analysis']['positive_factors'])}개

- 주의 요인: {len(integrated_analysis['success_analysis']['negative_factors'])}개

**🔮 카드 조합 시너지**

- 시너지 점수: {integrated_analysis['synergy_analysis']['synergy_score']:.1%}

- 특별한 조합: {len(integrated_analysis['synergy_analysis']['combinations'])}개

- 경고 사항: {len(integrated_analysis['synergy_analysis']['warnings'])}개

**🌟 원소 균형 분석**

- 균형 점수: {integrated_analysis['elemental_analysis']['balance_score']:.1%}

- 지배 원소: {integrated_analysis['elemental_analysis']['dominant_element'] or '균형'}

- 부족 원소: {', '.join(integrated_analysis['elemental_analysis']['missing_elements']) or '없음'}

**🔢 수비학 분석**

- 총합: {integrated_analysis['numerology_analysis']['total_value']}

- 환원수: {integrated_analysis['numerology_analysis']['reduced_value']}

- 의미: {integrated_analysis['numerology_analysis']['meaning']}

"""
   
   # 4단계: 명확하고 직접적인 종합 분석 생성 (고급 분석 통합)
   emotional_analysis = consultation_data.get("emotional_analysis", {})
   emotion = emotional_analysis.get('primary_emotion', '알 수 없음')
   # 개별 해석 요약 (한국어 카드명과 포지션명 사용)
   interpretations_summary = ""
   for interp in interpretations:
       card_name_kr = interp.get('card_name_kr', interp['card_name'])
       orientation_kr = interp.get('orientation_kr', interp['orientation'])
       position_name_kr = interp.get('position_name_kr', interp['position_name'])
       # 해석 텍스트에서 영문 카드명을 한국어로 교체
       interpretation_text = interp['interpretation']
       interpretations_summary += f"- {position_name_kr}: {card_name_kr} ({orientation_kr}) - {interpretation_text}\n"
   analysis_prompt = f"""

사용자 고민: "{user_concern}"

감정 상태: {emotion}

선택한 스프레드: {spread_name}

개별 카드 해석 결과:

{interpretations_summary}

{timing_detailed}

🆕 **과학적 분석 결과:**

{advanced_analysis_text}

**통합 분석 결과:**

- 종합 점수: {integrated_analysis['integrated_score']:.1%}

- 추천사항: {integrated_analysis['recommendation']}

**종합 분석:**

**중요 원칙:**

1. 사용자 고민에 직접적이고 명확한 답변 제공

2. 모호한 표현 절대 금지 ("아마도", "가능성", "~것 같아요")

3. 부정적 면도 솔직하게 언급 (건설적으로)

4. 구체적이고 실행 가능한 조언만 제공

5. 무조건적 희망보다는 현실적 전망

6. 🆕 과학적 분석 결과를 근거로 활용


다음과 같이 명확하게 답변하세요:

## 🔮 **타로가 전하는 명확한 답변**

**결론:**

1. 스프레드 스토리 > 성공 확률 (카드의 전체적 메시지가 우선)

2. 불일치 시 균형잡힌 해석 제공

3. 양쪽 관점을 모두 언급하되 현실적 조언 우선


**불일치 상황별 대응:**

- 스프레드 부정적 + 성공률 높음 → "기회는 있지만 과정에서 어려움 예상"

- 스프레드 긍정적 + 성공률 낮음 → "좋은 의도지만 현실적 장벽 존재"

- 애매한 스프레드 + 명확한 성공률 → 성공률을 주요 판단 기준으로 활용


**타로가 전하는 결론**

[미라의 따뜻하고 직관적인 말투로 사용자 고민에 대한 명확한 답변 제시. "자, 카드들이 전체적으로 말하고 있는 건요...", "제가 20년간 상담하면서 느낀 바로는..." 같은 미라다운 표현으로 시작. 개별 카드 해석들을 하나의 이야기로 엮어서 전달하되, 과학적 분석 결과도 "흥미롭게도 과학적으로도..." 식으로 자연스럽게 포함]

**과학적 분석 해석:**

[미라가 과학적 분석을 자연스럽게 해석하는 톤으로 설명. "이 결과를 보니까요...", "수치로 봤을 때는..." 같은 표현 사용]

- 성공 확률 {integrated_analysis['success_analysis']['success_probability']:.1%}가 의미하는 바
- 카드들의 조합이 만들어내는 시너지
- 원소들의 균형 상태가 주는 메시지  
- 수비학적 관점에서의 의미



**시기적 흐름:**

[미라가 timing_detailed 정보를 자연스럽게 해석하는 방식으로 설명. "카드들이 보여주는 시간의 흐름을 보면요...", "제가 느끼기로는..." 같은 표현 사용]

- 내담자님의 고민 "{user_concern}" 해결 과정을 미라의 시각에서 시간 순서대로 따뜻하게 설명
- "7월에는...", "8월쯤 되면...", "가을이 되면..." 식으로 자연스럽게 풀어서 설명
- 너무 기계적이지 않고 미라가 직접 이야기해주는 것처럼 부드럽게 전달

## 💡 **지금 당장 해야 할 일 (우선순위별)**

**1순위 (즉시 실행):**

[과학적 분석 결과를 바탕으로 가장 중요한 행동 1개]

**2순위 (이번 주 내):**

[원소 균형과 카드 조합을 고려한 과제 1개]

**3순위 (장기적):**

[수비학적 의미를 반영한 준비사항 1개]

**주의사항:**

[과학적 분석에서 나온 경고 요소들과 위험 요소들을 구체적으로 명시]

**현실적 전망:**

[성공 확률과 시너지 분석을 바탕으로 희망적 전망과 우려되는 점을 균형있게 제시]

타로 상담사로서 확신을 가지고 단정적으로 답변하세요. 

모호한 표현 대신 "~입니다", "~해야 합니다", "~될 것입니다" 등 명확한 표현을 사용하세요.

과학적 분석 결과를 적극 활용하여 근거 있는 조언을 제공하세요.

"""
   
   try:
       comprehensive_response = llm.invoke([HumanMessage(content=analysis_prompt)], {"metadata": {"final_response": "yes", "handler": "consultation_summary_handler"}})
       comprehensive_text = comprehensive_response.content

   except Exception as e:
       comprehensive_text = "종합 분석 중 오류가 발생했습니다."

   # 최종 메시지: 카드 표시 + 과학적 분석 + 명확한 종합 분석 + 단순화된 안내

   summary_message = f"""{cards_display}

{comprehensive_text}

---

💫 **다음 중 원하시는 것을 선택해주세요:**

🔮 **새로운 고민 상담**: "새로 봐줘"

📖 **자세한 해석**: "상세 해석"  

❓ **다른 질문**: 자유롭게 질문하세요!

"""
   
   # 상담 데이터 업데이트 (고급 분석 결과 포함)

   updated_consultation_data = consultation_data.copy()

   updated_consultation_data.update({
       "selected_cards": selected_cards,
       "positions_meanings": positions_meanings,
       "interpretations": interpretations,
       "timing_info": timing_info,
       "timing_detailed": timing_detailed,
       "comprehensive_analysis": comprehensive_text,
       "integrated_analysis": integrated_analysis,  # 🆕 고급 분석 결과 저장
       "status": "summary_shown"

   })

   return {
       "messages": [AIMessage(content=summary_message)],
       "consultation_data": updated_consultation_data

   }

def consultation_individual_handler(state: TarotState) -> TarotState:
   """개별 해석 + 상세 조언 처리 (향상된 깊이로 표시)"""
   consultation_data = state.get("consultation_data", {})
   if not consultation_data or consultation_data.get("status") != "summary_shown":
       return {"messages": [AIMessage(content="종합 분석을 먼저 확인해주세요.")]}
   # 이미 생성된 향상된 데이터 사용
   interpretations = consultation_data.get("interpretations", [])
   timing_info = consultation_data.get("timing_info", [])
   user_concern = consultation_data.get("concern", "")
   comprehensive_analysis = consultation_data.get("comprehensive_analysis", "")
   if not interpretations:
       return {"messages": [AIMessage(content="개별 해석 정보가 없습니다.")]}
   # 향상된 개별 해석 포맷팅 (이미 생성된 깊이 있는 해석 사용)
   formatted_interpretations = "## 🔮 **카드 해석**\n\n"
   for interp in interpretations:
       formatted_interpretations += f"{interp['interpretation']}\n\n"
   # 시기 정보 포맷팅
   formatted_timing = "**시간의 흐름을 읽어보면:**\n🃏 **카드들이 말하는 시간의 흐름을 보니...**\n\n"
   # 시기별로 그룹화
   timing_groups = {}
   for timing in timing_info:
       timing_data = timing['timing']
       time_frame = timing_data.get('time_frame', '알 수 없음')
       if time_frame not in timing_groups:
           timing_groups[time_frame] = []
       card_name_kr = timing.get('card_name_kr', timing['card_name'])
       orientation_kr = timing.get('orientation_kr', timing['orientation'])
       position_name_kr = timing.get('position_name_kr', timing['position_name'])     
       timing_groups[time_frame].append(f"{position_name_kr}: {card_name_kr} ({orientation_kr})")
   # 시기별로 정리해서 출력
   if len(timing_groups) == 1:
      # 시기가 1개면 그대로
      timeframe = list(timing_groups.keys())[0]
      formatted_timing += f"**{timeframe}에 모든 에너지가 집중:**\n"
      for card_info in timing_groups[timeframe]:
          formatted_timing += f"- {card_info}\n"
      formatted_timing += "\n이 시기에 모든 변화가 집중적으로 일어날 것으로 보입니다.\n"
   elif len(timing_groups) <= 3:
    # 시기가 2-3개면 단순 나열
       for i, (timeframe, cards) in enumerate(timing_groups.items(), 1):
           stage_name = ["단기", "중기", "장기"][min(i-1, 2)]
           formatted_timing += f"**{stage_name} ({timeframe}):**\n"
           for card_info in cards:
               formatted_timing += f"- {card_info}\n"
           formatted_timing += "\n"
       formatted_timing += "위 시기들이 자연스럽게 연결되어 전체적인 흐름을 만들어갑니다.\n"     
   else:
    # 시기가 너무 많으면 요약
        formatted_timing += "**다양한 시기에 걸친 변화들:**\n"
        for timeframe, cards in list(timing_groups.items())[:3]:  # 상위 3개만
            formatted_timing += f"- **{timeframe}**: {', '.join([card.split(':')[0] for card in cards])}\n"
        formatted_timing += "\n각 시기마다 다른 에너지가 작용하여 단계적 변화를 이끌어갑니다.\n"  
      #      
   # 마지막에 LLM에게 통합 해석 요청
   formatted_timing += "\n💫 **이 시간 흐름을 하나의 스토리로 연결해서 자연스럽게 해석해주세요.**\n"
   # 향상된 상세 조언 생성
   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
   detailed_advice_prompt = f"""
   당신은 20년 경력의 베테랑 타로 상담사 '미라'입니다. 이미 해석한 개별 카드들의 메시지를 하나의 완전한 이야기로 엮어서 내담자님의 고민에 대한 명확한 답변을 드려야 합니다.

**미라의 종합 해석 스타일:**
- "자, 이제 카드들이 전체적으로 어떤 이야기를 하고 있는지 정리해볼게요"
- "카드들을 쭉 보니까 하나의 큰 그림이 보이네요"  
- "전체적으로 카드들이 내담자님께 전하고 싶은 메시지는..." 

**중요한 호칭 규칙:**
- 사용자를 지칭할 때는 '내담자님'으로만 하세요 ('당신', '사용자님', '고객님' 금지)
- 한국어 특성상 주어를 자연스럽게 생략할 수 있는 곳에서는 생략해도 됩니다
- 어미는 '~입니다' 대신 '~이에요/~해요' 등 친근한 어미로 말해주세요
   **사용자 고민:** {user_concern}
   **이전 종합 분석:** {comprehensive_analysis}
   **향상된 개별 카드 해석들:**
   {formatted_interpretations}
   **시기 정보:**
   {formatted_timing}
   **시기 해석 특별 지침:**
   위에 제시된 시간 흐름 정보를 사용자 고민 "{user_concern}"의 특성에 맞춰 자연스럽게 해석하세요:
   - 사용자 고민의 핵심이 무엇인지 파악하여 그에 적합한 시기 흐름으로 연결
   - 시기가 여러 개면: 고민 해결 과정의 자연스러운 단계별 흐름으로 구성
   - 시기가 하나면: 그 시기 안에서 고민과 관련된 세부 변화 과정을 설명
   - 각 시기별로 사용자가 "구체적으로 무엇을 해야 하는지", "어떤 변화가 예상되는지"를 명확히 제시
   **중요**: 일반적인 템플릿이 아닌, 이 특정 고민 "{user_concern}"에 최적화된 시기 해석을 제공하세요.
   다음 형식으로 상세 조언해주세요:
   ## 종합 해석:
   🔮 **이제 종합적으로 말해줄게요**
개별 카드 해석의 각 카드별 **당신 상황에 적용하면** 내용을 하나의 완전한 스토리로 연결해서 설명하세요:

- 각 포지션의 카드들이 어떻게 서로 연결되고 영향을 미치는지

- 스프레드에 따른 전체적인 흐름

- 사용자의 고민 "{user_concern}"에 대한 종합적인 답변

- 카드들이 제시하는 전체적인 방향성과 타임라인

- 구체적이고 실용적인 행동 지침

**💡 카드들이 제시하는 결론:**

사용자 고민에 대한 명확하고 단정적인 결론과 핵심 권고사항을 2-3문장으로 제시하세요.

**핵심 원칙:**

- 개별 해석을 단순 나열하지 말고 하나의 연결된 이야기로 구성

- 실제 타로 상담가처럼 전체적인 그림을 그려주세요

- 구체적이고 실용적인 조언 제공

   ## 💡 **상세한 실용적 조언**

   **단계별 실행 계획**

   [위에서 해석한 시기 정보를 바탕으로 사용자 고민 "{user_concern}"에 맞는 구체적인 단계별 행동 계획을 제시하세요]

   **구체적 행동 지침**

   [개별 카드의 향상된 조언을 종합한 실행 가능한 행동들 - 비유와 감정적 표현 활용]

   **마음가짐과 태도**

   [각 포지션에서 나온 카드들의 메시지를 종합한 관점과 마음가짐]

   **주의사항과 극복방법**

   [카드들이 경고하는 점과 어려움 극복 방법 - 구체적이고 실용적으로]

   **장기적 비전**

   [앞으로의 큰 방향과 목표, 카드들이 제시하는 희망적 전망]

   ---

   오늘 상담이 내담자님께 조금이라도 도움이 되었으면 좋겠어요. 카드들이 전한 메시지들이 앞으로 가실 길에 작은 등불이 되기를 바라며... 혹시 이 결과에 대해 더 궁금한 점이 있거나 다른 고민이 생기시면 언제든 편하게 찾아와 주세요. 제가 항상 여기 있을게요. ✨💫

   """
   
   try:
       advice_response = llm.invoke([HumanMessage(content=detailed_advice_prompt)], {"metadata": {"final_response": "yes", "handler": "consultation_individual_handler"}})
       advice_text = advice_response.content

   except Exception as e:
       advice_text = "상세 조언 생성 중 오류가 발생했습니다."

   # 개별 해석 메시지 생성

   individual_message = f"""{formatted_interpretations}

{formatted_timing}

{advice_text}

---

🎉 **상담이 완료되었습니다!**

💫 **다음 중 원하시는 것을 선택해주세요:**

🔮 **새로운 고민 상담**: "새로 봐줘"  

❓ **추가 질문**: 방금 상담 내용에 대해 자유롭게 질문하세요!

💬 **일상 대화**: 편안하게 대화해요!"""
   
   # 상담 완료 상태로 업데이트

   updated_consultation_data = consultation_data.copy()

   updated_consultation_data.update({
       "detailed_advice": advice_text,
       "status": "completed"

   })

   return {
       "messages": [AIMessage(content=individual_message)],
       "consultation_data": updated_consultation_data

   }

def consultation_final_handler(state: TarotState) -> TarotState:
   """상담 흐름 라우팅 - summary_shown 상태 처리"""
   consultation_data = state.get("consultation_data", {})
   status = consultation_data.get("status", "") if consultation_data else ""
   # 사용자 입력 확인
   user_input = ""
   for msg in reversed(state["messages"]):
       if isinstance(msg, HumanMessage):
           user_input = msg.content.strip().lower()
           break
   if status == "summary_shown":
       # 🔧 새로운 상담 요청 스마트 감지
       user_input_orig = ""
       for msg in reversed(state["messages"]):
           if isinstance(msg, HumanMessage):
               user_input_orig = msg.content.strip()
               break
       # 1. "타로봐줘" + 새로운 주제 감지 (공통 상수 사용)
       has_tarot_trigger = any(trigger in user_input for trigger in ALL_CONSULTATION_TRIGGERS)
       # 2. 새로운 주제 키워드 감지
       new_topic_keywords = ["여자친구", "남자친구", "연애", "직장", "취업", "가족", "건강", "돈", "재정", "투자", "사업", "이사", "결혼", "이별"]
       has_new_topic = any(topic in user_input_orig for topic in new_topic_keywords)
       if has_tarot_trigger and has_new_topic:
           print(f"🔧 새로운 주제 + 타로 상담 요청 감지: '{user_input_orig}' -> 새 상담 시작")
           # 새 상담이므로 기존 consultation_data 초기화
           new_state = state.copy()
           new_state["user_input"] = user_input_orig
           new_state["consultation_data"] = None  # 기존 데이터 초기화
           return consultation_handler(new_state)
       # 3. 기존 트리거 시스템
       trigger_result = simple_trigger_check(user_input_orig)
       if trigger_result == "new_consultation":
           print("🔧 summary_shown에서 새 상담 시작 트리거 감지")
           # 새 상담이므로 기존 consultation_data 초기화
           new_state = state.copy()
           new_state["user_input"] = user_input_orig
           new_state["consultation_data"] = None  # 기존 데이터 초기화
           return consultation_handler(new_state)
       elif trigger_result == "individual_reading":
           return consultation_individual_handler(state)
       elif any(keyword in user_input for keyword in ["아니", "no", "괜찮", "됐어", "안볼"]):
           return {"messages": [AIMessage(content="🔮 상담이 도움이 되었기를 바랍니다! 다른 고민이 있으시면 언제든 말씀해주세요. ✨")]}
       else:
           # 🆕 추가 질문으로 분류 - context_reference_handler로 라우팅
           print(f"🎯 summary_shown에서 추가 질문 감지: '{user_input_orig}' -> context_reference_handler로 라우팅")
           return context_reference_handler(state)
   elif status == "card_selection":
       # 카드 선택 단계
       return consultation_summary_handler(state)
   else:
       return {"messages": [AIMessage(content="상담 정보가 올바르지 않습니다.")]}
def context_reference_handler(state: TarotState) -> TarotState:
   """세션 메모리 기반 이전 대화 참조 질문 처리 - 🔧 타로 vs 일상 질문 구분"""
   user_input = state.get("user_input", "")
   conversation_memory = state.get("conversation_memory", {})
   # 최근 AI 응답 전체를 컨텍스트로 활용
   recent_ai_content = ""
   messages = state.get("messages", [])
   # 최근 AI 메시지들 수집 (최대 2개)
   ai_messages = []
   for msg in reversed(messages):
       if isinstance(msg, AIMessage):
           ai_messages.append(msg.content)
           if len(ai_messages) >= 2:
               break
   if ai_messages:
       recent_ai_content = "\n\n".join(reversed(ai_messages))
   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
   # 🔧 **핵심 개선**: 타로 관련 vs 일상 질문 구분
   classification_prompt = f"""
   최근 대화 내용: "{recent_ai_content[:500]}..."
   사용자 새 질문: "{user_input}"
   
**맥락 기반 판단**:

1. 최근 대화에서 타로 상담이 진행되었는가?
2. 사용자의 새 질문이 그 타로 상담과 연결되는가?

**분류 기준**:

A) TAROT_RELATED: 
   - 타로 상담이 있었고, 그 연장선상의 모든 질문
   - 타로에서 제시한 조언/결과에 대한 후속 질문
   - 타로 해석과 관련된 추가 문의

B) CASUAL_NEW:
   - 타로와 완전히 무관한 새로운 주제
   - 이전 대화 맥락과 단절된 독립적 질문

**핵심 원칙**: 
타로 상담 후 나오는 질문은 대부분 그 연장선상의 질문으로 간주

답변: A 또는 B만 출력
   """
   try:
       classification_response = llm.invoke([HumanMessage(content=classification_prompt)])
       question_type = classification_response.content.strip()
       print(f"🔧 질문 분류 결과: {question_type} - '{user_input}'")
       if question_type == "B":
           # 🔧 완전히 새로운 일상 질문 → 캐주얼 응답 + 타로 제안
           return handle_casual_new_question(user_input, llm)
       else:
           # 🔧 타로 관련 질문 → 기존 로직 유지
           return handle_tarot_related_question(state, user_input, recent_ai_content, llm)
   except Exception as e:
       print(f"❌ Context Reference 오류: {e}")
       return {
           "messages": [AIMessage(content="🔮 질문을 이해하는 중 문제가 생겼어요. 다시 말씀해주시겠어요?")]
       }
def exception_handler(state: TarotState) -> TarotState:
   """예외 상황 처리"""
   user_input = state.get("user_input", "").lower()
   decision = state.get("supervisor_decision", {})
   # 중단/재시작 요청
   if any(keyword in user_input for keyword in ["그만", "중단", "취소", "다시", "처음"]):
       return {
           "messages": [AIMessage(content="🔮 알겠습니다. 새로운 상담을 시작할까요? 어떤 고민이 있으신지 말씀해주세요.")],
           "consultation_data": None,
           "user_intent": "unknown"
       }
   # 변경 요청
   elif any(keyword in user_input for keyword in ["바꿔", "다른", "변경"]):
       consultation_data = state.get("consultation_data", {})
       if consultation_data and consultation_data.get("status") == "spread_selection":
           return {"messages": [AIMessage(content="🔮 다른 스프레드를 원하신다면 새로운 고민을 말씀해주세요. 더 적합한 스프레드들을 찾아드릴게요!")]}
       else:
           return {"messages": [AIMessage(content="🔮 무엇을 바꾸고 싶으신가요? 구체적으로 말씀해주세요.")]}
   # 기타 예외
   else:
       return {"messages": [AIMessage(content="🔮 요청을 정확히 이해하지 못했습니다. 다시 말씀해주시겠어요?")]}
def emotional_support_handler(state: TarotState) -> TarotState:
    """감정 지원 핸들러 - 감정 분석 결과에 따른 따뜻한 메시지"""
    emotional_analysis = state.get("emotional_analysis", {})
    user_input = state.get("user_input") or get_last_user_input(state)
    emotion = emotional_analysis.get('primary_emotion', '알 수 없음')
    intensity = emotional_analysis.get('emotion_intensity', '보통')
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    if emotion == "불안" and intensity in ["높음", "매우높음"]:
        prompt = f"""
        사용자가 불안한 마음을 가지고 있습니다.
        고민: "{user_input}"
        감정 강도: {intensity}
        이 상황에 대해 당신의 감정을 이해하고, 
        당신의 마음을 위로하고 힘을 주는 메시지를 제시해주세요.
        �� 타로 상담사 톤으로 답변하세요.
        """
    elif emotion == "슬픔":
        prompt = f"""
        사용자가 슬픈 마음을 가지고 있습니다.
        고민: "{user_input}"
        감정 강도: {intensity}
        이 상황에 대해 당신의 감정을 이해하고, 
        당신의 마음을 위로하고 힘을 주는 메시지를 제시해주세요.
        🔮 타로 상담사 톤으로 답변하세요.
        """
    elif emotion == "걱정":
        prompt = f"""
        사용자가 걱정하는 마음을 가지고 있습니다.
        고민: "{user_input}"
        감정 강도: {intensity}
        이 상황에 대해 당신의 감정을 이해하고, 
        당신의 마음을 위로하고 힘을 주는 메시지를 제시해주세요.
        🔮 타로 상담사 톤으로 답변하세요.
        """
    else:
        prompt = f"""
        사용자가 평소와 다른 감정을 가지고 있습니다.
        고민: "{user_input}"
        감정 강도: {intensity}
        이 상황에 대해 당신의 감정을 이해하고, 
        당신의 마음을 위로하고 힘을 주는 메시지를 제시해주세요.
        🔮 타로 상담사 톤으로 답변하세요.
        """
    try:
        response = llm.invoke([HumanMessage(content=prompt)], {"metadata": {"final_response": "yes", "handler": "emotional_support_handler"}})
        return {"messages": [AIMessage(content=response.content)]}
    except Exception as e:
        fallback_msg = f"🔮 감정 지원 응답을 생성하는 중 문제가 발생했어요. 다시 시도해주세요!\n\n{e}"
        return {"messages": [AIMessage(content=fallback_msg)]}
def start_specific_spread_consultation(state: TarotState) -> TarotState:
   """리팩토링된 특정 스프레드 상담 핸들러 - 새로운 노드들을 순차 실행"""
   print("🔧 기존 start_specific_spread_consultation 호출 -> 리팩토링된 노드들로 처리")
   
   #  4개 노드를 순차 실행하여 동일한 결과 제공
   try:
       # 1. 스프레드 추출
       state = spread_extractor_node(state)
       
       # 2. 상황 분석
       state = situation_analyzer_node(state)
       
       # 3. 카드 수 추론
       state = card_count_inferrer_node(state)
       
       # 4. 상태 결정
       state = status_determiner_node(state)
       
       print("✅ 리팩토링된 start_specific_spread_consultation 성공적으로 완료")
       return state
       
   except Exception as e:
       print(f"❌ 리팩토링된 start_specific_spread_consultation 오류: {e}")
       # 기본 에러 처리
       return {
           "messages": [AIMessage(content="🔮 특정 스프레드 상담 처리 중 문제가 발생했습니다. 다시 시도해주세요.")],
           "consultation_data": {
               "status": "error"
           }
       }
    
def tool_result_handler(state: TarotState) -> TarotState:
   """도구 실행 후 결과를 AIMessage로 변환하여 사용자에게 전달"""
   messages = state.get("messages", [])
   
   if not messages:
       return {"messages": [AIMessage(content="도구 실행 결과를 찾을 수 없습니다.")]}
   
   # 마지막 메시지가 ToolMessage인지 확인
   last_message = messages[-1]
   
   if hasattr(last_message, 'name') and last_message.name in ['search_tarot_cards', 'search_tarot_spreads']:
       # 도구 결과를 LLM으로 처리하여 사용자 친화적인 답변 생성
       tool_result = last_message.content
       
       llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
       
       # 어떤 도구인지에 따라 프롬프트 변경
       if last_message.name == 'search_tarot_cards':
           prompt = f"""
           여러 타로 전문서에서 검색된 카드 의미들을 종합하여 사용자에게 완전하고 이해하기 쉬운 해석을 제공해주세요.

           검색 결과 (여러 타로책의 해석):
           {tool_result}

           **중요한 지침:**
           1. **여러 출처 통합**: 7개 타로책의 다양한 해석을 종합하여 완전한 의미 제공
           2. **이미지 설명 제외**: 카드 그림이나 시각적 묘사는 빼고 오직 **의미와 해석**만 포함
           3. **정방향/역방향 구분**: content(정방향)와 reversed(역방향) 의미를 명확히 분리
           4. **핵심 키워드 추출**: 각 방향별로 가장 중요한 키워드들 정리
           5. **실용적 조언**: 일상생활에서 이 카드가 나타났을 때의 의미와 조언

           **출력 형식:**
           🔮 **[카드명] 카드 의미**

           **✨ 정방향 (Upright)**
           - **핵심 의미**: [여러 책의 공통된 핵심 의미 통합]
           - **주요 키워드**: [중요 키워드 5-7개]
           - **상황별 해석**: 
             • 연애: [연애 관련 의미]
             • 직업: [직업/성공 관련 의미]  
             • 개인성장: [내적 성장 관련 의미]
           - **조언**: [이 카드가 나왔을 때 권하는 행동이나 마음가짐]

           **🔄 역방향 (Reversed)**
           - **핵심 의미**: [여러 책의 역방향 해석 통합]
           - **주요 키워드**: [역방향 키워드 5-7개]
           - **주의사항**: [조심해야 할 점들]
           - **극복방법**: [역방향 에너지를 긍정적으로 전환하는 방법]

           **💫 종합 메시지**
           [이 카드의 전체적인 메시지와 깊은 의미]

           **참고사항**: 
           - 카드 이미지나 그림 묘사는 완전히 제외
           - 여러 출처의 일치하지 않는 해석이 있다면 가장 일반적이고 전통적인 해석 우선
           - 따뜻하고 지지적인 타로 상담사 톤 유지

           마지막에 "다른 카드나 타로 상담이 필요하시면 언제든 말씀해주세요! 🌟"를 추가해주세요.
           """
       else:  # search_tarot_spreads
           prompt = f"""
           검색된 타로 스프레드 정보를 바탕으로 사용자에게 친근하고 실용적으로 설명해주세요.

           검색 결과 (스프레드 정보):
           {tool_result}

           **중요한 지침:**
           1. **스프레드 개요**: spread_name과 description을 활용하여 이 스프레드의 특징과 장점 설명
           2. **사용 상황**: keywords를 참고하여 어떤 상황에서 사용하면 좋은지 구체적으로 안내
           3. **포지션 설명**: positions 정보를 활용하여 각 카드 자리의 의미를 간략하고 이해하기 쉽게 설명
           4. **실용적 조언**: 이 스프레드가 어떤 질문이나 고민에 특히 효과적인지 안내

           **출력 형식:**
           🔮 **[스프레드명] 소개**

           **✨ 이 스프레드의 특징**
           [description과 keywords를 활용한 스프레드 특징 설명]

           **🎯 이런 상황에서 사용하세요**
           [keywords 기반으로 구체적인 사용 상황들 나열]
           - 예: "과거-현재-미래 흐름을 알고 싶을 때"
           - 예: "간단하고 명확한 답변이 필요할 때"

           **📍 카드 배치와 의미**
           [각 position의 position_meaning을 자연스럽게 조합해서 스프레드의 목적과 효과를 설명]
           - position_name을 직접 언급하지 말고, position_meaning의 내용을 바탕으로 자연스럽게 설명
           - 예: "Ambition 포지션을 통해..." (X) → "당신이 진정으로 원하는 것을 명확히 하고..." (O)
           - 예: "Fear or doubt 포지션에서..." (X) → "당신의 불안 요소를 인식할 수 있으며..." (O)

           **💡 이런 질문에 특히 좋아요**
           [keywords를 참고하여 적합한 질문 유형들 제시]

           **🌟 왜 추천하는가**
           [이 스프레드만의 장점과 효과]

           따뜻하고 친근한 타로 상담사 톤으로 작성하고, 마지막에 "이 스프레드로 상담받고 싶으시거나 다른 고민이 있으시면 언제든 말씀해주세요! ✨"를 추가해주세요.
           """
       
       try:
           response = llm.invoke([HumanMessage(content=prompt)], {"metadata": {"final_response": "yes", "handler": "tool_result_handler"}})
           print(f"🔧 도구 결과를 AIMessage로 변환 완료")
           
           # 기존 메시지들은 유지하고 마지막에 AI 응답 추가
           new_messages = messages[:-1]  # ToolMessage 제거
           new_messages.append(response)  # AIMessage 추가
           
           return {"messages": new_messages}
           
       except Exception as e:
           print(f"🔧 도구 결과 변환 중 오류: {e}")
           return {"messages": [AIMessage(content="🔮 검색 결과를 처리하는 중 문제가 발생했습니다. 다시 질문해주세요.")]}
   
   else:
       # ToolMessage가 아니면 그대로 반환
       print(f"🔧 도구 메시지가 아님: {type(last_message)}")
       return state
# =================================================================

# 기존 unified_processor_node, unified_tool_handler_node 등은 그대로 유지

# =================================================================

def unified_processor_node(state: TarotState) -> TarotState:
    """🆕 통합 처리기 - 모든 기존 핸들러 함수들을 조건부로 호출"""
    target_handler = state.get("target_handler", "unknown_handler")
    print(f"🔧 Unified Processor: 실행할 핸들러 = {target_handler}")
    function_map = {
        "card_info_handler": card_info_handler,
        "spread_info_handler": spread_info_handler,
        "consultation_handler": consultation_handler,
        "consultation_flow_handler": consultation_flow_handler,
        "consultation_continue_handler": consultation_continue_handler,
        "consultation_summary_handler": consultation_summary_handler,
        "consultation_individual_handler": consultation_individual_handler,
        "consultation_final_handler": consultation_final_handler,
        "general_handler": general_handler,
        "simple_card_handler": simple_card_handler,
        "context_reference_handler": context_reference_handler,
        "exception_handler": exception_handler,
        "emotional_support_handler": emotional_support_handler,
        "start_specific_spread_consultation": start_specific_spread_consultation,
        "unknown_handler": unknown_handler
    }
    handler_function = function_map.get(target_handler, unknown_handler)
    print(f"🔧 실행 중: {handler_function.__name__}")
    try:
        result = handler_function(state)
        print(f"✅ 핸들러 실행 완료: {handler_function.__name__}")
        return result
    except Exception as e:
        print(f"❌ 핸들러 실행 오류: {handler_function.__name__} - {e}")
        return {"messages": [AIMessage(content="처리 중 오류가 발생했습니다. 다시 시도해주세요.")]}
def unified_tool_handler_node(state: TarotState) -> TarotState:
    """🆕 통합 도구 처리기"""
    print("🔧 Tool Handler: 도구 실행 시작")
    tools = [search_tarot_spreads, search_tarot_cards]
    tool_node = ToolNode(tools)
    tool_result = tool_node.invoke(state)
    print("🔧 Tool Handler: 도구 실행 완료, 결과 처리 시작")
    final_result = tool_result_handler(tool_result)
    print("✅ Tool Handler: 최종 결과 생성 완료")
    return final_result
def classify_intent_node(state: TarotState) -> TarotState:
    """🆕 트리거 기반 의도 분류 - 명확한 키워드만 consultation으로 분류"""
    # 시간 맥락 설정
    state = ensure_temporal_context(state)
    # 마지막 사용자 메시지 추출
    user_input = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_input = msg.content.strip()
            break
    if not user_input:
        return {
            "user_intent": "unknown",
            "user_input": ""
        }
    user_input_lower = user_input.lower()
    # 🆕 1단계: 명확한 타로 상담 트리거 체크 (공통 상수 사용)
    if any(trigger in user_input_lower for trigger in ALL_CONSULTATION_TRIGGERS):
        print(f"🎯 타로 상담 트리거 감지: consultation")
        return {
            "user_intent": "consultation",
            "user_input": user_input
        }
    # 🆕 1-1단계: 간단한 카드 뽑기 트리거 체크 (공통 상수 사용)
    # 명확한 긍정 응답 체크 (카드 뽑기 제안에 대한 응답)
    clear_yes_triggers = ["네", "그래", "응", "해줘", "yes", "예"]
    has_clear_yes = any(trigger in user_input_lower for trigger in clear_yes_triggers)
    has_other_intent = any(keyword in user_input_lower for keyword in ["타로", "상담", "고민", "문제"])
    if any(trigger in user_input_lower for trigger in SIMPLE_CARD_TRIGGERS):
        print(f"🎯 간단한 카드 뽑기 트리거 감지: simple_card")
        return {
            "user_intent": "simple_card", 
            "user_input": user_input
        }
    elif has_clear_yes and not has_other_intent and len(user_input.strip()) <= 15:
        # 명확한 긍정 응답이면서 다른 의도가 없고 짧은 응답일 때
        print(f"🎯 카드 뽑기 제안에 대한 긍정 응답: simple_card")
        return {
            "user_intent": "simple_card",
            "user_input": user_input
        }
    # 🆕 2단계: 카드/스프레드 정보 질문 체크 (키워드 기반)
    card_keywords = ["카드 의미", "카드는", "역방향", "정방향", "메이저 아르카나", "마이너 아르카나"]
    spread_keywords = ["스프레드", "켈틱크로스", "3장", "5장", "배치"]
    if any(keyword in user_input_lower for keyword in card_keywords):
        print(f"🎯 카드 정보 질문 감지: card_info")
        return {
            "user_intent": "card_info",
            "user_input": user_input
        }
    if any(keyword in user_input_lower for keyword in spread_keywords):
        print(f"🎯 스프레드 정보 질문 감지: spread_info")
        return {
            "user_intent": "spread_info", 
            "user_input": user_input
        }
    # 🆕 3단계: 나머지는 모두 general로 분류 (일상 대화, 타로 일반 질문 등)
    print(f"🎯 일반 대화로 분류: general")
    return {
        "user_intent": "general",
        "user_input": user_input
    }
def consultation_router(state: TarotState) -> str:
    """상담 플로우의 조건부 라우팅"""
    consultation_status = state.get("consultation_status", "start")
    print(f"🔧 상담 라우터: 현재 상태 = {consultation_status}")
    if consultation_status == "emotion_analyzed":
        return "spread_recommender_node"  # 웹 검색 단계 건너뛰기
    elif consultation_status == "spreads_recommended":
        return "END"  # 스프레드 추천 완료
    else:
        return "emotion_analyzer_node"  # 시작점
def supervisor_llm_node(state: TarotState) -> TarotState:
    """기존 supervisor 함수 (그대로 유지)"""
    user_input = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_input = msg.content.strip()
            break
    # 최근 대화 맥락 간단히 추출
    recent_context = ""
    messages = state.get("messages", [])
    if len(messages) >= 2:
        last_ai = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                last_ai = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                break
        if last_ai:
            recent_context = f"직전 AI 응답: {last_ai}"
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.1,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    prompt = f"""
   당신은 대화 흐름과 사용자 감정 상태를 파악하는 전문가입니다.
   **현재 상황:**
   사용자 입력: "{user_input}"
   {recent_context}
   
   **1차 판단: 대화 흐름**
   사용자가 방금 전 답변에 대해 추가로 궁금해하는 것인지, 
   아니면 완전히 새로운 주제를 시작하는 것인지 판단하세요.
   
   **2차 판단: 감정 상태 및 지원 필요도**
   사용자 입력을 다음 중 하나로 분류하세요:
   
   **A) greeting** - 간단한 인사말
   - "안녕", "하이", "안뇽", "헬로", "hi", "hello" 등
   - 3글자 이하의 단순 인사
   
   **B) emotional_concern** - 감정적 고민 표현
   - "고민있어", "걱정돼", "우울해", "힘들어", "스트레스받아"
   - 구체적 상황 없이 감정만 표현
   
   **C) specific_concern** - 구체적 고민 상황
   - "연애 때문에 고민이야", "직장에서 힘들어", "돈 문제로 고생해"
   - "투자로 돈 많이 벌 수 있을까", "취업 잘 될까" 등 구체적 질문
   - 명확한 주제와 상황이 포함됨
   
   **D) consultation_inquiry** - 상담 관련 문의
   - "상담해줄 수 있어?", "고민 들어줄 수 있어?", "도와줄 수 있어?"
   - 상담 가능 여부나 서비스에 대한 문의
   
   **E) gratitude** - 감사 표현
   - "고마워", "감사해", "고맙다", "감사합니다", "thanks", "thank you"
   - 상담이나 도움에 대한 감사 인사
   
   **F) general** - 일반 질문이나 대화
   - 정보 요청, 일상 대화, 기타 모든 경우
   
   **특별 판단 규칙:**
   - 직전 AI가 카드 뽑기를 제안한 후 "네", "좋아", "그래" 등의 응답 → 새로운 주제
   - 타로 상담 키워드("타로 봐줘" 등) → 새로운 주제
   - 감정적 고민 표현 → 새로운 주제 + 감정 지원 필요
   
   다음 JSON으로 답변:
   {{
       "is_followup": true/false,
       "confidence": "high|medium|low",
       "reasoning": "판단 근거 (감정 상태 포함)",
       "emotion_type": "greeting|emotional_concern|specific_concern|consultation_inquiry|gratitude|general|unknown",
       "support_needed": "high|medium|low",
       "action": "handle_context_reference|route_to_intent"
   }}
   """
    try:
        # 🔧 스마트한 타로 상담 키워드 체크 (공통 상수 사용)
        has_tarot_trigger = any(trigger in user_input.lower() for trigger in ALL_CONSULTATION_TRIGGERS)
        if has_tarot_trigger:
            # 🔧 직전 AI 응답에서 "타로 봐줘"를 제안했는지 확인
            ai_suggested_tarot = False
            if recent_context and ("타로 봐줘" in recent_context or "타로 상담" in recent_context):
                ai_suggested_tarot = True
                print(f"🔧 AI가 타로 상담을 제안한 후 사용자가 응답함 - Follow-up으로 처리")
            if not ai_suggested_tarot:
                # AI가 제안하지 않았는데 사용자가 직접 "타로 봐줘" → 새 주제
                print(f"🎯 Supervisor: 사용자 주도 타로 상담 요청 → 의도 분류로 이동")
                return {
                    "user_input": user_input,
                    "supervisor_decision": {
                        "is_followup": False,
                        "confidence": "high", 
                        "reasoning": "사용자가 직접 새로운 타로 상담 요청",
                        "action": "route_to_intent"
                    }
                }
            # AI가 제안한 후 사용자 응답 → LLM이 판단하도록 진행
        # 🆕 단순 긍정 응답은 바로 의도 분류로 보내기
        simple_yes_responses = ["네", "좋아", "그래", "응", "해줘", "부탁해", "yes", "예"]
        if user_input.lower().strip() in simple_yes_responses:
            print(f"🎯 Supervisor: 단순 긍정 응답 감지 → 의도 분류로 이동")
            return {
                "user_input": user_input,
                "supervisor_decision": {
                    "is_followup": False,
                    "confidence": "high",
                    "reasoning": "단순 긍정 응답으로 의도 분류 필요",
                    "action": "route_to_intent"
                }
            }
        response = llm.invoke([HumanMessage(content=prompt)])
        decision = json.loads(response.content)
        is_followup = decision.get("is_followup", False)
        confidence = decision.get("confidence", "medium")
        emotion_type = decision.get("emotion_type", "general")
        support_needed = decision.get("support_needed", "medium")
        action = "handle_context_reference" if is_followup else "route_to_intent"
        print(f"🎯 Supervisor: {'Follow-up' if is_followup else 'New Topic'} (신뢰도: {confidence}) - 감정타입: {emotion_type}")
        return {
            "user_input": user_input,
            "supervisor_decision": {
                "is_followup": is_followup,
                "confidence": confidence,
                "reasoning": decision.get("reasoning", ""),
                "emotion_type": emotion_type,
                "support_needed": support_needed,
                "action": action
            }
        }
    except Exception as e:
        print(f"❌ Supervisor 오류: {e}")
        return {
            "user_input": user_input,
            "supervisor_decision": {
                "is_followup": True,  # 안전하게 follow-up으로 처리
                "confidence": "low",
                "action": "handle_context_reference"
            }
        }
def supervisor_master_node(state: TarotState) -> TarotState:
    """🆕 복잡한 경우만 전체 분석"""
    # 시간 맥락 설정
    state = ensure_temporal_context(state)
    # 기존 supervisor_llm_node 호출
    print("🧠 Supervisor Master: 전체 분석 시작")
    supervisor_result = supervisor_llm_node(state)
    state.update(supervisor_result)
    # 필요시 의도 분류
    supervisor_decision = state.get("supervisor_decision", {})
    if supervisor_decision.get("action") == "route_to_intent":
        print("🔍 의도 분류 실행")
        intent_result = classify_intent_node(state)
        state.update(intent_result)
    # 적절한 핸들러 결정
    target_handler = determine_target_handler(state)
    state["target_handler"] = target_handler
    print(f"🎯 Target Handler: {target_handler}")
    return state
def emotion_analyzer_node(state: TarotState) -> TarotState:
    """감정 분석 전용 노드 - LLM 1개만 사용"""
    user_input = state.get("user_input") or get_last_user_input(state)
    print("🔧 감정 분석 노드 실행")
    # 기존 로직 완전 보존
    emotional_analysis = analyze_emotion_and_empathy(user_input)
    empathy_message = generate_empathy_message(emotional_analysis, user_input)
    emotion = emotional_analysis.get('primary_emotion', '알 수 없음')
    intensity = emotional_analysis.get('emotion_intensity', '보통')
    # 감정에 따른 인사말 (기존 로직 완전 보존)
    if emotion == "불안" and intensity in ["높음", "매우높음"]:
        emotional_greeting = "🤗 불안한 마음을 달래드릴 수 있는 스프레드들을 준비했습니다."
    elif emotion == "슬픔":
        emotional_greeting = "💙 마음의 위로가 될 수 있는 스프레드들을 선별했습니다."
    elif emotion == "걱정":
        emotional_greeting = "🌟 걱정을 덜어드릴 수 있는 희망적인 스프레드들을 찾아드렸습니다."
    else:
        emotional_greeting = "🔮 상황에 가장 적합한 스프레드들을 찾아드렸습니다."
    return {
        "emotional_analysis": emotional_analysis,
        "empathy_message": empathy_message,
        "emotional_greeting": emotional_greeting,
        "consultation_status": "emotion_analyzed"
    }
def spread_recommender_node(state: TarotState) -> TarotState:
    """스프레드 추천 전용 노드 - 개선된 다층적 검색"""
    user_input = state.get("user_input") or get_last_user_input(state)
    print("🔧 스프레드 추천 노드 실행")
    state = ensure_temporal_context(state)
    print(f"🔍 고민별 스프레드 검색 시작: '{user_input}'")
    keywords = extract_concern_keywords(user_input)
    recommended_spreads = perform_multilayer_spread_search(keywords, user_input)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    emotional_analysis = state.get("emotional_analysis", {})
    emotional_greeting = state.get("emotional_greeting", "🔮 상황에 가장 적합한 스프레드들을 찾아드렸습니다.")
    emotion = emotional_analysis.get('primary_emotion', '알 수 없음')
    intensity = emotional_analysis.get('emotion_intensity', '보통')
    search_context = ""
    detailed_spreads_info = ""
    for i, spread in enumerate(recommended_spreads, 1):
        detailed_spreads_info += f"\n=== 스프레드 {i}: {spread['spread_name']} ===\n"
        detailed_spreads_info += f"카드 수: {spread['card_count']}장\n"
        detailed_spreads_info += f"설명: {spread.get('description', '설명 없음')}\n"
        positions = spread.get('positions', [])
        if positions:
            detailed_spreads_info += "포지션들:\n"
            for pos in positions[:5]:
                if isinstance(pos, dict):
                    pos_name = pos.get('position_name', '알 수 없음')
                    pos_meaning = pos.get('position_meaning', '설명 없음')
                    detailed_spreads_info += f"  - {pos_name}: {pos_meaning}\n"
        detailed_spreads_info += "\n"
    spread1_name_kr = translate_text_with_llm(recommended_spreads[0]['spread_name'], "spread_name") if len(recommended_spreads) > 0 else '첫 번째 스프레드'
    spread2_name_kr = translate_text_with_llm(recommended_spreads[1]['spread_name'], "spread_name") if len(recommended_spreads) > 1 else '두 번째 스프레드'
    spread3_name_kr = translate_text_with_llm(recommended_spreads[2]['spread_name'], "spread_name") if len(recommended_spreads) > 2 else '세 번째 스프레드'
    recommendation_prompt = f"""
    사용자의 고민: "{user_input}"
    사용자 감정 상태: {emotion} (강도: {intensity})
    추출된 키워드: {keywords}{search_context}
    다음은 사용자의 고민에 가장 적합하다고 판단되어 검색된 스프레드들입니다:
    {detailed_spreads_info}
    위 스프레드들의 실제 설명과 포지션 정보를 바탕으로, 사용자의 고민 "{user_input}"에 각 스프레드가 어떻게 도움이 될 수 있는지 구체적으로 설명해주세요.

    **⚠️ 중요한 작성 지침:**
         **중요한 호칭 규칙:**
     - 사용자를 지칭할 때는 '내담자님'으로만 하세요 ('당신', '사용자님', '고객님' 금지)
     - 한국어 특성상 주어를 자연스럽게 생략할 수 있는 곳에서는 생략해도 됩니다
     - 어미는 '~입니다' 대신 '~이에요/~해요' 등 친근한 어미로 말해주세요
    - position_name (예: "Ambition", "Fear or doubt", "Current situation")을 직접 언급하지 마세요
    - 대신 position_meaning의 내용을 자연스럽게 조합해서 스프레드의 목적과 효과를 설명하세요
    - 예시: "Ambition 포지션을 통해..." (X) → "내담자가 진정으로 원하는 것을 명확히 하고..." (O)
    - 예시: "Fear or doubt 포지션에서..." (X) → "내담자의 불안 요소를 인식할 수 있으며..." (O)
    - 예시: "Current situation에서..." (X) → "현재의 상황을 깊이 있게 분석하여..." (O)

    다음 형식으로 정확히 추천해주세요:
    {emotional_greeting}
    **1) {spread1_name_kr} ({recommended_spreads[0]['card_count']}장)**
    - 목적: [position_meaning들을 자연스럽게 조합해서 이 스프레드가 사용자 고민에 어떻게 도움이 될지 설명]
    - 효과: [감정 상태를 고려한 따뜻한 효과 설명]
    **2) {spread2_name_kr} ({recommended_spreads[1]['card_count'] if len(recommended_spreads) > 1 else 5}장)**  
    - 목적: [position_meaning들을 자연스럽게 조합해서 이 스프레드가 사용자 고민에 어떻게 도움이 될지 설명]
    - 효과: [감정 상태를 고려한 따뜻한 효과 설명]
    **3) {spread3_name_kr} ({recommended_spreads[2]['card_count'] if len(recommended_spreads) > 2 else 7}장)**
    - 목적: [position_meaning들을 자연스럽게 조합해서 이 스프레드가 사용자 고민에 어떻게 도움이 될지 설명]
    - 효과: [감정 상태를 고려한 따뜻한 효과 설명]
    💫 **마음에 드시는 스프레드로 타로 봐드릴게요. 하나 선택해주세요!**
    
    **반드시 다음 안내 문구를 마지막에 그대로 추가해주세요:**
    
    원하는 키워드의 스프레드를 말씀해주거나 기본 스프레드인 **세 장 스프레드**, **켈틱 크로스 스프레드**, **말굽 스프레드**로 볼 수도 있어요. 원하는 스프레드를 말씀해주세요.
    
    중요: position_name을 직접 언급하지 말고, position_meaning의 내용을 바탕으로 자연스럽고 매끄러운 설명을 작성해주세요.
    감정적으로 따뜻하고 희망적인 톤으로 작성해주세요.
    
    **📌 필수 요구사항: 응답 마지막에 반드시 다음 문구를 그대로 포함해주세요:**
    "원하는 키워드의 스프레드를 말씀해주거나 기본 스프레드인 **세 장 스프레드**, **켈틱 크로스 스프레드**, **말굽 스프레드**로 볼 수도 있어요. 원하는 스프레드를 말씀해주세요."
    """
    try:
        response = llm.invoke([HumanMessage(content=recommendation_prompt)], {"metadata": {"final_response": "yes", "handler": "spread_recommender_node"}})
        empathy_message = state.get("empathy_message", "")
        final_message = f"{empathy_message}\n\n{response.content}"
        updated_state = {
            "messages": [AIMessage(content=final_message)],
            "consultation_data": {
                "concern": user_input,
                "emotional_analysis": emotional_analysis,
                "recommended_spreads": recommended_spreads,
                "status": "spread_selection"
            },
            "consultation_status": "spreads_recommended"
        }
        return updated_state
    except Exception as e:
        empathy_message = state.get("empathy_message", "")
        fallback_message = f"{empathy_message}\n\n{emotional_greeting}\n\n스프레드 추천 중 일시적인 문제가 발생했습니다.\n하지만 걱정하지 마세요. 기본 스프레드로도 충분히 좋은 상담을 받으실 수 있습니다.\n\n어떤 스프레드를 선택하시겠어요? (1, 2, 3)"
        return {
            "messages": [AIMessage(content=fallback_message)],
            "consultation_data": {
                "concern": user_input,
                "emotional_analysis": emotional_analysis,
                "recommended_spreads": recommended_spreads,
                "status": "spread_selection"
            },
            "consultation_status": "spreads_recommended"
        }
def spread_extractor_node(state: TarotState) -> TarotState:
    """스프레드 추출 전용 노드 - LLM 1개만 사용"""
    user_input = state.get("user_input", "")
    print("🔧 스프레드 추출 노드 실행")
    # 1순위: Supervisor 결정 확인 (기존 로직 보존)
    supervisor_decision = state.get("supervisor_decision", {})
    specified_spread = supervisor_decision.get("specific_spread", "")
    # 2순위: LLM이 사용자 입력에서 스프레드 추출 (기존 로직 보존)
    if not specified_spread:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        extract_prompt = f"""
        사용자 입력에서 타로 스프레드를 추출해주세요: "{user_input}"
        사용자가 특정 스프레드를 언급했다면 그 이름을 답해주세요.
        언급하지 않았다면 "None"이라고 답해주세요.
        스프레드명만 답해주세요 (예: "One Card", "Celtic Cross", "Three Card", "None")
        """
        try:
            response = llm.invoke([HumanMessage(content=extract_prompt)])
            extracted_spread = response.content.strip()
            if extracted_spread != "None":
                specified_spread = extracted_spread
        except:
            pass
    # 3순위: 세션 메모리 (기존 로직 보존)
    if not specified_spread:
        session_memory = state.get("session_memory", {})
        explained_spreads = session_memory.get("explained_spreads", [])
        if explained_spreads:
            specified_spread = explained_spreads[-1]
    # 최종: LLM이 기본값도 결정 (기존 로직 보존)
    if not specified_spread:
        llm_default = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        default_prompt = f"""
        사용자가 타로 상담을 요청했지만 특정 스프레드를 지정하지 않았습니다.
        어떤 스프레드가 가장 적절할지 추천해주세요.
        일반적으로 초보자나 간단한 상담에는 어떤 스프레드가 좋은가요?
        스프레드명만 답해주세요 (예: "Three Card", "One Card", "Celtic Cross")
        """
        try:
            response = llm_default.invoke([HumanMessage(content=default_prompt)])
            specified_spread = response.content.strip()
        except:
            specified_spread = "Three Card"  # 진짜 최후의 수단
    print(f"🔧 추출된 스프레드: {specified_spread}")
    return {
        "extracted_spread": specified_spread,
        "specific_consultation_status": "spread_extracted"
    }
def situation_analyzer_node(state: TarotState) -> TarotState:
    """상황 분석 전용 노드 - LLM 1개만 사용"""
    user_input = state.get("user_input", "")
    extracted_spread = state.get("extracted_spread", "Three Card")
    print("🔧 상황 분석 노드 실행")
    # 기존 로직 완전 보존 - LLM이 상황을 판단하고 적절한 응답 생성
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    prompt = f"""
    사용자가 "{extracted_spread}" 스프레드로 상담을 요청했습니다.
    사용자 입력: "{user_input}"
    **한국어 자연어 이해:**
    한국어는 맥락 의존적이고 생략이 많습니다. 
    사용자의 진짜 의도를 파악하세요:
    - 짧은 표현도 명확한 의미를 담을 수 있음
    - 상담 의지와 구체적 주제 유무를 구분해서 판단
    - 한국인이 자연스럽게 사용하는 표현 방식 고려
    상황을 분석해서 적절히 응답해주세요:
    **만약 사용자가 이미 구체적인 고민이나 질문을 했다면:**
    - 바로 카드 선택 단계로 안내
    - "좋습니다! {user_input}에 대해 {extracted_spread} 스프레드로 봐드리겠습니다"
    - 카드 번호 선택 방법 안내 (1부터 78까지 X장 선택)
    **만약 상담은 원하지만 구체적 고민이 없다면:**
    - 스프레드 소개 후 구체적인 고민 질문
    - "어떤 고민에 대해 알아보고 싶으신지 구체적으로 말씀해주세요"
    **한국어 맥락과 사용자 의도를 자연스럽게 추론**해서 판단해주세요.
    타로 상담사 톤으로 따뜻하고 친근하게 작성해주세요.
    """
    try:
        response = llm.invoke([HumanMessage(content=prompt)], {"metadata": {"final_response": "yes", "handler": "situation_analyzer_node"}})
        return {
            "situation_analysis_response": response.content,
            "specific_consultation_status": "situation_analyzed"
        }
    except Exception as e:
        print(f"🔧 상황 분석 오류: {e}")
        return {
            "situation_analysis_response": f"🔮 {extracted_spread} 상담을 준비하는 중입니다. 어떤 고민을 봐드릴까요?",
            "specific_consultation_status": "situation_analyzed"
        }
def card_count_inferrer_node(state: TarotState) -> TarotState:
    """카드 수 추론 전용 노드 - LLM 1개만 사용"""
    extracted_spread = state.get("extracted_spread", "Three Card")
    print("🔧 카드 수 추론 노드 실행")
    # 기존 로직 완전 보존 - LLM이 카드 수도 추론하게 하기
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    card_count_prompt = f"""
    "{extracted_spread}" 스프레드는 몇 장의 카드를 사용하나요? 
    숫자만 답해주세요 (예: 3, 5, 10)
    """
    try:
        card_count_response = llm.invoke([HumanMessage(content=card_count_prompt)])
        card_count = int(card_count_response.content.strip())
    except:
        card_count = 3  # 기본값
    print(f"🔧 추론된 카드 수: {card_count}")
    return {
        "inferred_card_count": card_count,
        "specific_consultation_status": "card_count_inferred"
    }
def status_determiner_node(state: TarotState) -> TarotState:
    """상태 결정 전용 노드 - LLM 1개만 사용"""
    user_input = state.get("user_input", "")
    extracted_spread = state.get("extracted_spread", "Three Card")
    inferred_card_count = state.get("inferred_card_count", 3)
    situation_analysis_response = state.get("situation_analysis_response", "")
    print("🔧 상태 결정 노드 실행")
    # 기존 로직 완전 보존 - 상태 판단도 LLM에게 위임 (한국어 자연어 이해)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    status_prompt = f"""
    사용자 입력 "{user_input}"을 한국어 자연어로 이해해서 판단해주세요.
    **판단 기준:**
    한국어는 맥락 의존적이고 생략이 많습니다.
    사용자의 진짜 의도를 파악하세요:
    - 상담 의지가 있는가?
    - 구체적인 주제나 고민이 포함되어 있는가?
    - 한국인이 자연스럽게 사용하는 표현인가?
    다음 중 하나로 답해주세요:
    - "card_selection": 이미 구체적인 고민/질문이 있어서 바로 카드 선택 단계
    - "collecting_concern": 상담은 원하지만 아직 구체적인 고민을 물어봐야 함
    단어 하나만 답해주세요.
    """
    try:
        status_response = llm.invoke([HumanMessage(content=status_prompt)])
        status = status_response.content.strip()
        if status not in ["card_selection", "collecting_concern"]:
            status = "collecting_concern"  # 기본값
        print(f"🔧 LLM이 판단한 상태: {status}")
        # 기존 로직 완전 보존 - consultation_data 구성
        consultation_data = {
            "status": status,
            "selected_spread": {
                "spread_name": extracted_spread,
                "card_count": inferred_card_count,
                "description": f"{extracted_spread} 스프레드"
            }
        }
        # card_selection 상태면 concern도 저장 (기존 로직 보존)
        if status == "card_selection":
            consultation_data["concern"] = user_input
        return {
            "messages": [AIMessage(content=situation_analysis_response)],
            "consultation_data": consultation_data,
            "specific_consultation_status": "status_determined"
        }
    except Exception as e:
        print(f"🔧 상태 결정 오류: {e}")
        return {
            "messages": [AIMessage(content=situation_analysis_response)],
            "consultation_data": {
                "status": "collecting_concern",
                "selected_spread": {
                    "spread_name": extracted_spread,
                    "card_count": inferred_card_count,
                    "description": f"{extracted_spread} 스프레드"
                }
            },
            "specific_consultation_status": "status_determined"
        }
def specific_consultation_router(state: TarotState) -> str:
    """특정 스프레드 상담 플로우의 조건부 라우팅 - 🔧 스프레드 선택 상태 우선 체크"""
    # 🔧 1순위: consultation_data에서 스프레드 선택 상태 체크
    consultation_data = state.get("consultation_data", {})
    consultation_status = consultation_data.get("status", "")
    if consultation_status == "spread_selection":
        # 스프레드 선택 상태에서는 consultation_continue_handler로 이동
        print("🔧 스프레드 선택 상태 감지 → consultation_continue_handler로 라우팅")
        return "consultation_continue_handler"
    # 🔧 2순위: 기존 specific_consultation_status 체크
    specific_status = state.get("specific_consultation_status", "start")
    print(f"🔧 특정 상담 라우터: 현재 상태 = {specific_status}")
    if specific_status == "spread_extracted":
        return "situation_analyzer_node"
    elif specific_status == "situation_analyzed":
        return "card_count_inferrer_node"
    elif specific_status == "card_count_inferred":
        return "status_determiner_node"
    elif specific_status == "status_determined":
        return "END"  # 특정 스프레드 상담 완료
    else:
        return "spread_extractor_node"  # 시작점
def start_actual_consultation(state: TarotState) -> TarotState:
    """고민을 받은 후 실제 상담 진행"""
    user_input = state.get("user_input", "")
    # Phase 1 리팩토링: 웹 검색 제거 후 2개 노드 순차 실행
    try:
        # 1. 감정 분석
        result1 = emotion_analyzer_node(state)
        state.update(result1)
        # 2. 스프레드 추천 (웹 검색 관련 코드 제거됨)
        result4 = spread_recommender_node(state)
        state.update(result4)
        print("✅ 실제 상담 성공적으로 완료")
        return state
    except Exception as e:
        print(f"❌ 실제 상담 오류: {e}")
        # 기본 에러 처리
        return {
            "messages": [AIMessage(content="🔮 상담 처리 중 문제가 발생했습니다. 다시 시도해주세요.")],
            "consultation_data": {
                "status": "error"
            }
        }
def state_router(state: TarotState) -> str:
    """🆕 상태 기반 라우팅"""
    routing_decision = state.get("routing_decision", "NEW_SESSION")
    print(f"🔀 State Router: {routing_decision}")
    if routing_decision == "CONSULTATION_ACTIVE":
        return "consultation_direct"
    elif routing_decision == "FOLLOWUP_QUESTION":
        return "context_reference_direct"
    else:
        return "supervisor_master"
def processor_router(state: TarotState) -> str:
    """🆕 프로세서 후 라우팅 - 도구 호출 체크"""
    messages = state.get("messages", [])
    if not messages:
        return "end"
    last_message = messages[-1]
    # AIMessage이고 tool_calls가 있는지 체크
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print(f"🔧 도구 호출 감지: {len(last_message.tool_calls)}개")
        return "tools"
    print("🔧 도구 호출 없음 - 종료")
    return "end"
