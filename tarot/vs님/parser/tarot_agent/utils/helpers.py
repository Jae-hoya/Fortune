import numpy as np
import random
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from .state import TarotState
import pytz
from datetime import datetime, timedelta
import json

# from parsing.parser.tarot_agent.utils.tools import rag_system  # 순환 import 방지 위해 삭제

def convert_numpy_types(obj):
    """numpy 타입을 파이썬 기본 타입으로 변환"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(v) for v in obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        for attr_name in dir(obj):
            if not attr_name.startswith('_'):
                try:
                    attr_value = getattr(obj, attr_name)
                    if isinstance(attr_value, (np.floating, np.integer, np.ndarray)):
                        setattr(obj, attr_name, convert_numpy_types(attr_value))
                except:
                    continue
        return obj
    else:
        return obj
def safe_format_search_results(results) -> str:
    """검색 결과를 안전하게 포맷팅 (NumPy 타입 변환 포함)"""
    if not results:
        return "검색 결과가 없습니다."
    safe_results = convert_numpy_types(results)
    formatted = ""
    for i, result_item in enumerate(safe_results, 1):
        if isinstance(result_item, (tuple, list)) and len(result_item) >= 2:
            doc, score = result_item[0], result_item[1]
        else:
            doc = result_item
            score = 0.0
        if hasattr(doc, 'metadata'):
            metadata = doc.metadata
        else:
            metadata = {}
        if hasattr(doc, 'page_content'):
            content = doc.page_content
        else:
            content = str(doc)
        content = content[:300] + "..." if len(content) > 300 else content
        formatted += f"\n=== 결과 {i} (점수: {float(score):.3f}) ===\n"
        if metadata.get("card_name"):
            formatted += f"카드: {metadata['card_name']}\n"
        if metadata.get("spread_name"):
            formatted += f"스프레드: {metadata['spread_name']}\n"
        if metadata.get("source"):
            formatted += f"출처: {metadata['source']}\n"
        formatted += f"내용: {content}\n"
        formatted += "-" * 50 + "\n"
    return formatted
def parse_card_numbers(user_input: str, required_count: int) -> List[int]:
    """사용자 입력에서 카드 번호들을 파싱하고 중복 체크"""
    try:
        numbers = []
        parts = user_input.replace(" ", "").split(",")
        for part in parts:
            if part.isdigit():
                num = int(part)
                if 1 <= num <= 78:
                    if num not in numbers:
                        numbers.append(num)
                    else:
                        return None
        if len(numbers) == required_count:
            return numbers
        else:
            return None
    except:
        return None
# tarot_langgraph.py에서 100% 동일하게 복사: TAROT_CARDS 전체 정의

TAROT_CARDS = {

   1: "The Fool", 2: "The Magician", 3: "The High Priestess", 4: "The Empress",

   5: "The Emperor", 6: "The Hierophant", 7: "The Lovers", 8: "The Chariot",

   9: "Strength", 10: "The Hermit", 11: "Wheel of Fortune", 12: "Justice",

   13: "The Hanged Man", 14: "Death", 15: "Temperance", 16: "The Devil",

   17: "The Tower", 18: "The Star", 19: "The Moon", 20: "The Sun",

   21: "Judgement", 22: "The World",

   23: "Ace of Cups", 24: "Two of Cups", 25: "Three of Cups", 26: "Four of Cups",

   27: "Five of Cups", 28: "Six of Cups", 29: "Seven of Cups", 30: "Eight of Cups",

   31: "Nine of Cups", 32: "Ten of Cups", 33: "Page of Cups", 34: "Knight of Cups",

   35: "Queen of Cups", 36: "King of Cups",

   37: "Ace of Pentacles", 38: "Two of Pentacles", 39: "Three of Pentacles", 40: "Four of Pentacles",

   41: "Five of Pentacles", 42: "Six of Pentacles", 43: "Seven of Pentacles", 44: "Eight of Pentacles",

   45: "Nine of Pentacles", 46: "Ten of Pentacles", 47: "Page of Pentacles", 48: "Knight of Pentacles",

   49: "Queen of Pentacles", 50: "King of Pentacles",

   51: "Ace of Swords", 52: "Two of Swords", 53: "Three of Swords", 54: "Four of Swords",

   55: "Five of Swords", 56: "Six of Swords", 57: "Seven of Swords", 58: "Eight of Swords",

   59: "Nine of Swords", 60: "Ten of Swords", 61: "Page of Swords", 62: "Knight of Swords",

   63: "Queen of Swords", 64: "King of Swords",

   65: "Ace of Wands", 66: "Two of Wands", 67: "Three of Wands", 68: "Four of Wands",

   69: "Five of Wands", 70: "Six of Wands", 71: "Seven of Wands", 72: "Eight of Wands",

   73: "Nine of Wands", 74: "Ten of Wands", 75: "Page of Wands", 76: "Knight of Wands",

   77: "Queen of Wands", 78: "King of Wands"

}

# 번역 캐시 (메모리 절약을 위해 전역 변수로 관리)

_translation_cache = {}

def translate_card_info(english_name, direction_text):
    """카드명과 방향을 한국어로 번역하여 완전한 형태로 반환"""
    # 메이저 아르카나 수동 번역 (22장)
    major_arcana = {
        "The Fool": "바보",
        "The Magician": "마법사", 
        "The High Priestess": "여사제",
        "The Empress": "여황제",
        "The Emperor": "황제",
        "The Hierophant": "교황",
        "The Lovers": "연인",
        "The Chariot": "전차",
        "Strength": "힘",
        "The Hermit": "은둔자",
        "Wheel of Fortune": "운명의 수레바퀴",
        "Justice": "정의",
        "The Hanged Man": "매달린 사람",
        "Death": "죽음",
        "Temperance": "절제",
        "The Devil": "악마",
        "The Tower": "탑",
        "The Star": "별",
        "The Moon": "달",
        "The Sun": "태양",
        "Judgement": "심판",
        "The World": "세계"
    }
    # 카드명 번역
    if english_name in major_arcana:
        card_name_kr = major_arcana[english_name]
    else:
        # 마이너 아르카나 패턴 번역
        suits = {"Cups": "컵", "Pentacles": "펜타클", "Swords": "소드", "Wands": "완드"}
        ranks = {"Ace": "에이스", "Two": "2", "Three": "3", "Four": "4", "Five": "5", 
                 "Six": "6", "Seven": "7", "Eight": "8", "Nine": "9", "Ten": "10",
                 "Page": "페이지", "Knight": "기사", "Queen": "여왕", "King": "왕"}
        card_name_kr = english_name  # 기본값
        for eng_suit, kr_suit in suits.items():
            if f"of {eng_suit}" in english_name:
                for eng_rank, kr_rank in ranks.items():
                    if english_name.startswith(eng_rank):
                        card_name_kr = f"{kr_suit} {kr_rank}"
                        break
                break
    # 방향 번역
    if direction_text == "upright":
        direction_symbol = "⬆️"
        direction_kr = "정방향"
    elif direction_text == "reversed":
        direction_symbol = "⬇️"
        direction_kr = "역방향"
    else:
        direction_symbol = ""
        direction_kr = direction_text
    return {
        'name': card_name_kr,
        'symbol': direction_symbol,
        'direction': direction_kr,
        'full': f"{card_name_kr} {direction_symbol} ({direction_kr})"
    }
# select_cards_randomly_but_keep_positions에서 내부 참조로 변경

def select_cards_randomly_but_keep_positions(user_numbers: List[int], required_count: int) -> List[Dict[str, Any]]:
   """사용자 숫자는 무시하고 랜덤 카드 선택, 위치만 유지"""
   if len(user_numbers) != required_count:
       return []
   random_card_numbers = random.sample(range(1, 79), required_count)
   selected_cards = []
   for position_index, (user_num, random_card_num) in enumerate(zip(user_numbers, random_card_numbers)):
       card_name = TAROT_CARDS.get(random_card_num, f"Unknown Card {random_card_num}")
       orientation = random.choice(["upright", "reversed"])
       # 한국어 번역 추가
       translated_info = translate_card_info(card_name, orientation)
       selected_cards.append({
           "position": position_index + 1,
           "user_number": user_num,
           "card_number": random_card_num,
           "name": card_name,  # 영어 이름 (RAG 검색용)
           "name_kr": translated_info['name'],  # 한국어 이름 (사용자 표시용)
           "orientation": orientation,
           "orientation_kr": translated_info['direction'],  # 한국어 방향
           "orientation_symbol": translated_info['symbol']  # 방향 기호
       })
   return selected_cards
def analyze_emotion_and_empathy(user_input: str) -> Dict[str, Any]:
   """사용자 입력에서 감정 상태 분석 및 공감 톤 결정"""
   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
   emotion_prompt = f"""
   다음 사용자 입력에서 감정을 분석하고 적절한 공감 방식을 제안해주세요:
   "{user_input}"
   다음 JSON 형식으로 답변해주세요:
   {{
       "primary_emotion": "불안/슬픔/분노/혼란/기대/걱정/좌절/기타",
       "emotion_intensity": "낮음/보통/높음/매우높음",
       "empathy_tone": "gentle/supportive/encouraging/understanding",
       "comfort_message": "적절한 위로 메시지 (한 문장)",
       "response_style": "formal/casual/warm/professional"
   }}
   """
   try:
       response = llm.invoke([HumanMessage(content=emotion_prompt)])
       import json
       try:
           emotion_data = json.loads(response.content)
           return emotion_data
       except:
           return {
               "primary_emotion": "혼란",
               "emotion_intensity": "보통",
               "empathy_tone": "supportive",
               "comfort_message": "마음이 복잡하시겠어요. 함께 답을 찾아보겠습니다.",
               "response_style": "warm"
           }
   except Exception as e:
       print(f"🔍 감정 분석 오류: {e}")
       return {
           "primary_emotion": "혼란",
           "emotion_intensity": "보통", 
           "empathy_tone": "supportive",
           "comfort_message": "마음이 복잡하시겠어요. 함께 답을 찾아보겠습니다.",
           "response_style": "warm"
       }
def generate_empathy_message(emotional_analysis: Dict, user_concern: str) -> str:
   """감정 상태에 따른 공감 메시지 생성"""
   emotion = emotional_analysis.get('primary_emotion', '혼란')
   intensity = emotional_analysis.get('emotion_intensity', '보통')
   empathy_templates = {
       "불안": {
           "매우높음": "지금 정말 많이 불안하시겠어요. 그런 마음 충분히 이해합니다. 🤗 함께 차근차근 풀어보아요.",
           "높음": "많이 불안하시겠어요. 그런 마음이 드는 게 당연합니다. 타로가 좋은 방향을 제시해줄 거예요.",
           "보통": "걱정이 많으시군요. 마음이 복잡하실 텐데, 함께 해답을 찾아보아요.",
           "낮음": "약간의 불안감이 느껴지시는군요. 차근차근 살펴보겠습니다."
       },
       "슬픔": {
           "매우높음": "정말 많이 힘드시겠어요. 혼자가 아니니까 괜찮습니다. 💙 시간이 걸리더라도 함께 이겨내요.",
           "높음": "정말 힘드시겠어요. 마음이 아프시겠지만, 위로가 되는 답을 찾아드릴게요.",
           "보통": "마음이 무거우시겠어요. 슬픈 마음이 조금이라도 가벼워질 수 있도록 도와드릴게요.",
           "낮음": "조금 속상하시는 것 같아요. 함께 이야기해보면서 마음을 정리해보아요."
       },
       "걱정": {
           "매우높음": "정말 많이 걱정되시겠어요. 그런 마음이 드는 게 당연합니다. 함께 불안을 덜어보아요.",
           "높음": "많이 걱정되시는군요. 미래에 대한 두려움이 크시겠어요. 희망적인 답을 찾아보겠습니다.",
           "보통": "걱정이 되시는 상황이군요. 타로를 통해 안심할 수 있는 답을 찾아보아요.",
           "낮음": "조금 걱정되시는 것 같아요. 함께 살펴보면 마음이 편해질 거예요."
       }
   }
   emotion_messages = empathy_templates.get(emotion, empathy_templates.get("걱정", {}))
   message = emotion_messages.get(intensity, "마음이 복잡하시겠어요. 함께 답을 찾아보겠습니다.")
   return message
def get_default_spreads() -> List[Dict[str, Any]]:
   """기본 스프레드 3개 정의"""
   return [
       {
           "spread_name": "THE THREE CARD SPREAD",
           "normalized_name": "three card spread",
           "card_count": 3,
           "description": "A simple three-card spread perfect for quick insights and clarity on past, present, and future influences.",
           "positions": [
               {"position_num": 1, "position_name": "Past", "position_meaning": "Influences from the past that are affecting your situation"},
               {"position_num": 2, "position_name": "Present", "position_meaning": "Current energies and challenges you are facing now"},
               {"position_num": 3, "position_name": "Future", "position_meaning": "Potential outcomes and future developments"}
           ],
           "keywords": "simple, quick, past present future, clarity, beginner friendly, overview, direct, three cards"
       },
       {
           "spread_name": "THE CELTIC CROSS SPREAD",
           "normalized_name": "celtic cross spread",
           "card_count": 10,
           "description": "A comprehensive 10-card spread that examines multiple aspects of a situation, including obstacles, external influences, hopes, fears, and outcomes.",
           "positions": [
               {"position_num": 1, "position_name": "The Present", "position_meaning": "The current situation and core issue"},
               {"position_num": 2, "position_name": "The Challenge", "position_meaning": "Obstacles or challenges you're facing"},
               {"position_num": 3, "position_name": "The Past", "position_meaning": "Recent events that led to the current situation"},
               {"position_num": 4, "position_name": "The Future", "position_meaning": "What is coming in the near future"},
               {"position_num": 5, "position_name": "Above", "position_meaning": "Your conscious aims and ideals"},
               {"position_num": 6, "position_name": "Below", "position_meaning": "Subconscious influences and hidden factors"},
               {"position_num": 7, "position_name": "Advice", "position_meaning": "Recommended approach or attitude"},
               {"position_num": 8, "position_name": "External Influences", "position_meaning": "How others see you or outside factors"},
               {"position_num": 9, "position_name": "Hopes and Fears", "position_meaning": "Your inner hopes and fears about the situation"},
               {"position_num": 10, "position_name": "Outcome", "position_meaning": "The potential result if you continue on your current path"}
           ],
           "keywords": "detailed, comprehensive, traditional, complex, obstacles, influences, deep analysis, ten cards, classic, thorough"
       }
   ]
def extract_concern_keywords(user_concern: str) -> str:
   """사용자 고민에서 타로 스프레드 검색에 적합한 키워드 추출 - 개선된 버전"""
   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
   prompt = f"""
   사용자의 고민을 분석하여 타로 스프레드 검색에 적합한 키워드를 추출해주세요.
   사용자 고민: "{user_concern}"
   **단계별 분석:**
   1) 질문의 핵심 의도 파악:
   - 정보 요청 (Information): 단순히 알고 싶어하는 것
   - 결정 요청 (Decision): 선택이나 결정을 내려야 하는 상황  
   - 상황 파악 (Situation): 현재 상황에 대한 이해
   - 미래 예측 (Future): 앞으로 일어날 일에 대한 궁금증
   - 감정 지원 (Emotional): 위로나 격려가 필요한 상황
   2) 키워드 추출 (최대 6개, 우선순위 순서로):
   **의도 키워드 (필수 1-2개):**
   decision, choice, crossroads, dilemma, uncertainty, confusion, doubt, guidance, 
   advice, direction, clarity, insight, understanding, future, prediction, timing,
   information, knowledge, truth, revelation, confirmation, validation
   **주제 키워드 (필수 1-2개):**
   love, romance, relationship, dating, marriage, breakup, divorce, soulmate, partnership,
   career, job, work, business, promotion, interview, unemployment, success, failure,
   money, finance, investment, health, illness, healing, wellness, mental, physical,
   family, parents, children, siblings, friendship, social, community, conflict,
   spirituality, growth, purpose, destiny, travel, moving, home, education, creativity
   **감정/상황 키워드 (선택적 1-2개):**
   anxiety, fear, worry, stress, hope, excitement, joy, happiness, sadness, anger,
   frustration, guilt, regret, loneliness, peace, confidence, courage, depression,
   change, transition, transformation, crisis, challenge, obstacle, opportunity,
   new beginning, ending, closure, reconciliation, separation, commitment
   **분석 예시:**
   - "나 일 구할까 말까?" → "decision choice career job uncertainty opportunity"
   - "언제 결혼할까?" → "future timing love marriage relationship prediction"
   - "이 관계 계속해야 할까?" → "decision choice relationship love uncertainty guidance"
   결과를 영어 키워드로만 답해주세요 (정확히 5-6개, 공백으로 구분)
   키워드:"""
   try:
       response = llm.invoke([HumanMessage(content=prompt)])
       keywords = response.content.strip()
       print(f"🔍 개선된 키워드 추출: '{keywords}'")
       return keywords
   except Exception as e:
       print(f"🔍 키워드 추출 오류: {e}")
       # 기본 키워드에 decision 포함
       return "decision choice general situation guidance"
def extract_suit_from_name(card_name: str) -> str:
   """카드 이름에서 수트 추출"""
   if "Cups" in card_name:
       return "Cups"
   elif "Pentacles" in card_name:
       return "Pentacles"
   elif "Swords" in card_name:
       return "Swords"
   elif "Wands" in card_name:
       return "Wands"
   return ""
def extract_rank_from_name(card_name: str) -> str:
   """카드 이름에서 랭크 추출"""
   for rank in ["Ace", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Page", "Knight", "Queen", "King"]:
       if rank in card_name:
           return rank
   return ""
def is_major_arcana(card_name: str) -> bool:
   """메이저 아르카나 판별"""
   major_cards = ["The Fool", "The Magician", "The High Priestess", "The Empress", "The Emperor", "The Hierophant", "The Lovers", "The Chariot", "Strength", "The Hermit", "Wheel of Fortune", "Justice", "The Hanged Man", "Death", "Temperance", "The Devil", "The Tower", "The Star", "The Moon", "The Sun", "Judgement", "The World"]
   return card_name in major_cards
def get_last_user_input(state: TarotState) -> str:
   """마지막 사용자 입력 추출"""
   for msg in reversed(state.get("messages", [])):
       if isinstance(msg, HumanMessage):
           return msg.content.strip()
   return ""
def check_if_has_specific_concern(user_input: str, conversation_context: str = "") -> bool:
   """LLM을 사용해서 사용자 입력에 구체적인 고민이 포함되어 있는지 판단"""
   # 대화 맥락이 있으면 고민이 있는 것으로 간주
   if conversation_context and len(conversation_context.strip()) > 0:
       print(f"🤔 대화 맥락 감지: {conversation_context[:100]}...")
       return True
   
   # 단순한 타로 요청만 있는지 빠른 체크
   if user_input.strip() in ["타로 봐줘", "타로봐줘", "타로 상담", "점 봐줘", "운세 봐줘"]:
       return False
   llm = ChatOpenAI(
       model="gpt-4o-mini", 
       temperature=0.1,
       model_kwargs={"response_format": {"type": "json_object"}}
   )
   prompt = f"""
   사용자 입력을 분석해서 구체적인 고민이나 상황이 언급되어 있는지 판단해주세요.
   **사용자 입력:** "{user_input}"
   **판단 기준:**
   - 구체적인 고민이 있음: "연애 고민이 있어서 타로 봐줘", "취업 때문에 스트레스받아", "일 할까 말까 고민돼"
   - 구체적인 고민이 없음: "타로 봐줘", "타로 상담 받고 싶어", "점 봐줘"
   **고민이 있다고 판단되는 경우:**
   1. 구체적인 주제가 언급됨 (연애, 직업, 건강, 가족, 돈 등)
   2. 감정 상태가 언급됨 (걱정, 스트레스, 우울, 불안 등)
   3. 의사결정 상황이 언급됨 (할까 말까, 어떻게 해야 할지 등)
   4. 문제 상황이 구체적으로 서술됨
   다음 JSON 형식으로 답변:
   {{
       "has_specific_concern": true/false,
       "reasoning": "판단 근거"
   }}
   """
   try:
       response = llm.invoke([HumanMessage(content=prompt)])
       result = json.loads(response.content)
       has_concern = result.get("has_specific_concern", False)
       reasoning = result.get("reasoning", "")
       print(f"🤔 고민 포함 여부: {has_concern} - {reasoning}")
       return has_concern
   except Exception as e:
       print(f"❌ 고민 판단 오류: {e}")
       # 오류 시 안전하게 False 반환 (고민 없음으로 간주)
       return False
def simple_trigger_check(user_input: str) -> str:
    """
    Returns:
        "new_consultation" | "individual_reading" | "context_reference"
    """
    user_input_lower = user_input.lower()
    # 1. 새 상담 시작 트리거
    new_consultation_keywords = ["타로 봐줘", "새로 봐줘", "처음부터"]
    if any(keyword in user_input_lower for keyword in new_consultation_keywords):
        matched_keyword = next(keyword for keyword in new_consultation_keywords if keyword in user_input_lower)
        print(f"🎯 새 상담 트리거 감지: '{matched_keyword}' in '{user_input}'")
        return "new_consultation"
    # 2. 개별 해석 트리거 (기존 유지)
    individual_keywords = ["네", "yes", "보고싶"]
    if any(keyword in user_input_lower for keyword in individual_keywords):
        matched_keyword = next(keyword for keyword in individual_keywords if keyword in user_input_lower)
        print(f"🎯 개별 해석 트리거 감지: '{matched_keyword}' in '{user_input}'")
        return "individual_reading"
    # 3. 나머지는 모두 추가 질문으로 처리
    print(f"🎯 추가 질문으로 분류: '{user_input}'")
    return "context_reference"
def is_simple_followup(user_input: str) -> bool:
   """간단한 패턴으로 추가 질문 판단"""
   followup_patterns = [
       "어떻게", "왜", "그게", "그거", "아까", "방금", "더", "자세히", 
       "언제", "시기", "설명", "?", "뭐", "무엇", "좀 더", "추가로"
   ]
   user_lower = user_input.lower()
   return any(pattern in user_lower for pattern in followup_patterns)
def determine_consultation_handler(status: str) -> str:
    """상담 상태에 따른 핸들러 결정"""
    status_map = {
        "spread_selection": "consultation_continue_handler",
        "card_selection": "consultation_summary_handler", 
        "summary_shown": "consultation_final_handler"
    }
    return status_map.get(status, "consultation_flow_handler")
def determine_target_handler(state: TarotState) -> str:
    """어떤 기존 함수를 호출할지 결정 (기존 라우팅 로직 재사용)"""
    supervisor_decision = state.get("supervisor_decision", {})
    action = supervisor_decision.get("action", "route_to_intent")
    if action == "handle_consultation_flow":
        return "consultation_flow_handler"
    elif action == "start_specific_spread_consultation":
        return "start_specific_spread_consultation"
    elif action == "handle_context_reference":
        return "context_reference_handler"
    elif action == "handle_exception":
        return "exception_handler"
    elif action == "emotional_support":
        return "emotional_support_handler"
    elif action == "input_clarification":
        return "input_clarification_handler"
    else:
        intent = state.get("user_intent", "unknown")
        return {
            "card_info": "card_info_handler",
            "spread_info": "spread_info_handler",
            "consultation": "consultation_handler", 
            "general": "general_handler",
            "simple_card": "simple_card_handler"
        }.get(intent, "unknown_handler")
def perform_multilayer_spread_search(keywords: str, user_input: str) -> List[Dict]:
    global rag_system
    from parsing.parser.tarot_agent.utils.tools import rag_system
    """다층적 스프레드 검색 - 의도와 주제를 분리하여 검색"""
    recommended_spreads = []
    # 키워드를 의도와 주제로 분리
    keyword_parts = keywords.split()
    intent_keywords = []
    topic_keywords = []
    intent_terms = ['decision', 'choice', 'crossroads', 'dilemma', 'uncertainty', 'confusion', 
                   'doubt', 'guidance', 'advice', 'direction', 'clarity', 'insight', 
                   'understanding', 'future', 'prediction', 'timing']
    for word in keyword_parts:
        if word in intent_terms:
            intent_keywords.append(word)
        else:
            topic_keywords.append(word)
    print(f"🔍 키워드 분리 - 의도: {intent_keywords}, 주제: {topic_keywords}")
    try:
        if rag_system:
            search_attempts = [
                f"{keywords} tarot spread",
                f"{' '.join(intent_keywords)} {' '.join(topic_keywords)} spread",
                f"{' '.join(intent_keywords)} spread" if intent_keywords else None,
                f"{' '.join(topic_keywords)} spread" if topic_keywords else None,
                keywords
            ]
            search_attempts = [query for query in search_attempts if query]
            existing_names = set()
            for i, query in enumerate(search_attempts, 1):
                print(f"🔍 {i}차 검색: {query}")
                try:
                    results = rag_system.search_spreads(query, final_k=8)
                    safe_results = convert_numpy_types(results)
                    print(f"🔍 {i}차 검색 결과: {len(safe_results)}개")
                    if len(safe_results) > 0:
                        print(f"✅ {i}차 검색 성공")
                        for doc, score in safe_results:
                            if len(recommended_spreads) >= 15:
                                break
                            metadata = doc.metadata
                            spread_name = metadata.get('spread_name', f'스프레드 {len(recommended_spreads)+1}')
                            if spread_name not in existing_names:
                                existing_names.add(spread_name)
                                spread_data = {
                                    "number": len(recommended_spreads) + 1,
                                    "spread_name": spread_name,
                                    "card_count": metadata.get('card_count', 3),
                                    "positions": metadata.get("positions", []),
                                    "description": metadata.get("description", ""),
                                    "search_layer": i,
                                    "relevance_score": float(score) if hasattr(score, 'item') else score
                                }
                                recommended_spreads.append(spread_data)
                except Exception as e:
                    print(f"🔍 {i}차 검색 실패: {e}")
                    continue
        if len(recommended_spreads) < 3:
            raise Exception("모든 검색 시도 실패")
    except Exception as e:
        print(f"🔍 다층 검색 실패: {e}, 기본 스프레드 사용")
        default_spreads = get_default_spreads()
        for i, spread in enumerate(default_spreads[:3]):
            spread_data = {
                "number": i + 1,
                "spread_name": spread.get('spread_name', f'스프레드 {i+1}'),
                "card_count": spread.get('card_count', 3),
                "positions": spread.get("positions", []),
                "description": spread.get("description", ""),
                "search_layer": 0,
                "relevance_score": 0.5
            }
            recommended_spreads.append(spread_data)
    print(f"🔍 최종 추천 스프레드: {len(recommended_spreads)}개")
    for spread in recommended_spreads:
        print(f"  - {spread['spread_name']} (검색층: {spread['search_layer']}, 점수: {spread['relevance_score']:.3f})")
    intent_spreads = []
    topic_spreads = []
    mixed_spreads = []
    for spread in recommended_spreads:
        if spread['search_layer'] == 3:
            intent_spreads.append(spread)
        elif spread['search_layer'] == 4:
            topic_spreads.append(spread)
        else:
            mixed_spreads.append(spread)
    print(f"🎯 스프레드 분류 - 의도: {len(intent_spreads)}개, 주제: {len(topic_spreads)}개, 혼합: {len(mixed_spreads)}개")
    final_spreads = []
    if intent_spreads:
        final_spreads.append(intent_spreads[0])
        print(f"✅ 의도 스프레드 포함: {intent_spreads[0]['spread_name']}")
    available_topic = [s for s in topic_spreads if s not in final_spreads]
    if available_topic:
        final_spreads.append(available_topic[0])
        print(f"✅ 주제 스프레드 포함: {available_topic[0]['spread_name']}")
    remaining = [s for s in recommended_spreads if s not in final_spreads]
    if remaining:
        final_spreads.append(remaining[0])
        print(f"✅ 추가 스프레드 포함: {remaining[0]['spread_name']}")
    while len(final_spreads) < 3 and len(final_spreads) < len(recommended_spreads):
        remaining = [s for s in recommended_spreads if s not in final_spreads]
        if remaining:
            final_spreads.append(remaining[0])
        else:
            break
    print(f"🎯 최종 선택된 3개 스프레드:")
    for i, spread in enumerate(final_spreads, 1):
        print(f"  {i}. {spread['spread_name']} (검색층: {spread['search_layer']}, 점수: {spread['relevance_score']:.3f})")
    return final_spreads[:3]
def performance_monitor(func):
    """함수 실행 시간 측정 데코레이터"""
    from functools import wraps
    import time
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"⏱️ {func.__name__} 실행 시간: {end - start:.2f}초")
        return result
    return wrapper
def create_optimized_consultation_flow():
    """최적화된 상담 플로우 생성"""
    from concurrent.futures import ThreadPoolExecutor
    @performance_monitor
    def parallel_emotion_and_search_analysis(state: TarotState) -> TarotState:
        """감정 분석과 웹 검색 판단을 병렬로 실행"""
        user_input = state.get("user_input", "")
        print("🔧 병렬 분석 노드 실행 (감정 + 웹검색)")
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 병렬 실행
            emotion_future = executor.submit(analyze_emotion_and_empathy, user_input)
            search_future = executor.submit(web_search_decider_node, state)
            # 결과 병합
            emotion_result = emotion_future.result()
            search_result = search_future.result()
            # 두 결과를 병합
            combined_state = {**state}
            combined_state.update(emotion_result)
            combined_state.update(search_result)
            return combined_state
    @performance_monitor  
    def cached_spread_search(state: TarotState) -> TarotState:
        """캐시된 스프레드 검색"""
        user_input = state.get("user_input", "")
        # 간단한 메모리 캐시 (실제 운영에서는 Redis 등 사용)
        cache_key = f"spread_search_{hash(user_input)}"
        # 캐시 확인 (여기서는 state에 저장)
        cached_result = state.get("spread_cache", {}).get(cache_key)
        if cached_result:
            print("🔧 캐시된 스프레드 검색 결과 사용")
            return cached_result
        # 캐시 없으면 실제 검색 실행
        result = spread_recommender_node(state)
        # 캐시에 저장
        spread_cache = state.get("spread_cache", {})
        spread_cache[cache_key] = result
        result["spread_cache"] = spread_cache
        return result
    return {
        "parallel_emotion_and_search": parallel_emotion_and_search_analysis,
        "cached_spread_search": cached_spread_search
    }
def create_smart_routing_system():
    """스마트 라우팅 시스템 - 사용자 패턴 학습"""
    def analyze_user_pattern(state: TarotState) -> dict:
        """사용자 패턴 분석"""
        user_input = state.get("user_input", "")
        conversation_memory = state.get("conversation_memory", {})
        # 사용자 패턴 분석
        pattern = {
            "input_length": len(user_input),
            "has_specific_concern": any(keyword in user_input.lower() for keyword in ["고민", "문제", "걱정", "궁금"]),
            "emotional_intensity": "high" if any(keyword in user_input for keyword in ["너무", "정말", "진짜"]) else "normal",
            "previous_consultations": len(conversation_memory.get("consultations", [])),
            "preferred_style": conversation_memory.get("preferred_style", "detailed")
        }
        return pattern
    def smart_route_decision(state: TarotState) -> str:
        """스마트 라우팅 결정"""
        pattern = analyze_user_pattern(state)
        # 패턴 기반 라우팅 최적화
        if pattern["has_specific_concern"] and pattern["input_length"] > 10:
            return "fast_consultation_track"  # 빠른 상담 트랙
        elif pattern["previous_consultations"] > 3:
            return "experienced_user_track"  # 숙련 사용자 트랙
        else:
            return "standard_track"  # 표준 트랙
    return {
        "analyze_pattern": analyze_user_pattern,
        "smart_route": smart_route_decision
    }
def create_quality_assurance_system():
    """품질 보증 시스템"""
    def validate_consultation_quality(state: TarotState) -> dict:
        """상담 품질 검증"""
        consultation_data = state.get("consultation_data", {})
        messages = state.get("messages", [])
        quality_score = 0.0
        issues = []
        # 1. 메시지 품질 확인
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'content'):
                content = last_message.content
                # 길이 확인
                if len(content) > 100:
                    quality_score += 0.3
                else:
                    issues.append("응답이 너무 짧음")
                # 감정적 지원 확인
                emotional_keywords = ["마음", "느낌", "이해", "공감", "위로"]
                if any(keyword in content for keyword in emotional_keywords):
                    quality_score += 0.2
                # 구체적 조언 확인
                advice_keywords = ["추천", "제안", "조언", "방법", "해보세요"]
                if any(keyword in content for keyword in advice_keywords):
                    quality_score += 0.2
                # 타로 전문성 확인
                tarot_keywords = ["카드", "스프레드", "타로", "해석", "의미"]
                if any(keyword in content for keyword in tarot_keywords):
                    quality_score += 0.3
        # 2. 상담 데이터 완성도 확인
        if consultation_data:
            if consultation_data.get("emotional_analysis"):
                quality_score += 0.1
            if consultation_data.get("recommended_spreads"):
                quality_score += 0.1
        return {
            "quality_score": min(quality_score, 1.0),
            "issues": issues,
            "passed": quality_score >= 0.7
        }
    def auto_improvement_suggestions(quality_result: dict) -> list:
        """자동 개선 제안"""
        suggestions = []
        if quality_result["quality_score"] < 0.7:
            suggestions.append("응답의 감정적 지원 강화 필요")
            suggestions.append("더 구체적인 타로 해석 제공 필요")
        for issue in quality_result["issues"]:
            if "짧음" in issue:
                suggestions.append("응답 길이 증가 필요")
        return suggestions
    return {
        "validate_quality": validate_consultation_quality,
        "improve_suggestions": auto_improvement_suggestions
    }
def create_advanced_error_recovery():
    """고급 오류 복구 시스템"""
    def graceful_fallback(state: TarotState, error: Exception) -> TarotState:
        """우아한 폴백 처리"""
        user_input = state.get("user_input", "")
        print(f"🔧 우아한 폴백 실행: {type(error).__name__}")
        # 오류 유형별 맞춤 응답
        if "LLM" in str(error) or "OpenAI" in str(error):
            fallback_message = "🔮 잠시 마음을 가다듬고 있어요. 다시 한 번 말씀해주시겠어요?"
        elif "search" in str(error).lower():
            fallback_message = "🔮 검색 중 문제가 있었지만, 기본 지식으로 도움을 드릴 수 있어요. 어떤 고민이 있으신가요?"
        elif "rag" in str(error).lower():
            fallback_message = "🔮 자료 검색에 문제가 있었지만, 경험을 바탕으로 상담해드릴게요. 고민을 말씀해주세요."
        else:
            fallback_message = "🔮 예상치 못한 상황이 발생했지만, 최선을 다해 도움을 드리겠습니다. 다시 시도해주세요."
        return {
            "messages": [AIMessage(content=fallback_message)],
            "error_recovered": True,
            "error_type": type(error).__name__
        }
    def retry_with_backoff(func, max_retries=3):
        """백오프와 함께 재시도"""
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    wait_time = 2 ** attempt  # 지수적 백오프
                    print(f"🔧 재시도 {attempt + 1}/{max_retries}, {wait_time}초 대기")
                    time.sleep(wait_time)
        return wrapper
    return {
        "graceful_fallback": graceful_fallback,
        "retry_with_backoff": retry_with_backoff
    }
# 최적화된 시스템 인스턴스 생성

optimized_flows = create_optimized_consultation_flow()

smart_routing = create_smart_routing_system()

quality_assurance = create_quality_assurance_system()

error_recovery = create_advanced_error_recovery()

print("✅ Phase 3 최적화 시스템 초기화 완료")

def handle_casual_new_question(user_input: str, llm) -> TarotState:
    """🆕 일상적 새로운 질문 처리"""
    casual_prompt = f"""
    사용자가 일상적인 질문을 했습니다: "{user_input}"
    **요청사항**:
    1. 이 질문에 대해 **가볍고 친근하게** 답변해주세요
    2. 강요하지 않고 자연스럽게 타로 옵션을 제안해주세요
    **답변 형식**:
    [질문에 대한 가벼운 답변]
    만약 카드 한 장을 뽑아 [질문 관련 주제]를 더 깊이 알아보길 원하신다면 '네'라고 답해 주세요. 그리고 본격적인 타로 상담을 원하신다면 '타로 봐줘'라고 말씀해 주세요!
    **예시**:
    - "짬뽕 vs 짜장면" → "당신의 선택 성향을"
    - "오늘 뭐 입을까" → "당신의 스타일 감각을"
    - "비 올까?" → "날씨에 대한 직감을"
    """
    try:
        response = llm.invoke([HumanMessage(content=casual_prompt)])
        return {"messages": [response]}
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"🔮 {user_input}에 대해서는 여러 가지 관점이 있을 수 있어요! 만약 카드 한 장을 뽑아 이 선택에 대한 직감을 더 깊이 알아보길 원하신다면 '네'라고 답해 주세요. 그리고 본격적인 타로 상담을 원하신다면 '타로 봐줘'라고 말씀해 주세요!")]
        }
def handle_tarot_related_question(state: TarotState, user_input: str, recent_ai_content: str, llm) -> TarotState:
    """🔧 타로 관련 질문 처리 (기존 로직)"""
    conversation_memory = state.get("conversation_memory", {})
    # 🔧 개별 카드 해석이 이미 나왔는지 확인
    already_showed_individual = False
    if recent_ai_content:
        # 개별 카드 해석 완료를 나타내는 여러 패턴 확인
        completion_patterns = [
            "## 🔮 **카드 해석**",
            "🔮 **이제 종합적으로 말해줄게요**",
            "상담이 완료되었습니다!",
            "## 💡 **상세한 실용적 조언**",
            "종합 해석:",
            "🃏 **",  # 개별 카드 해석 시작 패턴
        ]
        # 패턴이 여러 개 발견되면 개별 해석이 이미 완료된 것으로 판단
        pattern_count = sum(1 for pattern in completion_patterns if pattern in recent_ai_content)
        if pattern_count >= 2:  # 2개 이상 패턴이 발견되면 개별 해석 완료로 판단
            already_showed_individual = True
            print(f"🔧 개별 해석 완료 감지: {pattern_count}개 패턴 발견")
    # 🔧 개별 해석 여부에 따라 다른 프롬프트 사용
    if already_showed_individual:
        ending_instruction = """자연스럽고 도움이 되는 답변을 해주세요.
**반드시 마지막에 다음 문구를 추가**:

"도움이 되셨나요? 추가로 궁금한 점이 있으시면 언제든 말씀해주세요! 😊" """
    else:
        ending_instruction = """자연스럽고 도움이 되는 답변을 해주세요.
**반드시 마지막에 다음 문구를 추가**:

"설명이 도움이 되셨을까요? 개별 카드 해석을 보고 싶으시다면 \"네\" 또는 \"보고 싶어\"라고 말해주세요! 😊" """
    prompt = f"""
    당신은 타로 상담사입니다. 사용자가 방금 전 답변에 대해 추가 질문을 했습니다.
    **사용자 추가 질문:** "{user_input}"
    **방금 전 내가 한 답변들:**
    {recent_ai_content}
    **핵심 원칙:**
    1. 사용자가 방금 전 답변의 **어떤 부분**에 대해 궁금해하는지 파악
    2. 그 부분을 **구체적이고 상세하게** 재설명
    3. 타로 상담사로서 **전문적이면서도 친근한** 톤 유지
    4. 필요하면 **추가 배경 지식**도 제공
    **가능한 질문 유형들:**
    - 시기 관련: "어떻게 그런 시기가 나온거야?"
    - 카드 의미: "그 카드가 정확히 뭘 의미하는거야?"
    - 조합 해석: "왜 그렇게 해석되는거야?"
    - 실용적 조언: "구체적으로 어떻게 해야 해?"
    - 배경 원리: "타로가 어떻게 그걸 알 수 있어?"
    - 확신도: "얼마나 확실한거야?"
    - 예외상황: "만약에 이렇게 되면 어떻게 해?"
    **답변 방식:**
    1. 먼저 사용자 질문에 **직접적으로** 답변
    2. 그 다음 **배경 설명**이나 **추가 정보** 제공
    3. **실용적 조언**이나 **격려** 추가
    {ending_instruction}
    """
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        # 이번 질문도 메모리에 추가
        updated_memory = conversation_memory.copy() if conversation_memory else {}
        updated_memory.setdefault("followup_questions", []).append({
            "question": user_input,
            "answered_about": extract_question_topic(user_input),
            "timestamp": len(state.get("messages", []))
        })
        return {
            "messages": [response],
            "conversation_memory": updated_memory
        }
    except Exception as e:
        print(f"❌ 타로 관련 질문 처리 오류: {e}")
        # 🔧 에러 메시지도 개별 해석 여부에 따라 조정
        if already_showed_individual:
            error_msg = "🔮 설명하는 중 문제가 생겼어요. 다른 방식으로 질문해주시겠어요? 추가로 궁금한 점이 있으시면 언제든 말씀해주세요! 😊"
        else:
            error_msg = "🔮 설명하는 중 문제가 생겼어요. 다른 방식으로 질문해주시겠어요? 개별 카드 해석을 보고 싶으시다면 \"네\" 또는 \"보고 싶어\"라고 말해주세요! 😊"
        return {
            "messages": [AIMessage(content=error_msg)]
        }

def extract_question_topic(user_input: str) -> str:
    """사용자 질문이 어떤 주제인지 간단히 추출"""
    input_lower = user_input.lower()
    if any(keyword in input_lower for keyword in ["시기", "언제", "타이밍", "시간"]):
        return "timing"
    elif any(keyword in input_lower for keyword in ["카드", "의미", "뜻"]):
        return "card_meaning"
    elif any(keyword in input_lower for keyword in ["어떻게", "왜", "근거"]):
        return "explanation"
    elif any(keyword in input_lower for keyword in ["조언", "해야", "방법"]):
        return "advice"
    elif any(keyword in input_lower for keyword in ["확실", "정확", "맞나"]):
        return "confidence"
    else:
        return "general"
# TAROT_CARDS는 별도 모듈에서 import 하도록 처리 필요 (실제 정의가 위에 추가됨)

def integrate_search_results_with_tarot(tarot_cards: List[Dict], search_results: dict, user_concern: str) -> str:
    """검색 결과를 타로 해석에 통합"""
    if not search_results.get("success") or not search_results.get("results"):
        return ""
    # 검색 결과 요약
    search_summary = ""
    results_data = search_results["results"]
    # 딕셔너리인 경우 처리
    if isinstance(results_data, dict):
        if "results" in results_data:
            results = results_data["results"]
        elif "data" in results_data:
            results = results_data["data"]
        else:
            results = [results_data]
    elif isinstance(results_data, list):
        results = results_data
    else:
        return ""
    if isinstance(results, list) and len(results) > 0:
        top_results = results[:3]
        search_summary = "\n".join([
            f"- {result.get('title', '제목 없음')}: {result.get('content', result.get('snippet', '내용 없음'))[:200]}"
            for result in top_results
            if isinstance(result, dict)
        ])
    if not search_summary:
        return ""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    card_summary = ", ".join([card.get("name", "알 수 없는 카드") for card in tarot_cards])
    prompt = f"""
    타로 카드 해석에 현실적 정보를 통합하여 조언을 제공해주세요.
    **사용자 고민:** {user_concern}
    **선택된 타로 카드:** {card_summary}
    **현실 정보 (웹 검색 결과):**
    {search_summary}
    **필수 요구사항:**
    1. 타로 카드의 상징적 의미와 현실 정보를 균형있게 결합하세요
    2. 검색 결과에서 얻은 구체적인 정보를 반드시 포함하세요
    3. 검색 결과를 단순히 언급하는 것이 아니라, 타로 해석과 깊이 통합하세요
    4. 사용자가 실제로 행동할 수 있는 구체적인 방향을 제시하세요
    5. 검색 결과와 타로 해석이 상충될 경우, 두 관점을 모두 제시하고 균형잡힌 조언을 제공하세요
    6. 검색 결과에서 얻은 구체적인 사실, 통계, 전문가 의견 등을 반드시 활용하세요
    7. 검색 결과의 주요 키워드와 개념을 타로 카드의 상징과 연결하세요
    **통합 해석 및 조언:**
    """
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        integration_result = response.content.strip()
        print(f"✨ 타로-현실 통합 해석 생성 완료 ({len(integration_result)}자)")
        return integration_result
    except Exception as e:
        print(f"❌ 통합 해석 생성 오류: {e}")
        return ""
def format_search_results_for_display(search_results: dict) -> str:
    """검색 결과를 사용자에게 보여줄 형태로 포맷"""
    if not search_results.get("success") or not search_results.get("results"):
        return ""
    results_data = search_results["results"]
    # 딕셔너리인 경우 처리
    if isinstance(results_data, dict):
        if "results" in results_data:
            results = results_data["results"]
        elif "data" in results_data:
            results = results_data["data"]
        else:
            results = [results_data]
    elif isinstance(results_data, list):
        results = results_data
    else:
        return ""
    if not isinstance(results, list) or len(results) == 0:
        return ""
    formatted = f"\n\n📊 **참고한 현실 정보** (출처: {search_results.get('source', '웹 검색')}):\n"
    for i, result in enumerate(results[:3], 1):
        if isinstance(result, dict):
            title = result.get('title', '제목 없음')
            content = result.get('content', result.get('snippet', '내용 없음'))
            url = result.get('url', '')
            if len(content) > 150:
                content = content[:150] + "..."
            formatted += f"{i}. **{title}**\n   {content}\n"
            if url:
                formatted += f"   🔗 {url}\n"
            formatted += "\n"
    return formatted
def get_current_context() -> dict:
   """현재 시간 맥락 정보 생성"""
   # 한국 시간 기준
   kst = pytz.timezone('Asia/Seoul')
   now = datetime.now(kst)
   return {
       "current_date": now.strftime("%Y년 %m월 %d일"),
       "current_year": now.year,
       "current_month": now.month,
       "current_day": now.day,
       "weekday": now.strftime("%A"),
       "weekday_kr": get_weekday_korean(now.weekday()),
       "season": get_season(now.month),
       "quarter": f"{now.year}년 {(now.month-1)//3 + 1}분기",
       "recent_period": f"최근 {get_recent_timeframe(now)}",
       "timestamp": now.isoformat(),
       "unix_timestamp": int(now.timestamp())
   }
def get_weekday_korean(weekday: int) -> str:
    """요일을 한국어로 변환 (0=월요일, 6=일요일)"""
    weekdays = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
    return weekdays[weekday]
def get_season(month: int) -> str:
    """계절 정보"""
    if month in [12, 1, 2]:
        return "겨울"
    elif month in [3, 4, 5]:
        return "봄"
    elif month in [6, 7, 8]:
        return "여름"
    else:
        return "가을"
def get_recent_timeframe(now) -> str:
    """최근 기간 표현"""
    return f"{now.year}년 {now.month}월 기준"
def calculate_days_until_target(target_month: int, target_day: int = 1) -> int:
   """특정 날짜까지 남은 일수 계산"""
   kst = pytz.timezone('Asia/Seoul')
   now = datetime.now(kst)
   # 올해 목표 날짜
   target_date = datetime(now.year, target_month, target_day, tzinfo=kst)
   # 이미 지났으면 내년 날짜로
   if target_date < now:
       target_date = datetime(now.year + 1, target_month, target_day, tzinfo=kst)
   delta = target_date - now
   return delta.days
def get_time_period_description(days: int) -> str:
    """일수를 기간 표현으로 변환"""
    if days <= 7:
        return f"{days}일 이내"
    elif days <= 30:
        weeks = days // 7
        return f"약 {weeks}주 후"
    elif days <= 365:
        months = days // 30
        return f"약 {months}개월 후"
    else:
        years = days // 365
        return f"약 {years}년 후"
def integrate_timing_with_current_date(tarot_timing: dict, current_context: dict) -> dict:
    """타로 시기 분석과 현재 날짜 정보 통합"""
    import pytz
    from datetime import datetime, timedelta
    kst = pytz.timezone('Asia/Seoul')
    current_date = datetime.now(kst)
    concrete_timing = []
    timing_list = tarot_timing.get("timing_predictions", [tarot_timing])
    for timing in timing_list:
        days_min = timing.get("days_min", 1)
        days_max = timing.get("days_max", 7)
        start_date = current_date + timedelta(days=days_min)
        end_date = current_date + timedelta(days=days_max)
        if start_date.year != current_date.year or end_date.year != current_date.year:
            period_str = f"{start_date.strftime('%Y년 %m월 %d일')} ~ {end_date.strftime('%Y년 %m월 %d일')}"
        elif start_date.month != end_date.month:
            period_str = f"{start_date.strftime('%m월 %d일')} ~ {end_date.strftime('%m월 %d일')}"
        else:
            period_str = f"{start_date.strftime('%m월 %d일')} ~ {end_date.strftime('%d일')}"
        concrete_timing.append({
            "period": period_str,
            "description": timing.get("description", ""),
            "confidence": timing.get("confidence", "보통"),
            "days_from_now": f"{days_min}-{days_max}일 후"
        })
    return {"concrete_timing": concrete_timing}
def ensure_temporal_context(state: TarotState) -> TarotState:
   """상태에 시간 맥락 정보가 없으면 추가"""
   if not state.get("temporal_context"):
       state["temporal_context"] = get_current_context()
   return state
def calculate_card_draw_probability(deck_size: int = 78, cards_of_interest: int = 1, 
                                 cards_drawn: int = 3, exact_matches: int = 1) -> dict:
   """하이퍼기하분포를 이용한 정확한 카드 확률 계산"""
   try:
       # 하이퍼기하분포: hypergeom(M, n, N)
       # M: 전체 개수 (78), n: 관심 카드 수, N: 뽑는 카드 수
       rv = hypergeom(deck_size, cards_of_interest, cards_drawn)
       # 정확히 exact_matches개 뽑을 확률
       exact_prob = rv.pmf(exact_matches)
       # 1개 이상 뽑을 확률
       at_least_one = 1 - rv.pmf(0)
       # 평균과 분산
       mean = rv.mean()
       variance = rv.var()
       return {
           "exact_probability": float(exact_prob),
           "at_least_one_probability": float(at_least_one),
           "expected_value": float(mean),
           "variance": float(variance),
           "distribution_type": "hypergeometric"
       }
   except Exception as e:
       return {
           "error": str(e),
           "exact_probability": 0.0,
           "at_least_one_probability": 0.0
       }
def calculate_success_probability_from_cards(selected_cards: List[Dict]) -> dict:
   """선택된 카드들을 기반으로 성공 확률 계산"""
   if not selected_cards:
       return {"success_probability": 0.5, "confidence": "low", "factors": []}
   total_weight = 0
   positive_factors = []
   negative_factors = []
   # 카드별 성공 가중치 (전통적 타로 해석 기반)
   success_weights = {
       # Major Arcana 성공 가중치
       "The Fool": 0.6,  # 새로운 시작
       "The Magician": 0.9,  # 의지력과 실행력
       "The High Priestess": 0.7,  # 직관과 지혜
       "The Empress": 0.8,  # 풍요와 창조
       "The Emperor": 0.8,  # 리더십과 안정
       "The Hierophant": 0.7,  # 전통과 지도
       "The Lovers": 0.8,  # 선택과 조화
       "The Chariot": 0.9,  # 의지와 승리
       "Strength": 0.8,  # 내적 힘
       "The Hermit": 0.6,  # 성찰과 지혜
       "Wheel of Fortune": 0.7,  # 운명의 변화
       "Justice": 0.8,  # 균형과 공정
       "The Hanged Man": 0.4,  # 정체와 희생
       "Death": 0.5,  # 변화와 전환
       "Temperance": 0.8,  # 조화와 절제
       "The Devil": 0.3,  # 속박과 유혹
       "The Tower": 0.2,  # 파괴와 충격
       "The Star": 0.9,  # 희망과 영감
       "The Moon": 0.4,  # 환상과 불안
       "The Sun": 0.95,  # 성공과 기쁨
       "Judgement": 0.7,  # 부활과 깨달음
       "The World": 0.95,  # 완성과 성취
       # Minor Arcana - Suit별 기본 가중치
       "Ace": 0.8,  # 새로운 시작
       "Two": 0.6,  # 균형과 협력
       "Three": 0.7,  # 창조와 성장
       "Four": 0.7,  # 안정과 기반
       "Five": 0.3,  # 갈등과 도전
       "Six": 0.8,  # 조화와 균형
       "Seven": 0.5,  # 도전과 시험
       "Eight": 0.7,  # 숙련과 발전
       "Nine": 0.8,  # 완성 근접
       "Ten": 0.6,  # 완성과 부담
       "Page": 0.6,  # 학습과 메시지
       "Knight": 0.7,  # 행동과 모험
       "Queen": 0.8,  # 성숙과 지혜
       "King": 0.8   # 마스터리와 리더십
   }
   # Suit별 보정 계수
   suit_modifiers = {
       "Wands": 0.1,    # 불 - 적극적 에너지
       "Cups": 0.05,    # 물 - 감정적 만족
       "Swords": -0.05, # 공기 - 갈등과 도전
       "Pentacles": 0.08 # 흙 - 실질적 성과
   }
   for card in selected_cards:
       card_name = card.get("name", "")
       orientation = card.get("orientation", "upright")
       # 기본 가중치 계산
       weight = 0.5  # 기본값
       # Major Arcana 체크
       if card_name in success_weights:
           weight = success_weights[card_name]
       else:
           # Minor Arcana - rank 기반
           for rank in success_weights:
               if rank in card_name and rank not in ["Ace", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"]:
                   weight = success_weights[rank]
                   break
           # Suit 보정
           for suit, modifier in suit_modifiers.items():
               if suit in card_name:
                   weight += modifier
                   break
       # 역방향 보정
       if orientation == "reversed":
           if weight > 0.5:
               weight = 1.0 - weight  # 긍정적 카드는 부정적으로
           else:
               weight = min(0.8, weight + 0.2)  # 부정적 카드는 약간 완화
       total_weight += weight
       # 요인 분석
       if weight >= 0.7:
           positive_factors.append(f"{card_name} ({orientation}) - 강한 긍정 에너지")
       elif weight >= 0.6:
           positive_factors.append(f"{card_name} ({orientation}) - 긍정적 영향")
       elif weight <= 0.3:
           negative_factors.append(f"{card_name} ({orientation}) - 주의 필요")
       elif weight <= 0.4:
           negative_factors.append(f"{card_name} ({orientation}) - 도전 요소")
   # 평균 성공 확률 계산
   avg_probability = total_weight / len(selected_cards) if selected_cards else 0.5
   # 신뢰도 계산
   if len(positive_factors) >= 2:
       confidence = "high"
   elif len(positive_factors) >= 1 and len(negative_factors) <= 1:
       confidence = "medium"
   else:
       confidence = "low"
   return {
       "success_probability": round(avg_probability, 3),
       "confidence": confidence,
       "positive_factors": positive_factors,
       "negative_factors": negative_factors,
       "total_cards_analyzed": len(selected_cards)
   }
def analyze_card_combination_synergy(selected_cards: List[Dict]) -> dict:
   """카드 조합의 시너지 효과 분석"""
   if len(selected_cards) < 2:
       return {"synergy_score": 0.5, "combinations": [], "warnings": []}
   synergy_score = 0.5
   combinations = []
   warnings = []
   # 원소 조합 분석
   elements = {"Wands": 0, "Cups": 0, "Swords": 0, "Pentacles": 0}
   major_count = 0
   for card in selected_cards:
       card_name = card.get("name", "")
       if any(major in card_name for major in ["The", "Fool", "Magician", "Priestess"]):
           major_count += 1
       else:
           for element in elements:
               if element in card_name:
                   elements[element] += 1
                   break
   # 원소 균형 보너스
   active_elements = sum(1 for count in elements.values() if count > 0)
   if active_elements >= 3:
       synergy_score += 0.1
       combinations.append("다양한 원소의 균형잡힌 조합")
   elif active_elements == 2:
       synergy_score += 0.05
       combinations.append("두 원소의 조화로운 결합")
   # Major Arcana 보너스
   if major_count >= 2:
       synergy_score += 0.15
       combinations.append("강력한 Major Arcana 에너지")
   elif major_count == 1:
       synergy_score += 0.05
       combinations.append("Major Arcana의 지도력")
   # 특별한 조합 패턴
   card_names = [card.get("name", "") for card in selected_cards]
   # 성공 조합
   success_combinations = [
       (["The Magician", "The Star"], 0.2, "의지력과 희망의 완벽한 조합"),
       (["The Sun", "The World"], 0.25, "성공과 완성의 최고 조합"),
       (["Ace of", "The Fool"], 0.15, "새로운 시작"),
       (["Queen", "King"], 0.1, "성숙한 리더십의 조화")
   ]
   for combo_cards, bonus, description in success_combinations:
       if all(any(combo_card in card_name for card_name in card_names) 
              for combo_card in combo_cards):
           synergy_score += bonus
           combinations.append(description)
   # 경고 조합
   warning_combinations = [
       (["The Tower", "Death"], "급격한 변화와 파괴의 이중 충격"),
       (["The Devil", "The Moon"], "혼란과 속박의 위험한 조합"),
       (["Five of", "Seven of"], "갈등과 도전이 겹치는 어려운 시기")
   ]
   for combo_cards, warning in warning_combinations:
        if all(any(combo_card in card_name for card_name in card_names) 
               for combo_card in combo_cards):
            synergy_score -= 0.1
            warnings.append(warning)
   return {
       "synergy_score": round(max(0.1, min(1.0, synergy_score)), 3),
       "combinations": combinations,
       "warnings": warnings,
       "element_distribution": elements,
       "major_arcana_count": major_count
   }
def analyze_elemental_balance(selected_cards: List[Dict]) -> dict:
   """카드의 원소 균형 분석"""
   elements = {
       "Fire": {"count": 0, "cards": [], "keywords": ["열정", "행동", "창조", "에너지"]},
       "Water": {"count": 0, "cards": [], "keywords": ["감정", "직감", "관계", "치유"]},
       "Air": {"count": 0, "cards": [], "keywords": ["사고", "소통", "갈등", "변화"]},
       "Earth": {"count": 0, "cards": [], "keywords": ["물질", "안정", "실용", "성장"]}
   }
   # 원소 매핑
   element_mapping = {
       "Wands": "Fire",
       "Cups": "Water", 
       "Swords": "Air",
       "Pentacles": "Earth"
   }
   # Major Arcana 원소 분류
   major_elements = {
       "The Fool": "Air", "The Magician": "Fire", "The High Priestess": "Water",
       "The Empress": "Earth", "The Emperor": "Fire", "The Hierophant": "Earth",
       "The Lovers": "Air", "The Chariot": "Water", "Strength": "Fire",
       "The Hermit": "Earth", "Wheel of Fortune": "Fire", "Justice": "Air",
       "The Hanged Man": "Water", "Death": "Water", "Temperance": "Fire",
       "The Devil": "Earth", "The Tower": "Fire", "The Star": "Air",
       "The Moon": "Water", "The Sun": "Fire", "Judgement": "Fire",
       "The World": "Earth"
   }
   total_cards = len(selected_cards)
   for card in selected_cards:
       card_name = card.get("name", "")
       element = None
       # Minor Arcana 원소 확인
       for suit, elem in element_mapping.items():
           if suit in card_name:
               element = elem
               break
       # Major Arcana 원소 확인
       if not element and card_name in major_elements:
           element = major_elements[card_name]
       if element:
           elements[element]["count"] += 1
           elements[element]["cards"].append(card_name)
   # 균형 분석
   if total_cards > 0:
       percentages = {elem: (data["count"] / total_cards) * 100 
                     for elem, data in elements.items()}
   else:
       percentages = {elem: 0 for elem in elements}
   # 지배적 원소 찾기
   dominant_element = max(elements.keys(), key=lambda x: elements[x]["count"])
   missing_elements = [elem for elem, data in elements.items() if data["count"] == 0]
   # 균형 점수 (0-1)
   balance_score = 1.0 - (max(percentages.values()) - 25) / 75 if max(percentages.values()) > 25 else 1.0
   return {
       "elements": elements,
       "percentages": percentages,
       "dominant_element": dominant_element if elements[dominant_element]["count"] > 0 else None,
       "missing_elements": missing_elements,
       "balance_score": round(balance_score, 3),
       "interpretation": generate_elemental_interpretation(elements, dominant_element, missing_elements)
   }
def generate_elemental_interpretation(elements: dict, dominant: str, missing: list) -> str:
   """원소 분석 결과 해석 생성"""
   interpretations = []
   if dominant and elements[dominant]["count"] > 0:
       element_meanings = {
           "Fire": "강한 행동력과 열정이 지배적입니다. 적극적으로 추진하되 성급함을 주의하세요.",
           "Water": "감정과 직감이 중요한 역할을 합니다. 관계와 내면의 소리에 귀 기울이세요.",
           "Air": "사고와 소통이 핵심입니다. 명확한 계획과 의사소통이 성공의 열쇠입니다.",
           "Earth": "실용적이고 안정적인 접근이 필요합니다. 차근차근 기반을 다지세요."
       }
       interpretations.append(element_meanings.get(dominant, ""))
   if missing:
       missing_advice = {
           "Fire": "더 적극적이고 열정적인 행동이 필요합니다.",
           "Water": "감정적 측면과 직감을 더 고려해보세요.",
           "Air": "논리적 사고와 소통을 강화하세요.",
           "Earth": "현실적이고 실용적인 계획이 부족합니다."
       }
       for elem in missing:
           interpretations.append(missing_advice.get(elem, ""))
   return " ".join(interpretations)
def calculate_numerological_significance(selected_cards: List[Dict]) -> dict:
   """카드의 수비학적 의미 분석"""
   if not selected_cards:
       return {"total_value": 0, "reduced_value": 0, "meaning": ""}
   # 카드별 수비학 값
   numerology_values = {
       # Major Arcana
       "The Fool": 0, "The Magician": 1, "The High Priestess": 2, "The Empress": 3,
       "The Emperor": 4, "The Hierophant": 5, "The Lovers": 6, "The Chariot": 7,
       "Strength": 8, "The Hermit": 9, "Wheel of Fortune": 10, "Justice": 11,
       "The Hanged Man": 12, "Death": 13, "Temperance": 14, "The Devil": 15,
       "The Tower": 16, "The Star": 17, "The Moon": 18, "The Sun": 19,
       "Judgement": 20, "The World": 21,
       # Minor Arcana ranks
       "Ace": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5,
       "Six": 6, "Seven": 7, "Eight": 8, "Nine": 9, "Ten": 10,
       "Page": 11, "Knight": 12, "Queen": 13, "King": 14
   }
   total_value = 0
   card_values = []
   for card in selected_cards:
       card_name = card.get("name", "")
       value = 0
       # Major Arcana 체크
       if card_name in numerology_values:
           value = numerology_values[card_name]
       else:
           # Minor Arcana rank 체크
           for rank, num_value in numerology_values.items():
               if rank in card_name:
                   value = num_value
                   break
       total_value += value
       card_values.append({"card": card_name, "value": value})
   # 수비학적 환원 (한 자리 수까지)
   reduced_value = total_value
   while reduced_value > 9 and reduced_value not in [11, 22, 33]:  # 마스터 넘버 제외
       reduced_value = sum(int(digit) for digit in str(reduced_value))
   # 수비학적 의미
   numerology_meanings = {
       0: "무한한 가능성과 새로운 시작",
       1: "리더십과 독립성, 새로운 시작",
       2: "협력과 균형, 파트너십",
       3: "창조성과 표현, 소통",
       4: "안정성과 질서, 근면",
       5: "자유와 모험, 변화",
       6: "책임과 보살핌, 조화",
       7: "영성과 내면 탐구, 완벽",
       8: "물질적 성공과 권력, 성취",
       9: "완성과 지혜, 봉사",
       11: "직감과 영감, 마스터 넘버",
       22: "마스터 빌더, 큰 꿈의 실현",
       33: "마스터 교사, 무조건적 사랑"
   }
   return {
       "total_value": total_value,
       "reduced_value": reduced_value,
       "meaning": numerology_meanings.get(reduced_value, "특별한 의미"),
       "card_values": card_values,
       "is_master_number": reduced_value in [11, 22, 33]
   }
def generate_integrated_analysis(selected_cards: List[Dict]) -> dict:
   """확률, 원소, 수비학을 통합한 종합 분석"""
   # 각 분석 실행
   success_analysis = calculate_success_probability_from_cards(selected_cards)
   synergy_analysis = analyze_card_combination_synergy(selected_cards)
   elemental_analysis = analyze_elemental_balance(selected_cards)
   numerology_analysis = calculate_numerological_significance(selected_cards)
   # 통합 점수 계산
   integrated_score = (
       success_analysis.get("success_probability", 0.5) * 0.4 +
       synergy_analysis.get("synergy_score", 0.5) * 0.3 +
       elemental_analysis.get("balance_score", 0.5) * 0.2 +
       min(1.0, numerology_analysis.get("reduced_value", 5) / 9) * 0.1
   )
   # 종합 해석 생성
   interpretation = []
   # 성공 확률 해석
   success_prob = success_analysis.get("success_probability", 0.5)
   if success_prob >= 0.7:
       interpretation.append("🌟 높은 성공 가능성을 보여줍니다")
   elif success_prob >= 0.6:
       interpretation.append("✨ 긍정적인 결과가 예상됩니다")
   elif success_prob <= 0.4:
       interpretation.append("⚠️ 신중한 접근이 필요합니다")
   # 원소 균형 해석
   if elemental_analysis.get("balance_score", 0) >= 0.7:
       interpretation.append("🔮 원소들이 조화롭게 균형을 이룹니다")
   elif elemental_analysis.get("dominant_element"):
       dominant = elemental_analysis["dominant_element"]
       interpretation.append(f"🔥 {dominant} 원소의 강한 영향을 받습니다")
   # 수비학 해석
   if numerology_analysis.get("is_master_number"):
       interpretation.append(f"✨ 마스터 넘버 {numerology_analysis['reduced_value']}의 특별한 에너지")
   return {
       "integrated_score": round(integrated_score, 3),
       "success_analysis": success_analysis,
       "synergy_analysis": synergy_analysis,
       "elemental_analysis": elemental_analysis,
       "numerology_analysis": numerology_analysis,
       "interpretation": " | ".join(interpretation),
       "recommendation": generate_integrated_recommendation(integrated_score, success_analysis, elemental_analysis)
   }
def generate_integrated_recommendation(score: float, success_analysis: dict, elemental_analysis: dict) -> str:
   """통합 분석 기반 추천사항 생성"""
   recommendations = []
   if score >= 0.7:
       recommendations.append("적극적으로 추진하세요")
   elif score >= 0.6:
       recommendations.append("신중하되 긍정적으로 접근하세요")
   elif score >= 0.5:
       recommendations.append("균형잡힌 접근이 필요합니다")
   else:
       recommendations.append("충분한 준비와 대안을 마련하세요")
   # 원소별 추천
   dominant = elemental_analysis.get("dominant_element")
   if dominant == "Fire":
       recommendations.append("열정을 조절하며 계획적으로 행동하세요")
   elif dominant == "Water":
       recommendations.append("직감을 믿되 현실적 판단도 함께 하세요")
   elif dominant == "Air":
       recommendations.append("소통과 정보 수집에 집중하세요")
   elif dominant == "Earth":
       recommendations.append("안정적이고 실용적인 방법을 선택하세요")
   return " | ".join(recommendations)
