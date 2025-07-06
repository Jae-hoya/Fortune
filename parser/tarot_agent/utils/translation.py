"""

번역 관련 유틸리티 함수들

"""

import json

from langchain_openai import ChatOpenAI

from langchain_core.messages import HumanMessage

_translation_cache = {}

def translate_text_with_llm(english_text: str, text_type: str = "general") -> str:
    """LLM을 사용해서 영어 텍스트를 한국어로 번역 (캐싱 포함)"""
    # 캐시 키 생성
    cache_key = f"{text_type}:{english_text}"
    
    # 캐시에서 확인
    if cache_key in _translation_cache:
        return _translation_cache[cache_key]
    
    try:
        if text_type == "spread_name":
            prompt = f"""
다음 타로 스프레드 이름을 자연스러운 한국어로 번역해주세요. 
타로 용어에 맞게 번역하되, 의미가 명확하게 전달되도록 해주세요.

영어 스프레드 이름: "{english_text}"

번역 결과만 출력해주세요.
"""
        elif text_type == "position_name":
            prompt = f"""
다음 타로 카드 포지션 이름을 자연스러운 한국어로 번역해주세요.
타로 상담에서 사용되는 용어로 번역하되, 의미가 명확하게 전달되도록 해주세요.

영어 포지션 이름: "{english_text}"

번역 결과만 출력해주세요.
"""
        else:
            prompt = f"다음 텍스트를 한국어로 번역해주세요: {english_text}"
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=100)
        response = llm.invoke([HumanMessage(content=prompt)])
        
        translated = response.content.strip()
        result = translated if translated else english_text
        
        # 캐시에 저장
        _translation_cache[cache_key] = result
        return result
        
    except Exception as e:
        print(f"번역 중 오류 발생: {e}")
        # 오류 시에도 캐시에 원본 저장 (재시도 방지)
        _translation_cache[cache_key] = english_text
        return english_text
    
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

def translate_korean_to_english_with_llm(korean_query: str) -> str:
   """LLM을 사용하여 한국어 타로 카드 질문을 영어로 번역"""
   
   llm = ChatOpenAI(
       model="gpt-4o-mini", 
       temperature=0.1,
       model_kwargs={"response_format": {"type": "json_object"}}
   )
   
   translation_prompt = f"""
   사용자의 한국어 타로 카드 질문을 영어 카드명으로 번역해주세요.

   사용자 질문: "{korean_query}"

   **번역 규칙:**
   - 한국어 카드명을 정확한 영어 타로 카드명으로 변환
   - 메이저 아르카나: "연인" → "The Lovers", "별" → "The Star"
   - 마이너 아르카나: "컵의 킹" → "King of Cups", "검의 에이스" → "Ace of Swords"
   - 방향: "역방향", "거꾸로" → "reversed", "정방향" → "upright"
   - 오타나 유사 표현도 추론해서 번역
   - 애매한 표현은 가장 가능성 높은 카드로 번역

   **주요 번역 예시:**
   - "연인", "러버", "사랑카드" → "The Lovers"
   - "별", "스타", "희망카드" → "The Star"  
   - "황제", "임페라토르" → "The Emperor"
   - "컵", "성배", "물의원소" → "Cups"
   - "소드", "검", "공기원소" → "Swords"

   JSON 형식으로 답변하세요:
   {{
       "original_query": "원본 질문",
       "translated_query": "번역된 영어 검색어",
       "card_name": "추론된 카드명 (있다면)",
       "orientation": "upright|reversed|both|unknown",
       "confidence": "high|medium|low"
   }}
   """
   
   try:
       response = llm.invoke([HumanMessage(content=translation_prompt)])
       result = json.loads(response.content)
       
       translated = result.get("translated_query", korean_query)
       confidence = result.get("confidence", "medium")
       
       print(f"🔧 LLM 번역: '{korean_query}' -> '{translated}' (신뢰도: {confidence})")
       
       return translated
       
   except Exception as e:
       print(f"🔧 LLM 번역 실패: {e}, 원본 반환")
       return korean_query
