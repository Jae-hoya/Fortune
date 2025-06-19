"""
LangGraph 시스템 유틸리티
공통 기능들을 모아둔 유틸리티 모듈
- 생년월일시 정보 추출
- 사용자 의도 분석
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# =============================================================================
# 생년월일시 정보 추출
# =============================================================================

def extract_birth_info(user_query: str) -> Optional[Dict[str, Any]]:
    """
    사용자 질문에서 생년월일시 정보 추출
    
    Args:
        user_query: 사용자 질문
    
    Returns:
        Dict: 추출된 생년월일시 정보 또는 None
    """
    
    # 다양한 날짜 형식 패턴
    patterns = [
        # 1995년 8월 26일 오전 10시 15분
        r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일.*?(\d{1,2})시\s*(\d{1,2})분',
        # 1995년 8월 26일 10시 15분
        r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일.*?(\d{1,2})시\s*(\d{1,2})분',
        # 1995년 8월 26일 오전 10시
        r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일.*?(\d{1,2})시',
        # 1995년 8월 26일
        r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일',
        # 1995-08-26 10:15
        r'(\d{4})-(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{1,2})',
        # 1995-08-26
        r'(\d{4})-(\d{1,2})-(\d{1,2})',
        # 1995.08.26
        r'(\d{4})\.(\d{1,2})\.(\d{1,2})',
        # 95년 8월 26일
        r'(\d{2})년\s*(\d{1,2})월\s*(\d{1,2})일'
    ]
    
    birth_info = None
    
    for pattern in patterns:
        match = re.search(pattern, user_query)
        if match:
            groups = match.groups()
            
            try:
                if len(groups) >= 3:
                    year = int(groups[0])
                    month = int(groups[1])
                    day = int(groups[2])
                    
                    # 2자리 년도를 4자리로 변환
                    if year < 100:
                        if year > 50:
                            year += 1900
                        else:
                            year += 2000
                    
                    # 시간 정보
                    hour = 12  # 기본값
                    minute = 0  # 기본값
                    
                    if len(groups) >= 4:
                        hour = int(groups[3])
                    if len(groups) >= 5:
                        minute = int(groups[4])
                    
                    # 오전/오후 처리
                    if "오후" in user_query and hour < 12:
                        hour += 12
                    elif "오전" in user_query and hour == 12:
                        hour = 0
                    
                    # 유효성 검사
                    if _is_valid_date(year, month, day, hour, minute):
                        birth_info = {
                            "year": year,
                            "month": month,
                            "day": day,
                            "hour": hour,
                            "minute": minute,
                            "is_male": _extract_gender(user_query)
                        }
                        break
                        
            except ValueError:
                continue
    
    return birth_info

def _extract_gender(user_query: str) -> bool:
    """
    성별 정보 추출
    
    Args:
        user_query: 사용자 질문
    
    Returns:
        bool: True(남성), False(여성)
    """
    
    # 남성 키워드
    male_keywords = ["남자", "남성", "남", "아들", "형", "동생", "아버지", "아빠"]
    # 여성 키워드  
    female_keywords = ["여자", "여성", "여", "딸", "언니", "누나", "어머니", "엄마"]
    
    male_count = sum(1 for keyword in male_keywords if keyword in user_query)
    female_count = sum(1 for keyword in female_keywords if keyword in user_query)
    
    if male_count > female_count:
        return True
    elif female_count > male_count:
        return False
    else:
        # 기본값은 남성 (통계적으로 더 많음)
        return True

def _is_valid_date(year: int, month: int, day: int, hour: int, minute: int) -> bool:
    """
    날짜 유효성 검사
    
    Args:
        year: 년
        month: 월
        day: 일
        hour: 시
        minute: 분
    
    Returns:
        bool: 유효한 날짜인지 여부
    """
    
    try:
        # 기본 범위 체크
        if not (1900 <= year <= 2100):
            return False
        if not (1 <= month <= 12):
            return False
        if not (1 <= day <= 31):
            return False
        if not (0 <= hour <= 23):
            return False
        if not (0 <= minute <= 59):
            return False
        
        # datetime으로 실제 유효성 검사
        datetime(year, month, day, hour, minute)
        return True
        
    except ValueError:
        return False

def format_birth_info(birth_info: Dict[str, Any]) -> str:
    """
    생년월일시 정보를 문자열로 포맷팅
    
    Args:
        birth_info: 생년월일시 정보
    
    Returns:
        str: 포맷팅된 문자열
    """
    
    if not birth_info:
        return "생년월일시 정보 없음"
    
    gender_str = "남성" if birth_info.get("is_male", True) else "여성"
    
    return (f"{birth_info['year']}년 {birth_info['month']}월 {birth_info['day']}일 "
            f"{birth_info['hour']}시 {birth_info['minute']}분 {gender_str}")

# =============================================================================
# 사용자 의도 분석
# =============================================================================

def analyze_user_intent(user_query: str, llm: ChatOpenAI) -> Dict[str, Any]:
    """
    사용자 질문 의도 분석
    
    Args:
        user_query: 사용자 질문
        llm: LLM 모델
    
    Returns:
        Dict: 의도 분석 결과
    """
    
    # 시스템 프롬프트
    system_prompt = """당신은 사주/운세 상담 시스템의 의도 분석 전문가입니다.
사용자의 질문을 분석하여 다음 중 하나로 분류해주세요:

1. saju_calculation: 사주팔자 계산이 필요한 경우
   - 생년월일시가 포함된 사주 계산 요청
   - "사주 봐주세요", "팔자 계산해주세요" 등

2. fortune_consultation: 운세 상담/해석이 필요한 경우  
   - 운세, 궁합, 택일 관련 질문
   - "올해 운세는?", "연애운은?" 등

3. general_search: 일반적인 정보 검색이 필요한 경우
   - 사주/운세와 관련 없는 일반 질문
   - 최신 정보가 필요한 질문

응답 형식:
{
    "primary_intent": "분류 결과",
    "confidence_score": 0.0-1.0,
    "complexity_level": "simple/medium/complex",
    "requires_birth_info": true/false,
    "extracted_entities": {"생년월일시 등 추출된 정보"}
}"""

    # 사용자 메시지
    user_message = f"다음 질문을 분석해주세요: {user_query}"
    
    try:
        # LLM 호출
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        
        response = llm.invoke(messages)
        result_text = response.content
        
        # JSON 파싱 시도
        try:
            result = json.loads(result_text)
        except:
            # JSON 파싱 실패시 규칙 기반 분석으로 대체
            result = _rule_based_intent_analysis(user_query)
        
        # 신뢰도 점수 계산
        confidence_scores = {
            result["primary_intent"]: result.get("confidence_score", 0.8)
        }
        
        return {
            "primary_intent": result["primary_intent"],
            "secondary_intents": [],
            "confidence_scores": confidence_scores,
            "extracted_entities": result.get("extracted_entities", {}),
            "requires_birth_info": result.get("requires_birth_info", False),
            "complexity_level": result.get("complexity_level", "medium")
        }
        
    except Exception as e:
        # LLM 호출 실패시 규칙 기반 분석
        return _rule_based_intent_analysis(user_query)

def _rule_based_intent_analysis(user_query: str) -> Dict[str, Any]:
    """
    규칙 기반 의도 분석 (LLM 실패시 백업)
    
    Args:
        user_query: 사용자 질문
    
    Returns:
        Dict: 의도 분석 결과
    """
    
    # 생년월일시 패턴 검사
    birth_patterns = [
        r'\d{4}년.*\d{1,2}월.*\d{1,2}일',  # 1995년 8월 26일
        r'\d{4}-\d{1,2}-\d{1,2}',         # 1995-08-26
        r'\d{4}\.\d{1,2}\.\d{1,2}',       # 1995.08.26
        r'\d{2}년.*\d{1,2}월.*\d{1,2}일'   # 95년 8월 26일
    ]
    
    has_birth_info = any(re.search(pattern, user_query) for pattern in birth_patterns)
    
    # 사주 계산 키워드
    saju_keywords = ["사주", "팔자", "명리", "계산", "봐주세요", "보세요"]
    
    # 운세 상담 키워드  
    fortune_keywords = ["운세", "운", "궁합", "택일", "올해", "내년", "연애", "결혼", "직업", "재물"]
    
    # 일반 검색 키워드
    general_keywords = ["뉴스", "날씨", "주식", "코인", "정보", "검색"]
    
    # 키워드 매칭
    saju_score = sum(1 for keyword in saju_keywords if keyword in user_query)
    fortune_score = sum(1 for keyword in fortune_keywords if keyword in user_query)
    general_score = sum(1 for keyword in general_keywords if keyword in user_query)
    
    # 의도 결정
    if has_birth_info and saju_score > 0:
        primary_intent = "saju_calculation"
        confidence = 0.9
    elif fortune_score > general_score:
        primary_intent = "fortune_consultation"  
        confidence = 0.7
    elif general_score > 0:
        primary_intent = "general_search"
        confidence = 0.8
    else:
        # 기본값
        primary_intent = "fortune_consultation"
        confidence = 0.5
    
    # 복잡도 판단
    if len(user_query) < 20:
        complexity = "simple"
    elif len(user_query) < 50:
        complexity = "medium"
    else:
        complexity = "complex"
    
    return {
        "primary_intent": primary_intent,
        "secondary_intents": [],
        "confidence_scores": {primary_intent: confidence},
        "extracted_entities": {},
        "requires_birth_info": has_birth_info,
        "complexity_level": complexity
    }

# =============================================================================
# 테스트 함수들
# =============================================================================

def test_birth_info_extraction():
    """생년월일시 추출 테스트"""
    
    test_cases = [
        "1995년 8월 26일 오전 10시 15분 남자 사주 봐주세요",
        "1995-08-26 10:15 여성 운세는?",
        "95년 8월 26일생 남자입니다",
        "1995.08.26 태어난 여자",
        "사주 봐주세요"  # 생년월일시 없음
    ]
    
    for test_case in test_cases:
        result = extract_birth_info(test_case)
        print(f"입력: {test_case}")
        print(f"결과: {format_birth_info(result)}")
        print("-" * 50)

def test_intent_analysis():
    """의도 분석 테스트 (규칙 기반)"""
    
    test_cases = [
        "1995년 8월 26일 오전 10시 15분 남자 사주 봐주세요",
        "올해 연애운은 어떤가요?",
        "오늘 날씨는 어떤가요?",
        "사주에서 십신이란 무엇인가요?"
    ]
    
    for test_case in test_cases:
        result = _rule_based_intent_analysis(test_case)
        print(f"입력: {test_case}")
        print(f"의도: {result['primary_intent']}")
        print(f"신뢰도: {result['confidence_scores']}")
        print(f"복잡도: {result['complexity_level']}")
        print("-" * 50)

if __name__ == "__main__":
    print("=== 생년월일시 추출 테스트 ===")
    test_birth_info_extraction()
    print("\n=== 의도 분석 테스트 ===")
    test_intent_analysis() 