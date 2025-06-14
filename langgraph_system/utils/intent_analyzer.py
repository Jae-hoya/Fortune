"""
사용자 의도 분석 유틸리티
사용자 질문을 분석하여 적절한 워커를 결정하는 로직
"""

import re
from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

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
        import json
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