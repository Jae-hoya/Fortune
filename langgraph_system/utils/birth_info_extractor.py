"""
생년월일시 정보 추출 유틸리티
사용자 질문에서 생년월일시 정보를 추출하는 로직
"""

import re
from typing import Dict, Any, Optional
from datetime import datetime

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

# 테스트용 함수
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

if __name__ == "__main__":
    test_birth_info_extraction() 