import re
from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class SajuQueryExpander:
    """한국어 사주 질문을 영어로 확장하는 클래스"""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        
        # 사주 전문 용어 매핑 (한국어 -> 영어)
        self.saju_terms = {
            # 십신 (Ten Gods)
            "정관": "Zheng Guan Direct Officer",
            "편관": "Pian Guan Indirect Officer",
            "정재": "Zheng Cai Direct Wealth",
            "편재": "Pian Cai Indirect Wealth",
            "정인": "Zheng Yin Direct Seal",
            "편인": "Pian Yin Indirect Seal",
            "비견": "Bi Jian Comparable",
            "겁재": "Jie Cai Rob Wealth",
            "식신": "Shi Shen Eating God",
            "상관": "Shang Guan Hurting Officer",
            
            # 오행 (Five Elements)
            "목": "Wood",
            "화": "Fire", 
            "토": "Earth",
            "금": "Metal",
            "수": "Water",
            
            # 천간 (Heavenly Stems)
            "갑": "Jia",
            "을": "Yi",
            "병": "Bing",
            "정": "Ding",
            "무": "Wu",
            "기": "Ji",
            "경": "Geng",
            "신": "Xin",
            "임": "Ren",
            "계": "Gui",
            
            # 지지 (Earthly Branches)
            "자": "Zi",
            "축": "Chou",
            "인": "Yin",
            "묘": "Mao",
            "진": "Chen",
            "사": "Si",
            "오": "Wu",
            "미": "Wei",
            "신": "Shen",
            "유": "You",
            "술": "Xu",
            "해": "Hai",
            
            # 사주 기본 용어
            "사주": "Four Pillars of Destiny",
            "팔자": "Ba Zi",
            "년주": "Year Pillar",
            "월주": "Month Pillar", 
            "일주": "Day Pillar",
            "시주": "Hour Pillar",
            "대운": "Great Luck Period",
            "소운": "Small Luck Period",
            "운세": "Fortune",
            "궁합": "Compatibility",
            "오행": "Five Elements",
            "십신": "Ten Gods",
            "천간": "Heavenly Stems",
            "지지": "Earthly Branches",
            "일간": "Day Master",
            "용신": "Useful God",
            "기신": "Taboo God",
            "희신": "Joy God",
            "한신": "Leisure God",
            "상생": "Mutual Generation",
            "상극": "Mutual Destruction",
        }
        
        # 질문 확장 프롬프트
        self.expansion_prompt = ChatPromptTemplate.from_template(
            """당신은 사주 전문가입니다. 다음 한국어 사주 질문을 영어로 확장해주세요.

원본 질문: {query}

확장 규칙:
1. 사주 전문 용어를 영어로 번역하세요
2. 질문의 의도를 명확히 하는 추가 컨텍스트를 포함하세요
3. Four Pillars of Destiny, Ba Zi 등의 표준 영어 용어를 사용하세요
4. 생년월일이 포함된 경우 birth chart analysis와 관련된 용어를 포함하세요

확장된 영어 질문을 생성하세요. 답변은 영어 질문만 제공하세요."""
        )
    
    def detect_birth_date(self, query: str) -> bool:
        """생년월일 패턴을 감지"""
        patterns = [
            r'\d{4}년.*\d{1,2}월.*\d{1,2}일',  # 1995년 3월 28일
            r'\d{4}-\d{1,2}-\d{1,2}',          # 1995-03-28
            r'\d{4}/\d{1,2}/\d{1,2}',          # 1995/03/28
            r'\d{4}\.\d{1,2}\.\d{1,2}',        # 1995.03.28
            r'\d{1,2}월.*\d{1,2}일',           # 3월 28일
        ]
        
        for pattern in patterns:
            if re.search(pattern, query):
                return True
        return False
    
    def replace_korean_terms(self, query: str) -> str:
        """한국어 사주 용어를 영어로 치환"""
        expanded_query = query
        
        for korean_term, english_term in self.saju_terms.items():
            if korean_term in expanded_query:
                expanded_query = expanded_query.replace(korean_term, english_term)
        
        return expanded_query
    
    def expand_query(self, query: str) -> str:
        """한국어 사주 질문을 영어로 확장"""
        try:
            # 1. 한국어 용어 치환
            partially_expanded = self.replace_korean_terms(query)
            
            # 2. 생년월일 포함 여부 확인
            has_birth_date = self.detect_birth_date(query)
            
            # 3. LLM을 사용한 질문 확장
            chain = self.expansion_prompt | self.llm
            response = chain.invoke({"query": query})
            
            expanded_query = response.content.strip()
            
            # 4. 생년월일이 포함된 경우 추가 컨텍스트 추가
            if has_birth_date:
                expanded_query += " Please analyze the birth chart and Great Luck Period (Da Yun) based on Four Pillars of Destiny."
            
            return expanded_query
            
        except Exception as e:
            # 오류 발생 시 기본 확장 수행
            print(f"LLM 확장 중 오류 발생: {e}")
            
            # 기본 확장: 한국어 용어만 영어로 치환
            basic_expansion = self.replace_korean_terms(query)
            
            if has_birth_date:
                return f"Please analyze the Four Pillars of Destiny for {basic_expansion} including birth chart interpretation and fortune analysis."
            else:
                return f"Please explain about {basic_expansion} in Four Pillars of Destiny context."
    
    def get_related_terms(self, main_term: str) -> List[str]:
        """주요 용어와 관련된 영어 용어들을 반환"""
        related_terms = {
            "정관": ["authority", "government", "official position", "career"],
            "편관": ["power", "pressure", "authority", "control"],
            "정재": ["stable wealth", "salary", "fixed income"],
            "편재": ["flexible wealth", "business", "investment"],
            "정인": ["education", "knowledge", "mother", "protection"],
            "편인": ["intuition", "creativity", "unconventional learning"],
            "비견": ["competition", "friendship", "siblings"],
            "겁재": ["partnership", "cooperation", "sharing"],
            "식신": ["creativity", "expression", "performance"],
            "상관": ["intelligence", "skills", "output"],
            "대운": ["life period", "fortune cycle", "decade luck"],
            "오행": ["elemental balance", "five elements theory"],
        }
        
        return related_terms.get(main_term, [])



