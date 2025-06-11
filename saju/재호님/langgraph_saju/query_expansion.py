from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Any
import re

class SajuQueryExpander:
    """사주 관련 한글 질문을 영어로 확장/변환하는 클래스"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.saju_terms_mapping = {
            # 기본 사주 용어
            "사주": "Four Pillars of Destiny, Saju, Chinese astrology",
            "사주팔자": "Four Pillars of Destiny, Eight Characters, Ba Zi",
            "오행": "Five Elements, Wu Xing",
            "십신": "Ten Gods, Shi Shen",
            "천간": "Heavenly Stems, Tian Gan",
            "지지": "Earthly Branches, Di Zhi",
            "대운": "Great Luck Period, Da Yun",
            "세운": "Annual Luck, Sui Yun",
            
            # 오행
            "목": "Wood element",
            "화": "Fire element", 
            "토": "Earth element",
            "금": "Metal element",
            "수": "Water element",
            
            # 십신
            "정관": "Zheng Guan, Correct Official, Direct Officer",
            "편관": "Pian Guan, Seven Killings, Indirect Officer",
            "정재": "Zheng Cai, Direct Wealth, Positive Wealth",
            "편재": "Pian Cai, Indirect Wealth, Partial Wealth",
            "정인": "Zheng Yin, Direct Seal, Correct Seal",
            "편인": "Pian Yin, Indirect Seal, Partial Seal",
            "비견": "Bi Jian, Friend, Shoulder to Shoulder",
            "겁재": "Jie Cai, Rob Wealth, Sibling",
            "식신": "Shi Shen, Eating God, Food God",
            "상관": "Shang Guan, Hurting Officer, Output",
            
            # 천간
            "갑": "Jia, Wood Yang",
            "을": "Yi, Wood Yin", 
            "병": "Bing, Fire Yang",
            "정": "Ding, Fire Yin",
            "무": "Wu, Earth Yang",
            "기": "Ji, Earth Yin",
            "경": "Geng, Metal Yang",
            "신": "Xin, Metal Yin",
            "임": "Ren, Water Yang",
            "계": "Gui, Water Yin",
            
            # 지지
            "자": "Zi, Rat, Water",
            "축": "Chou, Ox, Earth",
            "인": "Yin, Tiger, Wood",
            "묘": "Mao, Rabbit, Wood",
            "진": "Chen, Dragon, Earth",
            "사": "Si, Snake, Fire",
            "오": "Wu, Horse, Fire",
            "미": "Wei, Sheep, Earth",
            "신": "Shen, Monkey, Metal",
            "유": "You, Rooster, Metal",
            "술": "Xu, Dog, Earth",
            "해": "Hai, Pig, Water"
        }
        
        self.expansion_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in Four Pillars of Destiny (Saju). 
Your task is to expand Korean queries about Saju into comprehensive English search queries.

Rules:
1. If the query contains birth date/time information (like "1995년 3월 28일"), keep it as is and add relevant English terms
2. For conceptual Saju questions, translate and expand with related English terminology
3. Always include both Chinese pinyin terms and English translations
4. Add context-relevant synonyms and related concepts
5. Make the query comprehensive for better retrieval from English documents

Korean Saju Terms Mapping:
- 사주/사주팔자 → Four Pillars of Destiny, Saju, Ba Zi, Eight Characters
- 오행 → Five Elements, Wu Xing (Wood, Fire, Earth, Metal, Water)
- 십신 → Ten Gods, Shi Shen
- 정관 → Zheng Guan, Direct Officer, Correct Official
- 편관 → Pian Guan, Seven Killings, Indirect Officer
- 대운 → Great Luck Period, Da Yun
- And many more...

Examples:
Korean: "정관이 뭐야?"
English: "What is Zheng Guan Direct Officer Correct Official in Four Pillars of Destiny? What does positive authority mean in Saju?"

Korean: "1995년 3월 28일 남자 사주"
English: "1995년 3월 28일 male Four Pillars of Destiny Saju analysis birth date"
"""),
            ("user", "Korean query: {query}")
        ])
        
        self.output_parser = StrOutputParser()
        self.expansion_chain = self.expansion_prompt | self.llm | self.output_parser
    
    def is_birth_date_query(self, query: str) -> bool:
        """생년월일이 포함된 쿼리인지 확인"""
        # 년도 패턴 (19xx, 20xx)
        year_pattern = r'(19|20)\d{2}년?'
        # 월 패턴 (1~12월)
        month_pattern = r'(1[0-2]|[1-9])월'
        # 일 패턴 (1~31일)
        day_pattern = r'([1-2]?[0-9]|3[0-1])일'
        
        has_year = bool(re.search(year_pattern, query))
        has_month = bool(re.search(month_pattern, query))
        has_day = bool(re.search(day_pattern, query))
        
        return has_year or (has_month and has_day)
    
    def extract_saju_terms(self, query: str) -> List[str]:
        """쿼리에서 사주 용어를 추출하고 영어 용어로 매핑"""
        found_terms = []
        for korean_term, english_terms in self.saju_terms_mapping.items():
            if korean_term in query:
                found_terms.extend(english_terms.split(", "))
        return found_terms
    
    def expand_query(self, query: str) -> str:
        """한글 쿼리를 영어로 확장"""
        try:
            # LLM을 통한 쿼리 확장
            expanded_query = self.expansion_chain.invoke({"query": query})
            return expanded_query.strip()
        except Exception as e:
            print(f"Query expansion error: {e}")
            # 폴백: 기본적인 용어 매핑만 사용
            return self._fallback_expansion(query)
    
    def _fallback_expansion(self, query: str) -> str:
        """LLM 실패시 사용할 기본 확장 방법"""
        expanded_terms = self.extract_saju_terms(query)
        
        if self.is_birth_date_query(query):
            base_expansion = f"{query} Four Pillars Destiny Saju "
        else:
            base_expansion = f"{query} Four Pillars of Destiny Saju"
        
        if expanded_terms:
            base_expansion += " " + " ".join(expanded_terms)
            
        return base_expansion
    
    def get_multiple_expansions(self, query: str, num_expansions: int = 3) -> List[str]:
        """여러 버전의 확장된 쿼리 생성"""
        expansions = []
        
        # 원본 쿼리
        expansions.append(query)
        
        # LLM 확장 버전
        try:
            llm_expansion = self.expand_query(query)
            expansions.append(llm_expansion)
        except:
            pass
        
        # 기본 용어 매핑 버전
        fallback_expansion = self._fallback_expansion(query)
        expansions.append(fallback_expansion)
        
        # 중복 제거
        expansions = list(set(expansions))
        
        return expansions[:num_expansions]


# 사용 예시 및 테스트 함수
def test_query_expander():
    """쿼리 확장기 테스트"""
    expander = SajuQueryExpander()
    
    test_queries = [
        "1995년 3월 28일 남자 사주",
        "정관이 뭐야?",
        "오행에서 금의 의미",
        "대운이 언제 바뀌나요?",
        "사주팔자란?",
        "1990년 12월 15일 여자 오후 3시"
    ]
    
    print("=== Query Expansion Test ===")
    for query in test_queries:
        print(f"\n원본 쿼리: {query}")
        
        # 생년월일 쿼리 여부 확인
        is_birth = expander.is_birth_date_query(query)
        print(f"생년월일 쿼리: {is_birth}")
        
        # 사주 용어 추출
        terms = expander.extract_saju_terms(query)
        print(f"추출된 용어: {terms}")
        
        # 쿼리 확장
        expanded = expander.expand_query(query)
        print(f"확장된 쿼리: {expanded}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_query_expander() 