"""
FortuneAI Tools
모든 사주 상담 도구들을 통합 관리
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain_core.tools import tool, Tool
import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import quote_plus
import re

# 기존 모듈들 import
from vector_store import load_vector_store, get_all_documents
from search import create_hybrid_retriever
from reranker import get_flashrank_reranker, rerank_documents

# 사주 계산 모듈 import
from saju_calculator import SajuCalculator, format_saju_analysis
from typing import Dict

class WebSearcher:
    """사주 관련 일반 지식 웹 검색 클래스"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # 사주 관련 핵심 키워드
        self.saju_keywords = [
            '명리학', '사주명리', '음양오행', '십간십이지', '육십갑자',
            '십신', '용신', '희신', '기신', '원진살', '공망',
            '대운', '세운', '월운', '일운', '시운',
            '정관', '편관', '정재', '편재', '식신', '상관',
            '비견', '겁재', '정인', '편인', '천간', '지지',
            '오행', '상생', '상극', '합', '충', '형', '해'
        ]
    
    def enhance_search_query(self, query: str) -> str:
        """검색 쿼리를 사주 관련 검색에 최적화"""
        enhanced_query = query
        
        # 이미 사주 관련 키워드가 포함되어 있지 않으면 추가
        if not any(keyword in query for keyword in ['사주', '명리', '팔자']):
            enhanced_query += ' 사주 명리학'
        
        return enhanced_query
    
    def search_duckduckgo(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """DuckDuckGo를 사용한 사주 관련 웹 검색"""
        try:
            # 검색 쿼리 최적화
            enhanced_query = self.enhance_search_query(query)
            search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(enhanced_query)}"
            
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # 검색 결과 파싱
            for result in soup.find_all('div', class_='result')[:max_results]:
                title_elem = result.find('a', class_='result__a')
                snippet_elem = result.find('a', class_='result__snippet')
                
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    url = title_elem.get('href', '')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    # 사주 관련성 점수 계산
                    relevance_score = self.calculate_relevance(title, snippet, query)
                    
                    results.append({
                        'title': title,
                        'url': url,
                        'snippet': snippet,
                        'relevance_score': relevance_score
                    })
            
            # 관련성 점수로 정렬
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return results
            
        except Exception as e:
            print(f"웹 검색 오류: {e}")
            return []
    
    def calculate_relevance(self, title: str, snippet: str, original_query: str) -> float:
        """검색 결과의 사주 관련성 점수 계산"""
        score = 0.0
        text = (title + " " + snippet).lower()
        query_lower = original_query.lower()
        
        # 기본 사주 키워드 매칭
        basic_keywords = ['사주', '명리', '팔자', '운세', '명식', '사주명리']
        for keyword in basic_keywords:
            if keyword in text:
                score += 2.0
        
        # 사주 전문 용어 매칭
        for keyword in self.saju_keywords:
            if keyword in text:
                score += 1.5
        
        # 원본 쿼리와의 유사성
        query_words = query_lower.split()
        for word in query_words:
            if word in text:
                score += 1.0
        
        return score
    
    def get_page_content(self, url: str, max_chars: int = 1500) -> str:
        """웹 페이지 내용 가져오기 - 사주 관련 내용 중심으로"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 불필요한 태그 제거
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
                tag.decompose()
            
            # 텍스트 추출
            text = soup.get_text(separator=' ', strip=True)
            
            # 사주 관련 내용 우선 추출
            sentences = text.split('.')
            relevant_sentences = []
            
            for sentence in sentences:
                if any(keyword in sentence for keyword in ['사주', '명리', '팔자', '운세'] + self.saju_keywords):
                    relevant_sentences.append(sentence.strip())
            
            # 관련 문장이 있으면 우선 사용, 없으면 전체 텍스트 사용
            if relevant_sentences:
                filtered_text = '. '.join(relevant_sentences[:10])  # 최대 10문장
            else:
                filtered_text = text
            
            # 길이 제한
            if len(filtered_text) > max_chars:
                filtered_text = filtered_text[:max_chars] + "..."
            
            return filtered_text
            
        except Exception as e:
            return f"페이지 내용을 가져올 수 없습니다: {e}"
    
    def search_with_content(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """검색 결과와 함께 페이지 내용도 가져오기"""
        search_results = self.search_duckduckgo(query, max_results)
        
        for result in search_results:
            if result['url']:
                content = self.get_page_content(result['url'])
                result['content'] = content
        
        return search_results

class SmartSearchOrchestrator:
    """지능형 검색 오케스트레이터 - 질문 유형에 따라 최적의 검색 전략 결정"""
    
    def __init__(self, rag_search_func, web_search_func):
        self.rag_search = rag_search_func
        self.web_search = web_search_func
        
        # 사주 전문 용어 키워드
        self.saju_keywords = [
            '사주', '명리', '팔자', '운세', '천간', '지지', '간지', '육십갑자',
            '십신', '십성', '오행', '음양', '상생', '상극', '합', '충', '형', '해',
            '대운', '세운', '월운', '일운', '용신', '희신', '기신', '원진살',
            '정관', '편관', '정재', '편재', '식신', '상관', '비견', '겁재', '정인', '편인',
            '년주', '월주', '일주', '시주', '일간', '월령', '계절', '생시',
            '갑', '을', '병', '정', '무', '기', '경', '신', '임', '계',
            '자', '축', '인', '묘', '진', '사', '오', '미', '신', '유', '술', '해',
            '목', '화', '토', '금', '수', '장생', '목욕', '관대', '건록', '제왕',
            '공망', '도화', '역마', '천을귀인', '태극귀인', '문창', '문곡'
        ]
        
        # 일반 상담 키워드  
        self.general_keywords = [
            '안녕', '인사', '소개', '설명', '뭐야', '무엇', '어떻게', '왜',
            '현재', '요즘', '최근', '트렌드', '뉴스', '정보', '방법',
            '추천', '조언', '의견', '생각', '어떨까', '어때'
        ]
    
    def classify_query(self, query: str) -> str:
        """질문 유형 분류"""
        query_lower = query.lower()
        
        # 1. 생년월일시 정보가 포함된 경우 → 사주 분석
        birth_patterns = [
            r'\d{4}년.*\d{1,2}월.*\d{1,2}일',  # 1995년 8월 26일
            r'\d{4}-\d{1,2}-\d{1,2}',          # 1995-08-26
            r'\d{1,2}시.*\d{1,2}분',           # 10시 15분
            r'오전|오후.*\d{1,2}시',            # 오전 10시
            r'생년월일|생일|태어'                # 생년월일시 관련
        ]
        
        for pattern in birth_patterns:
            if re.search(pattern, query):
                return "saju_analysis"
        
        # 2. 사주 전문 용어가 포함된 경우 → 사주 지식
        saju_count = sum(1 for keyword in self.saju_keywords if keyword in query_lower)
        general_count = sum(1 for keyword in self.general_keywords if keyword in query_lower)
        
        # 3. 명확한 일반 질문 키워드 체크
        clear_general_keywords = ['날씨', '트렌드', '뉴스', '인사', '안녕']
        clear_general_count = sum(1 for keyword in clear_general_keywords if keyword in query_lower)
        
        # 4. 사주 관련 맥락 키워드
        saju_context_keywords = ['운세', '성격', '미래', '특성', '성향']
        saju_context_count = sum(1 for keyword in saju_context_keywords if keyword in query_lower)
        
        # 분류 로직 개선
        if clear_general_count >= 1 and saju_count == 0:  # 명확한 일반 질문
            return "general_web"
        elif saju_count >= 2:  # 사주 키워드 2개 이상
            return "saju_knowledge"
        elif saju_count >= 1 and general_count == 0:  # 사주 키워드 1개 + 일반 키워드 없음
            return "saju_knowledge"
        elif saju_context_count >= 1:  # 사주 맥락 키워드 포함
            return "saju_knowledge"
        elif general_count >= 2:  # 일반 키워드 2개 이상
            return "general_web"
        else:
            # 애매한 경우 사주 관련으로 분류 (사주 상담 AI이므로)
            return "saju_knowledge"
    
    def search_with_strategy(self, query: str) -> Dict[str, Any]:
        """최적 검색 전략 실행"""
        query_type = self.classify_query(query)
        
        if query_type == "saju_analysis":
            # 사주 분석: RAG만 사용 (정확성 우선)
            rag_result = self.rag_search(query)
            return {
                "primary_source": "rag",
                "rag_result": rag_result,
                "web_result": None,
                "strategy": "rag_only",
                "reason": "사주 분석 질문으로 전문 지식 필요"
            }
            
        elif query_type == "saju_knowledge":
            # 사주 지식: RAG 우선 → 부족시 웹 검색
            rag_result = self.rag_search(query)
            
            # RAG 결과 품질 평가
            rag_quality = self._evaluate_rag_quality(rag_result, query)
            
            if rag_quality < 0.3:  # RAG 결과가 부족한 경우
                web_result = self.web_search(query, max_results=3) if self.web_search else "웹 검색 기능 비활성화"
                return {
                    "primary_source": "web",
                    "rag_result": rag_result,
                    "web_result": web_result,
                    "strategy": "rag_then_web",
                    "reason": "RAG 결과 부족으로 웹 검색 보완"
                }
            else:
                return {
                    "primary_source": "rag",
                    "rag_result": rag_result,
                    "web_result": None,
                    "strategy": "rag_sufficient",
                    "reason": "RAG 검색 결과 충분"
                }
                
        else:  # general_web
            # 일반 질문: 웹 검색 우선
            web_result = self.web_search(query, max_results=3) if self.web_search else "웹 검색 기능 비활성화"
            return {
                "primary_source": "web",
                "rag_result": None,
                "web_result": web_result,
                "strategy": "web_only",
                "reason": "일반적인 질문으로 웹 검색 수행"
            }
    
    def _evaluate_rag_quality(self, rag_result: str, query: str) -> float:
        """RAG 검색 결과 품질 평가 (0.0 ~ 1.0)"""
        if not rag_result or "검색 중 오류" in rag_result or "검색 결과 없음" in rag_result:
            return 0.0
        
        # 길이 기반 평가 (더 관대하게)
        if len(rag_result) < 50:
            return 0.1
        elif len(rag_result) < 100:
            length_score = 0.3
        else:
            length_score = 0.5
        
        # 키워드 매칭 평가
        query_words = [word for word in query.lower().split() if len(word) > 1]
        result_lower = rag_result.lower()
        
        match_count = sum(1 for word in query_words if word in result_lower)
        match_ratio = match_count / len(query_words) if query_words else 0
        
        # 사주 전문 용어 포함 평가
        saju_term_count = sum(1 for term in self.saju_keywords if term in result_lower)
        saju_score = min(saju_term_count / 3, 1.0)  # 3개 이상이면 만점
        
        # 유용한 내용 패턴 체크
        useful_patterns = [
            '설명', '의미', '특성', '해석', '분석', '방법',
            '일간', '오행', '십신', '대운', '명리'
        ]
        useful_count = sum(1 for pattern in useful_patterns if pattern in result_lower)
        useful_score = min(useful_count / 3, 0.3)  # 최대 0.3점
        
        # 종합 점수 계산
        # 길이 40% + 키워드 매칭 30% + 사주 용어 20% + 유용성 10%
        final_score = (length_score * 0.4) + (match_ratio * 0.3) + (saju_score * 0.2) + (useful_score * 0.1)
        
        return min(final_score, 1.0)

class ToolManager:
    """도구 관리자 클래스"""
    
    def __init__(self, enable_rag: bool = True, enable_web: bool = False, enable_calendar: bool = False):
        """
        도구 관리자 초기화
        
        Args:
            enable_rag: RAG 도구 활성화 여부
            enable_web: 웹 검색 도구 활성화 여부
            enable_calendar: 만세력 도구 활성화 여부
        """
        self.enable_rag = enable_rag
        self.enable_web = enable_web
        self.enable_calendar = enable_calendar
        
        # RAG 컴포넌트들
        self.vectorstore = None
        self.hybrid_retriever = None
        self.reranker = None
        self.all_docs = None
        
        # 웹 검색 컴포넌트들
        self.web_searcher = None
        
        # 만세력 컴포넌트들
        self.calendar_api = None
        
        self.tools = []
        self.initialize()
    
    def initialize(self):
        """모든 활성화된 도구들 초기화"""
        print("🛠️ 도구 시스템 초기화 중...")
        
        if self.enable_rag:
            self._initialize_rag()
        
        if self.enable_web:
            self._initialize_web()
        
        if self.enable_calendar:
            self._initialize_calendar()
        
        self._setup_tools()
        print(f"✅ 총 {len(self.tools)}개 도구 초기화 완료!")
    
    def _initialize_rag(self):
        """RAG 컴포넌트들 초기화"""
        print("🔧 RAG 도구 초기화 중...")
        
        # 벡터 스토어 로드
        self.vectorstore = load_vector_store("saju_vectordb")
        
        # 모든 문서 가져오기
        self.all_docs = get_all_documents(self.vectorstore)
        
        # 하이브리드 검색기 초기화
        self.hybrid_retriever = create_hybrid_retriever(
            vectorstore=self.vectorstore, 
            documents=self.all_docs, 
            weights=[0.8, 0.2],
            top_k=20
        )
        
        # 리랭커 초기화
        self.reranker = get_flashrank_reranker()
        
        print("✅ RAG 도구 초기화 완료!")
    
    def _initialize_web(self):
        """웹 검색 도구 초기화"""
        print("🌐 웹 검색 도구 초기화 중...")
        self.web_searcher = WebSearcher()
        print("✅ 웹 검색 도구 초기화 완료!")
    
    def _initialize_calendar(self):
        """만세력 도구 초기화"""
        print("📅 만세력 도구 초기화 중...")
        # TODO: 만세력 API 클라이언트 초기화
        print("✅ 만세력 도구 초기화 완료!")
    
    def _setup_tools(self):
        """활성화된 도구들 설정"""
        self.tools = []
        
        if self.enable_rag:
            self.tools.extend(self._get_rag_tools())
        
        if self.enable_web:
            self.tools.extend(self._get_web_tools())
        
        if self.enable_calendar:
            self.tools.extend(self._get_calendar_tools())
    
    def _get_rag_tools(self) -> List[Tool]:
        """RAG 기반 도구들 반환"""
        
        @tool
        def search_saju_knowledge(query: str) -> str:
            """사주팔자, 운세, 오행, 십신 등에 대한 전문 지식을 검색합니다. 사주 관련 질문이나 개념 설명이 필요할 때 사용하세요."""
            try:
                # 하이브리드 검색 수행
                docs = self.hybrid_retriever.invoke(query)
                
                # 리랭킹 수행
                reranked_docs = rerank_documents(self.reranker, docs, query)
                
                # 상위 5개 문서만 사용
                top_docs = reranked_docs[:5]
                
                # 컨텍스트 생성
                context = "\n\n".join([doc.page_content for doc in top_docs])
                
                return f"검색된 사주 지식:\n{context}"
            except Exception as e:
                return f"검색 중 오류 발생: {str(e)}"
        
        @tool
        def smart_search_saju(query: str) -> str:
            """질문 유형을 자동 분석하여 최적의 검색 전략(RAG/웹)을 선택합니다. 모든 사주 관련 질문에 우선적으로 사용하세요."""
            try:
                # 지능형 검색 오케스트레이터 초기화
                if not hasattr(self, '_orchestrator'):
                    self._orchestrator = SmartSearchOrchestrator(
                        rag_search_func=search_saju_knowledge.func,
                        web_search_func=self.web_searcher.search_duckduckgo if hasattr(self, 'web_searcher') else None
                    )
                
                # 최적 검색 전략 실행
                result = self._orchestrator.search_with_strategy(query)
                
                # 결과 포맷팅
                response_parts = []
                response_parts.append(f"🔍 검색 전략: {result['strategy']}")
                response_parts.append(f"📝 사유: {result['reason']}")
                
                if result['primary_source'] == 'rag' and result['rag_result']:
                    response_parts.append(f"\n📚 RAG 검색 결과:\n{result['rag_result']}")
                    
                if result['web_result']:
                    response_parts.append(f"\n🌐 웹 검색 결과:\n{result['web_result']}")
                    
                if result['primary_source'] == 'web' and result['web_result']:
                    response_parts.append(f"\n🌐 웹 검색 결과:\n{result['web_result']}")
                
                return "\n".join(response_parts)
                
            except Exception as e:
                # 에러 시 기본 RAG 검색으로 폴백
                return search_saju_knowledge.func(query)
        
        return [search_saju_knowledge, smart_search_saju]
    
    def _parse_birth_info(self, birth_info: str) -> Dict:
        """생년월일시 정보 파싱"""
        try:
            # 정규식으로 생년월일시 추출
            # 예: "1995년 8월 26일 오전 10시 15분", "1995-08-26 10:15"
            
            # 년도 추출
            year_match = re.search(r'(\d{4})년?', birth_info)
            if not year_match:
                return None
            year = int(year_match.group(1))
            
            # 월 추출
            month_match = re.search(r'(\d{1,2})월', birth_info)
            if not month_match:
                return None
            month = int(month_match.group(1))
            
            # 일 추출
            day_match = re.search(r'(\d{1,2})일', birth_info)
            if not day_match:
                return None
            day = int(day_match.group(1))
            
            # 시간 추출
            hour = 12  # 기본값
            minute = 0  # 기본값
            
            # 오전/오후 처리
            if '오전' in birth_info or 'AM' in birth_info.upper():
                hour_match = re.search(r'오전\s*(\d{1,2})시?', birth_info)
                if hour_match:
                    hour = int(hour_match.group(1))
                    if hour == 12:
                        hour = 0
            elif '오후' in birth_info or 'PM' in birth_info.upper():
                hour_match = re.search(r'오후\s*(\d{1,2})시?', birth_info)
                if hour_match:
                    hour = int(hour_match.group(1))
                    if hour != 12:
                        hour += 12
            else:
                # 24시간 형식
                hour_match = re.search(r'(\d{1,2})시', birth_info)
                if hour_match:
                    hour = int(hour_match.group(1))
            
            # 분 추출
            minute_match = re.search(r'(\d{1,2})분', birth_info)
            if minute_match:
                minute = int(minute_match.group(1))
            
            # 성별 추출
            is_male = True  # 기본값
            if '여자' in birth_info or '여성' in birth_info or '여' in birth_info:
                is_male = False
            elif '남자' in birth_info or '남성' in birth_info or '남' in birth_info:
                is_male = True
            
            return {
                'year': year, 'month': month, 'day': day,
                'hour': hour, 'minute': minute, 'is_male': is_male
            }
            
        except Exception as e:
            print(f"생년월일시 파싱 오류: {e}")
            return None
    
    def _get_web_tools(self) -> List[Tool]:
        """웹 검색 기반 도구들 반환"""
        
        @tool
        def search_web_saju(query: str) -> str:
            """사주나 명리학 관련 일반 지식을 웹에서 검색합니다. RAG 시스템에 없는 정보나 추가적인 설명이 필요할 때 사용하세요."""
            try:
                results = self.web_searcher.search_duckduckgo(query, max_results=5)
                
                if not results:
                    return "웹 검색 결과를 찾을 수 없습니다."
                
                formatted_results = []
                for i, result in enumerate(results, 1):
                    formatted_results.append(
                        f"{i}. {result['title']}\n"
                        f"   {result['snippet']}\n"
                        f"   출처: {result['url']}"
                    )
                
                return f"웹 검색 결과:\n\n" + "\n\n".join(formatted_results)
                
            except Exception as e:
                return f"웹 검색 중 오류 발생: {str(e)}"
        
        return [search_web_saju]
    
    def _get_calendar_tools(self) -> List[Tool]:
        """사주 계산 도구들 반환"""
        
        @tool
        def parse_birth_info(birth_info: str) -> str:
            """생년월일시 정보를 파싱하여 구조화된 데이터로 변환합니다. 예: '1995년 8월 26일 오전 10시 15분'"""
            try:
                birth_data = self._parse_birth_info(birth_info)
                if not birth_data:
                    return "생년월일시 정보를 정확히 파악할 수 없습니다. 예: 1995년 8월 26일 오전 10시 15분"
                
                return f"""파싱된 생년월일시 정보:
- 년도: {birth_data['year']}년
- 월: {birth_data['month']}월  
- 일: {birth_data['day']}일
- 시간: {birth_data['hour']}시 {birth_data['minute']}분
- 성별: {'남성' if birth_data['is_male'] else '여성'}"""
                
            except Exception as e:
                return f"생년월일시 파싱 중 오류 발생: {str(e)}"
        
        @tool
        def calculate_saju_chart(birth_info: str) -> str:
            """생년월일시를 바탕으로 정확한 사주팔자를 계산합니다. 사주 기본 구조만 계산하고 해석은 별도 도구를 사용하세요."""
            try:
                # 생년월일시 정보 파싱
                birth_data = self._parse_birth_info(birth_info)
                if not birth_data:
                    return "생년월일시 정보를 정확히 파악할 수 없습니다. 예: 1995년 8월 26일 오전 10시 15분"
                
                # 사주 계산기 초기화
                calculator = SajuCalculator()
                
                # 사주팔자 계산
                saju_chart = calculator.calculate_saju(
                    year=birth_data['year'],
                    month=birth_data['month'], 
                    day=birth_data['day'],
                    hour=birth_data['hour'],
                    minute=birth_data['minute'],
                    is_male=birth_data.get('is_male', True)
                )
                
                # 기본 사주팔자만 반환 (해석 제외)
                result = []
                result.append("=== 사주팔자 계산 결과 ===")
                result.append(f"년주(年柱): {saju_chart.year_pillar}")
                result.append(f"월주(月柱): {saju_chart.month_pillar}")
                result.append(f"일주(日柱): {saju_chart.day_pillar}")
                result.append(f"시주(時柱): {saju_chart.hour_pillar}")
                result.append(f"일간(日干): {saju_chart.get_day_master()}")
                
                return "\n".join(result)
                
            except Exception as e:
                return f"사주 계산 중 오류 발생: {str(e)}"
        
        @tool
        def analyze_five_elements(birth_info: str) -> str:
            """사주팔자의 오행 강약을 분석합니다. 먼저 calculate_saju_chart로 사주를 계산한 후 사용하세요."""
            try:
                # 생년월일시 정보 파싱
                birth_data = self._parse_birth_info(birth_info)
                if not birth_data:
                    return "생년월일시 정보를 정확히 파악할 수 없습니다."
                
                # 사주 계산
                calculator = SajuCalculator()
                saju_chart = calculator.calculate_saju(
                    year=birth_data['year'],
                    month=birth_data['month'], 
                    day=birth_data['day'],
                    hour=birth_data['hour'],
                    minute=birth_data['minute'],
                    is_male=birth_data.get('is_male', True)
                )
                
                # 오행 분석 (현대 정밀 방식)
                elements = calculator.get_element_strength(saju_chart)
                elements_balanced = calculator.get_element_strength_balanced(saju_chart)
                elements_simple = calculator.get_element_strength_simple(saju_chart)
                
                result = []
                result.append("=== 오행 강약 분석 (정밀 분석) ===")
                for element, strength in elements.items():
                    result.append(f"{element}: {strength}점")
                
                result.append("\n=== 오행 강약 분석 (8점 절충 방식) ===")
                for element, strength in elements_balanced.items():
                    result.append(f"{element}: {strength}점")
                
                result.append("\n=== 오행 강약 분석 (전통 8점 방식) ===")
                for element, strength in elements_simple.items():
                    result.append(f"{element}: {strength}점")
                
                # 오행 균형 평가 (정밀 분석 기준)
                max_element = max(elements, key=elements.get)
                min_element = min(elements, key=elements.get)
                result.append(f"\n가장 강한 오행: {max_element} ({elements[max_element]}점)")
                result.append(f"가장 약한 오행: {min_element} ({elements[min_element]}점)")
                
                return "\n".join(result)
                
            except Exception as e:
                return f"오행 분석 중 오류 발생: {str(e)}"
        
        @tool
        def analyze_ten_gods(birth_info: str) -> str:
            """사주팔자의 십신을 분석합니다. 먼저 calculate_saju_chart로 사주를 계산한 후 사용하세요."""
            try:
                # 생년월일시 정보 파싱
                birth_data = self._parse_birth_info(birth_info)
                if not birth_data:
                    return "생년월일시 정보를 정확히 파악할 수 없습니다."
                
                # 사주 계산
                calculator = SajuCalculator()
                saju_chart = calculator.calculate_saju(
                    year=birth_data['year'],
                    month=birth_data['month'], 
                    day=birth_data['day'],
                    hour=birth_data['hour'],
                    minute=birth_data['minute'],
                    is_male=birth_data.get('is_male', True)
                )
                
                # 십신 분석
                ten_gods = calculator.analyze_ten_gods(saju_chart)
                
                result = []
                result.append("=== 십신 분석 ===")
                for pillar_name, gods in ten_gods.items():
                    if gods:
                        result.append(f"{pillar_name}: {', '.join(gods)}")
                
                return "\n".join(result)
                
            except Exception as e:
                return f"십신 분석 중 오류 발생: {str(e)}"
        
        @tool
        def calculate_great_fortune(birth_info: str) -> str:
            """대운을 계산합니다. 먼저 calculate_saju_chart로 사주를 계산한 후 사용하세요."""
            try:
                # 생년월일시 정보 파싱
                birth_data = self._parse_birth_info(birth_info)
                if not birth_data:
                    return "생년월일시 정보를 정확히 파악할 수 없습니다."
                
                # 사주 계산
                calculator = SajuCalculator()
                saju_chart = calculator.calculate_saju(
                    year=birth_data['year'],
                    month=birth_data['month'], 
                    day=birth_data['day'],
                    hour=birth_data['hour'],
                    minute=birth_data['minute'],
                    is_male=birth_data.get('is_male', True)
                )
                
                # 대운 계산
                great_fortunes = calculator.calculate_great_fortune_improved(saju_chart)
                
                result = []
                result.append("=== 대운 계산 ===")
                for gf in great_fortunes:
                    result.append(f"{gf['age']}세: {gf['pillar']} ({gf['years']}) - {gf['direction']}")
                
                return "\n".join(result)
                
            except Exception as e:
                return f"대운 계산 중 오류 발생: {str(e)}"
        
        @tool
        def get_comprehensive_saju_analysis(birth_info: str) -> str:
            """생년월일시를 바탕으로 종합적인 사주 분석을 수행합니다. 성별 정보도 자동으로 파싱하여 대운 계산에 반영합니다. 예: '1995년 8월 26일 오전 10시 15분 남성'"""
            try:
                # 생년월일시 정보 파싱
                birth_data = self._parse_birth_info(birth_info)
                if not birth_data:
                    return "생년월일시 정보를 정확히 파악할 수 없습니다. 예: 1995년 8월 26일 오전 10시 15분"
                
                # 사주 계산기 초기화
                calculator = SajuCalculator()
                
                # 사주팔자 계산
                saju_chart = calculator.calculate_saju(
                    year=birth_data['year'],
                    month=birth_data['month'], 
                    day=birth_data['day'],
                    hour=birth_data['hour'],
                    minute=birth_data['minute'],
                    is_male=birth_data.get('is_male', True)
                )
                
                # 종합 분석 결과 포맷팅
                analysis_result = format_saju_analysis(saju_chart, calculator)
                
                return analysis_result
                
            except Exception as e:
                return f"종합 사주 분석 중 오류 발생: {str(e)}"
        
        return [parse_birth_info, calculate_saju_chart, analyze_five_elements, 
                analyze_ten_gods, calculate_great_fortune, get_comprehensive_saju_analysis]
    
    def get_tools(self) -> List[Tool]:
        """모든 활성화된 도구들 반환"""
        return self.tools
    
    @property
    def calendar_tools(self) -> List[Tool]:
        """사주 계산 도구들 반환"""
        return self._get_calendar_tools()
    
    @property  
    def rag_tools(self) -> List[Tool]:
        """RAG 검색 도구들 반환"""
        return self._get_rag_tools()
    
    @property
    def web_tools(self) -> List[Tool]:
        """웹 검색 도구들 반환"""
        return self._get_web_tools()
    
    def get_all_tools(self) -> List[Tool]:
        """모든 활성화된 도구들 반환 (get_tools와 동일)"""
        return self.tools
    
    def get_tool_info(self) -> Dict[str, Any]:
        """도구 정보 반환"""
        return {
            "total_tools": len(self.tools),
            "rag_enabled": self.enable_rag,
            "web_enabled": self.enable_web,
            "calendar_enabled": self.enable_calendar,
            "tool_names": [tool.name for tool in self.tools]
        } 