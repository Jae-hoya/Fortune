# 타로 LangGraph 모듈화 구현 단계

이 문서는 `tarot_langgraph.py` 파일을 LangGraph 표준 구조에 맞게 모듈화하는 구체적인 구현 단계를 설명합니다.

## 1단계: 기본 디렉토리 구조 생성

### 디렉토리 생성
```bash
# Windows 환경
mkdir -p parsing\parser\tarot_agent\utils

# Linux/Mac 환경
mkdir -p parsing/parser/tarot_agent/utils
```

### __init__.py 파일 생성
```bash
# Windows 환경
type nul > parsing\parser\tarot_agent\__init__.py
type nul > parsing\parser\tarot_agent\utils\__init__.py

# Linux/Mac 환경
touch parsing/parser/tarot_agent/__init__.py
touch parsing/parser/tarot_agent/utils/__init__.py
```

## 2단계: 핵심 상태 및 도구 분리

### 2.1: state.py 생성

`parsing/parser/tarot_agent/utils/state.py` 파일을 생성하고 다음 내용을 추가:

```python
"""
타로 LangGraph 상태 정의 모듈
"""
from typing import Annotated, List, Dict, Any, Optional, Literal
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class TarotState(TypedDict):
    """최적화된 타로 상태"""
    # 기본 메시지 관리
    messages: Annotated[List[BaseMessage], add_messages]
    
    # 사용자 의도 (핵심!)
    user_intent: Literal["card_info", "spread_info", "consultation", "general", "simple_card", "unknown"]
    user_input: str
    
    # 상담 전용 데이터 (consultation일 때만 사용)
    consultation_data: Optional[Dict[str, Any]]
    
    # Supervisor 관련 필드
    supervisor_decision: Optional[Dict[str, Any]]
    
    # 라우팅 관련 (새로 추가)
    routing_decision: Optional[str]
    target_handler: Optional[str]
    needs_llm: Optional[bool]
    
    # 세션 메모리 (새로 추가)
    session_memory: Optional[Dict[str, Any]]
    conversation_memory: Optional[Dict[str, Any]]
    
    # 시간 맥락 정보 (새로 추가)
    temporal_context: Optional[Dict[str, Any]]
    search_timestamp: Optional[str]
    
    # 웹 검색 관련 필드 (새로 추가)
    search_results: Optional[Dict[str, Any]]
    search_decision: Optional[Dict[str, Any]]
```

### 2.2: tools.py 생성

`parsing/parser/tarot_agent/utils/tools.py` 파일을 생성하고 다음 내용을 추가:

```python
"""
타로 LangGraph 도구 모듈
LLM이 직접 호출하는 @tool 데코레이터 함수들과 RAG 시스템 초기화 함수 포함
"""
from langchain_core.tools import tool
import sys
import os

# 상위 디렉토리를 path에 추가하여 tarot_rag_system.py를 임포트할 수 있도록 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from tarot_rag_system import TarotRAGSystem

# 전역 변수 (다른 모듈에서 임포트하여 사용)
rag_system = None

def initialize_rag_system():
    """RAG 시스템 초기화"""
    global rag_system
    try:
        rag_system = TarotRAGSystem(
            card_faiss_path="parsing/parser/tarot_card_faiss_index",
            spread_faiss_path="parsing/parser/tarot_spread_faiss_index"
        )
        print("✅ RAG 시스템 초기화 성공!")
        return rag_system
    except Exception as e:
        print(f"⚠️ RAG 시스템 초기화 실패: {e}")
        print("📝 기본 모드로 계속 진행합니다...")
        return None

@tool
def search_tarot_spreads(query: str) -> str:
    """타로 스프레드를 검색합니다."""
    global rag_system
    if rag_system is None:
        return "RAG 시스템이 초기화되지 않았습니다."
    
    try:
        results = rag_system.search_spreads(query, final_k=3, show_details=False)
        
        # 결과 포맷팅
        response = "### 🔍 스프레드 검색 결과\n\n"
        
        if not results:
            return f"{response}검색어 '{query}'에 대한 결과가 없습니다."
        
        for i, (doc, score) in enumerate(results, 1):
            spread_info = rag_system.doc_to_spread_info(doc)
            response += f"**{i}. {spread_info['name']}** (관련도: {score:.2f})\n"
            response += f"- 카드 수: {spread_info['card_count']}장\n"
            response += f"- 난이도: {spread_info['difficulty']}\n"
            response += f"- 설명: {spread_info['description'][:150]}...\n\n"
            
        return response
    except Exception as e:
        return f"스프레드 검색 중 오류 발생: {str(e)}"

@tool  
def search_tarot_cards(query: str) -> str:
    """타로 카드의 의미를 검색합니다."""
    global rag_system
    if rag_system is None:
        return "RAG 시스템이 초기화되지 않았습니다."
    
    try:
        results = rag_system.search_cards(query, final_k=3, show_details=False)
        
        # 결과 포맷팅
        response = "### 🔍 카드 검색 결과\n\n"
        
        if not results:
            return f"{response}검색어 '{query}'에 대한 결과가 없습니다."
        
        for i, (doc, score) in enumerate(results, 1):
            card_info = rag_system.doc_to_card_info(doc)
            response += f"**{i}. {card_info['name_ko']}** ({card_info['name']}, 관련도: {score:.2f})\n"
            response += f"- 정방향: {card_info['upright_meaning_ko'][:100]}...\n"
            response += f"- 역방향: {card_info['reversed_meaning_ko'][:100]}...\n\n"
            
        return response
    except Exception as e:
        return f"카드 검색 중 오류 발생: {str(e)}"
```

## 3단계: 유틸리티 모듈 분리

### 3.1: web_search.py 생성

`parsing/parser/tarot_agent/utils/web_search.py` 파일을 생성하고 웹 검색 관련 함수들을 추가:

```python
"""
웹 검색 관련 유틸리티 함수들
"""
import os
from typing import Dict, List, Any
from datetime import datetime
from langchain_openai import ChatOpenAI

# 웹 검색 관련 imports
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
try:
   from langchain_tavily import TavilySearch
   TAVILY_AVAILABLE = True
except ImportError:
   TAVILY_AVAILABLE = False
   print("⚠️ Tavily 라이브러리를 사용하려면 'pip install langchain-tavily' 설치 필요")

# 번역 함수 임포트 (순환 참조 방지를 위해 함수 내에서 임포트)
def translate_korean_to_english_with_llm(korean_query: str) -> str:
    """한국어 -> 영어 번역 함수"""
    # 순환 참조 방지를 위해 여기서 임포트
    from .translation import translate_korean_to_english_with_llm as translate_func
    return translate_func(korean_query)

def initialize_search_tools():
    """웹 검색 도구들을 초기화 (이중 백업 시스템)"""
    
    search_tools = {}
    
    # 1순위: Tavily Search (더 정확하고 신뢰할 수 있는 결과)
    if TAVILY_AVAILABLE:
        try:
            search_tools["tavily"] = TavilySearch(
                max_results=5,
            )
            print("✅ Tavily Search 도구 초기화 완료 (1순위)")
        except Exception as e:
            print(f"⚠️ Tavily Search 초기화 실패: {e}")
            search_tools["tavily"] = None
    else:
        search_tools["tavily"] = None
        print("⚠️ Tavily 라이브러리 없음")
    
    # 2순위: DuckDuckGo Search (백업 도구)
    try:
        search_tools["duckduckgo_results"] = DuckDuckGoSearchResults(max_results=5)
        search_tools["duckduckgo_run"] = DuckDuckGoSearchRun()
        print("✅ DuckDuckGo Search 도구 초기화 완료 (2순위 백업)")
    except Exception as e:
        print(f"⚠️ DuckDuckGo Search 초기화 실패: {e}")
        search_tools["duckduckgo_results"] = None
        search_tools["duckduckgo_run"] = None
    
    return search_tools

# 전역 검색 도구 초기화
SEARCH_TOOLS = initialize_search_tools()

def perform_web_search(query: str, search_type: str = "general") -> dict:
    """웹 검색 수행"""
    
    results = {
        "query": query,
        "search_type": search_type,
        "results": [],
        "source": None,
        "success": False,
        "error": None
    }
    
    # 한국 관련 키워드 및 최신 정보 키워드 추가
    current_year = datetime.now().year
    
    search_query = query
    
    # 한국 관련 키워드가 없으면 추가
    if "한국" not in query and "korea" not in query.lower() and "kr" not in query.lower():
        search_query = f"{query} 한국"
    
    # 최신 정보 키워드 추가 (년도가 이미 포함되어 있지 않은 경우만)
    if str(current_year) not in search_query and str(current_year-1) not in search_query:
        search_query = f"{search_query} {current_year} 최신"
    
    print(f"🔄 검색어 최종: {query} → {search_query}")
    
    # 한국어 우선, 필요시에만 영어 번역 추가
    if any(ord(char) > 127 for char in search_query):  # 한국어 포함 검사
        try:
            search_query_en = translate_korean_to_english_with_llm(search_query)
            # 한국어를 우선하고 영어를 보조로 사용
            search_query = f"{search_query} OR {search_query_en}"
            print(f"🔄 번역 추가: {search_query}")
        except:
            pass  # 번역 실패시 원본 사용
    
    # 1순위: Tavily Search 시도 (최신 정보 우선)
    if SEARCH_TOOLS.get("tavily"):
        try:
            # 최신 정보 우선 검색 (최근 1년 내)
            tavily_params = {
                "query": search_query,
                "max_results": 6,
                "days": 365  # 최근 1년 내 정보만
            }
            
            tavily_results = SEARCH_TOOLS["tavily"].invoke(tavily_params)
            if tavily_results:
                results["results"] = tavily_results
                results["source"] = "tavily"
                results["success"] = True
                print(f"✅ Tavily 최신 검색 성공: {len(tavily_results)}개 결과")
                return results
        except Exception as e:
            print(f"⚠️ Tavily 최신 검색 실패, 일반 검색 시도: {e}")
            # 시간 제한 검색이 실패하면 일반 검색 시도
            try:
                tavily_results = SEARCH_TOOLS["tavily"].invoke(search_query)
                if tavily_results:
                    results["results"] = tavily_results
                    results["source"] = "tavily"
                    results["success"] = True
                    print(f"✅ Tavily 일반 검색 성공: {len(tavily_results)}개 결과")
                    return results
            except Exception as e2:
                print(f"⚠️ Tavily 일반 검색도 실패, DuckDuckGo로 전환: {e2}")
    
    # 2순위: DuckDuckGo Search 시도 (Tavily 실패 시 백업)
    if SEARCH_TOOLS.get("duckduckgo_results"):
        try:
            # 최신 정보 우선 검색 (검색어에 최신 키워드 이미 포함됨)
            ddg_results = SEARCH_TOOLS["duckduckgo_results"].invoke(search_query)
            if ddg_results:
                results["results"] = ddg_results
                results["source"] = "duckduckgo"
                results["success"] = True
                print(f"✅ DuckDuckGo 최신 검색 성공: {len(ddg_results)}개 결과")
                return results
        except Exception as e:
            print(f"⚠️ DuckDuckGo 최신 검색 실패, 일반 검색 시도: {e}")
            # 최신 키워드 제거하고 일반 검색 시도
            try:
                # 최신 키워드 제거
                fallback_query = search_query.replace(f" {current_year} 최신", "").replace(" 최신", "")
                ddg_results = SEARCH_TOOLS["duckduckgo_results"].invoke(fallback_query)
                if ddg_results:
                    results["results"] = ddg_results
                    results["source"] = "duckduckgo"
                    results["success"] = True
                    print(f"✅ DuckDuckGo 일반 검색 성공: {len(ddg_results)}개 결과")
                    return results
            except Exception as e2:
                print(f"⚠️ DuckDuckGo 일반 검색도 실패: {e2}")
    
    # 모든 검색 실패
    results["error"] = "Tavily와 DuckDuckGo 모든 검색 도구 실패"
    print("❌ 모든 웹 검색 도구 실패")
    return results

def decide_web_search_need_with_llm(user_query: str, conversation_context: str = "") -> dict:
    """LLM을 활용한 지능적 웹 검색 필요성 판단"""
    
    llm = ChatOpenAI(
        model="gpt-4o", 
        temperature=0.1,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    
    prompt = f"""
    사용자 질문을 분석하여 웹 검색이 필요한지 판단해주세요.
    
    질문: {user_query}
    대화 맥락: {conversation_context}
    
    다음 기준으로 판단하세요:
    1. 현재 시장/경제 상황이 필요한가? (예: 취업, 창업, 투자)
    2. 최신 트렌드나 뉴스가 도움이 되는가? (예: 업계 동향, 사회 이슈)
    3. 객관적 데이터가 조언에 도움이 되는가? (예: 통계, 사실 확인)
    4. 개인적/감정적 문제로 내면 탐구가 더 중요한가? (예: 연애, 가족 관계)
    
    웹 검색이 필요한 경우:
    - 직업/취업/이직 관련 질문
    - 창업/사업 관련 질문  
    - 투자/재정 관련 질문
    - 최신 동향이 중요한 질문
    - 현실적 조건/환경 파악이 필요한 질문
    
    웹 검색이 불필요한 경우:
    - 순수한 감정/연애 문제
    - 개인적 내면 탐구
    - 타로 카드 자체에 대한 질문
    - 스프레드 방법 문의
    - 철학적/영적 질문
    
    JSON 형태로 답변:
    {
        "need_search": true/false,
        "reason": "판단 이유",
        "search_keywords": ["검색어1", "검색어2"],
        "confidence": 0.0~1.0
    }
    """
    
    try:
        response = llm.invoke(prompt)
        result = response.content
        
        # 문자열을 딕셔너리로 변환
        import json
        decision = json.loads(result)
        
        # 필수 필드 확인 및 기본값 설정
        if "need_search" not in decision:
            decision["need_search"] = False
        
        if "reason" not in decision:
            decision["reason"] = "판단 불가"
            
        if "search_keywords" not in decision:
            decision["search_keywords"] = [user_query]
            
        if "confidence" not in decision:
            decision["confidence"] = 0.5
            
        return decision
    
    except Exception as e:
        print(f"⚠️ 웹 검색 필요성 판단 중 오류: {e}")
        # 오류 시 기본값 반환
        return {
            "need_search": False,
            "reason": f"판단 중 오류 발생: {str(e)}",
            "search_keywords": [user_query],
            "confidence": 0.0
        }

def integrate_search_results_with_tarot(tarot_cards: List[Dict], search_results: dict, user_concern: str) -> str:
    """검색 결과와 타로 결합"""
    # 이 함수 구현은 다른 모듈에서 필요한 함수를 임포트하여 사용
    pass

def format_search_results_for_display(search_results: dict) -> str:
    """검색 결과 포맷팅"""
    # 이 함수 구현은 다른 모듈에서 필요한 함수를 임포트하여 사용
    pass
```

### 3.2: 나머지 유틸리티 모듈 생성

`analysis.py`, `timing.py`, `translation.py`, `helpers.py` 모듈도 위와 같은 방식으로 생성합니다. 각 모듈은 관련 함수들을 포함하고, 필요한 임포트와 전역 변수를 처리합니다.

## 4단계: 노드 함수 분리

`parsing/parser/tarot_agent/utils/nodes.py` 파일을 생성하고 모든 노드 및 핸들러 함수들을 추가합니다. 필요한 유틸리티 모듈을 임포트하고 전역 변수를 처리합니다.

## 5단계: 그래프 구성 및 메인 실행

`parsing/parser/tarot_agent/agent.py` 파일을 생성하고 그래프 구성 및 메인 실행 함수를 추가합니다. 필요한 모듈을 임포트하고 라우터 함수를 정의합니다.

## 테스트 및 디버깅

각 단계마다 다음 테스트를 진행하여 기능이 올바르게 작동하는지 확인합니다:

1. 모듈 임포트 테스트
2. 기능 단위 테스트
3. 통합 테스트

## 주의사항

1. 임포트 순환 참조 방지
2. 전역 변수 처리
3. 기존 코드 보존
4. RAG 시스템 로딩 확인 