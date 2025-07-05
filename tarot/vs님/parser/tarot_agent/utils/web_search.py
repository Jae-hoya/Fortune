"""

웹 검색 관련 유틸리티 함수들

"""

import os

from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults

from typing import List, Dict, Any

import json

# Tavily 라이브러리 조건부 임포트

try:
    from langchain_tavily import TavilySearch
    TAVILY_AVAILABLE = True

except ImportError:
    TAVILY_AVAILABLE = False
    print("⚠️ Tavily 라이브러리를 사용하려면 'pip install langchain-tavily' 설치 필요")

# 번역 함수 임포트

from .translation import translate_korean_to_english_with_llm

# SEARCH_TOOLS는 서비스 초기화 시 반드시 세팅되어야 함 (예: Tavily, DuckDuckGo 등)

SEARCH_TOOLS = {}

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
   
   # 한국 정보 우선 검색 로직 강화
   from datetime import datetime
   current_year = datetime.now().year
   
   # 1단계: 한국 키워드 강화
   search_query = query
   if "한국" not in query and "korea" not in query.lower() and "kr" not in query.lower():
       search_query = f"{query} 한국"
   
   # 2단계: 한국 사이트 우선 검색어 생성
   korean_sites = "site:naver.com OR site:daum.net OR site:donga.com OR site:chosun.com OR site:joongang.co.kr OR site:hani.co.kr OR site:khan.co.kr OR site:go.kr"
   korean_priority_query = f"({search_query}) AND ({korean_sites})"
   
   # 3단계: 최신 정보 키워드 추가
   if str(current_year) not in search_query and str(current_year-1) not in search_query:
       korean_priority_query = f"{korean_priority_query} {current_year}"
   
   print(f"🔄 한국 우선 검색어: {query} → {korean_priority_query}")
   
   # 4단계: 백업 검색어 (한국 사이트 제한 없이)
   fallback_query = f"{search_query} {current_year} 최신"
   print(f"🔄 백업 검색어: {fallback_query}")
   
   # 1순위: Tavily Search - 한국 사이트 우선 검색
   if SEARCH_TOOLS.get("tavily"):
       try:
           print("🇰🇷 한국 사이트 우선 검색 시도...")
           tavily_results = SEARCH_TOOLS["tavily"].invoke(korean_priority_query)
           if tavily_results and len(tavily_results) >= 2:  # 한국 사이트에서 충분한 결과
               filtered_results = filter_korean_results(tavily_results, query)
               results["results"] = filtered_results
               results["source"] = "tavily (한국 사이트 우선)"
               results["success"] = True
               print(f"✅ Tavily 한국 사이트 검색 성공: {len(tavily_results)}개 결과")
               return results
           else:
               print("⚠️ 한국 사이트 결과 부족, 백업 검색 진행...")
       except Exception as e:
           print(f"⚠️ Tavily 한국 사이트 검색 실패: {e}")
       
       # 백업: 일반 검색 (한국 키워드 포함)
       try:
           print("🔄 백업 검색 시도...")
           tavily_results = SEARCH_TOOLS["tavily"].invoke(fallback_query)
           if tavily_results:
               filtered_results = filter_korean_results(tavily_results, query)
               results["results"] = filtered_results
               results["source"] = "tavily (백업)"
               results["success"] = True
               print(f"✅ Tavily 백업 검색 성공: {len(tavily_results)}개 결과")
               return results
       except Exception as e:
           print(f"⚠️ Tavily 백업 검색도 실패, DuckDuckGo로 전환: {e}")
   
   # 2순위: DuckDuckGo Search - 한국 정보 우선
   if SEARCH_TOOLS.get("duckduckgo_results"):
       try:
           print("🇰🇷 DuckDuckGo 한국 우선 검색 시도...")
           # DuckDuckGo는 site: 문법이 제한적이므로 한국어 키워드 강화
           korea_enhanced_query = f"{search_query} 한국 사이트 네이버 다음 {current_year}"
           ddg_results = SEARCH_TOOLS["duckduckgo_results"].invoke(korea_enhanced_query)
           if ddg_results:
               filtered_results = filter_korean_results(ddg_results, query)
               results["results"] = filtered_results
               results["source"] = "duckduckgo (한국 우선)"
               results["success"] = True
               print(f"✅ DuckDuckGo 한국 우선 검색 성공: {len(ddg_results)}개 결과")
               return results
       except Exception as e:
           print(f"⚠️ DuckDuckGo 한국 우선 검색 실패: {e}")
           
       # 백업: 기본 검색
       try:
           print("🔄 DuckDuckGo 백업 검색 시도...")
           ddg_results = SEARCH_TOOLS["duckduckgo_results"].invoke(fallback_query)
           if ddg_results:
               filtered_results = filter_korean_results(ddg_results, query)
               results["results"] = filtered_results
               results["source"] = "duckduckgo (백업)"
               results["success"] = True
               print(f"✅ DuckDuckGo 백업 검색 성공: {len(ddg_results)}개 결과")
               return results
       except Exception as e:
           print(f"⚠️ DuckDuckGo 백업 검색도 실패: {e}")
   
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
   {{
       "need_search": true/false,
       "confidence": 0.0-1.0,
       "search_type": "market/news/data/trend/none",
       "search_query": "구체적 검색어 (need_search가 true인 경우만, 한국 정보 우선)",
       "reasoning": "판단 근거"
   }}
   
   **검색어 생성 시 주의사항:**
   - 한국 관련 정보를 우선적으로 검색하도록 "한국" 키워드 포함
   - 한국 사이트(naver.com, daum.net, nate.com, google.co.kr, korea.kr, go.kr)에서 정보 우선 검색
   - 예: "공무원 투잡" → "공무원 투잡 한국", "창업 아이디어" → "창업 아이디어 한국"
   """
   
   try:
       response = llm.invoke([HumanMessage(content=prompt)])
       decision = json.loads(response.content)
       
       print(f"🧠 웹 검색 필요성 판단: {decision.get('need_search', False)} (신뢰도: {decision.get('confidence', 0.0):.2f})")
       print(f"📝 판단 근거: {decision.get('reasoning', '')}")
       
       return decision
       
   except Exception as e:
       print(f"❌ 웹 검색 판단 오류: {e}")
       return {
           "need_search": False,
           "confidence": 0.0,
           "search_type": "none",
           "search_query": "",
           "reasoning": "판단 함수 오류로 인해 검색 안함"
       }

def extract_relevant_keywords(user_query: str) -> list:
   """사용자 질문에서 검색 관련 키워드를 동적으로 추출"""
   if not user_query:
       return []
   
   try:
       llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
       prompt = f"""
       다음 사용자 질문에서 웹 검색 결과 필터링에 유용한 핵심 키워드들을 추출해주세요:
       
       질문: "{user_query}"
       
       **추출 기준:**
       1. 직업/업종 관련 키워드 (예: 공무원, 교사, 의사, 간호사, 회사원 등)
       2. 활동/행위 관련 키워드 (예: 투잡, 부업, 창업, 투자, 취업 등) 
       3. 분야/영역 관련 키워드 (예: IT, 교육, 의료, 금융, 예술 등)
       4. 상황/맥락 관련 키워드 (예: 재택근무, 온라인, 오프라인 등)
       
       **주의사항:**
       - 한국어와 영어 둘 다 포함
       - 동의어나 유사어도 포함
       - 너무 일반적인 단어는 제외 (예: 사람, 일, 돈 등)
       - 최대 10개까지만
       
       JSON 형식으로 답변:
       {{"keywords": ["키워드1", "키워드2", ...]}}
       """
       
       response = llm.invoke([HumanMessage(content=prompt)])
       result = json.loads(response.content)
       keywords = result.get("keywords", [])
       
       print(f"🔍 동적 키워드 추출: {keywords}")
       return keywords
       
   except Exception as e:
       print(f"⚠️ 동적 키워드 추출 실패: {e}")
       return []

def filter_korean_results(results: list, user_query: str = "") -> list:
   """검색 결과에서 한국 관련 결과를 우선적으로 필터링"""
   if not results:
       print("🇰🇷 필터링: 입력 결과가 비어있음")
       return results
   
   print(f"🇰🇷 필터링 시작: 총 {len(results)}개 결과")
   print(f"🔍 검색 결과 구조 확인: {type(results)}")
   
   # Tavily API 응답 구조 확인 및 정규화
   normalized_results = []
   
   # 결과가 딕셔너리인 경우 (Tavily 응답 구조)
   if isinstance(results, dict):
       if 'results' in results:
           normalized_results = results['results']
       elif 'data' in results:
           normalized_results = results['data']
       else:
           normalized_results = [results]
   # 결과가 리스트인 경우
   elif isinstance(results, list):
       for item in results:
           if isinstance(item, dict):
               normalized_results.append(item)
           elif isinstance(item, str):
               # 문자열 결과를 딕셔너리로 변환
               normalized_results.append({
                   'title': '검색 결과',
                   'content': item,
                   'url': '',
                   'snippet': item
               })
           else:
               print(f"⚠️ 예상치 못한 결과 타입: {type(item)}")
   
   if not normalized_results:
       print("⚠️ 정규화된 결과가 없음, 원본 반환")
       return results[:3] if isinstance(results, list) else []
   
   print(f"🔍 정규화 완료: {len(normalized_results)}개 결과")
   
   korean_results = []
   other_results = []
   
   korean_domains = ['naver.com', 'daum.net', 'donga.com', 'chosun.com', 'joongang.co.kr', 
                    'hani.co.kr', 'khan.co.kr', 'go.kr', 'or.kr', 'co.kr', 'gov.kr', 'korea.net']
   
   # 기본 한국 키워드
   korean_keywords = ['한국', '국내', '서울', '부산', '대구', '인천', '광주', '대전', '울산', 'korea']
   
   # 사용자 질문에서 동적으로 관련 키워드 추출
   dynamic_keywords = extract_relevant_keywords(user_query) if user_query else []
   all_keywords = korean_keywords + dynamic_keywords
   
   print(f"🔍 사용된 키워드: 기본 {len(korean_keywords)}개 + 동적 {len(dynamic_keywords)}개 = 총 {len(all_keywords)}개")
   
   for i, result in enumerate(normalized_results):
       if isinstance(result, dict):
           url = result.get('url', '').lower()
           title = result.get('title', '').lower()
           content = result.get('content', result.get('snippet', '')).lower()
           
           print(f"🔍 결과 {i+1}: URL={url[:50]}..., 제목={title[:30]}...")
           
           # 한국 도메인 확인
           is_korean_domain = any(domain in url for domain in korean_domains)
           
           # 한국 키워드 확인 (기본 + 동적 키워드)
           is_korean_content = any(keyword in title + content for keyword in all_keywords)
           
           # 추가 조건: 동적으로 추출된 키워드가 있으면 관련성 높음
           is_relevant_content = len(dynamic_keywords) > 0 and any(keyword in title + content for keyword in dynamic_keywords)
           
           if is_korean_domain or is_korean_content or is_relevant_content:
               korean_results.append(result)
               print(f"   ✅ 한국 관련으로 분류 (도메인:{is_korean_domain}, 키워드:{is_korean_content}, 관련:{is_relevant_content})")
           else:
               other_results.append(result)
               print(f"   ❌ 기타로 분류")
       else:
           print(f"🔍 결과 {i+1}: 딕셔너리가 아님 - {type(result)}")
           # 딕셔너리가 아닌 경우도 처리
           if isinstance(result, str):
               other_results.append({
                   'title': '검색 결과',
                   'content': result,
                   'url': '',
                   'snippet': result
               })
           else:
               other_results.append(result)
   
   # 한국 결과 우선, 부족하면 다른 결과로 보완
   filtered_results = korean_results[:3]  # 한국 결과 최대 3개
   if len(filtered_results) < 3:
       remaining_slots = 3 - len(filtered_results)
       filtered_results.extend(other_results[:remaining_slots])
   
   # 최소 1개는 보장 (모든 필터링이 실패해도)
   if len(filtered_results) == 0 and len(normalized_results) > 0:
       filtered_results = normalized_results[:3]
       print("⚠️ 모든 필터링 실패, 정규화된 결과 사용")
   
   print(f"🇰🇷 결과 필터링: 한국 관련 {len(korean_results)}개, 기타 {len(other_results)}개 → 최종 {len(filtered_results)}개 선택")
   return filtered_results

def integrate_search_results_with_tarot(tarot_cards: List[Dict], search_results: dict, user_concern: str) -> str:
   """검색 결과를 타로 해석에 통합"""
   
   if not search_results.get("success") or not search_results.get("results"):
       return ""
   
   # 검색 결과 요약
   search_summary = ""
   results_data = search_results["results"]
   
   # 딕셔너리인 경우 처리
   if isinstance(results_data, dict):
       # 딕셔너리에서 실제 결과 리스트 찾기
       if "results" in results_data:
           results = results_data["results"]
       elif "data" in results_data:
           results = results_data["data"]
       else:
           # 딕셔너리 자체가 하나의 결과일 수 있음
           results = [results_data]
   elif isinstance(results_data, list):
       results = results_data
   else:
       return ""
   
   if isinstance(results, list) and len(results) > 0:
       # 상위 3개 결과만 사용
       top_results = results[:3]
       search_summary = "\n".join([
           f"- {result.get('title', '제목 없음')}: {result.get('content', result.get('snippet', '내용 없음'))[:200]}"
           for result in top_results
           if isinstance(result, dict)
       ])
   
   if not search_summary:
       return ""
   
   llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
   
   # 카드 정보 요약
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
       # 딕셔너리에서 실제 결과 리스트 찾기
       if "results" in results_data:
           results = results_data["results"]
       elif "data" in results_data:
           results = results_data["data"]
       else:
           # 딕셔너리 자체가 하나의 결과일 수 있음
           results = [results_data]
   elif isinstance(results_data, list):
       results = results_data
   else:
       return ""
   
   if not isinstance(results, list) or len(results) == 0:
       return ""
   
   formatted = f"\n\n📊 **참고한 현실 정보** (출처: {search_results.get('source', '웹 검색')}):\n"
   
   # 상위 3개 결과 표시 (2개에서 3개로 증가)
   for i, result in enumerate(results[:3], 1):
       if isinstance(result, dict):
           title = result.get('title', '제목 없음')
           content = result.get('content', result.get('snippet', '내용 없음'))
           url = result.get('url', '')
           
           # 내용이 너무 길면 자르기
           if len(content) > 150:
               content = content[:150] + "..."
           
           formatted += f"{i}. **{title}**\n   {content}\n"
           if url:
               formatted += f"   🔗 {url}\n"
           formatted += "\n"
   
   return formatted
