from dotenv import load_dotenv
load_dotenv()

import os
import random
import re
import json
from typing import Annotated, List, Dict, Any, Optional, Literal
from typing_extensions import TypedDict
import numpy as np
import scipy.stats as stats
from scipy.stats import hypergeom
import math
from collections import Counter
from datetime import datetime, timedelta
import pytz

# LangChain 및 LangGraph 관련 imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# 기존 RAG 시스템 import
from tarot_rag_system import TarotRAGSystem

# 웹 검색 관련 imports 추가
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
try:
   from langchain_tavily import TavilySearch
   TAVILY_AVAILABLE = True
except ImportError:
   TAVILY_AVAILABLE = False
   print("⚠️ Tavily 라이브러리를 사용하려면 'pip install langchain-tavily' 설치 필요")

# =================================================================
# State 정의 
# =================================================================

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

# =================================================================
# 웹 검색 도구 설정 (기존 그대로)
# =================================================================

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
   from datetime import datetime
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

# =================================================================
# 시간 맥락 관리 함수들 (기존 그대로)
# =================================================================

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

def get_recent_timeframe(now: datetime) -> str:
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
   
   # 현재 날짜 가져오기 (한국 시간)
   kst = pytz.timezone('Asia/Seoul')
   current_date = datetime.now(kst)
   
   concrete_timing = []
   
   # 단일 타이밍 객체도 처리
   timing_list = tarot_timing.get("timing_predictions", [tarot_timing])
   
   for timing in timing_list:
       # days_min, days_max 값 추출
       days_min = timing.get("days_min", 1)
       days_max = timing.get("days_max", 7)
       
       # 실제 날짜 계산
       start_date = current_date + timedelta(days=days_min)
       end_date = current_date + timedelta(days=days_max)
       
       # 연도가 바뀌는 경우 처리 (시작일 또는 종료일이 현재 년도와 다른 경우)
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
   
   return {
       "abstract_timing": tarot_timing,
       "concrete_timing": concrete_timing,
       "current_context": current_context
   }

def ensure_temporal_context(state: TarotState) -> TarotState:
   """상태에 시간 맥락 정보가 없으면 추가"""
   if not state.get("temporal_context"):
       state["temporal_context"] = get_current_context()
   return state

# =================================================================
# 확률 계산 정교화 (기존 그대로)
# =================================================================

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


# =================================================================
# 원소/수비학 통합 분석 (기존 그대로)
# =================================================================

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

# =================================================================
# 유틸리티 함수들 (기존 그대로)
# =================================================================

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

# 기존 시기 예측 함수들 그대로 유지
def predict_timing_from_card_metadata(card_info: dict) -> dict:
   """카드 메타데이터로 시기 예측 - 개선된 버전"""
   timing_info = {
       "time_frame": "알 수 없음",
       "days_min": 0,
       "days_max": 365,
       "speed": "보통",
       "description": "시기 정보가 부족합니다.",
       "confidence": "낮음"
   }
   
   suit = card_info.get("suit", "")
   suit_timing = {
       "Wands": {
           "days_min": 1, "days_max": 7,
           "time_frame": "1-7일",
           "speed": "매우 빠름", 
           "description": "불의 원소 - 즉각적이고 에너지 넘치는 변화"
       },
       "Cups": {
           "days_min": 7, "days_max": 30,
           "time_frame": "1-4주",
           "speed": "보통",
           "description": "물의 원소 - 감정적 변화, 점진적 발전"
       },
       "Swords": {
           "days_min": 3, "days_max": 14,
           "time_frame": "3일-2주", 
           "speed": "빠름",
           "description": "공기의 원소 - 정신적 변화, 빠른 의사결정"
       },
       "Pentacles": {
           "days_min": 30, "days_max": 180,
           "time_frame": "1-6개월",
           "speed": "느림",
           "description": "흙의 원소 - 물질적 변화, 실제적이고 지속적인 결과"
       }
   }
   
   if suit in suit_timing:
       timing_info.update(suit_timing[suit])
       timing_info["confidence"] = "중간"
   
   rank = card_info.get("rank", "")
   rank_multipliers = {
       "Ace": 0.5, "Two": 0.7, "Three": 0.8, "Four": 1.0, "Five": 1.3,
       "Six": 1.1, "Seven": 1.4, "Eight": 1.2, "Nine": 1.5, "Ten": 1.6,
       "Page": 0.6, "Knight": 0.4, "Queen": 1.3, "King": 1.5
   }
   
   if rank in rank_multipliers:
       multiplier = rank_multipliers[rank]
       timing_info["days_min"] = int(timing_info["days_min"] * multiplier)
       timing_info["days_max"] = int(timing_info["days_max"] * multiplier)
       timing_info["time_frame"] = format_time_range(timing_info["days_min"], timing_info["days_max"])
       timing_info["confidence"] = "높음"
   
   if card_info.get("is_major_arcana"):
       major_timing = {
           "The Fool": (1, 3), "The Magician": (1, 7), "The High Priestess": (30, 90),
           "The Empress": (90, 180), "The Emperor": (30, 90), "The Hierophant": (90, 180),
           "The Lovers": (14, 56), "The Chariot": (7, 14), "Strength": (30, 90),
           "The Hermit": (90, 270), "Wheel of Fortune": (90, 180), "Justice": (30, 180),
           "The Hanged Man": (180, 365), "Death": (90, 365), "Temperance": (90, 180),
           "The Devil": (1, 90), "The Tower": (1, 7), "The Star": (180, 730),
           "The Moon": (30, 180), "The Sun": (30, 90), "Judgement": (90, 365),
           "The World": (180, 730)
       }
       
       card_name = card_info.get("card_name", "")
       if card_name in major_timing:
           timing_info["days_min"], timing_info["days_max"] = major_timing[card_name]
           timing_info["time_frame"] = format_time_range(timing_info["days_min"], timing_info["days_max"])
           timing_info["description"] = "메이저 아르카나 - 인생의 중요한 변화"
           timing_info["confidence"] = "높음"
   
   orientation = card_info.get("orientation", "")
   if orientation == "reversed":
       timing_info["days_min"] = int(timing_info["days_min"] * 1.5)
       timing_info["days_max"] = int(timing_info["days_max"] * 1.5)
       timing_info["time_frame"] = format_time_range(timing_info["days_min"], timing_info["days_max"])
       timing_info["description"] += " (역방향 - 지연 또는 내적 변화)"
   
   return timing_info

def predict_timing_with_current_date(card_info: dict, temporal_context: dict = None) -> dict:
   """현재 날짜를 고려한 개선된 시기 예측"""
   
   # 기본 타로 시기 분석
   basic_timing = predict_timing_from_card_metadata(card_info)
   
   # 현재 시간 맥락 확보
   if not temporal_context:
       temporal_context = get_current_context()
   
   # 현재 날짜와 통합
   enhanced_timing = integrate_timing_with_current_date(
       {"timing_predictions": [basic_timing]}, 
       temporal_context
   )
   
   # 결과 통합
   result = {
       "basic_timing": basic_timing,
       "current_context": temporal_context,
       "concrete_dates": enhanced_timing["concrete_timing"],
       "recommendations": generate_timing_recommendations(basic_timing, temporal_context)
   }
   
   return result

def generate_timing_recommendations(timing_info: dict, temporal_context: dict) -> list:
   """시간 맥락을 고려한 타이밍 추천"""
   recommendations = []
   
   current_season = temporal_context.get("season", "")
   current_month = temporal_context.get("current_month", 1)
   
   # 계절별 추천
   season_advice = {
       "봄": "새로운 시작과 성장의 에너지가 강한 시기입니다.",
       "여름": "활발한 활동과 결실을 맺기 좋은 시기입니다.", 
       "가을": "수확과 정리, 준비의 시기입니다.",
       "겨울": "내적 성찰과 계획 수립에 적합한 시기입니다."
   }
   
   if current_season in season_advice:
       recommendations.append(f"🌱 현재 {current_season}철: {season_advice[current_season]}")
   
   # 타로 시기와 현재 시기 조합
   speed = timing_info.get("speed", "보통")
   if speed == "매우 빠름":
       recommendations.append("⚡ 즉각적인 행동이 필요한 시기입니다.")
   elif speed == "빠름":
       recommendations.append("🏃 신속한 결정과 실행이 중요합니다.")
   elif speed == "느림":
       recommendations.append("🐌 인내심을 갖고 차근차근 준비하세요.")
   
   # 월별 특성 고려
   if current_month in [1, 2]:  # 신년
       recommendations.append("🎊 새해 새로운 계획을 세우기 좋은 시기입니다.")
   elif current_month in [3, 4]:  # 봄
       recommendations.append("🌸 변화와 새로운 도전을 시작하기 좋습니다.")
   elif current_month in [9, 10]:  # 가을
       recommendations.append("🍂 성과를 정리하고 다음 단계를 준비하세요.")
   elif current_month == 12:  # 연말
       recommendations.append("🎄 올해를 마무리하고 내년을 준비하는 시기입니다.")
   
   return recommendations

def format_time_range(days_min: int, days_max: int) -> str:
   """일수를 사용자 친화적 시간 표현으로 변환"""
   if days_max <= 7:
       return f"{days_min}-{days_max}일"
   elif days_max <= 30:
       weeks_min = max(1, days_min // 7)
       weeks_max = days_max // 7
       if weeks_min == weeks_max:
           return f"{weeks_min}주"
       return f"{weeks_min}-{weeks_max}주"
   elif days_max <= 365:
       months_min = max(1, days_min // 30)
       months_max = days_max // 30
       if months_min == months_max:
           return f"{months_min}개월"
       return f"{months_min}-{months_max}개월"
   else:
       years_min = max(1, days_min // 365)
       years_max = days_max // 365
       if years_min == years_max:
           return f"{years_min}년"
       return f"{years_min}-{years_max}년"

# 기존 카드 데이터베이스 그대로 유지
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
    
# 기존 카드/스프레드 처리 함수들 그대로 유지
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

# 기존 감정 분석 함수들 그대로 유지
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

# 기존 스프레드 관련 함수들 그대로 유지
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

# 헬퍼 함수들 (기존 그대로)
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

def check_if_has_specific_concern(user_input: str) -> bool:
   """LLM을 사용해서 사용자 입력에 구체적인 고민이 포함되어 있는지 판단"""
   
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
    단순화된 트리거 체크 - triggerplan.md에 따른 핵심 트리거만
    
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

# =================================================================
# 최적화된 핵심 노드들
# =================================================================

def state_classifier_node(state: TarotState) -> TarotState:
   """🆕 상태 기반 빠른 분류 - LLM 호출 최소화"""
   
   # Step 1: 명확한 상태는 바로 분류 (LLM 없이)
   consultation_data = state.get("consultation_data", {})
   status = consultation_data.get("status", "") if consultation_data else ""
   
   print(f"🔍 State Classifier: status='{status}'")
   
   # 상담 진행 중이면 바로 라우팅
   if status in ["spread_selection", "card_selection", "summary_shown"]:
       handler = determine_consultation_handler(status)
       print(f"🚀 Fast Track: CONSULTATION_ACTIVE -> {handler}")
       return {
           "routing_decision": "CONSULTATION_ACTIVE",
           "target_handler": handler,
           "needs_llm": False
       }
   
   # 상담 완료 후 추가 질문 판단
   if status == "completed":
       user_input = get_last_user_input(state)
       if is_simple_followup(user_input):  # 간단한 패턴 매칭
           print(f"🚀 Fast Track: FOLLOWUP_QUESTION")
           return {
               "routing_decision": "FOLLOWUP_QUESTION", 
               "target_handler": "context_reference_handler",
               "needs_llm": False
           }
   
   # Step 2: 애매한 경우만 LLM 사용
   print(f"🧠 Complex Analysis: NEW_SESSION")
   return {
       "routing_decision": "NEW_SESSION",
       "needs_llm": True
   }

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

def supervisor_master_node(state: TarotState) -> TarotState:
   """🆕 복잡한 경우만 전체 분석"""
   
   # 시간 맥락 설정
   state = ensure_temporal_context(state)
   
   # 기존 supervisor_llm_node 호출
   print("🧠 Supervisor Master: 전체 분석 시작")
   supervisor_result = supervisor_llm_node(state)
   state.update(supervisor_result)
   
   # 필요시 의도 분류
   supervisor_decision = state.get("supervisor_decision", {})
   if supervisor_decision.get("action") == "route_to_intent":
       print("🔍 의도 분류 실행")
       intent_result = classify_intent_node(state)
       state.update(intent_result)
   
   # 적절한 핸들러 결정
   target_handler = determine_target_handler(state)
   state["target_handler"] = target_handler
   
   print(f"🎯 Target Handler: {target_handler}")
   return state

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
       # 의도 기반 라우팅 (기존 route_by_intent 로직)
       intent = state.get("user_intent", "unknown")
       return {
           "card_info": "card_info_handler",
           "spread_info": "spread_info_handler",
           "consultation": "consultation_handler", 
           "general": "general_handler",
           "simple_card": "simple_card_handler"  # 🆕 간단한 카드 뽑기 핸들러 추가
       }.get(intent, "unknown_handler")

def unified_processor_node(state: TarotState) -> TarotState:
   """🆕 통합 처리기 - 모든 기존 핸들러 함수들을 조건부로 호출"""
   
   target_handler = state.get("target_handler", "unknown_handler")
   print(f"🔧 Unified Processor: 실행할 핸들러 = {target_handler}")
   
   # 기존 함수들을 매핑으로 호출 (코드 그대로 재사용)
   function_map = {
       "card_info_handler": card_info_handler,
       "spread_info_handler": spread_info_handler,
       "consultation_handler": consultation_handler,
       "consultation_flow_handler": consultation_flow_handler,
       "consultation_continue_handler": consultation_continue_handler,
       "consultation_summary_handler": consultation_summary_handler,
       "consultation_individual_handler": consultation_individual_handler,
       "consultation_final_handler": consultation_final_handler,
       "general_handler": general_handler,
       "simple_card_handler": simple_card_handler,  # 🆕 간단한 카드 뽑기 핸들러 추가
       "context_reference_handler": context_reference_handler,
       "exception_handler": exception_handler,
       "emotional_support_handler": emotional_support_handler,
       "start_specific_spread_consultation": start_specific_spread_consultation,
       "unknown_handler": unknown_handler
   }
   
   # 해당 함수 실행 (기존 코드 그대로)
   handler_function = function_map.get(target_handler, unknown_handler)
   print(f"🔧 실행 중: {handler_function.__name__}")
   
   try:
       result = handler_function(state)
       print(f"✅ 핸들러 실행 완료: {handler_function.__name__}")
       return result
   except Exception as e:
       print(f"❌ 핸들러 실행 오류: {handler_function.__name__} - {e}")
       return {"messages": [AIMessage(content="처리 중 오류가 발생했습니다. 다시 시도해주세요.")]}

def unified_tool_handler_node(state: TarotState) -> TarotState:
   """🆕 통합 도구 처리기"""
   
   print("🔧 Tool Handler: 도구 실행 시작")
   
   # Step 1: 기존 tool_node 실행
   tools = [search_tarot_spreads, search_tarot_cards]
   tool_node = ToolNode(tools)
   tool_result = tool_node.invoke(state)
   
   print("🔧 Tool Handler: 도구 실행 완료, 결과 처리 시작")
   
   # Step 2: 기존 tool_result_handler 실행  
   final_result = tool_result_handler(tool_result)
   
   print("✅ Tool Handler: 최종 결과 생성 완료")
   return final_result

# =================================================================
# 기존 핸들러 함수들 (그대로 유지)
# =================================================================

def supervisor_llm_node(state: TarotState) -> TarotState:
   """기존 supervisor 함수 (그대로 유지)"""
   
   user_input = ""
   for msg in reversed(state["messages"]):
       if isinstance(msg, HumanMessage):
           user_input = msg.content.strip()
           break
   
   # 최근 대화 맥락 간단히 추출
   recent_context = ""
   messages = state.get("messages", [])
   if len(messages) >= 2:
       last_ai = None
       for msg in reversed(messages):
           if isinstance(msg, AIMessage):
               last_ai = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
               break
       
       if last_ai:
           recent_context = f"직전 AI 응답: {last_ai}"
   
   llm = ChatOpenAI(
       model="gpt-4o", 
       temperature=0.1,
       model_kwargs={"response_format": {"type": "json_object"}}
   )
   
   prompt = f"""
   당신은 대화 흐름을 파악하는 전문가입니다.

   **현재 상황:**
   사용자 입력: "{user_input}"
   {recent_context}
   
   **특별 판단 규칙:**
   만약 직전 AI 응답에 "카드 한 장 뽑아서 알아보길 원하시면 '네'라고 답해주세요"가 포함되어 있고,
   사용자 입력이 "네", "좋아", "그래", "응", "해줘" 등의 단순 긍정 응답이라면,
   이는 카드 뽑기 요청이므로 새로운 주제로 판단해야 합니다.

   **판단 기준:**
   사용자가 방금 전 답변에 대해 추가로 궁금해하는 것인지, 
   아니면 완전히 새로운 주제를 시작하는 것인지 판단하세요.

   **추가 질문의 신호들:**
   - "어떻게", "왜", "그게", "그거", "아까", "방금"
   - 구체적 설명 요구: "더 자세히", "설명해봐"
   - 의문 표현: "?", "하냐고", "거야"
   - 짧고 직접적인 질문

   **새로운 주제의 신호들:**
   - 완전히 다른 카드나 스프레드 언급
   - 새로운 고민이나 상황 설명
   - 정중한 새 요청: "다른 것도", "이번엔"
   - **🔥 타로 상담 키워드**: "타로 봐줘", "타로봐줢", "타로 상담", "점 봐줘", "운세 봐줢" 등이 포함된 경우 **무조건** 새로운 주제로 판단
   - **중요**: 단순한 긍정 응답("네", "좋아", "그래" 등)이 직전 AI가 카드 뽑기를 제안한 후 나왔다면 새로운 주제로 판단

   다음 JSON으로 답변:
   {{
       "is_followup": true/false,
       "confidence": "high|medium|low",
       "reasoning": "판단 근거",
       "action": "handle_context_reference|route_to_intent"
   }}
   """
   
   try:
       # 🔧 스마트한 타로 상담 키워드 체크 (컨텍스트 고려)
       tarot_triggers = ["타로 봐줘", "타로봐줘", "타로 상담", "점 봐줘", "운세 봐줢", "새로 봐줘"]
       has_tarot_trigger = any(trigger in user_input.lower() for trigger in tarot_triggers)
       
       if has_tarot_trigger:
           # 🔧 직전 AI 응답에서 "타로 봐줘"를 제안했는지 확인
           ai_suggested_tarot = False
           if recent_context and ("타로 봐줘" in recent_context or "타로 상담" in recent_context):
               ai_suggested_tarot = True
               print(f"🔧 AI가 타로 상담을 제안한 후 사용자가 응답함 - Follow-up으로 처리")
           
           if not ai_suggested_tarot:
               # AI가 제안하지 않았는데 사용자가 직접 "타로 봐줘" → 새 주제
               print(f"🎯 Supervisor: 사용자 주도 타로 상담 요청 → 의도 분류로 이동")
               return {
                   "user_input": user_input,
                   "supervisor_decision": {
                       "is_followup": False,
                       "confidence": "high", 
                       "reasoning": "사용자가 직접 새로운 타로 상담 요청",
                       "action": "route_to_intent"
                   }
               }
           # AI가 제안한 후 사용자 응답 → LLM이 판단하도록 진행
       
       # 🆕 단순 긍정 응답은 바로 의도 분류로 보내기
       simple_yes_responses = ["네", "좋아", "그래", "응", "해줘", "부탁해", "yes", "예"]
       if user_input.lower().strip() in simple_yes_responses:
           print(f"🎯 Supervisor: 단순 긍정 응답 감지 → 의도 분류로 이동")
           return {
               "user_input": user_input,
               "supervisor_decision": {
                   "is_followup": False,
                   "confidence": "high",
                   "reasoning": "단순 긍정 응답으로 의도 분류 필요",
                   "action": "route_to_intent"
               }
           }
       
       response = llm.invoke([HumanMessage(content=prompt)])
       decision = json.loads(response.content)
       
       is_followup = decision.get("is_followup", False)
       confidence = decision.get("confidence", "medium")
       action = "handle_context_reference" if is_followup else "route_to_intent"
       
       print(f"🎯 Supervisor: {'Follow-up' if is_followup else 'New Topic'} (신뢰도: {confidence})")
       
       return {
           "user_input": user_input,
           "supervisor_decision": {
               "is_followup": is_followup,
               "confidence": confidence,
               "reasoning": decision.get("reasoning", ""),
               "action": action
           }
       }
       
   except Exception as e:
       print(f"❌ Supervisor 오류: {e}")
       return {
           "user_input": user_input,
           "supervisor_decision": {
               "is_followup": True,  # 안전하게 follow-up으로 처리
               "confidence": "low",
               "action": "handle_context_reference"
           }
       }

def classify_intent_node(state: TarotState) -> TarotState:
   """🆕 트리거 기반 의도 분류 - 명확한 키워드만 consultation으로 분류"""
   
   # 시간 맥락 설정
   state = ensure_temporal_context(state)
   
   # 마지막 사용자 메시지 추출
   user_input = ""
   for msg in reversed(state["messages"]):
       if isinstance(msg, HumanMessage):
           user_input = msg.content.strip()
           break
   
   if not user_input:
       return {
           "user_intent": "unknown",
           "user_input": ""
       }
   
   user_input_lower = user_input.lower()
   
   # 🆕 1단계: 명확한 타로 상담 트리거 체크
   tarot_triggers = ["타로 봐줘", "타로봐줘", "새로 봐줘", "처음부터", "타로 상담", "점 봐줘", "운세 봐줘"]
   if any(trigger in user_input_lower for trigger in tarot_triggers):
       print(f"🎯 타로 상담 트리거 감지: consultation")
       return {
           "user_intent": "consultation",
           "user_input": user_input
       }
   
   # 🆕 1-1단계: 간단한 카드 뽑기 트리거 체크
   simple_card_triggers = ["카드 한장", "카드 뽑", "카드뽑", "간단히", "가볍게", "알아볼까", "뽑아서", "한장만", "뽑아줘", "뽑아주", "뽑아"]
   
   # 명확한 긍정 응답 체크 (카드 뽑기 제안에 대한 응답)
   clear_yes_triggers = ["네", "좋아", "그래", "응", "해줘", "부탁해", "yes", "예"]
   has_clear_yes = any(trigger in user_input_lower for trigger in clear_yes_triggers)
   has_other_intent = any(keyword in user_input_lower for keyword in ["타로", "상담", "고민", "문제"])
   
   if any(trigger in user_input_lower for trigger in simple_card_triggers):
       print(f"🎯 간단한 카드 뽑기 트리거 감지: simple_card")
       return {
           "user_intent": "simple_card", 
           "user_input": user_input
       }
   elif has_clear_yes and not has_other_intent and len(user_input.strip()) <= 15:
       # 명확한 긍정 응답이면서 다른 의도가 없고 짧은 응답일 때
       print(f"🎯 카드 뽑기 제안에 대한 긍정 응답: simple_card")
       return {
           "user_intent": "simple_card",
           "user_input": user_input
       }
   
   # 🆕 2단계: 카드/스프레드 정보 질문 체크 (키워드 기반)
   card_keywords = ["카드 의미", "카드는", "역방향", "정방향", "메이저 아르카나", "마이너 아르카나"]
   spread_keywords = ["스프레드", "켈틱크로스", "3장", "5장", "배치"]
   
   if any(keyword in user_input_lower for keyword in card_keywords):
       print(f"🎯 카드 정보 질문 감지: card_info")
       return {
           "user_intent": "card_info",
           "user_input": user_input
       }
   
   if any(keyword in user_input_lower for keyword in spread_keywords):
       print(f"🎯 스프레드 정보 질문 감지: spread_info")
       return {
           "user_intent": "spread_info", 
           "user_input": user_input
       }
   
   # 🆕 3단계: 나머지는 모두 general로 분류 (일상 대화, 타로 일반 질문 등)
   print(f"🎯 일반 대화로 분류: general")
   return {
       "user_intent": "general",
       "user_input": user_input
   }

# 기존 핸들러 함수들 (모두 그대로 유지)
def card_info_handler(state: TarotState) -> TarotState:
   """카드 정보 핸들러 - 기존 RAG 기능 완전 통합"""
   user_input = state["user_input"]
   
   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
   llm_with_tools = llm.bind_tools([search_tarot_cards])
   
   prompt = f"""
   사용자가 타로 카드에 대해 질문했습니다: "{user_input}"
   
   search_tarot_cards 도구를 사용해서 관련 카드 정보를 검색하고, 
   친근하고 이해하기 쉽게 설명해주세요.
   
   마지막에 "다른 카드나 타로 상담이 필요하시면 언제든 말씀해주세요!"라고 덧붙여주세요.
   
   🔮 타로 상담사 톤으로 답변하세요.
   """
   
   try:
       response = llm_with_tools.invoke([HumanMessage(content=prompt)])
       return {"messages": [response]}
   except Exception as e:
       fallback_msg = f"🔮 카드 정보를 찾는 중 문제가 생겼어요. 다시 질문해주시면 더 정확히 답변드릴게요!\n\n다른 궁금한 점이나 고민이 있으시면 언제든 말씀해주세요!"
       return {"messages": [AIMessage(content=fallback_msg)]}

def spread_info_handler(state: TarotState) -> TarotState:
   """스프레드 정보 핸들러 - 기존 RAG 기능 완전 통합"""
   user_input = state["user_input"]
   
   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
   llm_with_tools = llm.bind_tools([search_tarot_spreads])
   
   prompt = f"""
   사용자가 타로 스프레드에 대해 질문했습니다: "{user_input}"
   
   search_tarot_spreads 도구를 사용해서 관련 스프레드 정보를 검색하고,
   스프레드의 특징, 사용법, 언제 사용하면 좋은지 등을 친근하게 설명해주세요.
   
          마지막에 "카드 한 장 뽑아서 알아보길 원하시면 '네'라고 답해주세요. 이 스프레드로 본격 상담을 원하시면 '타로 봐줘'라고 말씀해주세요!"라고 덧붙여주세요.
   
   🔮 타로 상담사 톤으로 답변하세요.
   """
   
   try:
       response = llm_with_tools.invoke([HumanMessage(content=prompt)])
       return {"messages": [response]}
   except Exception as e:
       fallback_msg = f"🔮 스프레드 정보를 찾는 중 문제가 생겼어요. 다시 질문해주시면 더 정확히 답변드릴게요!\n\n이 스프레드로 상담받고 싶으시거나 다른 고민이 있으시면 언제든 말씀해주세요!"
       return {"messages": [AIMessage(content=fallback_msg)]}

def simple_card_handler(state: TarotState) -> TarotState:
   """🆕 간단한 카드 한 장 뽑기 핸들러"""
   user_input = state["user_input"]
   
   # 🆕 대화 히스토리에서 원래 질문 찾기
   original_question = ""
   messages = state.get("messages", [])
   
   # 최근 사용자 메시지들을 역순으로 검색 (현재 "네" 제외)
   for msg in reversed(messages):
       if isinstance(msg, HumanMessage) and msg.content.strip().lower() not in ["네", "좋아", "그래", "응", "해줘", "부탁해", "yes", "예"]:
           original_question = msg.content.strip()
           break
   
   # 실제 질문이 있으면 그것을 사용, 없으면 현재 입력 사용
   question_for_interpretation = original_question if original_question else user_input
   
   print(f"🎯 원래 질문: '{original_question}' | 해석용 질문: '{question_for_interpretation}'")
   
   # 랜덤으로 카드 한 장 선택
   import random
   card_number = random.randint(1, 78)
   orientation = random.choice(["정방향", "역방향"])
   
   # 카드 정보 가져오기
   selected_card = select_cards_randomly_but_keep_positions([card_number], 1)[0]
   card_name = selected_card['name']
   card_name_kr = translate_card_info(card_name, orientation)
   
   # 카드 검색으로 의미 가져오기
   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
   llm_with_tools = llm.bind_tools([search_tarot_cards])
   
   search_prompt = f"{card_name} {orientation} meaning"
   
   try:
       # 카드 의미 검색
       search_response = llm_with_tools.invoke([HumanMessage(content=f"search_tarot_cards('{search_prompt}')")])
       
       # 간단한 해석 생성
       interpretation_prompt = f"""
       사용자가 간단한 질문을 했습니다: "{question_for_interpretation}"
       
       뽑힌 카드: {card_name_kr} ({orientation})
       
       **중요: 반드시 사용자의 구체적인 질문에 카드로 답변해주세요!**
       
       **답변 구조:**
       1. 뽑힌 카드 간단 소개 (1줄)
       2. **사용자 질문에 대한 카드의 직접적인 답변** (2-3줄) - 가장 중요!
       3. 카드가 제시하는 간단한 조언이나 방향성 (1-2줄)
       
       **예시 (만약 "짬뽕? 짜장?" 질문이라면):**
       - "운명의 수레바퀴가 말하길, 지금은 변화를 받아들일 때라고 하네요. 매운 짬뽕으로 가세요!"
       - "이 카드는 새로운 도전을 의미해요. 평소와 다른 선택을 해보는 건 어떨까요?"
       
       **사용자의 질문이 선택 관련이면:** 카드가 어떤 선택을 제시하는지
       **사용자의 질문이 상황 관련이면:** 카드가 그 상황을 어떻게 보는지
       **사용자의 질문이 감정 관련이면:** 카드가 그 감정에 어떤 메시지를 주는지
       
       마지막에 "도움이 되셨나요? 더 자세한 상담이 필요하시면 언제든 말씀해주세요!"라고 덧붙여주세요.
       
       🔮 친근하고 실용적인 톤으로, 사용자 질문과 카드를 확실히 연결해서 답변하세요.
       """
       
       interpretation_response = llm.invoke([HumanMessage(content=interpretation_prompt)])
       
       # 카드 정보를 해석에 자연스럽게 포함 (별도 표시 제거)
       final_message = interpretation_response.content
       
       return {"messages": [AIMessage(content=final_message)]}
       
   except Exception as e:
       # 간단한 폴백 응답
       fallback_msg = f"""🃏 뽑힌 카드는 **{card_name_kr} ({orientation})**이에요!

카드 해석 중 문제가 생겼지만, 이 카드는 분명 당신의 질문 "{user_input}"에 대한 답을 가지고 있을 거예요. 다시 시도해주시거나, 더 자세한 상담이 필요하시면 언제든 말씀해주세요!"""
       
       return {"messages": [AIMessage(content=fallback_msg)]}
   
def consultation_handler(state: TarotState) -> TarotState:
   """리팩토링된 상담 핸들러 - 새로운 노드들을 순차 실행"""
   print("🔧 기존 consultation_handler 호출 -> 리팩토링된 노드들로 처리")
   
   # 🔧 핵심 수정: user_input을 state에 설정
   user_input = get_last_user_input(state)
   state["user_input"] = user_input
   print(f"🔧 사용자 입력 설정: '{user_input}'")
   
   # 🆕 단순한 "타로 봐줘"만 있는 경우 처리 개선
   simple_triggers = ["타로 봐줘", "타로봐줘", "타로 상담", "점 봐줘", "운세 봐줘"]
   # LLM으로 고민이 포함되어 있는지 동적 판단
   has_specific_concern = check_if_has_specific_concern(user_input)
   
   # 🔧 대화 히스토리에서 최근 고민이 있는지 확인
   recent_concern = None
   if user_input.strip() in simple_triggers and not has_specific_concern:
       messages = state.get("messages", [])
       # 최근 3개의 사용자 메시지에서 고민 찾기
       user_messages = []
       for msg in reversed(messages):
           if isinstance(msg, HumanMessage):
               user_messages.append(msg.content)
               if len(user_messages) >= 3:
                   break
       
       # 최근 메시지들에서 고민이 있는지 확인
       for recent_msg in user_messages:
           if check_if_has_specific_concern(recent_msg):
               recent_concern = recent_msg
               print(f"🔧 대화 히스토리에서 고민 발견: '{recent_concern}'")
               break
   
   # 고민이 없으면 물어보기
   if user_input.strip() in simple_triggers and not has_specific_concern and not recent_concern:
       print("🔧 단순 트리거 감지 - 고민 문의")
       return {
           "messages": [AIMessage(content="""🔮 

어떤 고민이나 궁금한 점이 있으신가요? 편하게 말씀해주세요.

예를 들어:
• 연애나 인간관계 고민
• 진로나 직업 관련 고민  
• 현재 상황에 대한 조언
• 미래에 대한 궁금증
• 중요한 결정을 앞둔 상황

무엇이든 편하게 이야기해주시면, 가장 적합한 타로 스프레드로 답을 찾아드릴게요! ✨""")],
           "consultation_data": {
               "status": "waiting_for_concern"
           }
       }
   
   # 🔧 최근 고민이 있으면 그것을 사용해서 상담 시작
   if recent_concern:
       print(f"🔧 최근 고민으로 상담 시작: '{recent_concern}'")
       state["user_input"] = recent_concern  # 최근 고민으로 교체
   
   # Phase 1 리팩토링: 4개 노드를 순차 실행하여 동일한 결과 제공
   try:
       # 1. 감정 분석
       result1 = emotion_analyzer_node(state)
       state.update(result1)
       
       # 2. 웹 검색 판단
       result2 = web_search_decider_node(state)
       state.update(result2)
       
       # 3. 웹 검색 실행
       result3 = web_searcher_node(state)
       state.update(result3)
       
       # 4. 스프레드 추천
       result4 = spread_recommender_node(state)
       state.update(result4)
       
       print("✅ 리팩토링된 consultation_handler 성공적으로 완료")
       return state
       
   except Exception as e:
       import traceback
       error_details = traceback.format_exc()
       print(f"❌ 리팩토링된 consultation_handler 오류: {e}")
       print(f"❌ 오류 상세 정보:\n{error_details}")
       # 기본 에러 처리
       return {
           "messages": [AIMessage(content="🔮 상담 처리 중 문제가 발생했습니다. 다시 시도해주세요.")],
           "consultation_data": {
               "status": "error"
           }
       }

# =================================================================
# Phase 1 리팩토링: consultation_handler 분해 (표준 LangGraph 패턴)
# =================================================================

def emotion_analyzer_node(state: TarotState) -> TarotState:
    """감정 분석 전용 노드 - LLM 1개만 사용"""
    user_input = state.get("user_input") or get_last_user_input(state)
    
    print("🔧 감정 분석 노드 실행")
    
    # 기존 로직 완전 보존
    emotional_analysis = analyze_emotion_and_empathy(user_input)
    empathy_message = generate_empathy_message(emotional_analysis, user_input)
    
    emotion = emotional_analysis.get('primary_emotion', '알 수 없음')
    intensity = emotional_analysis.get('emotion_intensity', '보통')
    
    # 감정에 따른 인사말 (기존 로직 완전 보존)
    if emotion == "불안" and intensity in ["높음", "매우높음"]:
        emotional_greeting = "🤗 불안한 마음을 달래드릴 수 있는 스프레드들을 준비했습니다."
    elif emotion == "슬픔":
        emotional_greeting = "💙 마음의 위로가 될 수 있는 스프레드들을 선별했습니다."
    elif emotion == "걱정":
        emotional_greeting = "🌟 걱정을 덜어드릴 수 있는 희망적인 스프레드들을 찾아드렸습니다."
    else:
        emotional_greeting = "🔮 상황에 가장 적합한 스프레드들을 찾아드렸습니다."
    
    return {
        "emotional_analysis": emotional_analysis,
        "empathy_message": empathy_message,
        "emotional_greeting": emotional_greeting,
        "consultation_status": "emotion_analyzed"
    }

def web_search_decider_node(state: TarotState) -> TarotState:
    """웹 검색 필요성 판단 전용 노드 - LLM 1개만 사용"""
    user_input = state.get("user_input") or get_last_user_input(state)
    
    print("🔧 웹 검색 판단 노드 실행")
    
    # 기존 로직 완전 보존 - 대화 맥락 구성
    conversation_context = ""
    messages = state.get("messages", [])
    if len(messages) >= 2:
        last_ai = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                last_ai = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                break
        if last_ai:
            conversation_context = f"직전 대화: {last_ai}"
    
    # 기존 함수 그대로 사용
    search_decision = decide_web_search_need_with_llm(user_input, conversation_context)
    
    return {
        "search_decision": search_decision,
        "needs_web_search": search_decision.get("need_search", False) and search_decision.get("confidence", 0) > 0.4,
        "consultation_status": "search_decided"
    }

def web_searcher_node(state: TarotState) -> TarotState:
    """웹 검색 실행 전용 노드 - LLM 없음"""
    
    print("🔧 웹 검색 실행 노드 실행")
    
    # 웹 검색이 필요하지 않은 경우
    if not state.get("needs_web_search", False):
        return {
            "search_results": None,
            "consultation_status": "search_completed"
        }
    
    # 기존 로직 완전 보존
    search_decision = state.get("search_decision", {})
    user_input = state.get("user_input", "")
    
    search_query = search_decision.get("search_query", user_input)
    search_type = search_decision.get("search_type", "general")
    
    print(f"🔍 상담 중 웹 검색 실행: {search_query} (타입: {search_type})")
    search_results = perform_web_search(search_query, search_type)
    
    # 검색 결과 로깅 추가
    if search_results and search_results.get("success") and search_results.get("results"):
        results_data = search_results.get("results", [])
        print(f"🔍 검색 결과 구조 확인: {type(results_data)}")
        
        # 딕셔너리인 경우 처리
        if isinstance(results_data, dict):
            # 딕셔너리에서 실제 결과 리스트 찾기
            if "results" in results_data:
                results_list = results_data["results"]
            elif "data" in results_data:
                results_list = results_data["data"]
            else:
                # 딕셔너리 자체가 하나의 결과일 수 있음
                results_list = [results_data]
        elif isinstance(results_data, list):
            results_list = results_data
        else:
            results_list = []
            
        if isinstance(results_list, list) and len(results_list) > 0:
            result_count = len(results_list)
            print(f"✅ 웹 검색 성공: {result_count}개 결과 발견")
            
            # 첫 번째 결과 미리보기
            try:
                first_result = results_list[0]
                title = first_result.get('title', '제목 없음') if isinstance(first_result, dict) else '제목 없음'
                print(f"🔍 첫 번째 결과: {title}")
            except (IndexError, KeyError, TypeError) as e:
                print(f"🔍 첫 번째 결과 처리 중 오류: {e}")
        else:
            print(f"❌ 웹 검색 결과 처리 실패: {type(results_data)}")
    else:
        print("❌ 웹 검색 실패 또는 결과 없음")
    
    return {
        "search_results": search_results,
        "consultation_status": "search_completed"
    }

def perform_multilayer_spread_search(keywords: str, user_input: str) -> List[Dict]:
    """다층적 스프레드 검색 - 의도와 주제를 분리하여 검색"""
    recommended_spreads = []
    
    # 키워드를 의도와 주제로 분리
    keyword_parts = keywords.split()
    intent_keywords = []
    topic_keywords = []
    
    # 의도 관련 키워드 분리
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
                # 1차: 전체 키워드 + spread
                f"{keywords} tarot spread",
                # 2차: 의도 키워드 + 주제 키워드 + spread  
                f"{' '.join(intent_keywords)} {' '.join(topic_keywords)} spread",
                # 3차: 의도 키워드만 + spread (결정/선택 관련 스프레드)
                f"{' '.join(intent_keywords)} spread" if intent_keywords else None,
                # 4차: 주제 키워드만 + spread (직업/연애 등 주제별 스프레드)
                f"{' '.join(topic_keywords)} spread" if topic_keywords else None,
                # 5차: 전체 키워드만
                keywords
            ]
            
            # None 제거
            search_attempts = [query for query in search_attempts if query]
            
            # 전체 검색에서 공통으로 사용할 중복 제거 세트
            existing_names = set()
            
            for i, query in enumerate(search_attempts, 1):
                print(f"🔍 {i}차 검색: {query}")
                try:
                    results = rag_system.search_spreads(query, final_k=8)  # 더 많은 결과 요청
                    safe_results = convert_numpy_types(results)
                    print(f"🔍 {i}차 검색 결과: {len(safe_results)}개")
                    
                    if len(safe_results) > 0:
                        print(f"✅ {i}차 검색 성공")
                        
                        for doc, score in safe_results:
                            if len(recommended_spreads) >= 15:  # 전체 검색에서 최대 15개까지
                                break
                                
                            metadata = doc.metadata
                            spread_name = metadata.get('spread_name', f'스프레드 {len(recommended_spreads)+1}')
                            
                            # 중복 제거
                            if spread_name not in existing_names:
                                existing_names.add(spread_name)
                                spread_data = {
                                    "number": len(recommended_spreads) + 1,
                                    "spread_name": spread_name,
                                    "card_count": metadata.get('card_count', 3),
                                    "positions": metadata.get("positions", []),
                                    "description": metadata.get("description", ""),
                                    "search_layer": i,  # 어느 검색에서 나왔는지 추적
                                    "relevance_score": float(score) if hasattr(score, 'item') else score
                                }
                                recommended_spreads.append(spread_data)
                        
                        # 각 검색층에서 최소 1개씩은 확보하도록 계속 진행
                            
                except Exception as e:
                    print(f"🔍 {i}차 검색 실패: {e}")
                    continue
                    
        if len(recommended_spreads) < 3:
            raise Exception("모든 검색 시도 실패")
            
    except Exception as e:
        print(f"🔍 다층 검색 실패: {e}, 기본 스프레드 사용")
        # 기본 스프레드 반환
        default_spreads = get_default_spreads()
        
        for i, spread in enumerate(default_spreads[:3]):
            spread_data = {
                "number": i + 1,
                "spread_name": spread.get('spread_name', f'스프레드 {i+1}'),
                "card_count": spread.get('card_count', 3),
                "positions": spread.get("positions", []),
                "description": spread.get("description", ""),
                "search_layer": 0,  # 기본 스프레드 표시
                "relevance_score": 0.5
            }
            recommended_spreads.append(spread_data)
    
    print(f"🔍 최종 추천 스프레드: {len(recommended_spreads)}개")
    for spread in recommended_spreads:
        print(f"  - {spread['spread_name']} (검색층: {spread['search_layer']}, 점수: {spread['relevance_score']:.3f})")
    
    # 균형잡힌 스프레드 선택 로직
    intent_spreads = []  # 3차 검색(의도만)에서 나온 것들
    topic_spreads = []   # 4차 검색(주제만)에서 나온 것들  
    mixed_spreads = []   # 1-2차 검색(혼합)에서 나온 것들

    for spread in recommended_spreads:
        if spread['search_layer'] == 3:  # 의도 키워드만으로 검색
            intent_spreads.append(spread)
        elif spread['search_layer'] == 4:  # 주제 키워드만으로 검색
            topic_spreads.append(spread)
        else:
            mixed_spreads.append(spread)

    print(f"🎯 스프레드 분류 - 의도: {len(intent_spreads)}개, 주제: {len(topic_spreads)}개, 혼합: {len(mixed_spreads)}개")

    # 균형잡힌 3개 선택
    final_spreads = []

    # 1. 의도 스프레드 최소 1개 보장
    if intent_spreads:
        final_spreads.append(intent_spreads[0])
        print(f"✅ 의도 스프레드 포함: {intent_spreads[0]['spread_name']}")

    # 2. 주제 스프레드 1개 추가 (의도 스프레드와 중복 방지)
    available_topic = [s for s in topic_spreads if s not in final_spreads]
    if available_topic:
        final_spreads.append(available_topic[0])
        print(f"✅ 주제 스프레드 포함: {available_topic[0]['spread_name']}")

    # 3. 나머지 1개는 전체에서 점수 순으로 (이미 선택된 것 제외)
    remaining = [s for s in recommended_spreads if s not in final_spreads]
    if remaining:
        final_spreads.append(remaining[0])
        print(f"✅ 추가 스프레드 포함: {remaining[0]['spread_name']}")

    # 부족한 경우 전체에서 채우기
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

def spread_recommender_node(state: TarotState) -> TarotState:
    """스프레드 추천 전용 노드 - 개선된 다층적 검색"""
    # user_input을 올바르게 추출 - 메시지에서 가져오기
    user_input = state.get("user_input") or get_last_user_input(state)
    
    print("🔧 스프레드 추천 노드 실행")
    
    # 시간 맥락 설정 (기존 로직 보존)
    state = ensure_temporal_context(state)
    
    # 개선된 키워드 추출 및 다층적 검색
    print(f"🔍 고민별 스프레드 검색 시작: '{user_input}'")
    keywords = extract_concern_keywords(user_input)
    
    # 다층적 검색 수행
    recommended_spreads = perform_multilayer_spread_search(keywords, user_input)
    
    # 기존 로직 완전 보존 - LLM 추천 메시지 생성
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    # 이전 노드에서 받은 감정 정보
    emotional_analysis = state.get("emotional_analysis", {})
    emotional_greeting = state.get("emotional_greeting", "🔮 상황에 가장 적합한 스프레드들을 찾아드렸습니다.")
    
    emotion = emotional_analysis.get('primary_emotion', '알 수 없음')
    intensity = emotional_analysis.get('emotion_intensity', '보통')
    
    # 웹 검색 결과 컨텍스트 추가 (기존 로직 완전 보존)
    search_context = ""
    search_results = state.get("search_results")
    if search_results and search_results.get("success"):
        search_summary = ""
        results = search_results.get("results", [])
        if isinstance(results, list) and len(results) > 0:
            top_results = results[:2]
            search_summary = "\n".join([
                f"- {result.get('title', '제목 없음')}: {result.get('content', result.get('snippet', '내용 없음'))[:150]}"
                for result in top_results
                if isinstance(result, dict)
            ])
        
        if search_summary:
            search_context = f"\n\n**최신 정보 (웹 검색 결과):**\n{search_summary}\n\n위 최신 정보를 참고하여 더 현실적이고 구체적인 조언을 제공해주세요."

    # 스프레드 상세 정보 구성 (기존 로직 완전 보존)
    detailed_spreads_info = ""
    for i, spread in enumerate(recommended_spreads, 1):
        detailed_spreads_info += f"\n=== 스프레드 {i}: {spread['spread_name']} ===\n"
        detailed_spreads_info += f"카드 수: {spread['card_count']}장\n"
        detailed_spreads_info += f"설명: {spread.get('description', '설명 없음')}\n"
        
        positions = spread.get('positions', [])
        if positions:
            detailed_spreads_info += "포지션들:\n"
            for pos in positions[:5]:
                if isinstance(pos, dict):
                    pos_name = pos.get('position_name', '알 수 없음')
                    pos_meaning = pos.get('position_meaning', '설명 없음')
                    detailed_spreads_info += f"  - {pos_name}: {pos_meaning}\n"
        detailed_spreads_info += "\n"

    # 스프레드 이름 번역 (프롬프트 생성 전에 미리 실행)
    spread1_name_kr = translate_text_with_llm(recommended_spreads[0]['spread_name'], "spread_name") if len(recommended_spreads) > 0 else '첫 번째 스프레드'
    spread2_name_kr = translate_text_with_llm(recommended_spreads[1]['spread_name'], "spread_name") if len(recommended_spreads) > 1 else '두 번째 스프레드'
    spread3_name_kr = translate_text_with_llm(recommended_spreads[2]['spread_name'], "spread_name") if len(recommended_spreads) > 2 else '세 번째 스프레드'
    
    # 기존 프롬프트 완전 보존
    recommendation_prompt = f"""
    사용자의 고민: "{user_input}"
    사용자 감정 상태: {emotion} (강도: {intensity})
    추출된 키워드: {keywords}{search_context}

    다음은 사용자의 고민에 가장 적합하다고 판단되어 검색된 스프레드들입니다:
    {detailed_spreads_info}

    위 스프레드들의 실제 설명과 포지션 정보를 바탕으로, 사용자의 고민 "{user_input}"에 각 스프레드가 어떻게 도움이 될 수 있는지 구체적으로 설명해주세요.

    다음 형식으로 정확히 추천해주세요:

    {emotional_greeting}

    **1) {spread1_name_kr} ({recommended_spreads[0]['card_count']}장)**
    - 목적: [실제 스프레드 설명을 바탕으로 사용자 고민과의 연관성을 구체적으로 설명]
    - 효과: [감정 상태를 고려한 따뜻한 효과 설명]

    **2) {spread2_name_kr} ({recommended_spreads[1]['card_count'] if len(recommended_spreads) > 1 else 5}장)**  
    - 목적: [실제 스프레드 설명을 바탕으로 사용자 고민과의 연관성을 구체적으로 설명]
    - 효과: [감정 상태를 고려한 따뜻한 효과 설명]

    **3) {spread3_name_kr} ({recommended_spreads[2]['card_count'] if len(recommended_spreads) > 2 else 7}장)**
    - 목적: [실제 스프레드 설명을 바탕으로 사용자 고민과의 연관성을 구체적으로 설명] 
    - 효과: [감정 상태를 고려한 따뜻한 효과 설명]

    💫 **어떤 스프레드가 마음에 드시나요? 번호로 답해주세요 (1, 2, 3).**
    
    중요: 각 스프레드의 실제 설명과 포지션 정보를 반드시 활용하여, 사용자의 구체적인 고민 "{user_input}"에 어떻게 도움이 될 수 있는지 명확하고 설득력 있게 설명해주세요.
    감정적으로 따뜻하고 희망적인 톤으로 작성해주세요.
    """
    
    try:
        response = llm.invoke([HumanMessage(content=recommendation_prompt)])
        empathy_message = state.get("empathy_message", "")
        final_message = f"{empathy_message}\n\n{response.content}"
        
        # 상태 업데이트 (기존 로직 완전 보존)
        updated_state = {
            "messages": [AIMessage(content=final_message)],
            "consultation_data": {
                "concern": user_input,
                "emotional_analysis": emotional_analysis,
                "recommended_spreads": recommended_spreads,
                "status": "spread_selection"
            },
            "consultation_status": "spreads_recommended"
        }
        
        # 검색 결과가 있으면 포함 (기존 로직 보존)
        if search_results:
            updated_state["search_results"] = search_results
            updated_state["search_decision"] = state.get("search_decision")
        
        return updated_state
        
    except Exception as e:
        # 기존 에러 처리 로직 완전 보존
        empathy_message = state.get("empathy_message", "")
        fallback_message = f"{empathy_message}\n\n{emotional_greeting}\n\n스프레드 추천 중 일시적인 문제가 발생했습니다.\n하지만 걱정하지 마세요. 기본 스프레드로도 충분히 좋은 상담을 받으실 수 있습니다.\n\n어떤 스프레드를 선택하시겠어요? (1, 2, 3)"
        
        return {
            "messages": [AIMessage(content=fallback_message)],
            "consultation_data": {
                "concern": user_input,
                "emotional_analysis": emotional_analysis,
                "recommended_spreads": recommended_spreads,
                "status": "spread_selection"
            },
            "consultation_status": "spreads_recommended"
        }

def consultation_router(state: TarotState) -> str:
    """상담 플로우의 조건부 라우팅"""
    consultation_status = state.get("consultation_status", "start")
    
    print(f"🔧 상담 라우터: 현재 상태 = {consultation_status}")
    
    if consultation_status == "emotion_analyzed":
        return "web_search_decider_node"
    elif consultation_status == "search_decided":
        return "web_searcher_node"
    elif consultation_status == "search_completed":
        return "spread_recommender_node"
    elif consultation_status == "spreads_recommended":
        return "END"  # 스프레드 추천 완료
    else:
                 return "emotion_analyzer_node"  # 시작점

# =================================================================
# Phase 2 리팩토링: start_specific_spread_consultation 분해
# =================================================================

def spread_extractor_node(state: TarotState) -> TarotState:
    """스프레드 추출 전용 노드 - LLM 1개만 사용"""
    user_input = state.get("user_input", "")
    
    print("🔧 스프레드 추출 노드 실행")
    
    # 1순위: Supervisor 결정 확인 (기존 로직 보존)
    supervisor_decision = state.get("supervisor_decision", {})
    specified_spread = supervisor_decision.get("specific_spread", "")
    
    # 2순위: LLM이 사용자 입력에서 스프레드 추출 (기존 로직 보존)
    if not specified_spread:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        
        extract_prompt = f"""
        사용자 입력에서 타로 스프레드를 추출해주세요: "{user_input}"
        
        사용자가 특정 스프레드를 언급했다면 그 이름을 답해주세요.
        언급하지 않았다면 "None"이라고 답해주세요.
        
        스프레드명만 답해주세요 (예: "One Card", "Celtic Cross", "Three Card", "None")
        """
        
        try:
            response = llm.invoke([HumanMessage(content=extract_prompt)])
            extracted_spread = response.content.strip()
            if extracted_spread != "None":
                specified_spread = extracted_spread
        except:
            pass
    
    # 3순위: 세션 메모리 (기존 로직 보존)
    if not specified_spread:
        session_memory = state.get("session_memory", {})
        explained_spreads = session_memory.get("explained_spreads", [])
        if explained_spreads:
            specified_spread = explained_spreads[-1]
    
    # 최종: LLM이 기본값도 결정 (기존 로직 보존)
    if not specified_spread:
        llm_default = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        
        default_prompt = f"""
        사용자가 타로 상담을 요청했지만 특정 스프레드를 지정하지 않았습니다.
        
        어떤 스프레드가 가장 적절할지 추천해주세요.
        일반적으로 초보자나 간단한 상담에는 어떤 스프레드가 좋은가요?
        
        스프레드명만 답해주세요 (예: "Three Card", "One Card", "Celtic Cross")
        """
        
        try:
            response = llm_default.invoke([HumanMessage(content=default_prompt)])
            specified_spread = response.content.strip()
        except:
            specified_spread = "Three Card"  # 진짜 최후의 수단
    
    print(f"🔧 추출된 스프레드: {specified_spread}")
    
    return {
        "extracted_spread": specified_spread,
        "specific_consultation_status": "spread_extracted"
    }

def situation_analyzer_node(state: TarotState) -> TarotState:
    """상황 분석 전용 노드 - LLM 1개만 사용"""
    user_input = state.get("user_input", "")
    extracted_spread = state.get("extracted_spread", "Three Card")
    
    print("🔧 상황 분석 노드 실행")
    
    # 기존 로직 완전 보존 - LLM이 상황을 판단하고 적절한 응답 생성
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    prompt = f"""
    사용자가 "{extracted_spread}" 스프레드로 상담을 요청했습니다.
    사용자 입력: "{user_input}"
    
    **한국어 자연어 이해:**
    한국어는 맥락 의존적이고 생략이 많습니다. 
    사용자의 진짜 의도를 파악하세요:
    - 짧은 표현도 명확한 의미를 담을 수 있음
    - 상담 의지와 구체적 주제 유무를 구분해서 판단
    - 한국인이 자연스럽게 사용하는 표현 방식 고려
    
    상황을 분석해서 적절히 응답해주세요:
    
    **만약 사용자가 이미 구체적인 고민이나 질문을 했다면:**
    - 바로 카드 선택 단계로 안내
    - "좋습니다! {user_input}에 대해 {extracted_spread} 스프레드로 봐드리겠습니다"
    - 카드 번호 선택 방법 안내 (1부터 78까지 X장 선택)
    
    **만약 상담은 원하지만 구체적 고민이 없다면:**
    - 스프레드 소개 후 구체적인 고민 질문
    - "어떤 고민에 대해 알아보고 싶으신지 구체적으로 말씀해주세요"
    
    **한국어 맥락과 사용자 의도를 자연스럽게 추론**해서 판단해주세요.
    타로 상담사 톤으로 따뜻하고 친근하게 작성해주세요.
    """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        
        return {
            "situation_analysis_response": response.content,
            "specific_consultation_status": "situation_analyzed"
        }
        
    except Exception as e:
        print(f"🔧 상황 분석 오류: {e}")
        return {
            "situation_analysis_response": f"🔮 {extracted_spread} 상담을 준비하는 중입니다. 어떤 고민을 봐드릴까요?",
            "specific_consultation_status": "situation_analyzed"
        }

def card_count_inferrer_node(state: TarotState) -> TarotState:
    """카드 수 추론 전용 노드 - LLM 1개만 사용"""
    extracted_spread = state.get("extracted_spread", "Three Card")
    
    print("🔧 카드 수 추론 노드 실행")
    
    # 기존 로직 완전 보존 - LLM이 카드 수도 추론하게 하기
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    card_count_prompt = f"""
    "{extracted_spread}" 스프레드는 몇 장의 카드를 사용하나요? 
    숫자만 답해주세요 (예: 3, 5, 10)
    """
    
    try:
        card_count_response = llm.invoke([HumanMessage(content=card_count_prompt)])
        card_count = int(card_count_response.content.strip())
    except:
        card_count = 3  # 기본값
    
    print(f"🔧 추론된 카드 수: {card_count}")
    
    return {
        "inferred_card_count": card_count,
        "specific_consultation_status": "card_count_inferred"
    }

def status_determiner_node(state: TarotState) -> TarotState:
    """상태 결정 전용 노드 - LLM 1개만 사용"""
    user_input = state.get("user_input", "")
    extracted_spread = state.get("extracted_spread", "Three Card")
    inferred_card_count = state.get("inferred_card_count", 3)
    situation_analysis_response = state.get("situation_analysis_response", "")
    
    print("🔧 상태 결정 노드 실행")
    
    # 기존 로직 완전 보존 - 상태 판단도 LLM에게 위임 (한국어 자연어 이해)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    status_prompt = f"""
    사용자 입력 "{user_input}"을 한국어 자연어로 이해해서 판단해주세요.
    
    **판단 기준:**
    한국어는 맥락 의존적이고 생략이 많습니다.
    사용자의 진짜 의도를 파악하세요:
    - 상담 의지가 있는가?
    - 구체적인 주제나 고민이 포함되어 있는가?
    - 한국인이 자연스럽게 사용하는 표현인가?
    
    다음 중 하나로 답해주세요:
    - "card_selection": 이미 구체적인 고민/질문이 있어서 바로 카드 선택 단계
    - "collecting_concern": 상담은 원하지만 아직 구체적인 고민을 물어봐야 함
    
    단어 하나만 답해주세요.
    """
    
    try:
        status_response = llm.invoke([HumanMessage(content=status_prompt)])
        status = status_response.content.strip()
        
        if status not in ["card_selection", "collecting_concern"]:
            status = "collecting_concern"  # 기본값
        
        print(f"🔧 LLM이 판단한 상태: {status}")
        
        # 기존 로직 완전 보존 - consultation_data 구성
        consultation_data = {
            "status": status,
            "selected_spread": {
                "spread_name": extracted_spread,
                "card_count": inferred_card_count,
                "description": f"{extracted_spread} 스프레드"
            }
        }
        
        # card_selection 상태면 concern도 저장 (기존 로직 보존)
        if status == "card_selection":
            consultation_data["concern"] = user_input
        
        return {
            "messages": [AIMessage(content=situation_analysis_response)],
            "consultation_data": consultation_data,
            "specific_consultation_status": "status_determined"
        }
        
    except Exception as e:
        print(f"🔧 상태 결정 오류: {e}")
        # 기본 에러 처리 (기존 로직 보존)
        return {
            "messages": [AIMessage(content=f"🔮 {extracted_spread} 상담을 준비하는 중입니다. 어떤 고민을 봐드릴까요?")],
            "consultation_data": {
                "status": "collecting_concern",
                "selected_spread": {
                    "spread_name": extracted_spread,
                    "card_count": inferred_card_count,
                    "description": f"{extracted_spread} 스프레드"
                }
            },
            "specific_consultation_status": "status_determined"
        }

def specific_consultation_router(state: TarotState) -> str:
    """특정 스프레드 상담 플로우의 조건부 라우팅 - 🔧 스프레드 선택 상태 우선 체크"""
    
    # 🔧 1순위: consultation_data에서 스프레드 선택 상태 체크
    consultation_data = state.get("consultation_data", {})
    consultation_status = consultation_data.get("status", "")
    
    if consultation_status == "spread_selection":
        # 스프레드 선택 상태에서는 consultation_continue_handler로 이동
        print("🔧 스프레드 선택 상태 감지 → consultation_continue_handler로 라우팅")
        return "consultation_continue_handler"
    
    # 🔧 2순위: 기존 specific_consultation_status 체크
    specific_status = state.get("specific_consultation_status", "start")
    
    print(f"🔧 특정 상담 라우터: 현재 상태 = {specific_status}")
    
    if specific_status == "spread_extracted":
        return "situation_analyzer_node"
    elif specific_status == "situation_analyzed":
        return "card_count_inferrer_node"
    elif specific_status == "card_count_inferred":
        return "status_determiner_node"
    elif specific_status == "status_determined":
        return "END"  # 특정 스프레드 상담 완료
    else:
        return "spread_extractor_node"  # 시작점

# =================================================================
# Phase 3: 성능 최적화 및 추가 기능
# =================================================================

import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from functools import wraps

def performance_monitor(func):
    """성능 모니터링 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"⏱️ {func.__name__} 실행 시간: {execution_time:.3f}초")
        
        # 성능 경고 (2초 이상)
        if execution_time > 2.0:
            print(f"⚠️ {func.__name__} 성능 주의: {execution_time:.3f}초")
        
        return result
    return wrapper

def create_optimized_consultation_flow():
    """최적화된 상담 플로우 생성"""
    
    @performance_monitor
    def parallel_emotion_and_search_analysis(state: TarotState) -> TarotState:
        """감정 분석과 웹 검색 판단을 병렬로 실행"""
        user_input = state.get("user_input", "")
        
        print("🔧 병렬 분석 노드 실행 (감정 + 웹검색)")
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 병렬 실행
            emotion_future = executor.submit(emotion_analyzer_node, state)
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

def general_handler(state: TarotState) -> TarotState:
   """일반 질문 핸들러 - 날짜 질문 특별 처리 및 웹 검색 통합"""
   user_input = state["user_input"]
   
   # 🔧 LLM 기반 날짜 질문 감지 (하드코딩 제거)
   def is_date_question(text: str) -> bool:
       """LLM으로 날짜/시간 관련 질문인지 판단"""
       try:
           llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
           prompt = f"""
           다음 사용자 입력이 **순수하게 현재 날짜나 시간 정보 자체**를 알고 싶어하는 질문인지 판단해주세요:
           "{text}"
           
           **True (순수 날짜/시간 질문)**: 
           - 오늘 몇일?, 지금 몇시?, 오늘 날짜 알려줘, 현재 시간?, 며칠이야?, 몇시야?
           - 단순히 날짜나 시간 정보만 원하는 경우
           
           **False (다른 목적이 있는 질문)**:
           - 오늘부터 투자해도 돼?, 지금 당장 들어가도 돼?, 현재 상황에서 투자할까?
           - 언제 투자해야해?, 언제 만나?, 언제가 좋을까?, 타이밍이 언제?, 시기가 언제?
           - "오늘/지금/현재" 등이 포함되어도 투자/조언/결정을 묻는 질문
           
           **핵심 판단 기준**: 사용자가 **날짜/시간 정보 자체**를 원하는가? 아니면 **그 시점과 관련된 조언/결정**을 원하는가?
           
           JSON 형식으로 답변:
           {{"is_date_question": true/false, "reasoning": "판단 근거"}}
           """
           response = llm.invoke([HumanMessage(content=prompt)])
           import json
           result = json.loads(response.content.strip())
           return result.get("is_date_question", False)
       except Exception as e:
           # 🔧 LLM 실패시 False 반환 (하드코딩 완전 제거)
           print(f"⚠️ 날짜 질문 판단 실패: {e}")
           return False  # 확실하지 않으면 일반 질문으로 처리
   
   if is_date_question(user_input):
       # 시간 맥락 설정
       state = ensure_temporal_context(state)
       current_context = state.get("temporal_context", {})
       
       current_date = current_context.get("current_date", "날짜 정보를 가져올 수 없습니다")
       weekday = current_context.get("weekday_kr", "")
       season = current_context.get("season", "")
       
       date_response = f"""🔮

오늘은 **{current_date} {weekday}**입니다. 
현재 {season}철이네요! ✨

매일매일이 새로운 가능성으로 가득 차 있으니, 오늘도 좋은 하루 되시길 바랍니다!

타로 상담을 받고 싶으시거나 다른 궁금한 점이 있으시면 언제든 말씀해주세요! 🔮"""
       
       return {"messages": [AIMessage(content=date_response)]}
   
   # 웹 검색 필요성 판단
   conversation_context = ""
   messages = state.get("messages", [])
   if len(messages) >= 2:
       last_ai = None
       for msg in reversed(messages):
           if isinstance(msg, AIMessage):
               last_ai = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
               break
       if last_ai:
           conversation_context = f"직전 대화: {last_ai}"
   
   search_decision = decide_web_search_need_with_llm(user_input, conversation_context)
   
   # 웹 검색 실행 (필요한 경우)
   search_results = None
   if search_decision.get("need_search", False) and search_decision.get("confidence", 0) > 0.5:
       search_query = search_decision.get("search_query", user_input)
       search_type = search_decision.get("search_type", "general")
       
       print(f"🔍 웹 검색 실행: {search_query} (타입: {search_type})")
       search_results = perform_web_search(search_query, search_type)
   
   # 일반 질문 처리
   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
   
   # 검색 결과가 있으면 프롬프트에 포함
   search_context = ""
   if search_results and search_results.get("success"):
       search_summary = ""
       results = search_results.get("results", [])
       if isinstance(results, list) and len(results) > 0:
           top_results = results[:2]
           search_summary = "\n".join([
               f"- {result.get('title', '제목 없음')}: {result.get('content', result.get('snippet', '내용 없음'))[:150]}"
               for result in top_results
               if isinstance(result, dict)
           ])
       
       if search_summary:
           search_context = f"\n\n**참고 정보 (웹 검색 결과):**\n{search_summary}\n\n위 정보를 참고하여 더 현실적이고 구체적인 조언을 제공해주세요."
   
   # 🆕 일상 대화 감지 및 자연스러운 응답
   casual_keywords = ["먹", "날씨", "안녕", "뭐해", "어때", "좋아", "싫어", "피곤", "행복"]
   is_casual_chat = any(keyword in user_input.lower() for keyword in casual_keywords)
   
   if is_casual_chat:
       prompt = f"""
       사용자가 일상적인 대화를 시작했습니다: "{user_input}"
       
       타로 상담사로서 친근하고 자연스럽게 응답해주세요. 
       타로적 관점을 살짝 섞되, 과하지 않게 일상 대화처럼 답변하세요.{search_context}
       
       마지막에 "카드 한 장 뽑아서 알아보길 원하시면 '네'라고 답해주세요."라고 명확하게 제안해주세요.
       
       😊 친근하고 따뜻한 톤으로 답변하세요.
       """
   else:
       prompt = f"""
       사용자가 타로나 점술에 대한 일반적인 질문을 했습니다: "{user_input}"
       
       타로 상담사로서 친근하고 도움이 되는 답변을 해주세요.{search_context}
       
       마지막에 "카드 한 장 뽑아서 알아보길 원하시면 '네'라고 답해주세요. 본격적인 타로 상담을 원하시면 '타로 봐줘'라고 말씀해주세요!"라고 덧붙여주세요.
       
       🔮 따뜻하고 전문적인 톤으로 답변하세요.
       """
   
   try:
       response = llm.invoke([HumanMessage(content=prompt)])
       
       # 검색 결과 표시 추가 (있는 경우)
       final_response = response.content
       if search_results and search_results.get("success"):
           search_display = format_search_results_for_display(search_results)
           if search_display:
               final_response += search_display
       
       # 상태에 검색 정보 저장
       updated_state = {"messages": [AIMessage(content=final_response)]}
       if search_results:
           updated_state["search_results"] = search_results
           updated_state["search_decision"] = search_decision
       
       return updated_state
       
   except Exception as e:
       fallback_msg = "🔮 질문에 답변드리는 중 문제가 생겼어요. 다시 질문해주시면 더 정확히 답변드릴게요!\n\n카드 한 장 뽑아서 알아보길 원하시면 '네'라고 답해주세요. 다른 궁금한 점이 있으시면 언제든 말씀해주세요!"
       return {"messages": [AIMessage(content=fallback_msg)]}

def unknown_handler(state: TarotState) -> TarotState:
   """알 수 없는 입력 핸들러"""
   return {
       "messages": [AIMessage(content="""🔮

어떤 도움이 필요하신가요? 
- 타로 카드 의미가 궁금하시거나
- 스프레드 정보를 알고 싶으시거나
- 타로 상담을 받고 싶으시면 언제든 말씀해주세요!

편안하게 대화해요 😊""")]
   }

def consultation_flow_handler(state: TarotState) -> TarotState:
   """상담 진행 중 처리 - 안전성 강화"""
   
   # 안전성 체크
   if not state:
       print("❌ state가 None입니다")
       return {"messages": [AIMessage(content="🔮 상담을 시작하겠습니다. 어떤 고민이 있으신가요?")]}
   
   consultation_data = state.get("consultation_data", {})
   if not consultation_data:
       consultation_data = {}
       
   status = consultation_data.get("status", "")
   user_input = state.get("user_input", "")
   
   print(f"🔧 상담 흐름 처리: status={status}, user_input='{user_input}'")
   
   if status == "waiting_for_concern":
       # 🆕 고민을 받은 후 실제 상담 시작
       print("🔧 고민 접수 - 실제 상담 시작")
       # 원래 consultation_handler의 로직 실행 (고민 체크 제외)
       return start_actual_consultation(state)
   
   elif status == "spread_selection":
       if any(num in user_input for num in ["1", "2", "3"]):
           return consultation_continue_handler(state)
       else:
           return {"messages": [AIMessage(content="1, 2, 3 중에서 선택해주세요.")]}
   
   elif status == "card_selection":
       if any(char.isdigit() or char == ',' for char in user_input):
           return consultation_summary_handler(state)
       else:
           return {"messages": [AIMessage(content="카드 번호를 입력해주세요. (예: 7, 23, 45)")]}
   
   elif status == "summary_shown":
       user_input_lower = user_input.lower()
       if any(keyword in user_input_lower for keyword in ["네", "yes", "보고싶", "보고 싶", "개별", "자세히", "더"]):
           return consultation_individual_handler(state)
       elif any(keyword in user_input_lower for keyword in ["아니", "no", "괜찮", "됐어", "안볼"]):
           return {"messages": [AIMessage(content="🔮 상담이 도움이 되었기를 바랍니다! 다른 고민이 있으시면 언제든 말씀해주세요. ✨")]}
       else:
           return {"messages": [AIMessage(content="개별 해석을 보고 싶으시면 '네' 또는 '보고싶어'라고 말씀해주세요!")]}
   
   elif status == "completed":
       # 🆕 개별 해석 완료 후 처리 - triggerplan.md 핵심 개선사항
       print(f"🔧 상담 완료 후 처리: user_input='{user_input}'")
       
       trigger_result = simple_trigger_check(user_input)
       print(f"🎯 트리거 결과: {trigger_result}")
       
       if trigger_result == "new_consultation":
           print("🔧 새 상담 시작 트리거 감지")
           return consultation_handler(state)
       elif trigger_result == "individual_reading":
           # 이미 완료된 상태에서 개별 해석 재요청 - 안내 메시지
           return {"messages": [AIMessage(content="이미 개별 해석을 모두 보여드렸습니다. 새로운 고민이 있으시면 '타로 봐줘'라고 말씀해주세요!")]}
       else:
           # context_reference - 추가 질문으로 처리
           print("🔧 추가 질문으로 처리")
           return context_reference_handler(state)
   
   else:
       # 새로운 상담 시작
       print("🔧 새로운 상담 시작")
       return consultation_handler(state)

def start_actual_consultation(state: TarotState) -> TarotState:
   """고민을 받은 후 실제 상담 진행"""
   user_input = state.get("user_input", "")
   
   # Phase 1 리팩토링: 4개 노드를 순차 실행하여 동일한 결과 제공
   try:
       # 1. 감정 분석
       result1 = emotion_analyzer_node(state)
       state.update(result1)
       
       # 2. 웹 검색 판단
       result2 = web_search_decider_node(state)
       state.update(result2)
       
       # 3. 웹 검색 실행
       result3 = web_searcher_node(state)
       state.update(result3)
       
       # 4. 스프레드 추천
       result4 = spread_recommender_node(state)
       state.update(result4)
       
       print("✅ 실제 상담 성공적으로 완료")
       return state
       
   except Exception as e:
       print(f"❌ 실제 상담 오류: {e}")
       # 기본 에러 처리
       return {
           "messages": [AIMessage(content="🔮 상담 처리 중 문제가 발생했습니다. 다시 시도해주세요.")],
           "consultation_data": {
               "status": "error"
           }
       }

def consultation_continue_handler(state: TarotState) -> TarotState:
   """상담 계속 진행 핸들러 - 스프레드 선택 후"""
   
   consultation_data = state.get("consultation_data", {})
   if not consultation_data or consultation_data.get("status") != "spread_selection":
       return {"messages": [AIMessage(content="상담 정보가 없습니다. 새로운 고민을 말씀해주세요.")]}
   
   # 마지막 사용자 메시지에서 스프레드 선택 추출
   user_input = ""
   for msg in reversed(state["messages"]):
       if isinstance(msg, HumanMessage):
           user_input = msg.content.strip()
           break
   
   # 🔧 사용자 맞춤 스프레드 요청 감지
   custom_request_keywords = ["원하는", "다른", "새로운", "특별한", "맞춤", "추천", "더", "별도"]
   has_custom_request = any(keyword in user_input for keyword in custom_request_keywords)
   
   if has_custom_request:
       print(f"🔧 사용자 맞춤 스프레드 요청 감지: '{user_input}'")
       
       # 기존 고민 정보 유지하면서 새 스프레드 검색
       user_concern = consultation_data.get("concern", "")
       keywords = extract_concern_keywords(user_concern + " " + user_input)
       
       # 새로운 스프레드 검색 수행
       new_spreads = perform_multilayer_spread_search(keywords, user_input)
       
       if new_spreads:
           # 상담 데이터 업데이트 (새 스프레드로)
           updated_consultation_data = consultation_data.copy()
           updated_consultation_data.update({
               "recommended_spreads": new_spreads,
               "status": "spread_selection"
           })
           
           # 새 스프레드 옵션 제시
           spread_msg = "🔮 **맞춤 스프레드를 새로 찾았습니다!**\n\n"
           
           for idx, spread in enumerate(new_spreads[:3], 1):
               spread_name_kr = translate_text_with_llm(spread['spread_name'], "spread_name")
               spread_msg += f"**{idx}. {spread_name_kr}** ({spread['card_count']}장)\n"
               spread_msg += f"   📝 {spread['description']}\n\n"
           
           spread_msg += "어떤 스프레드로 진행하시겠어요? (1, 2, 3 중 선택)"
           
           return {
               "messages": [AIMessage(content=spread_msg)],
               "consultation_data": updated_consultation_data
           }
       else:
           # 검색 실패시 기존 스프레드 유지
           return {"messages": [AIMessage(content="죄송합니다. 추가 스프레드를 찾지 못했어요. 기존 옵션 중에서 선택해주세요. (1, 2, 3)")]}
   
   # 기존 숫자 선택 로직
   selected_number = None
   if "1" in user_input:
       selected_number = 1
   elif "2" in user_input:
       selected_number = 2
   elif "3" in user_input:
       selected_number = 3
   
   if selected_number is None:
       return {"messages": [AIMessage(content="1, 2, 3 중에서 선택해주세요.")]}
   
   # 선택된 스프레드 정보 (인덱스 기반으로 수정)
   recommended_spreads = consultation_data.get("recommended_spreads", [])
   
   if not recommended_spreads or selected_number < 1 or selected_number > len(recommended_spreads):
       return {"messages": [AIMessage(content="선택한 스프레드 정보를 찾을 수 없습니다.")]}
   
   # 인덱스 기반으로 선택 (사용자 선택 1,2,3 → 배열 인덱스 0,1,2)
   selected_spread = recommended_spreads[selected_number - 1]
   
   # 카드 선택 안내 메시지
   emotional_analysis = consultation_data.get("emotional_analysis", {})
   emotion = emotional_analysis.get('primary_emotion', '알 수 없음')
   
   # 감정별 카드 선택 안내
   if emotion == "불안":
       emotional_guidance = "🌟 마음을 진정시키고, 직감을 믿어보세요. 처음 떠오르는 숫자들이 당신에게 필요한 메시지를 담고 있을 거예요."
   elif emotion == "슬픔":
       emotional_guidance = "💙 힘든 마음이지만, 카드가 위로와 희망의 메시지를 전해줄 거예요. 마음이 이끄는 대로 숫자를 선택해보세요."
   elif emotion == "걱정":
       emotional_guidance = "🌟 걱정이 많으시겠지만, 카드가 안심할 수 있는 답변을 제시해줄 거예요. 직감적으로 떠오르는 숫자들을 선택해보세요."
   else:
       emotional_guidance = "✨ 직감을 믿고 마음이 이끄는 대로 숫자들을 선택해보세요. 카드가 당신에게 필요한 메시지를 전해줄 거예요."
   
   card_count = selected_spread.get("card_count", 3)
   card_selection_msg = f"""
✅ **{selected_spread['spread_name']}**를 선택하셨습니다!

{emotional_guidance}

🃏 **카드 선택 방법:**
타로 카드는 총 78장이 있습니다. 
1부터 78 사이의 숫자를 **{card_count}장** 선택해주세요.

**예시:** 7, 23, 45, 12, 56

💫 **팁:** 숫자를 고민하지 마시고, 직감적으로 떠오르는 숫자들을 말씀해주세요. 
당신의 무의식이 이미 답을 알고 있을 거예요.
"""
   
   # 상담 데이터 업데이트
   updated_consultation_data = consultation_data.copy()
   updated_consultation_data.update({
       "selected_spread": selected_spread,
       "status": "card_selection"
   })
   
   return {
       "messages": [AIMessage(content=card_selection_msg)],
       "consultation_data": updated_consultation_data
   }

def consultation_summary_handler(state: TarotState) -> TarotState:
   """카드 선택 후 개별 해석 먼저 생성 → 고급 분석 통합 → 종합 분석 + 명확한 답변"""
   
   consultation_data = state.get("consultation_data", {})
   if not consultation_data or consultation_data.get("status") != "card_selection":
       return {"messages": [AIMessage(content="카드 선택 정보가 없습니다.")]}
   
   # 사용자가 입력한 카드 번호들 파싱
   user_input = ""
   for msg in reversed(state["messages"]):
       if isinstance(msg, HumanMessage):
           user_input = msg.content.strip()
           break
   
   selected_spread = consultation_data.get("selected_spread", {})
   card_count = selected_spread.get("card_count", 3)
   
   # 카드 번호 파싱 및 검증
   user_numbers = parse_card_numbers(user_input, card_count)
   
   if user_numbers is None:
       error_msg = f"""
❌ **입력 오류**

다음 중 하나의 문제가 있습니다:
- 같은 숫자를 두 번 입력했습니다
- {card_count}개의 숫자가 필요합니다
- 1-78 범위의 숫자만 입력 가능합니다

다시 입력해주세요. **{card_count}개의 서로 다른 숫자**를 선택해주세요.
**예시:** 7, 23, 45, 12, 56, 33, 71
"""
       return {"messages": [AIMessage(content=error_msg)]}
   
   # 카드 선택
   selected_cards = select_cards_randomly_but_keep_positions(user_numbers, card_count)
   selected_cards = convert_numpy_types(selected_cards)
   
   # 🆕 고급 분석 실행
   integrated_analysis = generate_integrated_analysis(selected_cards)
   
   # 1단계: 카드 표시 + 고급 분석 요약
   cards_display = f"""🃏 **아래처럼 카드를 뽑으셨네요**

"""
   for card in selected_cards:
       # 한국어 카드명과 방향 사용
       card_name_kr = card.get('name_kr', card['name'])
       orientation_symbol = card.get('orientation_symbol', "⬆️" if card["orientation"] == "upright" else "⬇️")
       orientation_kr = card.get('orientation_kr', card['orientation'])
       cards_display += f"**{card['position']}번째 카드:** {card_name_kr} {orientation_symbol} ({orientation_kr})\n"
   
   # 🆕 고급 분석 요약 추가
   success_prob = integrated_analysis["success_analysis"]["success_probability"]
   integrated_score = integrated_analysis["integrated_score"]
   
   cards_display += f"""
📊 **과학적 분석 결과**
- 성공 확률: {success_prob:.1%}
- 종합 점수: {integrated_score:.1%}
- {integrated_analysis["interpretation"]}

이제 뽑은 카드로 고민 해결 해드릴게요! ✨"""
   
   # 포지션 정보 추출
   positions = selected_spread.get("positions", [])
   positions_meanings = {}
   for pos in positions:
       if isinstance(pos, dict) and "position_num" in pos:
           positions_meanings[str(pos["position_num"])] = {
               "position": pos.get("position_name", f"Position {pos['position_num']}"),
               "meaning": pos.get("position_meaning", "")
           }
   
   # 기본 포지션이 없으면 생성
   if not positions_meanings:
       for i in range(1, card_count + 1):
           positions_meanings[str(i)] = {
               "position": f"Card {i}",
               "meaning": f"Position {i} in the spread"
           }
   
   # 개별 해석 생성
   llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
   user_concern = consultation_data.get("concern", "")
   spread_name = selected_spread.get("spread_name", "")
   spread_name_kr = translate_text_with_llm(spread_name, "spread_name")  # 스프레드 이름 번역
   
   interpretations = []
   timing_info = []
   
   # 웹 검색 결과 가져오기 (있는 경우)
   search_results = state.get("search_results", {})
   search_integration = ""
   
   # 검색 결과가 있으면 통합 해석 생성
   if search_results and search_results.get("success") and selected_cards:
       search_integration = integrate_search_results_with_tarot(selected_cards, search_results, user_concern)
       print(f"🌐 웹 검색 결과 통합: {len(search_integration)}자")
       
       # 검색 결과 표시도 추가
       search_display = format_search_results_for_display(search_results)
       if search_display:
           cards_display += f"\n\n{search_display}"
   
   for card in selected_cards:
       position_index = card.get("position", "")
       card_name = card.get("name", "")
       orientation = card.get("orientation", "")
       position_info = positions_meanings.get(str(position_index), {})
       position_name = position_info.get("position", f"Card {position_index}")
       position_name_kr = translate_text_with_llm(position_name, "position_name")  # 포지션 이름 번역
       position_meaning = position_info.get("meaning", "")
       
       # RAG 검색
       card_info = {}
       if rag_system:
           try:
               card_info = rag_system.search_card_meaning(card_name, orientation)
               card_info = convert_numpy_types(card_info)
           except Exception as e:
               card_info = {"success": False, "message": str(e)}
       
       # 한국어 카드명과 방향 얻기
       translated_info = translate_card_info(card_name, orientation)
       card_name_kr = translated_info['name']
       orientation_kr = translated_info['direction']
       
       # 카드 해석 프롬프트
       interpretation_prompt = f"""
당신은 정확하고 솔직한 타로 상담사입니다. 현실을 직시하게 하되 건설적인 방향을 제시하세요.

[사용자 상황]
- 고민: "{user_concern}"
- 선택한 스프레드: {spread_name_kr} (영문: {spread_name})

[카드별 해석]
- 카드명: {card_name_kr} (영문: {card_name})
- 방향: {orientation_kr}

[포지션 정보]
- 위치: {position_index}번째 카드 ({position_name_kr})
- 위치 의미: {position_meaning}

**핵심 원칙:**
1. 카드가 보여주는 현실을 있는 그대로 전달
2. 부정적인 메시지도 회피하지 말고 직접적으로 언급
3. 단순한 위로보다는 실용적 통찰 제공
4. 사용자가 스스로 판단할 수 있는 명확한 정보 전달
5. 무조건적 긍정 금지 - 균형잡힌 시각 유지

**해석 구조:**

🃏 **{card_name_kr}가 뽑힌 위치의 의미: {position_name_kr}**
이 위치가 무엇을 나타내는지 {position_meaning}을 참고해 명확히 설명하세요.

**{card_name_kr}({orientation_kr})의 메시지**
카드가 전하는 핵심 메시지를 직설적으로 전달하세요. 

**당신 상황에 적용하면**
카드가 {position_index}에 위치한다면 어떤 걸 의미하는 지 2-3문장으로 설명하세요.   
- 당신은 타로 상담가가 되어 포지션별 카드의 의미를 사용자 고민과 구체적으로 연결해 해석해주세요.


**톤:**
- 따뜻하되 현실적
- 희망적이되 맹목적이지 않음
- 직설적이되 잔인하지 않음
"""

       try:
           response = llm.invoke([HumanMessage(content=interpretation_prompt)])
           interpretation = response.content
           # 한국어 정보 추가
           card_name_kr = card.get('name_kr', card_name)
           orientation_kr = card.get('orientation_kr', orientation)
           
           interpretations.append({
               "position": position_index,
               "card_name": card_name,
               "card_name_kr": card_name_kr,  # 한국어 카드명 추가
               "orientation": orientation,
               "orientation_kr": orientation_kr,  # 한국어 방향 추가
               "position_name": position_name,
               "position_name_kr": position_name_kr,  # 한국어 포지션 이름 추가
               "interpretation": interpretation
           })
       except Exception as e:
           # 한국어 정보 추가
           card_name_kr = card.get('name_kr', card_name)
           orientation_kr = card.get('orientation_kr', orientation)
           
           interpretations.append({
               "position": position_index,
               "card_name": card_name,
               "card_name_kr": card_name_kr,  # 한국어 카드명 추가
               "orientation": orientation,
               "orientation_kr": orientation_kr,  # 한국어 방향 추가
               "position_name": position_name,
               "position_name_kr": position_name_kr,  # 한국어 포지션 이름 추가
               "interpretation": f"카드 해석 중 오류가 발생했습니다: {str(e)}"
           })
       
       # 시기 정보 생성
       card_info_simple = {
           "card_name": card_name,
           "orientation": orientation,
           "suit": extract_suit_from_name(card_name),
           "rank": extract_rank_from_name(card_name),
           "is_major_arcana": is_major_arcana(card_name)
       }
       
       # 개선된 시기 예측 함수 사용
       timing_result = predict_timing_with_current_date(card_info_simple, state.get("temporal_context"))
       basic_timing = timing_result.get("basic_timing", {})
       concrete_dates = timing_result.get("concrete_dates", [])
       
       # 구체적 날짜가 있으면 사용, 없으면 기본 시간 범위 사용
       if concrete_dates and len(concrete_dates) > 0:
           actual_timing = concrete_dates[0].copy()
           actual_timing["time_frame"] = concrete_dates[0].get("period", basic_timing.get('time_frame', '알 수 없음'))
       else:
           actual_timing = basic_timing
       
       # 한국어 정보 추가
       card_name_kr = card.get('name_kr', card_name)
       orientation_kr = card.get('orientation_kr', orientation)
       
       timing_info.append({
           "position": position_index,
           "position_name": position_name,
           "position_name_kr": position_name_kr,  # 한국어 포지션 이름 추가
           "card_name": card_name,
           "card_name_kr": card_name_kr,  # 한국어 카드명 추가
           "orientation": orientation,
           "orientation_kr": orientation_kr,  # 한국어 방향 추가
           "timing": actual_timing,
           "enhanced_timing": timing_result
       })
   
   # 시기 정보 구조화
   timing_detailed = "**정확한 시기 정보 (절대 변경 금지):**\n"
   timing_by_period = {}
   
   for timing in timing_info:
       timing_data = timing['timing']
       
       time_frame = timing_data.get('time_frame', '알 수 없음')
       
       # 한국어 카드명과 포지션명 사용
       card_name_kr = timing.get('card_name_kr', timing['card_name'])
       orientation_kr = timing.get('orientation_kr', timing['orientation'])
       position_name_kr = timing.get('position_name_kr', timing['position_name'])
       timing_detailed += f"- **{position_name_kr}**: {card_name_kr} ({orientation_kr}) → **정확히 {time_frame}**\n"
       
       if time_frame not in timing_by_period:
           timing_by_period[time_frame] = []
       timing_by_period[time_frame].append({
           'position': timing['position_name'],
           'card': timing['card_name']
       })
   
   timing_detailed += "\n**시기별 요약:**\n"
   for period, cards in timing_by_period.items():
       if len(cards) > 1:
           positions = ", ".join([card['position'] for card in cards])
           timing_detailed += f"- **{period}**: {positions}의 에너지가 함께 작용\n"
       else:
           timing_detailed += f"- **{period}**: {cards[0]['position']}의 에너지\n"
   
   # 🆕 고급 분석 상세 정보 포맷팅
   advanced_analysis_text = f"""
## 🔬 **과학적 타로 분석**

**📊 성공 확률 분석**
- 전체 성공 확률: {integrated_analysis['success_analysis']['success_probability']:.1%}
- 신뢰도: {integrated_analysis['success_analysis']['confidence']}
- 긍정 요인: {len(integrated_analysis['success_analysis']['positive_factors'])}개
- 주의 요인: {len(integrated_analysis['success_analysis']['negative_factors'])}개

**🔮 카드 조합 시너지**
- 시너지 점수: {integrated_analysis['synergy_analysis']['synergy_score']:.1%}
- 특별한 조합: {len(integrated_analysis['synergy_analysis']['combinations'])}개
- 경고 사항: {len(integrated_analysis['synergy_analysis']['warnings'])}개

**🌟 원소 균형 분석**
- 균형 점수: {integrated_analysis['elemental_analysis']['balance_score']:.1%}
- 지배 원소: {integrated_analysis['elemental_analysis']['dominant_element'] or '균형'}
- 부족 원소: {', '.join(integrated_analysis['elemental_analysis']['missing_elements']) or '없음'}

**🔢 수비학 분석**
- 총합: {integrated_analysis['numerology_analysis']['total_value']}
- 환원수: {integrated_analysis['numerology_analysis']['reduced_value']}
- 의미: {integrated_analysis['numerology_analysis']['meaning']}
"""
   
   # 4단계: 명확하고 직접적인 종합 분석 생성 (고급 분석 통합)
   emotional_analysis = consultation_data.get("emotional_analysis", {})
   emotion = emotional_analysis.get('primary_emotion', '알 수 없음')
   
   # 개별 해석 요약 (한국어 카드명과 포지션명 사용)
   interpretations_summary = ""
   for interp in interpretations:
       card_name_kr = interp.get('card_name_kr', interp['card_name'])
       orientation_kr = interp.get('orientation_kr', interp['orientation'])
       position_name_kr = interp.get('position_name_kr', interp['position_name'])
       # 해석 텍스트에서 영문 카드명을 한국어로 교체
       interpretation_text = interp['interpretation']
       interpretations_summary += f"- {position_name_kr}: {card_name_kr} ({orientation_kr}) - {interpretation_text}\n"
   
   analysis_prompt = f"""
사용자 고민: "{user_concern}"
감정 상태: {emotion}
선택한 스프레드: {spread_name}

개별 카드 해석 결과:
{interpretations_summary}

{timing_detailed}

🆕 **과학적 분석 결과:**
{advanced_analysis_text}

**통합 분석 결과:**
- 종합 점수: {integrated_analysis['integrated_score']:.1%}
- 추천사항: {integrated_analysis['recommendation']}

**웹 검색 통합 분석:**
{search_integration if search_integration else ""}

**중요 원칙:**
1. 사용자 고민에 직접적이고 명확한 답변 제공
2. 모호한 표현 절대 금지 ("아마도", "가능성", "~것 같아요")
3. 부정적 면도 솔직하게 언급 (건설적으로)
4. 구체적이고 실행 가능한 조언만 제공
5. 무조건적 희망보다는 현실적 전망
6. 🆕 과학적 분석 결과를 근거로 활용
7. 🌐 웹 검색 결과가 있다면 반드시 활용하여 현실적 정보 통합

다음과 같이 명확하게 답변하세요:

## 🔮 **타로가 전하는 명확한 답변**

**결론 우선순위:**
1. 스프레드 스토리 > 성공 확률 (카드의 전체적 메시지가 우선)
2. 불일치 시 균형잡힌 해석 제공
3. 양쪽 관점을 모두 언급하되 현실적 조언 우선
4. 웹 검색 결과가 있다면 반드시 타로 해석과 통합하여 현실적 조언 제공

**불일치 상황별 대응:**
- 스프레드 부정적 + 성공률 높음 → "기회는 있지만 과정에서 어려움 예상"
- 스프레드 긍정적 + 성공률 낮음 → "좋은 의도지만 현실적 장벽 존재"
- 애매한 스프레드 + 명확한 성공률 → 성공률을 주요 판단 기준으로 활용
- 타로 해석과 웹 검색 결과 불일치 → 두 관점을 모두 제시하고 균형잡힌 조언 제공

**단도직입적으로 말할게요**
[사용자 고민에 대해 네/아니오 또는 구체적 결론을 명확히 제시. 개별 카드 해석들의 "당신 상황에 적용하면" 내용들을 종합해서 나온 사용자 고민에 대한 스프레드 해석을 우선으로 하되, 과학적 분석의 성공 확률({integrated_analysis['success_analysis']['success_probability']:.1%})과 상충될 경우 양쪽을 모두 고려한 현실적 결론 제시. 웹 검색 결과가 있다면 반드시 통합하여 현실적 정보를 제공.]

**과학적 근거:**
[통합 분석 결과를 바탕으로 논리적 설명]
- 성공 확률 {integrated_analysis['success_analysis']['success_probability']:.1%}의 의미
- 카드 조합 시너지 효과
- 원소 균형이 미치는 영향
- 수비학적 의미
- 웹 검색에서 발견한 현실적 정보 (있는 경우)

**시기적으로 언제 어떻게 될 것인가:**
위에 제시된 timing_detailed 데이터를 활용하여 간단하고 명확한 시기 요약을 제공하세요:
- 사용자 고민 "{user_concern}"의 해결 과정을 시간 순서대로 간단히 서술
- 중복되거나 비슷한 시기는 하나로 통합하여 설명  
- "7월 초-중순: 정보 수집", "8월: 결정 실행", "가을: 결과 확인" 식으로 핵심만 간략하게
- 개별 카드 시기는 언급하지 말고, 전체 여정의 핵심 타임라인만 3-4줄로 제시
- 너무 상세하지 말고 사용자가 한눈에 파악할 수 있는 수준으로 요약

**중요**: 상세한 시기 해석은 개별 해석에서 다룰 예정이므로, 여기서는 핵심 흐름만 간단히 제시하세요.

## 💡 **지금 당장 해야 할 일 (우선순위별)**

**1순위 (즉시 실행):**
[과학적 분석 결과를 바탕으로 가장 중요한 행동 1개]

**2순위 (이번 주 내):**
[원소 균형과 카드 조합을 고려한 과제 1개]

**3순위 (장기적):**
[수비학적 의미를 반영한 준비사항 1개]

**주의사항:**
[과학적 분석에서 나온 경고 요소들과 위험 요소들을 구체적으로 명시]

**현실적 전망:**
[성공 확률과 시너지 분석을 바탕으로 희망적 전망과 우려되는 점을 균형있게 제시]

타로 상담사로서 확신을 가지고 단정적으로 답변하세요. 
모호한 표현 대신 "~입니다", "~해야 합니다", "~될 것입니다" 등 명확한 표현을 사용하세요.
과학적 분석 결과를 적극 활용하여 근거 있는 조언을 제공하세요.
"""
   
   try:
       comprehensive_response = llm.invoke([HumanMessage(content=analysis_prompt)])
       comprehensive_text = comprehensive_response.content
   except Exception as e:
       comprehensive_text = "종합 분석 중 오류가 발생했습니다."
   
   # 최종 메시지: 카드 표시 + 과학적 분석 + 명확한 종합 분석 + 단순화된 안내
   summary_message = f"""{cards_display}

{comprehensive_text}

---

💫 **다음 중 원하시는 것을 선택해주세요:**

🔮 **새로운 고민 상담**: "타로 봐줘" 또는 "새로 봐줘"
📖 **개별 카드 해석**: "네" 또는 "보고싶어"  
❓ **다른 질문**: 자유롭게 질문하세요!
"""
   
   # 상담 데이터 업데이트 (고급 분석 결과 포함)
   updated_consultation_data = consultation_data.copy()
   updated_consultation_data.update({
       "selected_cards": selected_cards,
       "positions_meanings": positions_meanings,
       "interpretations": interpretations,
       "timing_info": timing_info,
       "timing_detailed": timing_detailed,
       "comprehensive_analysis": comprehensive_text,
       "integrated_analysis": integrated_analysis,  # 🆕 고급 분석 결과 저장
       "status": "summary_shown"
   })
   
   return {
       "messages": [AIMessage(content=summary_message)],
       "consultation_data": updated_consultation_data
   }

def consultation_individual_handler(state: TarotState) -> TarotState:
   """개별 해석 + 상세 조언 처리 (향상된 깊이로 표시)"""
   
   consultation_data = state.get("consultation_data", {})
   if not consultation_data or consultation_data.get("status") != "summary_shown":
       return {"messages": [AIMessage(content="종합 분석을 먼저 확인해주세요.")]}
   
   # 이미 생성된 향상된 데이터 사용
   interpretations = consultation_data.get("interpretations", [])
   timing_info = consultation_data.get("timing_info", [])
   user_concern = consultation_data.get("concern", "")
   comprehensive_analysis = consultation_data.get("comprehensive_analysis", "")
   
   if not interpretations:
       return {"messages": [AIMessage(content="개별 해석 정보가 없습니다.")]}
   
   # 향상된 개별 해석 포맷팅 (이미 생성된 깊이 있는 해석 사용)
   formatted_interpretations = "## 🔮 **카드 해석**\n\n"
   for interp in interpretations:
       formatted_interpretations += f"{interp['interpretation']}\n\n"
   
   # 시기 정보 포맷팅
   formatted_timing = "**시간의 흐름을 읽어보면:**\n🃏 **카드들이 말하는 시간의 흐름을 보니...**\n\n"

   # 시기별로 그룹화
   timing_groups = {}
   for timing in timing_info:
       timing_data = timing['timing']
       time_frame = timing_data.get('time_frame', '알 수 없음')

       if time_frame not in timing_groups:
           timing_groups[time_frame] = []

       card_name_kr = timing.get('card_name_kr', timing['card_name'])
       orientation_kr = timing.get('orientation_kr', timing['orientation'])
       position_name_kr = timing.get('position_name_kr', timing['position_name'])     

       timing_groups[time_frame].append(f"{position_name_kr}: {card_name_kr} ({orientation_kr})")

   # 시기별로 정리해서 출력
   if len(timing_groups) == 1:
      # 시기가 1개면 그대로
      timeframe = list(timing_groups.keys())[0]
      formatted_timing += f"**{timeframe}에 모든 에너지가 집중:**\n"
      for card_info in timing_groups[timeframe]:
          formatted_timing += f"- {card_info}\n"
      formatted_timing += "\n이 시기에 모든 변화가 집중적으로 일어날 것으로 보입니다.\n"

   elif len(timing_groups) <= 3:
    # 시기가 2-3개면 단순 나열
       for i, (timeframe, cards) in enumerate(timing_groups.items(), 1):
           stage_name = ["단기", "중기", "장기"][min(i-1, 2)]
           formatted_timing += f"**{stage_name} ({timeframe}):**\n"
           for card_info in cards:
               formatted_timing += f"- {card_info}\n"
           formatted_timing += "\n"
       formatted_timing += "위 시기들이 자연스럽게 연결되어 전체적인 흐름을 만들어갑니다.\n"     

   else:
    # 시기가 너무 많으면 요약
        formatted_timing += "**다양한 시기에 걸친 변화들:**\n"
        for timeframe, cards in list(timing_groups.items())[:3]:  # 상위 3개만
            formatted_timing += f"- **{timeframe}**: {', '.join([card.split(':')[0] for card in cards])}\n"
        formatted_timing += "\n각 시기마다 다른 에너지가 작용하여 단계적 변화를 이끌어갑니다.\n"  
      #      
   # 마지막에 LLM에게 통합 해석 요청
   formatted_timing += "\n💫 **이 시간 흐름을 하나의 스토리로 연결해서 자연스럽게 해석해주세요.**\n"
   
   # 향상된 상세 조언 생성
   llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
   
   detailed_advice_prompt = f"""
   당신은 정확하고 솔직한 타로 상담사입니다. 이미 생성된 개별 카드 해석들을 종합해서 전체적인 스토리로 연결하여 사용자의 고민을 해결해주세요.

   **사용자 고민:** {user_concern}
   
   **이전 종합 분석:** {comprehensive_analysis}
   
   **향상된 개별 카드 해석들:**
   {formatted_interpretations}
   
   **시기 정보:**
   {formatted_timing}
   
   **시기 해석 특별 지침:**
   위에 제시된 시간 흐름 정보를 사용자 고민 "{user_concern}"의 특성에 맞춰 자연스럽게 해석하세요:
   - 사용자 고민의 핵심이 무엇인지 파악하여 그에 적합한 시기 흐름으로 연결
   - 시기가 여러 개면: 고민 해결 과정의 자연스러운 단계별 흐름으로 구성
   - 시기가 하나면: 그 시기 안에서 고민과 관련된 세부 변화 과정을 설명
   - 각 시기별로 사용자가 "구체적으로 무엇을 해야 하는지", "어떤 변화가 예상되는지"를 명확히 제시
   
   **중요**: 일반적인 템플릿이 아닌, 이 특정 고민 "{user_concern}"에 최적화된 시기 해석을 제공하세요.

   다음 형식으로 상세 조언해주세요:

   ## 종합 해석:

   🔮 **이제 종합적으로 말해줄게요**

개별 카드 해석의 각 카드별 **당신 상황에 적용하면** 내용을 하나의 완전한 스토리로 연결해서 설명하세요:
- 각 포지션의 카드들이 어떻게 서로 연결되고 영향을 미치는지
- 스프레드에 따른 전체적인 흐름
- 사용자의 고민 "{user_concern}"에 대한 종합적인 답변
- 카드들이 제시하는 전체적인 방향성과 타임라인
- 구체적이고 실용적인 행동 지침

**💡 카드들이 제시하는 결론:**
사용자 고민에 대한 명확하고 단정적인 결론과 핵심 권고사항을 2-3문장으로 제시하세요.

**핵심 원칙:**
- 개별 해석을 단순 나열하지 말고 하나의 연결된 이야기로 구성
- 실제 타로 상담가처럼 전체적인 그림을 그려주세요
- 구체적이고 실용적인 조언 제공
   
   ## 💡 **상세한 실용적 조언**
   
   **단계별 실행 계획**
   [위에서 해석한 시기 정보를 바탕으로 사용자 고민 "{user_concern}"에 맞는 구체적인 단계별 행동 계획을 제시하세요]
   
   **구체적 행동 지침**
   [개별 카드의 향상된 조언을 종합한 실행 가능한 행동들 - 비유와 감정적 표현 활용]
   
   **마음가짐과 태도**
   [각 포지션에서 나온 카드들의 메시지를 종합한 관점과 마음가짐]
   
   **주의사항과 극복방법**
   [카드들이 경고하는 점과 어려움 극복 방법 - 구체적이고 실용적으로]
   
   **장기적 비전**
   [앞으로의 큰 방향과 목표, 카드들이 제시하는 희망적 전망]
   
   ---
   
   상담이 도움이 되셨나요? 이 결과에 대해 더 궁금한 점이나 다른 고민이 있으시면 언제든 말씀해 주세요. ✨
   
   """
   
   try:
       advice_response = llm.invoke([HumanMessage(content=detailed_advice_prompt)])
       advice_text = advice_response.content
   except Exception as e:
       advice_text = "상세 조언 생성 중 오류가 발생했습니다."
   
   # 개별 해석 메시지 생성
   individual_message = f"""{formatted_interpretations}

{formatted_timing}

{advice_text}

---

🎉 **상담이 완료되었습니다!**

💫 **다음 중 원하시는 것을 선택해주세요:**

🔮 **새로운 고민 상담**: "타로 봐줘" 또는 "새로 봐줘"  
❓ **추가 질문**: 방금 상담 내용에 대해 자유롭게 질문하세요!
💬 **일상 대화**: 편안하게 대화해요!"""
   
   # 상담 완료 상태로 업데이트
   updated_consultation_data = consultation_data.copy()
   updated_consultation_data.update({
       "detailed_advice": advice_text,
       "status": "completed"
   })
   
   return {
       "messages": [AIMessage(content=individual_message)],
       "consultation_data": updated_consultation_data
   }

def consultation_final_handler(state: TarotState) -> TarotState:
   """상담 흐름 라우팅 - summary_shown 상태 처리"""
   
   consultation_data = state.get("consultation_data", {})
   status = consultation_data.get("status", "") if consultation_data else ""
   
   # 사용자 입력 확인
   user_input = ""
   for msg in reversed(state["messages"]):
       if isinstance(msg, HumanMessage):
           user_input = msg.content.strip().lower()
           break
   
   if status == "summary_shown":
       # 🔧 새로운 상담 요청 스마트 감지
       user_input_orig = ""
       for msg in reversed(state["messages"]):
           if isinstance(msg, HumanMessage):
               user_input_orig = msg.content.strip()
               break
       
       # 1. "타로봐줘" + 새로운 주제 감지
       tarot_triggers = ["타로 봐줘", "타로봐줘", "타로 상담", "점 봐줘", "운세 봐줘", "새로 봐줘"]
       has_tarot_trigger = any(trigger in user_input for trigger in tarot_triggers)
       
       # 2. 새로운 주제 키워드 감지
       new_topic_keywords = ["여자친구", "남자친구", "연애", "직장", "취업", "가족", "건강", "돈", "재정", "투자", "사업", "이사", "결혼", "이별"]
       has_new_topic = any(topic in user_input_orig for topic in new_topic_keywords)
       
       if has_tarot_trigger and has_new_topic:
           print(f"🔧 새로운 주제 + 타로 상담 요청 감지: '{user_input_orig}' -> 새 상담 시작")
           # 새 상담이므로 기존 consultation_data 초기화
           new_state = state.copy()
           new_state["user_input"] = user_input_orig
           new_state["consultation_data"] = None  # 기존 데이터 초기화
           return consultation_handler(new_state)
       
       # 3. 기존 트리거 시스템
       trigger_result = simple_trigger_check(user_input)
       
       if trigger_result == "new_consultation":
           print("🔧 summary_shown에서 새 상담 시작 트리거 감지")
           return consultation_handler(state)
       elif trigger_result == "individual_reading":
           return consultation_individual_handler(state)
       elif any(keyword in user_input for keyword in ["아니", "no", "괜찮", "됐어", "안볼"]):
           return {"messages": [AIMessage(content="🔮 상담이 도움이 되었기를 바랍니다! 다른 고민이 있으시면 언제든 말씀해주세요. ✨")]}
       else:
           # 🆕 추가 질문으로 분류 - context_reference_handler로 라우팅
           print(f"🎯 summary_shown에서 추가 질문 감지: '{user_input}' -> context_reference_handler로 라우팅")
           return context_reference_handler(state)
   elif status == "card_selection":
       # 카드 선택 단계
       return consultation_summary_handler(state)
   else:
       return {"messages": [AIMessage(content="상담 정보가 올바르지 않습니다.")]}

def context_reference_handler(state: TarotState) -> TarotState:
   """세션 메모리 기반 이전 대화 참조 질문 처리 - 🔧 타로 vs 일상 질문 구분"""
   
   user_input = state.get("user_input", "")
   conversation_memory = state.get("conversation_memory", {})
   
   # 최근 AI 응답 전체를 컨텍스트로 활용
   recent_ai_content = ""
   messages = state.get("messages", [])
   
   # 최근 AI 메시지들 수집 (최대 2개)
   ai_messages = []
   for msg in reversed(messages):
       if isinstance(msg, AIMessage):
           ai_messages.append(msg.content)
           if len(ai_messages) >= 2:
               break
   
   if ai_messages:
       recent_ai_content = "\n\n".join(reversed(ai_messages))
   
   llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
   
   # 🔧 **핵심 개선**: 타로 관련 vs 일상 질문 구분
   classification_prompt = f"""
   최근 대화 내용: "{recent_ai_content[:500]}..."
   사용자 새 질문: "{user_input}"
   
   이 질문이 다음 중 어떤 유형인지 판단해주세요:
   
   A) TAROT_RELATED: 최근 타로 상담/해석과 관련된 추가 질문
      - 예: "그 카드 의미가 뭐야?", "왜 그렇게 해석되는거야?", "시기는 언제야?"
      
   B) CASUAL_NEW: 완전히 새로운 일상적 질문  
      - 예: "짬뽕 vs 짜장면?", "오늘 뭐 입을까?", "비 올까?"
   
   **판단 기준**:
   - 최근 대화에 타로 카드/해석이 있고, 새 질문이 그것과 연관되면 → A
   - 완전히 다른 주제의 가벼운 질문이면 → B
   
   답변: A 또는 B만 출력
   """
   
   try:
       classification_response = llm.invoke([HumanMessage(content=classification_prompt)])
       question_type = classification_response.content.strip()
       
       print(f"🔧 질문 분류 결과: {question_type} - '{user_input}'")
       
       if question_type == "B":
           # 🔧 완전히 새로운 일상 질문 → 캐주얼 응답 + 타로 제안
           return handle_casual_new_question(user_input, llm)
       else:
           # 🔧 타로 관련 질문 → 기존 로직 유지
           return handle_tarot_related_question(state, user_input, recent_ai_content, llm)
           
   except Exception as e:
       print(f"❌ Context Reference 오류: {e}")
       return {
           "messages": [AIMessage(content="🔮 질문을 이해하는 중 문제가 생겼어요. 다시 말씀해주시겠어요?")]
       }

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

def exception_handler(state: TarotState) -> TarotState:
   """예외 상황 처리"""
   user_input = state.get("user_input", "").lower()
   decision = state.get("supervisor_decision", {})
   
   # 중단/재시작 요청
   if any(keyword in user_input for keyword in ["그만", "중단", "취소", "다시", "처음"]):
       return {
           "messages": [AIMessage(content="🔮 알겠습니다. 새로운 상담을 시작할까요? 어떤 고민이 있으신지 말씀해주세요.")],
           "consultation_data": None,
           "user_intent": "unknown"
       }
   
   # 변경 요청
   elif any(keyword in user_input for keyword in ["바꿔", "다른", "변경"]):
       consultation_data = state.get("consultation_data", {})
       if consultation_data and consultation_data.get("status") == "spread_selection":
           return {"messages": [AIMessage(content="🔮 다른 스프레드를 원하신다면 새로운 고민을 말씀해주세요. 더 적합한 스프레드들을 찾아드릴게요!")]}
       else:
           return {"messages": [AIMessage(content="🔮 무엇을 바꾸고 싶으신가요? 구체적으로 말씀해주세요.")]}
   
   # 기타 예외
   else:
       return {"messages": [AIMessage(content="🔮 요청을 정확히 이해하지 못했습니다. 다시 말씀해주시겠어요?")]}

def emotional_support_handler(state: TarotState) -> TarotState:
   """감정적 지원 처리"""
   user_input = state.get("user_input", "")
   decision = state.get("supervisor_decision", {})
   
   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
   
   emotional_prompt = f"""
   사용자가 감정적인 상태입니다. 타로 상담사로서 공감하고 위로해주세요.
   사용자 입력: "{user_input}"
   추천 톤: {decision.get('emotional_tone', 'supportive')}
   
   따뜻하고 이해심 있게 응답한 후, 타로 상담으로 어떻게 도움을 줄 수 있는지 제안해주세요.
   """
   
   try:
       response = llm.invoke([HumanMessage(content=emotional_prompt)])
       return {"messages": [response]}
   except Exception as e:
       return {"messages": [AIMessage(content="🔮 마음이 힘드시는군요. 함께 이야기하면서 위로가 되는 답을 찾아보아요. 어떤 고민이 있으신가요?")]}

def start_specific_spread_consultation(state: TarotState) -> TarotState:
   """리팩토링된 특정 스프레드 상담 핸들러 - 새로운 노드들을 순차 실행"""
   print("🔧 기존 start_specific_spread_consultation 호출 -> 리팩토링된 노드들로 처리")
   
   # Phase 2 리팩토링: 4개 노드를 순차 실행하여 동일한 결과 제공
   try:
       # 1. 스프레드 추출
       state = spread_extractor_node(state)
       
       # 2. 상황 분석
       state = situation_analyzer_node(state)
       
       # 3. 카드 수 추론
       state = card_count_inferrer_node(state)
       
       # 4. 상태 결정
       state = status_determiner_node(state)
       
       print("✅ 리팩토링된 start_specific_spread_consultation 성공적으로 완료")
       return state
       
   except Exception as e:
       print(f"❌ 리팩토링된 start_specific_spread_consultation 오류: {e}")
       # 기본 에러 처리
       return {
           "messages": [AIMessage(content="🔮 특정 스프레드 상담 처리 중 문제가 발생했습니다. 다시 시도해주세요.")],
           "consultation_data": {
               "status": "error"
           }
       }

# =================================================================
# 도구 처리 함수들
# =================================================================

def tool_result_handler(state: TarotState) -> TarotState:
   """도구 실행 후 결과를 AIMessage로 변환하여 사용자에게 전달"""
   messages = state.get("messages", [])
   
   if not messages:
       return {"messages": [AIMessage(content="도구 실행 결과를 찾을 수 없습니다.")]}
   
   # 마지막 메시지가 ToolMessage인지 확인
   last_message = messages[-1]
   
   if hasattr(last_message, 'name') and last_message.name in ['search_tarot_cards', 'search_tarot_spreads']:
       # 도구 결과를 LLM으로 처리하여 사용자 친화적인 답변 생성
       tool_result = last_message.content
       
       llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
       
       # 어떤 도구인지에 따라 프롬프트 변경
       if last_message.name == 'search_tarot_cards':
           prompt = f"""
           여러 타로 전문서에서 검색된 카드 의미들을 종합하여 사용자에게 완전하고 이해하기 쉬운 해석을 제공해주세요.

           검색 결과 (여러 타로책의 해석):
           {tool_result}

           **중요한 지침:**
           1. **여러 출처 통합**: 7개 타로책의 다양한 해석을 종합하여 완전한 의미 제공
           2. **이미지 설명 제외**: 카드 그림이나 시각적 묘사는 빼고 오직 **의미와 해석**만 포함
           3. **정방향/역방향 구분**: content(정방향)와 reversed(역방향) 의미를 명확히 분리
           4. **핵심 키워드 추출**: 각 방향별로 가장 중요한 키워드들 정리
           5. **실용적 조언**: 일상생활에서 이 카드가 나타났을 때의 의미와 조언

           **출력 형식:**
           🔮 **[카드명] 카드 의미**

           **✨ 정방향 (Upright)**
           - **핵심 의미**: [여러 책의 공통된 핵심 의미 통합]
           - **주요 키워드**: [중요 키워드 5-7개]
           - **상황별 해석**: 
             • 연애: [연애 관련 의미]
             • 직업: [직업/성공 관련 의미]  
             • 개인성장: [내적 성장 관련 의미]
           - **조언**: [이 카드가 나왔을 때 권하는 행동이나 마음가짐]

           **🔄 역방향 (Reversed)**
           - **핵심 의미**: [여러 책의 역방향 해석 통합]
           - **주요 키워드**: [역방향 키워드 5-7개]
           - **주의사항**: [조심해야 할 점들]
           - **극복방법**: [역방향 에너지를 긍정적으로 전환하는 방법]

           **💫 종합 메시지**
           [이 카드의 전체적인 메시지와 깊은 의미]

           **참고사항**: 
           - 카드 이미지나 그림 묘사는 완전히 제외
           - 여러 출처의 일치하지 않는 해석이 있다면 가장 일반적이고 전통적인 해석 우선
           - 따뜻하고 지지적인 타로 상담사 톤 유지

           마지막에 "다른 카드나 타로 상담이 필요하시면 언제든 말씀해주세요! 🌟"를 추가해주세요.
           """
       else:  # search_tarot_spreads
           prompt = f"""
           검색된 타로 스프레드 정보를 바탕으로 사용자에게 친근하고 실용적으로 설명해주세요.

           검색 결과 (스프레드 정보):
           {tool_result}

           **중요한 지침:**
           1. **스프레드 개요**: spread_name과 description을 활용하여 이 스프레드의 특징과 장점 설명
           2. **사용 상황**: keywords를 참고하여 어떤 상황에서 사용하면 좋은지 구체적으로 안내
           3. **포지션 설명**: positions 정보를 활용하여 각 카드 자리의 의미를 간략하고 이해하기 쉽게 설명
           4. **실용적 조언**: 이 스프레드가 어떤 질문이나 고민에 특히 효과적인지 안내

           **출력 형식:**
           🔮 **[스프레드명] 소개**

           **✨ 이 스프레드의 특징**
           [description과 keywords를 활용한 스프레드 특징 설명]

           **🎯 이런 상황에서 사용하세요**
           [keywords 기반으로 구체적인 사용 상황들 나열]
           - 예: "과거-현재-미래 흐름을 알고 싶을 때"
           - 예: "간단하고 명확한 답변이 필요할 때"

           **📍 카드 배치와 의미**
           [각 position의 position_name과 position_meaning을 간략히 설명]
           - **1번째 카드 (포지션명)**: [position_meaning 요약]
           - **2번째 카드 (포지션명)**: [position_meaning 요약]
           - **3번째 카드 (포지션명)**: [position_meaning 요약]

           **💡 이런 질문에 특히 좋아요**
           [keywords를 참고하여 적합한 질문 유형들 제시]

           **🌟 왜 추천하는가**
           [이 스프레드만의 장점과 효과]

           따뜻하고 친근한 타로 상담사 톤으로 작성하고, 마지막에 "이 스프레드로 상담받고 싶으시거나 다른 고민이 있으시면 언제든 말씀해주세요! ✨"를 추가해주세요.
           """
       
       try:
           response = llm.invoke([HumanMessage(content=prompt)])
           print(f"🔧 도구 결과를 AIMessage로 변환 완료")
           
           # 기존 메시지들은 유지하고 마지막에 AI 응답 추가
           new_messages = messages[:-1]  # ToolMessage 제거
           new_messages.append(response)  # AIMessage 추가
           
           return {"messages": new_messages}
           
       except Exception as e:
           print(f"🔧 도구 결과 변환 중 오류: {e}")
           return {"messages": [AIMessage(content="🔮 검색 결과를 처리하는 중 문제가 발생했습니다. 다시 질문해주세요.")]}
   
   else:
       # ToolMessage가 아니면 그대로 반환
       print(f"🔧 도구 메시지가 아님: {type(last_message)}")
       return state

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

# =================================================================
# 라우팅 함수들
# =================================================================

def state_router(state: TarotState) -> str:
   """🆕 상태 기반 라우팅"""
   routing_decision = state.get("routing_decision", "NEW_SESSION")
   
   print(f"🔀 State Router: {routing_decision}")
   
   if routing_decision == "CONSULTATION_ACTIVE":
       return "consultation_direct"
   elif routing_decision == "FOLLOWUP_QUESTION":
       return "context_reference_direct"
   else:
       return "supervisor_master"

def processor_router(state: TarotState) -> str:
   """🆕 프로세서 후 라우팅 - 도구 호출 체크"""
   messages = state.get("messages", [])
   if not messages:
       return "end"
   
   last_message = messages[-1]
   
   # AIMessage이고 tool_calls가 있는지 체크
   if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
       print(f"🔧 도구 호출 감지: {len(last_message.tool_calls)}개")
       return "tools"
   
   print("🔧 도구 호출 없음 - 종료")
   return "end"

# =================================================================
# 최적화된 그래프 생성 함수
# =================================================================

def create_optimized_tarot_graph():
   """🆕 최적화된 타로 그래프 - 기존 함수들 100% 재사용"""
   
   workflow = StateGraph(TarotState)
   
   # === 3개 핵심 노드만 추가 ===
   workflow.add_node("state_classifier", state_classifier_node)
   workflow.add_node("supervisor_master", supervisor_master_node)
   workflow.add_node("unified_processor", unified_processor_node)
   workflow.add_node("unified_tool_handler", unified_tool_handler_node)
   
   # === 간단한 연결 구조 ===
   workflow.add_edge(START, "state_classifier")
   
   # 상태 기반 라우팅
   workflow.add_conditional_edges(
       "state_classifier",
       state_router,
       {
           "consultation_direct": "unified_processor",  # Fast Track
           "context_reference_direct": "unified_processor",  # Fast Track
           "supervisor_master": "supervisor_master"  # Full Analysis
       }
   )
   
   workflow.add_edge("supervisor_master", "unified_processor")
   
   # 도구 호출 체크
   workflow.add_conditional_edges(
       "unified_processor",
       processor_router,
       {
           "tools": "unified_tool_handler",
           "end": END
       }
   )
   
   workflow.add_edge("unified_tool_handler", END)
   
   return workflow

# =================================================================
# RAG 시스템 관련
# =================================================================

rag_system = None

def initialize_rag_system():
   """RAG 시스템 초기화"""
   global rag_system
   if rag_system is None:
       rag_system = TarotRAGSystem(
           card_faiss_path="tarot_card_faiss_index",
           spread_faiss_path="tarot_spread_faiss_index"
       )
       print("✅ RAG 시스템 초기화 완료")

@tool
def search_tarot_spreads(query: str) -> str:
   """타로 스프레드를 검색합니다 - LLM 번역 사용"""
   if rag_system is None:
       return "RAG 시스템이 초기화되지 않았습니다."
   
   try:
       # 스프레드도 LLM 번역 적용
       english_query = translate_korean_to_english_with_llm(query)
       
       results = rag_system.search_spreads(english_query, final_k=5)
       safe_results = convert_numpy_types(results)
       
       print(f"🔮 SPREAD SEARCH: {query} -> {english_query}")
       print(f"🔍 검색 결과: {len(safe_results)}개")
       
       return safe_format_search_results(safe_results)
   except Exception as e:
       return f"스프레드 검색 중 오류가 발생했습니다: {str(e)}"

@tool  
def search_tarot_cards(query: str) -> str:
   """타로 카드의 의미를 검색합니다 - LLM 번역 사용"""
   if rag_system is None:
       return "RAG 시스템이 초기화되지 않았습니다."
   
   try:
       # LLM으로 번역
       english_query = translate_korean_to_english_with_llm(query)
       
       # 영어 쿼리로 검색
       results = rag_system.search_cards(english_query, final_k=5)
       safe_results = convert_numpy_types(results)
       
       print(f"🃏 CARD SEARCH: {query} -> {english_query}")
       print(f"🔍 검색 결과: {len(safe_results)}개")
       
       return safe_format_search_results(safe_results)
   except Exception as e:
       return f"카드 검색 중 오류가 발생했습니다: {str(e)}"

# =================================================================
# 메인 실행 함수
# =================================================================

def main():
   """🆕 최적화된 메인 실행 함수"""
   print("🔮 최적화된 타로 시스템을 초기화하는 중...")
   
   # RAG 시스템 초기화
   global rag_system
   try:
       initialize_rag_system()
       print("✅ RAG 시스템 초기화 성공!")
   except Exception as e:
       print(f"⚠️ RAG 시스템 초기화 실패: {e}")
       print("📝 기본 모드로 계속 진행합니다...")
       rag_system = None
   
   # 그래프 생성
   try:
       app = create_optimized_tarot_graph().compile()
       print("✅ 최적화된 타로 시스템 초기화 완료!")
       print("🚀 Fast Track 기능으로 멀티턴 성능 대폭 향상!")
       print("=" * 50)
   except Exception as e:
       print(f"❌ 그래프 초기화 실패: {e}")
       return
   
   # 초기 상태
   current_state = {
       "messages": [AIMessage(content="🔮 안녕하세요! 타로 상담사입니다. 오늘은 어떤 도움이 필요하신가요?")],
       "user_intent": "unknown",
       "user_input": "",
       "consultation_data": None,
       "supervisor_decision": None
   }
   
   # 첫 인사 출력
   first_message = current_state["messages"][0]
   print(f"\n🔮 타로 상담사: {first_message.content}")
   
   # 대화 루프
   while True:
       user_input = input("\n사용자: ").strip()
       
       # 🔧 빈 입력 처리 개선 - 종료 의사가 명확한 경우만 종료
       if user_input.lower() in ['quit', 'exit', '종료', '끝', '그만', 'bye']:
           print("🔮 타로 상담이 도움이 되었기를 바랍니다. 좋은 하루 되세요! ✨")
           break
       
       # 🔧 빈 입력일 경우 계속 대기
       if not user_input:
           print("💬 무엇이든 편하게 말씀해주세요!")
           continue
       
       # 사용자 메시지 추가
       current_state["messages"].append(HumanMessage(content=user_input))
       current_state["user_input"] = user_input
       
       # 🔧 성능 측정
       import time
       start_time = time.time()
       
       try:
           # 그래프 실행
           result = app.invoke(current_state)
           current_state = result
           
           # 성능 측정 완료
           end_time = time.time()
           response_time = end_time - start_time
           
           # 응답 출력
           messages = current_state.get("messages", [])
           if messages:
               last_message = messages[-1]
               if isinstance(last_message, AIMessage):
                   print(f"\n🔮 타로 상담사: {last_message.content}")
                   
                   
                   # 성능 정보 출력 (Fast Track 여부 표시)
                   routing_decision = current_state.get("routing_decision", "unknown")
                   if routing_decision in ["CONSULTATION_ACTIVE", "FOLLOWUP_QUESTION"]:
                       print(f"⚡ Fast Track ({response_time:.2f}초)")
                   else:
                       print(f"🧠 Full Analysis ({response_time:.2f}초)")
               else:
                   print(f"🔍 마지막 메시지가 AIMessage가 아님: {last_message}")
           else:
               print("🔍 메시지가 없습니다")
               
       except Exception as e:
           print(f"❌ 처리 중 오류 발생: {e}")
           continue

if __name__ == "__main__":
   main()      
   