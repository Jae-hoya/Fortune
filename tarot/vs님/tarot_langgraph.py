from dotenv import load_dotenv
load_dotenv()

import os
import random
import re
import difflib
import json
from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
import numpy as np

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

# =================================================================
# 1. 카드 데이터베이스 및 유틸리티 함수
# =================================================================

# 78장 타로 카드 데이터베이스
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

def parse_card_numbers(user_input: str, required_count: int) -> List[int]:
    """사용자 입력에서 카드 번호들을 파싱하고 중복 체크"""
    try:
        numbers = []
        parts = user_input.replace(" ", "").split(",")
        
        for part in parts:
            if part.isdigit():
                num = int(part)
                if 1 <= num <= 78:
                    if num not in numbers:  # 중복 체크
                        numbers.append(num)
                    else:
                        # 중복된 숫자가 있으면 None 반환 (다시 입력 요청용)
                        return None
        
        # 필요한 개수만큼 입력되었는지 확인
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
    
    # 78장 중에서 랜덤으로 중복 없이 선택
    random_card_numbers = random.sample(range(1, 79), required_count)
    
    selected_cards = []
    for position_index, (user_num, random_card_num) in enumerate(zip(user_numbers, random_card_numbers)):
        card_name = TAROT_CARDS.get(random_card_num, f"Unknown Card {random_card_num}")
        orientation = random.choice(["upright", "reversed"])
        selected_cards.append({
            "position": position_index + 1,  # 스프레드에서의 위치 (Card 1, Card 2, ...)
            "user_number": user_num,         # 사용자가 입력한 숫자 (기록용)
            "card_number": random_card_num,  # 실제 랜덤 선택된 카드 번호
            "name": card_name,
            "orientation": orientation
        })
    
    return selected_cards

def format_search_results(results) -> str:
    """검색 결과를 문자열로 포맷팅"""
    if not results:
        return "검색 결과가 없습니다."
    
    formatted = ""
    for i, (doc, score) in enumerate(results, 1):
        metadata = doc.metadata
        content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
        
        formatted += f"\n=== 결과 {i} (점수: {score:.3f}) ===\n"
        
        if metadata.get("card_name"):
            formatted += f"카드: {metadata['card_name']}\n"
        if metadata.get("spread_name"):
            formatted += f"스프레드: {metadata['spread_name']}\n"
        if metadata.get("source"):
            formatted += f"출처: {metadata['source']}\n"
        
        formatted += f"내용: {content}\n"
        formatted += "-" * 50 + "\n"
    
    return formatted

def extract_positions_from_spread(spread_info: dict) -> Dict[str, Dict[str, str]]:
    """스프레드에서 카드 위치 정보 추출 - FAISS 메타데이터 구조에 맞게 수정"""
    positions = {}
    
    # positions 리스트가 있는 경우 (FAISS 메타데이터 형식)
    if "positions" in spread_info and isinstance(spread_info["positions"], list):
        for pos in spread_info["positions"]:
            if isinstance(pos, dict) and "position_num" in pos and "position_name" in pos:
                positions[str(pos["position_num"])] = {
                    "position": pos["position_name"],
                    "meaning": pos.get("position_meaning", "")
                }
    
    # 폴백 스프레드의 경우
    elif "positions_table" in spread_info:
        table_text = spread_info["positions_table"]
        lines = table_text.strip().split("\n")
        
        for line in lines:
            if "|" not in line or "---" in line or "Card #" in line:
                continue
                
            parts = line.split("|")
            if len(parts) >= 4:
                card_num = parts[1].strip()
                position = parts[2].strip().replace("**", "")
                meaning = parts[3].strip()
                
                positions[card_num] = {
                    "position": position,
                    "meaning": meaning
                }
    
    return positions

# =================================================================
# 타로 데이터 처리 클래스
# =================================================================

class TarotDataProcessor:
    """타로 카드 및 스프레드 데이터 처리를 위한 클래스"""
    
    def __init__(self):
        """초기화 함수"""
        self.fallback_spreads = self._define_fallback_spreads()
        
    def _define_fallback_spreads(self) -> List[Dict[str, Any]]:
        """기본 폴백 스프레드 정의"""
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
            },
            {
                "spread_name": "THE HORSESHOE TAROT CARD SPREAD",
                "normalized_name": "horseshoe spread",
                "card_count": 7,
                "description": "A seven-card spread in the shape of a horseshoe that explores the past, present, future, obstacles, external influences, advice, and final outcome.",
                "positions": [
                    {"position_num": 1, "position_name": "Past Influences", "position_meaning": "Events from the past that have shaped the present situation"},
                    {"position_num": 2, "position_name": "Present", "position_meaning": "Current circumstances and energies at work"},
                    {"position_num": 3, "position_name": "Hidden Influences", "position_meaning": "Unseen factors or subconscious elements affecting the situation"},
                    {"position_num": 4, "position_name": "Obstacles", "position_meaning": "Challenges that need to be overcome"},
                    {"position_num": 5, "position_name": "External Influences", "position_meaning": "How others or outside circumstances are affecting the situation"},
                    {"position_num": 6, "position_name": "Advice", "position_meaning": "Guidance on how to approach the situation"},
                    {"position_num": 7, "position_name": "Outcome", "position_meaning": "The likely result if current trends continue"}
                ],
                "keywords": "balanced, moderate, seven cards, obstacles, advice, outcome, horseshoe shape, luck, external influences, guidance"
            }
        ]
    
    def get_fallback_spreads(self, concern: str = "") -> List[Dict[str, Any]]:
        """
        사용자 관심사에 기반하여 폴백 스프레드 반환
        
        Args:
            concern: 사용자 관심사/질문
            
        Returns:
            적합한 폴백 스프레드 리스트
        """
        # 모든 폴백 스프레드
        all_fallbacks = self._define_fallback_spreads()
        
        if not concern:
            # 관심사가 없으면 모든 폴백 반환
            return all_fallbacks
            
        # 관심사 키워드 기반 적합성 점수 계산
        concern = concern.lower()
        
        # 키워드와 스프레드 매핑
        keyword_spread_mapping = {
            "simple": "three card spread",
            "quick": "three card spread", 
            "basic": "three card spread",
            "beginner": "three card spread",
            "complex": "celtic cross spread",
            "detailed": "celtic cross spread",
            "comprehensive": "celtic cross spread",
            "deep": "celtic cross spread",
            "thorough": "celtic cross spread",
            "balanced": "horseshoe spread",
            "moderate": "horseshoe spread"
        }
        
        # 키워드 확인 및 점수 계산
        scores = {"three card spread": 0, "celtic cross spread": 0, "horseshoe spread": 0}
        
        for keyword, spread in keyword_spread_mapping.items():
            if keyword in concern:
                scores[spread] += 1
                
        # 관심사 길이 기반 추가 점수 부여
        concern_word_count = len(concern.split())
        if concern_word_count < 10:
            scores["three card spread"] += 2
        elif concern_word_count > 25:
            scores["celtic cross spread"] += 2
        else:
            scores["horseshoe spread"] += 1
        
        # 점수 기반 정렬
        sorted_spreads = sorted(
            all_fallbacks, 
            key=lambda x: scores.get(x.get("normalized_name", ""), 0),
            reverse=True
        )
        
        return sorted_spreads
    
    def normalize_spread_name(self, spread_name: str) -> str:
        """스프레드 이름 정규화"""
        if not spread_name:
            return ""
            
        # 소문자 변환 및 불필요한 단어 제거
        normalized = spread_name.lower()
        normalized = re.sub(r'(the|tarot|card|reading)\s+', '', normalized)
        normalized = re.sub(r'\s+spread$', ' spread', normalized)
        normalized = re.sub(r'[^\w\s]', '', normalized)  # 특수문자 제거
        normalized = re.sub(r'\s+', ' ', normalized).strip()  # 공백 정리
        
        return normalized
    
    def get_fallback_spread_by_name(self, spread_name: str) -> Optional[Dict[str, Any]]:
        """
        이름으로 폴백 스프레드 검색
        
        Args:
            spread_name: 스프레드 이름
            
        Returns:
            폴백 스프레드 정보 (없으면 None)
        """
        normalized_name = self.normalize_spread_name(spread_name)
        
        for spread in self.fallback_spreads:
            if self.normalize_spread_name(spread["spread_name"]) == normalized_name:
                return spread
                
            # 부분 일치도 허용
            if normalized_name in self.normalize_spread_name(spread["spread_name"]) or \
               self.normalize_spread_name(spread["spread_name"]) in normalized_name:
                return spread
        
        # 가장 유사한 이름 찾기
        best_match = None
        best_score = 0
        
        for spread in self.fallback_spreads:
            spread_norm_name = self.normalize_spread_name(spread["spread_name"])
            score = difflib.SequenceMatcher(None, normalized_name, spread_norm_name).ratio()
            
            if score > best_score:
                best_score = score
                best_match = spread
        
        # 유사도가 0.5 이상이면 반환
        if best_score >= 0.5 and best_match:
            return best_match
            
        # 기본값으로 THREE CARD SPREAD 반환
        return self.fallback_spreads[0]
    
    def extract_positions_from_table(self, positions_table: str) -> List[Dict[str, Any]]:
        """
        포지션 테이블에서 포지션 정보 추출
        
        Args:
            positions_table: 마크다운 형식의 포지션 테이블
            
        Returns:
            포지션 정보 리스트
        """
        if not positions_table or not isinstance(positions_table, str):
            return []
            
        positions = []
        
        # 패턴 1: 기본 테이블 형식 (| 1 | **Position Name** | Description |)
        pattern1 = r"\|\s*(\d+)\s*\|\s*\*\*(.*?)\*\*\s*\|\s*(.*?)\s*\|"
        matches1 = re.findall(pattern1, positions_table)
        
        if matches1:
            for match in matches1:
                positions.append({
                    "position_num": int(match[0]),
                    "position_name": match[1].strip(),
                    "position_meaning": match[2].strip()
                })
            return positions
            
        # 패턴 2: 단순 테이블 형식 (| 1 | Position Name | Description |)
        pattern2 = r"\|\s*(\d+)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|"
        matches2 = re.findall(pattern2, positions_table)
        
        if matches2:
            for match in matches2:
                positions.append({
                    "position_num": int(match[0]),
                    "position_name": match[1].strip(),
                    "position_meaning": match[2].strip()
                })
            return positions
            
        # 패턴 3: 번호와 의미 형식 (1. Position Name - Description)
        pattern3 = r"(\d+)\.\s*(.*?)\s*[-–—:]\s*(.*?)(?:\n|$)"
        matches3 = re.findall(pattern3, positions_table)
        
        if matches3:
            for match in matches3:
                positions.append({
                    "position_num": int(match[0]),
                    "position_name": match[1].strip(),
                    "position_meaning": match[2].strip()
                })
            return positions
            
        # 패턴 4: 카드 형식 (Card #1: Position Name - Description)
        pattern4 = r"Card\s*#(\d+):\s*(.*?)\s*[-–—:]\s*(.*?)(?:\n|$)"
        matches4 = re.findall(pattern4, positions_table)
        
        if matches4:
            for match in matches4:
                positions.append({
                    "position_num": int(match[0]),
                    "position_name": match[1].strip(),
                    "position_meaning": match[2].strip()
                })
            return positions
        
        return positions
    
    def get_default_positions(self, spread_name: str, card_count: int) -> List[Dict[str, Any]]:
        """
        스프레드 이름과 카드 수를 기반으로 기본 포지션 정보 생성
        
        Args:
            spread_name: 스프레드 이름
            card_count: 카드 수
            
        Returns:
            기본 포지션 정보 리스트
        """
        # 폴백 스프레드 확인
        fallback = self.get_fallback_spread_by_name(spread_name)
        if fallback and "positions" in fallback:
            return fallback["positions"]
            
        # 기본 포지션 생성
        positions = []
        
        # 3장 스프레드 기본값
        if card_count == 3:
            positions = [
                {"position_num": 1, "position_name": "Past", "position_meaning": "Influences from the past"},
                {"position_num": 2, "position_name": "Present", "position_meaning": "Current situation"},
                {"position_num": 3, "position_name": "Future", "position_meaning": "Potential outcomes"}
            ]
        # 기타 스프레드는 단순 번호 부여
        else:
            for i in range(1, card_count + 1):
                positions.append({
                    "position_num": i,
                    "position_name": f"Position {i}",
                    "position_meaning": f"Card position {i} in the {spread_name}"
                })
                
        return positions

# =================================================================
# 2. LangGraph State 정의
# =================================================================

class TarotState(TypedDict):
    # 기본 메시지 관리
    messages: Annotated[List[BaseMessage], add_messages]
    
    # 대화 단계 관리
    current_step: str
    is_first_conversation: bool
    
    # 고민 분석
    user_concern: str
    concern_analysis: Dict[str, Any]
    
    # 스프레드 관련
    search_results: List[Any]
    recommended_spreads: List[Dict[str, Any]]
    selected_spread: Optional[Dict[str, Any]]
    spread_card_count: int
    
    # 카드 관련
    user_card_input: str
    selected_cards: List[Dict[str, Any]]
    
    # 해석 관련
    card_interpretations: List[str]
    comprehensive_analysis: str
    practical_advice: str
    
    # 추가 처리
    needs_followup: bool

# =================================================================
# 3. RAG 검색 도구들
# =================================================================

rag_system = None
data_processor = TarotDataProcessor()

def initialize_rag_system():
    """RAG 시스템 초기화"""
    global rag_system
    if rag_system is None:
        try:
            rag_system = TarotRAGSystem(
                card_faiss_path="tarot_card_faiss_index",
                spread_faiss_path="tarot_spread_faiss_index"
            )
            print("✅ RAG 시스템 초기화 완료")
        except Exception as e:
            print(f"⚠️ RAG 시스템 초기화 실패: {e}")
            print("폴백 스프레드만 사용할 수 있습니다.")

def search_spreads_with_fallback(query: str) -> dict:
    """
    RAG 시스템으로 스프레드 검색 시도 후, 실패 시 3가지 기본 스프레드를 폴백으로 제공
    """
    
    # 1. RAG 시스템으로 검색 시도
    try:
        search_results = rag_system.search_spreads(query, final_k=3)
        if search_results and len(search_results) > 0:
            # FAISS 메타데이터 구조 확인
            for doc, score in search_results:
                # 메타데이터에서 positions 정보 확인
                if "positions" in doc.metadata:
                    print(f"🔍 메타데이터에서 포지션 정보 발견: {len(doc.metadata['positions'])}개")
                
                # normalized_name과 keywords 확인
                if "normalized_name" in doc.metadata:
                    print(f"🔍 정규화된 이름 발견: {doc.metadata['normalized_name']}")
                if "keywords" in doc.metadata:
                    print(f"🔍 키워드 발견: {doc.metadata['keywords']}")
            
            return {
                "success": True,
                "spread_data": format_search_results(search_results),
                "source": "rag",
                "raw_results": search_results
            }
    except Exception as e:
        print(f"🔍 RAG 검색 실패: {e}")
    
    # 2. RAG 실패 시 폴백 스프레드 제공 (TarotDataProcessor 활용)
    data_processor = TarotDataProcessor()
    fallback_spreads = data_processor.get_fallback_spreads(query)
    
    # 폴백 스프레드 정보 포맷팅
    formatted_spreads = []
    for spread in fallback_spreads:
        formatted_spread = {
            "name": spread["spread_name"],
            "description": spread["description"],
            "card_count": spread["card_count"],
            "positions_table": "",
            "normalized_name": spread.get("normalized_name", ""),
            "keywords": spread.get("keywords", "")
        }
        
        # 포지션 정보를 테이블 형식으로 변환
        positions_table = "| Card # | Position | Meaning |\n"
        positions_table += "| ------ | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |\n"
        
        for pos in spread["positions"]:
            positions_table += f"| {pos['position_num']} | **{pos['position_name']}** | {pos['position_meaning']} |\n"
            
        formatted_spread["positions_table"] = positions_table
        formatted_spread["positions"] = spread["positions"]  # embedding.py 구조와 일치
        formatted_spreads.append(formatted_spread)
    
    # 선택된 스프레드 정보 반환
    return {
        "success": False, 
        "spread_data": formatted_spreads,
        "source": "fallback"
    }

@tool
def search_tarot_spreads(query: str) -> str:
    """타로 스프레드를 검색합니다."""
    if rag_system is None:
        return "RAG 시스템이 초기화되지 않았습니다."
    
    try:
        results = rag_system.search_spreads(query, final_k=5)
        return format_search_results(results)
    except Exception as e:
        return f"스프레드 검색 중 오류가 발생했습니다: {str(e)}"

@tool
def search_tarot_cards(query: str) -> str:
    """타로 카드의 의미를 검색합니다."""
    if rag_system is None:
        return "RAG 시스템이 초기화되지 않았습니다."
    
    try:
        results = rag_system.search_cards(query, final_k=5)
        return format_search_results(results)
    except Exception as e:
        return f"카드 검색 중 오류가 발생했습니다: {str(e)}"

@tool
def search_spread_positions(spread_name: str) -> str:
    """위치 정보 검색 최적화 함수"""
    if rag_system is None:
        return "RAG 시스템이 초기화되지 않았습니다."
    
    # 최적화된 검색 쿼리 변형
    variations = [
        f"{spread_name} positions table",
        f"{spread_name} card positions",
        f"{spread_name} position meanings",
        f"{spread_name} spread layout",
        f"{spread_name} card layout",
        f"{spread_name} card meaning positions",
        f"positions in {spread_name}",
        f"how to read {spread_name}",
        f"{spread_name} tarot meaning"
    ]
    
    try:
        # 각 변형으로 시도하고 가장 관련성 높은 결과 반환
        best_result = ""
        for query in variations:
            print(f"🔍 위치 검색 시도: '{query}'")
            results = rag_system.search_spreads(query, final_k=3)
            result_text = format_search_results(results)
            
            # 위치 관련 정보가 포함된 결과인지 확인
            if "position" in result_text.lower() or "card #" in result_text.lower() or "card 1" in result_text.lower():
                # 테이블 형식이나 위치 정보가 풍부한 결과 우선
                if "|" in result_text or "position" in result_text.lower():
                    return result_text
                
                # 백업으로 저장
                if not best_result:
                    best_result = result_text
        
        return best_result or f"스프레드 '{spread_name}'의 위치 정보를 찾을 수 없습니다."
    except Exception as e:
        return f"스프레드 위치 검색 중 오류가 발생했습니다: {str(e)}"

@tool
def get_fallback_spreads() -> str:
    """기본 스프레드 3개를 가져옵니다."""
    if rag_system is None:
        return "RAG 시스템이 초기화되지 않았습니다."
    
    try:
        # 기본 스프레드 3개 검색 - 정확한 이름 지정
        fallback_names = [
            "THREE CARD SPREAD", 
            "HORSESHOE TAROT CARD SPREAD", 
            "CELTIC CROSS"
        ]
        
        all_results = []
        
        for spread_name in fallback_names:
            # 다양한 키워드 변형으로 검색 시도
            variations = [
                spread_name, 
                f"THE {spread_name}", 
                f"{spread_name} SPREAD",
                spread_name.replace(" SPREAD", "")
            ]
            
            found = False
            for variation in variations:
                print(f"🔍 Fallback 스프레드 검색 시도: '{variation}'")
                results = rag_system.search_spreads(variation, final_k=1)
                if results:
                    all_results.extend(results)
                    print(f"🔍 '{variation}' 검색 성공!")
                    found = True
                    break
            
            if not found:
                print(f"🔍 '{spread_name}' 모든 변형 검색 실패")
        
        if not all_results:
            # 모든 검색이 실패하면 아무 스프레드나 3개 가져오기
            print("🔍 Fallback: 일반 스프레드 검색")
            all_results = rag_system.search_spreads("spread", final_k=3)
        
        return format_search_results(all_results)
    except Exception as e:
        print(f"🔍 기본 스프레드 검색 오류: {e}")
        return f"기본 스프레드 검색 중 오류가 발생했습니다: {str(e)}"

def get_fallback_spread_positions(spread_name: str) -> Dict[str, Dict[str, str]]:
    """폴백 스프레드의 위치 정보 제공 - FAISS 메타데이터 구조에 맞게 수정"""
    data_processor = TarotDataProcessor()
    fallback_spread = data_processor.get_fallback_spread_by_name(spread_name)
    
    # 폴백 스프레드에서 포지션 정보 추출
    positions = {}
    if fallback_spread and "positions" in fallback_spread:
        for pos in fallback_spread["positions"]:
            positions[str(pos["position_num"])] = {
                "position": pos["position_name"],
                "meaning": pos["position_meaning"]
            }
    
    # 포지션 정보가 없으면 기본값 생성
    if not positions:
        # 카드 수 추정
        card_count = 3  # 기본값
        if "celtic cross" in spread_name.lower():
            card_count = 10
        elif "horseshoe" in spread_name.lower():
            card_count = 7
        
        # 기본 포지션 생성
        default_positions = data_processor.get_default_positions(spread_name, card_count)
        for pos in default_positions:
            positions[str(pos["position_num"])] = {
                "position": pos["position_name"],
                "meaning": pos["position_meaning"]
            }
    
    return positions

# =================================================================
# 4. LangGraph 노드 함수들 (수정됨)
# =================================================================

def greeting_node(state: TarotState) -> TarotState:
    """첫 인사 노드"""
    print("🔍 greeting_node 실행됨!")
    
    greeting_msg = "🔮 안녕하세요! 타로 상담사입니다. 오늘은 어떤 고민이 있으신가요?"
    
    return {
        "messages": [AIMessage(content=greeting_msg)],
        "current_step": "waiting_concern",
        "is_first_conversation": False
    }

def concern_analysis_node(state: TarotState) -> TarotState:
    """고민 분석 노드 - 사용자 입력이 있을 때만 실행"""
    print("🔍 concern_analysis_node 실행됨!")
    
    messages = state.get("messages", [])
    current_step = state.get("current_step", "")
    
    print(f"🔍 current_step: {current_step}")
    print(f"🔍 messages 개수: {len(messages)}")
    
    # 마지막 사용자 메시지 찾기
    last_user_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_message = msg.content.strip()
            break
    
    # 사용자 메시지가 없으면 아무것도 하지 않음
    if not last_user_message:
        print("🔍 사용자 메시지가 없어서 대기")
        return state
    
    # LLM을 사용한 고민 분석
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    analysis_prompt = f"""
    사용자의 고민을 분석해주세요: "{last_user_message}"
    
    다음 형식으로 분석해주세요:
    1. 고민 주제: (연애/진로/인간관계/건강/가족/재정/기타)
    2. 감정 상태: (현재 느끼는 감정)
    3. 원하는 답변: (조언/예측/선택도움/내면탐구)
    4. 상황 요약: (구체적 상황 한 줄 요약)
    
    간단명료하게 분석해주세요.
    """
    
    try:
        print(f"🔍 LLM에 보낼 프롬프트: {analysis_prompt[:100]}...")
        analysis_response = llm.invoke([HumanMessage(content=analysis_prompt)])
        analysis_content = analysis_response.content
        
        return {
            "user_concern": last_user_message,
            "concern_analysis": {"analysis": analysis_content},
            "current_step": "spread_search",
            "messages": [AIMessage(content=f"고민을 이해했습니다. 적합한 타로 스프레드를 찾아보겠습니다.\n\n{analysis_content}")]
        }
    except Exception as e:
        error_msg = f"분석 중 오류: {str(e)}"
        print(f"🔍 {error_msg}")
        return {
            "user_concern": last_user_message,
            "concern_analysis": {"analysis": error_msg},
            "current_step": "spread_search",
            "messages": [AIMessage(content=f"고민을 이해했습니다. 적합한 타로 스프레드를 찾아보겠습니다.\n\n{error_msg}")]
        }

def spread_search_node(state: TarotState) -> TarotState:
    print("🔍 spread_search_node 실행됨!")
    user_concern = state.get("user_concern", "")
    all_spreads = []
    if hasattr(rag_system, 'spread_bm25'):
        all_spreads = rag_system.spread_bm25.documents
    else:
        all_spreads = []
    spread_info_list = []
    for doc in all_spreads:
        meta = doc.metadata
        info = dict(meta)
        info["content"] = doc.page_content
        info = convert_numpy_types(info)
        info["_normalized_name"] = normalize_spread_name(info.get("spread_name", info.get("name", "")))
        # positions 필드 포함 보장
        if "positions" in meta:
            info["positions"] = meta["positions"]
        spread_info_list.append(info)
    # spread_name(영문) 기준으로만 추천 목록 생성
    recommended_spreads = []
    spreads_info = ""
    for i, info in enumerate(spread_info_list[:3]):
        spread_name = clean_spread_name(info.get('spread_name', info.get('name', '')))
        card_count = info.get('card_count', 3)
        recommended_spreads.append({
            "number": i + 1,
            "spread_name": spread_name,
            "_normalized_name": normalize_spread_name(spread_name),
            "card_count": card_count,
            "positions": info.get("positions", [])
        })
        spreads_info += f"{i+1}. {spread_name} {card_count}장\n"
    messages = state.get("messages", [])
    llm = ChatOpenAI(temperature=0.7)
    # LLM은 추천 이유/설명만 담당
    response = llm.invoke(
        messages + [
            HumanMessage(content=f"""
            당신은 전문 타로 상담사입니다. 사용자의 고민에 가장 적합한 타로 스프레드를 추천해야 합니다.
            사용자 고민: {user_concern}
            다음은 추천 가능한 타로 스프레드 목록입니다:
            {spreads_info}
            위 스프레드 중에서 사용자의 고민에 가장 적합한 것을 선택하고, 왜 그 스프레드가 적합한지 간단히 설명해주세요.
            형식: "**추천 스프레드: [스프레드 이름]**\n[추천 이유]"
            """)
        ]
    )
    # spread_name(영문) 기준으로만 추천 스프레드 선택
    recommended_spread_name = ""
    for spread in recommended_spreads:
        if spread.get("spread_name", "") in response.content:
            recommended_spread_name = spread.get("spread_name", "")
            break
    if not recommended_spread_name and recommended_spreads:
        recommended_spread_name = recommended_spreads[0].get("spread_name", "")
    recommended_spread = next((s for s in recommended_spreads if s.get("spread_name", "") == recommended_spread_name), recommended_spreads[0])
    return {
        **state,
        "search_results": spreads_info,
        "recommended_spreads": recommended_spreads,
        "current_step": "recommend_spread"
    }

def spread_recommendation_node(state: TarotState) -> TarotState:
    """스프레드 추천 노드 - RAG 검색 결과 파싱 포함"""
    print("🔍 spread_recommendation_node 실행됨!")
    
    messages = state["messages"]
    user_concern = state.get("user_concern", "")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    # 도구 실행 결과에서 스프레드 정보 추출
    search_results = ""
    for msg in reversed(messages):
        if hasattr(msg, 'content') and msg.content and 'spread' in msg.content.lower():
            search_results = msg.content
            break
    
    # 검색 결과를 바탕으로 추천 및 스프레드 정보 추출
    recommendation_prompt = f"""
    사용자의 고민 "{user_concern}"에 대해 검색된 스프레드들을 바탕으로 3개의 스프레드를 추천해주세요.

    검색 결과: {search_results}

    다음 형식으로 정확히 추천해주세요:

    🔮 **고민 분석 결과를 바탕으로 다음 스프레드들을 추천드립니다:**

    **1) [스프레드이름] ([카드수]장)**
    - 목적: [이 스프레드의 특징과 적용상황]
    - 효과: [어떤 도움을 받을 수 있는지]

    **2) [스프레드이름] ([카드수]장)**  
    - 목적: [이 스프레드의 특징과 적용상황]
    - 효과: [어떤 도움을 받을 수 있는지]

    **3) [스프레드이름] ([카드수]장)**
    - 목적: [이 스프레드의 특징과 적용상황] 
    - 효과: [어떤 도움을 받을 수 있는지]

    어떤 스프레드를 선택하시겠어요? 번호로 답해주세요 (1, 2, 3).
    
    중요: 응답 마지막에 다음 형식으로 스프레드 정보를 추가해주세요:
    SPREAD_INFO:
    1|[스프레드이름1]|[카드수1]
    2|[스프레드이름2]|[카드수2]
    3|[스프레드이름3]|[카드수3]
    """
    
    try:
        response = llm.invoke([HumanMessage(content=recommendation_prompt)] + messages[-3:])
        response_content = response.content
        
        # 스프레드 정보 파싱
        recommended_spreads = []
        if "SPREAD_INFO:" in response_content:
            info_section = response_content.split("SPREAD_INFO:")[1].strip()
            for line in info_section.split('\n'):
                if '|' in line:
                    parts = line.strip().split('|')
                    if len(parts) >= 3:
                        try:
                            spread_info = {
                                "number": int(parts[0]),
                                "name": parts[1].strip(),
                                "card_count": int(parts[2].strip())
                            }
                            recommended_spreads.append(spread_info)
                        except ValueError:
                            continue
        
        # 기본값 설정 (파싱 실패 시)
        if not recommended_spreads:
            recommended_spreads = [
                {"number": 1, "name": "첫 번째 스프레드", "card_count": 3},
                {"number": 2, "name": "두 번째 스프레드", "card_count": 5},
                {"number": 3, "name": "세 번째 스프레드", "card_count": 7}
            ]
        
        # SPREAD_INFO 부분을 사용자에게 보이지 않도록 제거
        if "SPREAD_INFO:" in response_content:
            response_content = response_content.split("SPREAD_INFO:")[0].strip()
        
        print(f"🔍 파싱된 스프레드 정보: {recommended_spreads}")
        
        return {
            "messages": [AIMessage(content=response_content)],
            "current_step": "waiting_spread_selection",
            "user_concern": user_concern,
            "recommended_spreads": recommended_spreads
        }
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"스프레드 추천 중 오류가 발생했습니다: {str(e)}")],
            "current_step": "waiting_spread_selection",
            "user_concern": user_concern,
            "recommended_spreads": [
                {"number": 1, "name": "기본 스프레드 1", "card_count": 3},
                {"number": 2, "name": "기본 스프레드 2", "card_count": 5},
                {"number": 3, "name": "기본 스프레드 3", "card_count": 7}
            ]
        }

def card_selection_node(state: TarotState) -> TarotState:
    print("🔍 card_selection_node 실행됨!")
    messages = state["messages"]
    user_concern = state.get("user_concern", "")
    current_step = state.get("current_step", "")
    recommended_spreads = state.get("recommended_spreads", [])
    print(f"🔍 card_selection_node - current_step: {current_step}")
    print(f"🔍 추천된 스프레드 정보: {recommended_spreads}")
    if current_step != "waiting_spread_selection":
        print("🔍 스프레드 선택 대기 상태가 아님")
        return state
    last_user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_message = msg.content.strip()
            print(f"🔍 마지막 사용자 메시지: '{last_user_message}'")
            break
    if not any(num in last_user_message for num in ["1", "2", "3"]):
        print("🔍 스프레드 선택되지 않음 - 대기")
        return state
    selected_number = None
    if "1" in last_user_message:
        selected_number = 1
    elif "2" in last_user_message:
        selected_number = 2
    elif "3" in last_user_message:
        selected_number = 3
    selected_spread_info = None
    for spread in recommended_spreads:
        if spread.get("number") == selected_number:
            norm_name = spread.get("_normalized_name", normalize_spread_name(spread.get("spread_name", "")))
            # spread_info_list에서 정규화 이름으로 정확히 매칭
            all_spreads = []
            if hasattr(rag_system, 'spread_bm25'):
                all_spreads = rag_system.spread_bm25.documents
            spread_info_list = []
            for doc in all_spreads:
                meta = doc.metadata
                info = dict(meta)
                info["content"] = doc.page_content
                info = convert_numpy_types(info)
                info["_normalized_name"] = normalize_spread_name(info.get("spread_name", info.get("name", "")))
                # positions 필드 포함 보장
                if "positions" in meta:
                    info["positions"] = meta["positions"]
                spread_info_list.append(info)
            matched = next((s for s in spread_info_list if s["_normalized_name"] == norm_name), None)
            if matched:
                matched["spread_name"] = clean_spread_name(matched.get("spread_name", spread.get("spread_name", "")))
                matched["card_count"] = spread.get("card_count", 3)
                selected_spread_info = matched
            else:
                selected_spread_info = spread
            break
    if not selected_spread_info:
        selected_spread_info = {
            "number": selected_number,
            "spread_name": f"Spread {selected_number}",
            "card_count": 3,
            "positions": []
        }
    selected_spread_name = selected_spread_info.get("spread_name", "")
    card_count = selected_spread_info.get("card_count", 3)
    print(f"🔍 선택된 스프레드: {selected_spread_name}, 카드 수: {card_count}")
    card_selection_msg = f"""
✅ **{selected_spread_name}**를 선택하셨습니다!
🃏 **카드 선택 방법:**
타로 카드는 총 78장이 있습니다. 
1부터 78 사이의 숫자를 **{card_count}장** 선택해주세요.
**예시:** 7, 23, 45
직감으로 떠오르는 숫자들을 말씀해주세요.
"""
    return {
        "messages": [AIMessage(content=card_selection_msg)],
        "selected_spread": selected_spread_info,
        "spread_card_count": card_count,
        "current_step": "waiting_card_numbers",
        "user_concern": user_concern,
        "recommended_spreads": recommended_spreads
    }

def card_processing_node(state: TarotState) -> TarotState:
    """카드 처리 노드 - 수정: 사용자 숫자를 실제 카드로 사용"""
    print("🔍 card_processing_node 실행됨!")
    
    messages = state["messages"]
    card_count = state.get("spread_card_count", 3)
    
    # 마지막 사용자 입력에서 숫자 추출
    last_user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_message = msg.content
            break
    
    # 카드 번호 파싱 및 중복 체크
    user_numbers = parse_card_numbers(last_user_message, card_count)
    
    if user_numbers is None:
        # 중복이나 잘못된 입력
        error_msg = f"""
❌ **입력 오류**

다음 중 하나의 문제가 있습니다:
- 같은 숫자를 두 번 입력했습니다
- {card_count}개의 숫자가 필요합니다
- 1-78 범위의 숫자만 입력 가능합니다

다시 입력해주세요. **{card_count}개의 서로 다른 숫자**를 선택해주세요.
**예시:** 7, 23, 45, 12, 56, 33, 71
"""
        return {
            "messages": [AIMessage(content=error_msg)],
            "current_step": "waiting_card_numbers"  # 다시 입력 대기
        }
    
    # 수정: 사용자 숫자를 실제 카드로 사용
    selected_cards = select_cards_randomly_but_keep_positions(user_numbers, card_count)
    
    # 선택된 카드들 표시
    cards_display = "🃏 **선택된 카드들:**\n\n"
    for card in selected_cards:
        orientation_emoji = "⬆️" if card["orientation"] == "upright" else "⬇️"
        cards_display += f"**{card['position']}번째 카드:** {card['name']} {orientation_emoji} ({card['orientation']})\n"
    
    cards_display += "\n🔮 카드들의 의미를 해석해드리겠습니다. 잠시만 기다려주세요..."
    
    return {
        "messages": [AIMessage(content=cards_display)],
        "selected_cards": selected_cards,
        "user_card_input": last_user_message,
        "current_step": "card_interpretation"
    }

def card_interpretation_node(state: TarotState) -> TarotState:
    """카드 해석 노드 - FAISS 메타데이터 활용"""
    print("🔍 card_interpretation_node 실행됨!")
    
    selected_cards = state.get("selected_cards", [])
    user_concern = state.get("user_concern", "")
    selected_spread = state.get("selected_spread", {})
    spread_name = selected_spread.get("spread_name", selected_spread.get("name", ""))
    norm_name = normalize_spread_name(spread_name)
    positions_meanings = {}
    # positions 필드가 있으면 반드시 우선 사용
    if "positions" in selected_spread and selected_spread["positions"]:
        for pos in selected_spread["positions"]:
            positions_meanings[str(pos.get("position_num", ""))] = {
                "position": pos.get("position_name", ""),
                "meaning": pos.get("position_meaning", "")
            }
    else:
        # spread_info_list에서 정규화 이름으로 정확히 매칭되는 spread의 positions 사용
        all_spreads = []
        if hasattr(rag_system, 'spread_bm25'):
            all_spreads = rag_system.spread_bm25.documents
        spread_info_list = []
        for doc in all_spreads:
            meta = doc.metadata
            info = dict(meta)
            info["content"] = doc.page_content
            info = convert_numpy_types(info)
            info["_normalized_name"] = normalize_spread_name(info.get("spread_name", info.get("name", "")))
            spread_info_list.append(info)
        matched = next((s for s in spread_info_list if s["_normalized_name"] == norm_name), None)
        if matched and "positions" in matched and matched["positions"]:
            for pos in matched["positions"]:
                positions_meanings[str(pos.get("position_num", ""))] = {
                    "position": pos.get("position_name", ""),
                    "meaning": pos.get("position_meaning", "")
                }
    # positions가 여전히 없으면 폴백 사용
    if not positions_meanings:
        print(f"🔍 모든 검색 실패. 폴백 위치 정보 사용")
        positions_meanings = get_fallback_spread_positions(spread_name)
    if not positions_meanings:
        print(f"🔍 폴백도 실패. 기본 위치 정보 생성")
        for i in range(1, len(selected_cards) + 1):
            positions_meanings[str(i)] = {
                "position": f"Card {i}",
                "meaning": f"Position {i} in the {spread_name} spread"
            }
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    # 스프레드 위치 정보 추출 - FAISS 메타데이터에서 직접 가져오기
    interpretations = []
    
    for card in selected_cards:
        position_index = card.get("position", "")
        card_name = card.get("name", "")
        orientation = card.get("orientation", "")
        position_info = positions_meanings.get(str(position_index), {})
        position_name = position_info.get("position", f"Card {position_index}")
        position_meaning = position_info.get("meaning", "")
        card_info = {}
        if rag_system:
            try:
                card_info = rag_system.search_card_meaning(card_name, orientation)
            except Exception as e:
                print(f"🔍 카드 의미 검색 오류: {e}")
                card_info = {"success": False, "message": str(e)}
        card_info_clean = convert_numpy_types(card_info)
        card_info_json = json.dumps(card_info_clean, ensure_ascii=False, indent=2)

        # 프롬프트
        interpretation_prompt = f"""
당신은 전문 타로 상담사입니다.

[카드별 해석]
- 카드명: {card_name}
- 방향: {orientation}
- 카드 메타데이터: {card_info_json}

[포지션 정보]
- 위치: {position_index}번째 카드 ({position_name})
- 위치 의미: {position_meaning}

[해석 지침]
1. 카드의 기본 의미(정방향/역방향, 여러 출처 통합)를 2~3문장 이내로 요약해서 설명하세요.
2. 카드의 의미와 포지션 의미를 결합해, 이 카드가 해당 위치에 놓였을 때의 핵심 메시지를 1~2문장으로 요약하세요.
3. 추상적이지 않고, 구체적이고 실용적인 언어를 사용하세요.
4. 부정적인 카드도 성장과 배움의 기회로 긍정적으로 해석하세요.

[출력 형식 예시]
**{position_index}번째 카드:** {card_name} {('⬆️' if orientation == 'upright' else '⬇️')} ({orientation}): [카드 기본 해석 요약]
→ 위치 의미: {position_meaning}
→ 이 카드가 이 위치에 놓였을 때의 메시지: [결합 해석 요약]
"""

        try:
            response = llm.invoke([HumanMessage(content=interpretation_prompt)])
            interpretations.append(response.content)
        except Exception as e:
            fallback_interpretation = f"""
**🃏 Card {position_index}: {card_name} ({orientation})**
**위치**: {position_name}
**카드 의미**: 정보를 가져오는 중 오류가 발생했습니다.
**해석**: 이 카드는 일반적으로 {orientation} 방향일 때 의미가 있으나, 자세한 해석을 제공하지 못했습니다. 다른 카드들의 맥락에서 해석해보세요.
"""
            interpretations.append(fallback_interpretation)
    
    # 메시지 생성
    card_message = f"## 🔮 카드 해석\n\n"
    for interp in interpretations:
        card_message += f"{interp}\n\n"
    
    return {
        **state,
        "messages": [AIMessage(content=card_message)],
        "current_step": "comprehensive_analysis",
        "card_interpretations": interpretations,
        "positions_meanings": positions_meanings
    }

def comprehensive_analysis_node(state: TarotState) -> TarotState:
    """종합 분석 노드 - FAISS 메타데이터 활용"""
    print("🔍 comprehensive_analysis_node 실행됨!")
    
    card_interpretations = state.get("card_interpretations", [])
    user_concern = state.get("user_concern", "")
    selected_cards = state.get("selected_cards", [])
    selected_spread = state.get("selected_spread", {})
    spread_name = selected_spread.get("spread_name", selected_spread.get("name", ""))
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    llm_with_tools = llm.bind_tools([search_tarot_cards, search_tarot_spreads])
    
    # 카드 정보 정리
    cards_summary = []
    card_keywords_all = []
    
    for card in selected_cards:
        card_pos = card['position']
        card_name = card['name']
        orientation = card['orientation']
        cards_summary.append(f"Card {card_pos}: {card_name} ({orientation})")
        
        # embedding.py 메타데이터 구조에 맞게 카드 키워드 수집 시도
        if rag_system:
            try:
                card_info = rag_system.search_card_meaning(card_name, orientation)
                if card_info and card_info.get("success", False):
                    if orientation == "upright" and "upright_keywords" in card_info:
                        card_keywords_all.extend(card_info["upright_keywords"].split(", "))
                    elif orientation == "reversed" and "reversed_keywords" in card_info:
                        card_keywords_all.extend(card_info["reversed_keywords"].split(", "))
                    elif "tarot_keywords" in card_info:
                        card_keywords_all.extend(card_info["tarot_keywords"])
            except Exception as e:
                print(f"🔍 카드 키워드 수집 오류: {e}")
    
    cards_info = "\n".join(cards_summary)
    interpretations_text = "\n\n".join(card_interpretations)
    
    # 수집된 키워드 추가
    keywords_text = ""
    if card_keywords_all:
        # 중복 제거 및 최대 10개로 제한
        unique_keywords = list(set(card_keywords_all))[:10]
        keywords_text = f"\n\n주요 키워드: {', '.join(unique_keywords)}"
    
    analysis_prompt = f"""
    당신은 전문 타로 상담사입니다. 사용자의 고민에 대해 명쾌하고 직접적인 답변을 제공해야 합니다.

    **사용자 고민:** "{user_concern}"
    
    **선택된 스프레드:** {spread_name}
    
    **선택된 카드들:**
    {cards_info}
    {keywords_text}
    
    **개별 카드 해석들:**
    {interpretations_text}
    
    **중요한 요구사항:**
    1. 사용자의 질문에 대해 명확한 YES/NO 또는 구체적인 답변을 제시하세요
    2. 타로 상담사 말투로 자신감 있게 답변하세요
    3. 카드들의 종합적 의미를 바탕으로 결론을 내리세요
    4. 구체적인 시기나 방법도 제시하세요

    **출력 형식:**

    ## 🔮 **타로가 전하는 답변**

    **🎯 직접적인 답변:**
    [사용자 질문에 대한 명쾌한 답변. 예: "네, 사업은 성공할 것으로 보입니다" 또는 "아니요, 현재로서는 재회가 어려워 보입니다"]

    **📊 카드들이 말하는 이유:**
    [왜 그런 결론에 이르렀는지 카드들의 의미를 종합해서 설명]

    **⏰ 시기와 조건:**
    [언제, 어떤 조건하에 그 결과가 나타날지]

    **💡 구체적 행동 지침:**
    [사용자가 해야 할 구체적인 행동 2-3가지]

    **⚠️ 주의사항:**
    [조심해야 할 점이나 피해야 할 것들]

    반드시 타로 상담사답게 확신을 가지고 명확한 답변을 제시하세요.
    """
    
    try:
        response = llm_with_tools.invoke([HumanMessage(content=analysis_prompt)])
        comprehensive_text = response.content
        
        return {
            "messages": [AIMessage(content=comprehensive_text)],
            "comprehensive_analysis": comprehensive_text,
            "current_step": "practical_advice"
        }
    except Exception as e:
        error_msg = f"종합 분석 중 오류가 발생했습니다: {str(e)}"
        return {
            "messages": [AIMessage(content=error_msg)],
            "comprehensive_analysis": error_msg,
            "current_step": "practical_advice"
        }

def practical_advice_node(state: TarotState) -> TarotState:
    """실용적 조언 노드"""
    print("🔍 practical_advice_node 실행됨!")
    
    user_concern = state.get("user_concern", "")
    comprehensive_analysis = state.get("comprehensive_analysis", "")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    advice_prompt = f"""
    종합 분석을 바탕으로 실용적이고 구체적인 조언을 제공해주세요.

    **사용자 고민:** {user_concern}
    
    **종합 분석 결과:** {comprehensive_analysis}
    
    다음 형식으로 조언해주세요:
    
    ## 💡 **실용적 조언**
    
    **즉시 실행할 수 있는 행동:**
    [구체적이고 실행 가능한 1-2가지 행동 방안]
    
    **마음가짐의 변화:**
    [어떤 관점이나 태도로 접근하면 좋을지]
    
    **타이밍과 기회:**
    [언제, 어떤 기회를 놓치지 말아야 할지]
    
    **장기적 방향성:**
    [앞으로의 큰 방향과 목표]
    
    ## 🌟 **마무리 메시지**
    [희망적이고 격려하는 한 줄 메시지]
    
    ---
    
    상담이 도움이 되셨나요? 이 결과에 대해 더 궁금한 점이나 다른 고민이 있으시면 언제든 말씀해 주세요. 
    대화를 끝내고 싶으시면 'esc'를 입력해주세요.
    """
    
    try:
        response = llm.invoke([HumanMessage(content=advice_prompt)])
        advice_text = response.content
        
        return {
            "messages": [AIMessage(content=advice_text)],
            "practical_advice": advice_text,
            "current_step": "consultation_complete",
            "needs_followup": True
        }
    except Exception as e:
        error_msg = f"조언 생성 중 오류가 발생했습니다: {str(e)}"
        return {
            "messages": [AIMessage(content=error_msg)],
            "practical_advice": error_msg,
            "current_step": "consultation_complete",
            "needs_followup": True
        }

def followup_node(state: TarotState) -> TarotState:
    """추가 질문 처리 노드 - LLM 기반 꼬리질문 vs 새로운고민 판단"""
    print("🔍 followup_node 실행됨!")
    
    messages = state["messages"]
    user_concern = state.get("user_concern", "")
    
    # 마지막 사용자 메시지 확인
    last_user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_message = msg.content.strip().lower()
            break
    
    if last_user_message == "esc":
        farewell_msg = "🔮 오늘 타로 상담이 도움이 되었기를 바랍니다. 언제든 고민이 생기시면 다시 찾아주세요. 좋은 하루 되세요! ✨"
        return {
            "messages": [AIMessage(content=farewell_msg)],
            "current_step": "ended",
            "needs_followup": False
        }
    
    # 현재 사용자 입력
    original_user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            original_user_message = msg.content.strip()
            break
    
    print(f"🔍 이전 고민: '{user_concern}'")
    print(f"🔍 현재 입력: '{original_user_message}'")
    
    # LLM을 사용한 질문 유형 판단
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    classification_prompt = f"""
    사용자의 이전 고민과 현재 입력을 분석해서 다음 중 하나로 분류해주세요.

    **이전 고민:** "{user_concern}"
    **현재 입력:** "{original_user_message}"

    **분류 기준:**
    1. **FOLLOWUP** - 이전 고민과 관련된 추가 질문이나 세부 사항 문의
       예: "언제쯤 성공할까?", "돈 얼마정도 벌 수 있을까?", "어떤 방법이 좋을까?"
    
    2. **NEW_CONCERN** - 완전히 새로운 주제의 고민이나 상담 요청
       예: "남자친구와 헤어졌어", "직장을 바꿀까?", "건강이 안 좋아"

    **답변 형식:** 반드시 "FOLLOWUP" 또는 "NEW_CONCERN" 중 하나만 답하세요.
    
    분류:"""
    
    try:
        print("🔍 LLM에게 질문 유형 판단 요청...")
        response = llm.invoke([HumanMessage(content=classification_prompt)])
        classification = response.content.strip().upper()
        
        print(f"🔍 LLM 판단 결과: '{classification}'")
        
        if "FOLLOWUP" in classification:
            is_followup_question = True
            is_new_concern = False
        elif "NEW_CONCERN" in classification:
            is_followup_question = False
            is_new_concern = True
        else:
            print(f"🔍 예상치 못한 LLM 응답: {classification}, 기본값(FOLLOWUP) 사용")
            is_followup_question = True
            is_new_concern = False
            
    except Exception as e:
        print(f"🔍 LLM 판단 오류: {e}, 기본값(FOLLOWUP) 사용")
        is_followup_question = True
        is_new_concern = False
    
    print(f"🔍 최종 판단 - 새로운 고민: {is_new_concern}, 꼬리 질문: {is_followup_question}")
    
    if is_new_concern:
        # 새로운 타로 상담 시작
        new_consultation_msg = """🔮 새로운 고민이 있으시군요! 

새로운 타로 상담을 시작하겠습니다. 이전 상담 내용은 저장되었으니 언제든 다시 참고하실 수 있습니다.

고민을 자세히 말씀해주세요. 어떤 상황인지, 무엇이 궁금한지 구체적으로 설명해주시면 더 정확한 상담을 드릴 수 있습니다."""
        
        return {
            "messages": [AIMessage(content=new_consultation_msg)],
            "current_step": "waiting_concern",
            "is_first_conversation": False,
            # 이전 상담 내용 초기화
            "user_concern": "",
            "concern_analysis": {},
            "recommended_spreads": [],
            "selected_spread": None,
            "spread_card_count": 3,
            "selected_cards": [],
            "card_interpretations": [],
            "comprehensive_analysis": "",
            "practical_advice": "",
            "needs_followup": False
        }
    else:
        # 이전 상담에 대한 꼬리 질문으로 처리
        print("🔍 꼬리 질문으로 처리 → detailed_explanation으로 이동")
        return {
            "current_step": "detailed_explanation"
        }

# 문제가 되는 부분을 찾아서 수정
def detailed_explanation_node(state: TarotState) -> TarotState:
    """상세 설명 노드 - 꼬리질문에 대한 상담사 답변"""
    print("🔍 detailed_explanation_node 실행됨!")
    
    messages = state["messages"]
    user_concern = state.get("user_concern", "")
    comprehensive_analysis = state.get("comprehensive_analysis", "")
    card_interpretations = state.get("card_interpretations", [])
    
    # 마지막 사용자 질문
    last_user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_message = msg.content
            break
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    llm_with_tools = llm.bind_tools([search_tarot_cards, search_tarot_spreads])
    
    # f-string 내에서 join() 사용시 백슬래시 문제 해결
    card_interpretations_text = "\n".join(card_interpretations)
    
    detailed_prompt = f"""
    당신은 전문 타로 상담사입니다. 사용자가 이전 타로 상담 결과에 대해 추가 질문을 했습니다.

    **원래 고민:** "{user_concern}"
    **사용자 추가 질문:** "{last_user_message}"
    
    **이전 타로 해석:**
    {comprehensive_analysis}

    **개별 카드 해석들:**
    {card_interpretations_text}

    **답변 방식:**
    1. 이전 타로 해석을 바탕으로 답변하세요
    2. 필요하면 추가 카드 상담을 제안하세요
    3. 타로 상담사 말투로 공감하며 답변하세요
    4. 구체적이고 실용적인 조언을 포함하세요

    **출력 형식:**

    🔮 **타로 상담사가 답합니다**

    **질문에 대한 답변:**
    [사용자 질문에 직접적으로 답변하며, 이전 카드 해석을 근거로 제시]

    **카드가 말하는 바:**
    [이전 해석에서 관련된 부분을 인용하며 설명]

    **추가 조언:**
    [상황에 맞는 구체적 행동 지침]

    **추가 상담 제안:**
    [필요시] "이 부분에 대해 더 자세히 알고 싶으시다면, 추가 카드를 뽑아보는 것을 권합니다. 어떻게 하시겠어요?"

    타로 상담사답게 따뜻하고 지혜로운 톤으로 답변하세요.
    """
    
    try:
        response = llm_with_tools.invoke([HumanMessage(content=detailed_prompt)])
        
        return {
            "messages": [response],
            "current_step": "consultation_complete"
        }
    except Exception as e:
        error_msg = f"상세 설명 중 오류가 발생했습니다: {str(e)}"
        return {
            "messages": [AIMessage(content=error_msg)],
            "current_step": "consultation_complete"
        }

# =================================================================
# 5. 조건부 엣지 (라우팅) 함수들 (수정됨)
# =================================================================

def should_analyze_concern(state: TarotState) -> str:
    """고민 분석 여부 결정"""
    messages = state.get("messages", [])
    current_step = state.get("current_step", "")
    
    print(f"🔍 should_analyze_concern - current_step: {current_step}")
    print(f"🔍 should_analyze_concern - messages 개수: {len(messages)}")
    
    # greeting 직후 waiting_concern 상태에서만 고민 분석 진행
    if current_step == "waiting_concern":
        # 마지막 메시지가 사용자 메시지인지 확인
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_input = msg.content.strip()
                print(f"🔍 사용자 입력: '{user_input}'")
                
                # 스프레드 선택 번호나 카드 번호가 아닌 실제 고민인지 확인
                if len(user_input) > 3 and not user_input.isdigit():  # 실제 고민 문장
                    print("🔍 실제 고민 발견 - analyze_concern으로 이동")
                    return "analyze_concern"
                else:
                    print("🔍 고민이 아닌 입력 - 대기")
                    return "wait_for_input"
        print("🔍 사용자 메시지 없음 - 대기")
        return "wait_for_input"
    
    return "wait_for_input"

def should_continue_after_spread_search(state: TarotState) -> str:
    """스프레드 검색 후 도구 호출 여부 확인"""
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    
    print(f"🔍 도구 호출 확인 - last_message: {last_message}")
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print(f"🔍 도구 호출 감지: {last_message.tool_calls}")
        return "tools"
    else:
        print("🔍 도구 호출 없음, recommend_spreads로 이동")
        return "recommend_spreads"

def should_select_cards(state: TarotState) -> str:
    """카드 선택 여부 결정"""
    current_step = state.get("current_step", "")
    messages = state.get("messages", [])
    
    print(f"🔍 should_select_cards - current_step: {current_step}")
    
    if current_step == "waiting_spread_selection":
        # 사용자가 스프레드를 선택했는지 확인
        # 스프레드 추천 이후의 메시지만 확인
        user_selected = False
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_input = msg.content.strip()
                print(f"🔍 사용자 입력 확인: '{user_input}'")
                if any(num in user_input for num in ["1", "2", "3"]):
                    print("🔍 스프레드 선택됨 - select_cards로 이동")
                    user_selected = True
                    break
                # 다른 입력이면 더 이상 확인하지 않음
                break
        
        if not user_selected:
            print("🔍 스프레드 선택 대기")
            return "wait_for_selection"
        else:
            return "select_cards"
    
    return "wait_for_selection"

def should_process_cards(state: TarotState) -> str:
    """카드 처리 여부 결정"""
    current_step = state.get("current_step", "")
    messages = state.get("messages", [])
    
    print(f"🔍 should_process_cards - current_step: {current_step}")
    
    if current_step == "waiting_card_numbers":
        # 사용자가 카드 번호를 입력했는지 확인
        # 카드 선택 안내 이후의 메시지만 확인
        user_selected = False
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_input = msg.content.strip()
                print(f"🔍 사용자 입력 확인: '{user_input}'")
                # 숫자나 쉼표가 포함되어 있으면 카드 번호로 간주
                if any(char.isdigit() or char == ',' for char in user_input):
                    print("🔍 카드 번호 입력됨 - process_cards로 이동")
                    user_selected = True
                    break
                # 다른 입력이면 더 이상 확인하지 않음
                break
        
        if not user_selected:
            print("🔍 카드 번호 입력 대기")
            return "wait_for_cards"
        else:
            return "process_cards"
    
    return "wait_for_cards"

def route_after_card_interpretation(state: TarotState) -> str:
    """카드 해석 후 라우팅"""
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    else:
        return "analyze_comprehensive"

def route_after_followup(state: TarotState) -> str:
    """추가 질문 처리 후 라우팅"""
    current_step = state.get("current_step", "")
    
    if current_step == "ended":
        return END
    elif current_step == "detailed_explanation":
        return "explain_details"
    elif current_step == "waiting_concern":
        return "wait_for_new_concern"
    else:
        return "handle_followup"

def route_after_detailed_explanation(state: TarotState) -> str:
    """상세 설명 후 라우팅"""
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    else:
        return "handle_followup"

def route_after_tools(state: TarotState) -> str:
    """도구 실행 후 라우팅"""
    current_step = state.get("current_step", "")
    
    if current_step == "spread_search":
        return "recommend_spreads"
    elif current_step == "card_interpretation":
        return "analyze_comprehensive"
    elif current_step == "detailed_explanation":
        return "handle_followup"
    else:
        return "recommend_spreads"

# =================================================================
# 6. LangGraph 구성 및 실행 (수정됨)
# =================================================================

def create_tarot_graph():
    """타로 LangGraph 생성"""
    
    workflow = StateGraph(TarotState)
    
    # 노드들 추가 (상태 키와 중복되지 않도록 수정)
    workflow.add_node("greeting", greeting_node)
    workflow.add_node("analyze_concern", concern_analysis_node)  # 이름 변경
    workflow.add_node("search_spreads", spread_search_node)  # 이름 변경
    workflow.add_node("recommend_spreads", spread_recommendation_node)  # 이름 변경
    workflow.add_node("select_cards", card_selection_node)  # 이름 변경
    workflow.add_node("process_cards", card_processing_node)  # 이름 변경
    workflow.add_node("interpret_cards", card_interpretation_node)  # 이름 변경
    workflow.add_node("analyze_comprehensive", comprehensive_analysis_node)  # 이름 변경
    workflow.add_node("give_advice", practical_advice_node)  # 이름 변경
    workflow.add_node("handle_followup", followup_node)  # 이름 변경
    workflow.add_node("explain_details", detailed_explanation_node)  # 이름 변경
    
    # 도구 노드 추가 (수정됨)
    tools = [search_tarot_spreads, search_tarot_cards, search_spread_positions]
    tool_node = ToolNode(tools)
    workflow.add_node("tools", tool_node)
    
    # 시작점 설정
    workflow.add_edge(START, "greeting")
    
    # 기본 엣지들 (노드 이름 수정)
    workflow.add_edge("analyze_concern", "search_spreads")
    workflow.add_edge("process_cards", "interpret_cards")
    workflow.add_edge("analyze_comprehensive", "give_advice")
    workflow.add_edge("give_advice", "handle_followup")
    
    # 조건부 엣지들 (수정됨)
    workflow.add_conditional_edges(
        "greeting",
        should_analyze_concern,
        {
            "analyze_concern": "analyze_concern",
            "wait_for_input": END  # 사용자 입력 대기
        }
    )
    
    # 다른 상태에서도 사용자 입력 대기하도록 추가
    workflow.add_conditional_edges(
        "recommend_spreads", 
        lambda state: "select_cards" if state.get("current_step") == "waiting_spread_selection" else END,
        {
            "select_cards": END,  # 사용자 스프레드 선택 대기
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "select_cards",
        lambda state: "process_cards" if state.get("current_step") == "waiting_card_numbers" else END,
        {
            "process_cards": END,  # 사용자 카드 번호 입력 대기
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "search_spreads",
        should_continue_after_spread_search,
        {
            "tools": "tools",
            "recommend_spreads": "recommend_spreads"
        }
    )
    
    workflow.add_conditional_edges(
        "interpret_cards",
        route_after_card_interpretation,
        {
            "tools": "tools",
            "analyze_comprehensive": "analyze_comprehensive"
        }
    )
    
    workflow.add_conditional_edges(
        "handle_followup",
        route_after_followup,
        {
            END: END,
            "explain_details": "explain_details",
            "wait_for_new_concern": END,  # 새로운 고민 입력 대기
            "handle_followup": "handle_followup"
        }
    )
    
    workflow.add_conditional_edges(
        "explain_details",
        route_after_detailed_explanation,
        {
            "tools": "tools",
            "handle_followup": "handle_followup"
        }
    )
    
    workflow.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "recommend_spreads": "recommend_spreads",
            "analyze_comprehensive": "analyze_comprehensive",
            "handle_followup": "handle_followup"
        }
    )
    
    return workflow

def main():
    """메인 실행 함수 (수정됨)"""
    print("🔮 타로 LangGraph 시스템을 초기화하는 중...")
    
    # RAG 시스템 초기화
    try:
        initialize_rag_system()
    except Exception as e:
        print(f"❌ RAG 시스템 초기화 실패: {str(e)}")
        return
    
    # 그래프 생성 및 컴파일
    try:
        workflow = create_tarot_graph()
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
        
        print("✅ 초기화 완료!")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ 그래프 초기화 실패: {str(e)}")
        return
    
    # 대화형 실행 (수정됨)
    config = {"configurable": {"thread_id": "tarot_session_1"}}
    
    try:
        # 초기 상태 설정
        initial_state = {
            "messages": [],
            "current_step": "greeting",
            "is_first_conversation": True,
            "user_concern": "",
            "concern_analysis": {},
            "search_results": [],
            "recommended_spreads": [],
            "selected_spread": None,
            "spread_card_count": 3,
            "user_card_input": "",
            "selected_cards": [],
            "card_interpretations": [],
            "comprehensive_analysis": "",
            "practical_advice": "",
            "needs_followup": False
        }
        
        print("🔍 첫 인사를 시작합니다...")
        
        # 첫 인사 실행
        result = app.invoke(initial_state, config)
        
        # AI 메시지 출력
        if result.get("messages"):
            last_message = result["messages"][-1]
            if isinstance(last_message, AIMessage):
                print(f"\n🔮 타로 상담사: {last_message.content}")
        
        # 메인 대화 루프 (완전 수정됨)
        while True:
            current_step = result.get("current_step", "")
            print(f"\n🔍 현재 단계: {current_step}")
            
            if current_step == "ended":
                print("상담이 종료되었습니다.")
                break
            
            # 사용자 입력 받기
            user_input = input("\n사용자: ").strip()
            if not user_input:
                print("입력을 해주세요.")
                continue
                
            if user_input.lower() in ['quit', 'exit', '종료']:
                print("대화를 종료합니다.")
                break
            
            # 현재 상태에 따라 다르게 처리
            if current_step == "waiting_concern":
                # 고민 입력 단계
                new_message = HumanMessage(content=user_input)
                result["messages"].append(new_message)
                
                # 고민 분석 진행
                result = app.invoke(result, config)
                
            elif current_step == "waiting_spread_selection":
                # 스프레드 선택 단계
                if user_input in ["1", "2", "3"]:
                    new_message = HumanMessage(content=user_input)
                    result["messages"].append(new_message)
                    
                    # 카드 선택 노드 실행
                    updated_result = card_selection_node(result)
                    result.update(updated_result)
                else:
                    print("1, 2, 3 중에서 선택해주세요.")
                    continue
                    
            elif current_step == "waiting_card_numbers":
                # 카드 번호 입력 단계
                if any(char.isdigit() or char == ',' for char in user_input):
                    new_message = HumanMessage(content=user_input)
                    result["messages"].append(new_message)
                    
                    # 5단계: 카드 처리 (카드 추출 및 안내) - 중복 체크 포함
                    print("🔍 5단계: 카드 추출 및 안내")
                    card_result = card_processing_node(result)
                    result.update(card_result)
                    
                    # 중복이나 오류가 있으면 다시 입력 대기
                    if result.get("current_step") == "waiting_card_numbers":
                        # 오류 메시지 출력
                        for msg in result.get("messages", []):
                            if isinstance(msg, AIMessage) and msg.content and not hasattr(msg, '_printed'):
                                print(f"\n🔮 타로 상담사: {msg.content}")
                                msg._printed = True
                        continue  # 다시 입력 받기
                    
                    # 새로 추가된 메시지 출력 (카드 추출 안내)
                    for msg in result.get("messages", []):
                        if isinstance(msg, AIMessage) and msg.content and not hasattr(msg, '_printed'):
                            print(f"\n🔮 타로 상담사: {msg.content}")
                            msg._printed = True
                    
                    # 6단계: 카드 해석
                    print("🔍 6단계: 카드 해석")
                    interpret_result = card_interpretation_node(result)
                    result.update(interpret_result)
                    
                    # 새로 추가된 메시지 출력 (카드 해석)
                    for msg in result.get("messages", []):
                        if isinstance(msg, AIMessage) and msg.content and not hasattr(msg, '_printed'):
                            print(f"\n🔮 타로 상담사: {msg.content}")
                            msg._printed = True
                    
                    # 7단계: 종합 분석
                    print("🔍 7단계: 종합 분석")
                    analysis_result = comprehensive_analysis_node(result)
                    result.update(analysis_result)
                    
                    # 새로 추가된 메시지 출력 (종합 분석)
                    for msg in result.get("messages", []):
                        if isinstance(msg, AIMessage) and msg.content and not hasattr(msg, '_printed'):
                            print(f"\n🔮 타로 상담사: {msg.content}")
                            msg._printed = True
                    
                    # 8단계: 실용적 조언
                    print("🔍 8단계: 실용적 조언")
                    advice_result = practical_advice_node(result)
                    result.update(advice_result)
                    
                    # 새로 추가된 메시지 출력 (실용적 조언)
                    for msg in result.get("messages", []):
                        if isinstance(msg, AIMessage) and msg.content and not hasattr(msg, '_printed'):
                            print(f"\n🔮 타로 상담사: {msg.content}")
                            msg._printed = True
                    
                    # 9단계: 상담 완료 상태로 변경
                    result["current_step"] = "consultation_complete"
                    
                else:
                    print("카드 번호를 입력해주세요. (예: 7, 23, 45)")
                    continue
                    
            elif current_step == "consultation_complete":
           # 상담 완료 후 추가 질문 처리
                new_message = HumanMessage(content=user_input)
                result["messages"].append(new_message)
    
           # followup_node 실행 후 라우팅
                followup_result = followup_node(result)
                result.update(followup_result)
    
              # 새로운 상담 시작인지 확인
                if result.get("current_step") == "waiting_concern":
                # 새로운 상담이면 concern_analysis_node 실행
                   pass  # 다음 루프에서 처리
                elif result.get("current_step") == "detailed_explanation":
                  # 꼬리 질문이면 detailed_explanation_node 실행
                  detailed_result = detailed_explanation_node(result)
                  result.update(detailed_result)
                    
            else:
                # 기타 상황
                new_message = HumanMessage(content=user_input)
                result["messages"].append(new_message)
                result = app.invoke(result, config)
            
            # 새로 추가된 AI 메시지 출력 (기타 경우만)
            if current_step not in ["waiting_card_numbers"]:  # 카드 번호 입력 단계는 위에서 개별 처리
                current_messages = result.get("messages", [])
                for msg in current_messages:
                    if isinstance(msg, AIMessage) and msg.content and not hasattr(msg, '_printed'):
                        print(f"\n🔮 타로 상담사: {msg.content}")
                        msg._printed = True  # 출력 완료 표시
                
    except Exception as e:
        print(f"❌ 실행 중 오류: {str(e)}")
        import traceback
        traceback.print_exc()

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    else:
        return obj

def clean_spread_name(spread_name):
    # 괄호 안에 (숫자 CARDS) 패턴 제거
    return re.sub(r'\s*\(\d+\s*CARDS?\)', '', spread_name, flags=re.IGNORECASE).strip()

def normalize_spread_name(spread_name: str) -> str:
    if not spread_name:
        return ""
    normalized = spread_name.lower()
    normalized = re.sub(r'^the ', '', normalized)
    normalized = re.sub(r' tarot card spread$', '', normalized)
    normalized = re.sub(r' tarot card$', '', normalized)
    normalized = re.sub(r' spread$', '', normalized)
    normalized = re.sub(r'[^a-z0-9 ]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

if __name__ == "__main__":
    print("🚀 스크립트 시작됨!")
    
    print("=" * 60)
    print("🔮 TAROT LANGGRAPH SYSTEM")
    print("=" * 60)
    print("필요한 환경 변수:")
    print("- OPENAI_API_KEY: OpenAI API 키")
    print()
    print("필요한 파일들:")
    print("- tarot_rag_system.py (기존 RAG 시스템)")
    print("- tarot_card_faiss_index/ (카드 FAISS 인덱스)")
    print("- tarot_spread_faiss_index/ (스프레드 FAISS 인덱스)")
    print("=" * 60)
    
    try:
        main()
    except Exception as e:
        print(f"❌ main() 함수에서 오류 발생: {e}")
        import traceback
        traceback.print_exc()