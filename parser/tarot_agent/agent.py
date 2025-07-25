# LangGraph 메인 에이전트 및 실행

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

from Fortune.parser.tarot_rag_system import TarotRAGSystem

# 웹 검색 관련 imports 제거됨

# 내부 모듈 imports

from .utils.state import TarotState

from .utils.tools import initialize_rag_system, search_tarot_spreads, search_tarot_cards

from .utils.nodes import (

    state_classifier_node, supervisor_master_node, unified_processor_node, unified_tool_handler_node,

    state_router, processor_router

)

from .utils.nodes import (

    supervisor_llm_node, classify_intent_node, card_info_handler, spread_info_handler, simple_card_handler,

    consultation_handler, emotion_analyzer_node, perform_multilayer_spread_search,

    spread_recommender_node, consultation_router, spread_extractor_node, situation_analyzer_node, card_count_inferrer_node,

    status_determiner_node, specific_consultation_router, general_handler, unknown_handler, consultation_flow_handler,

    start_actual_consultation, consultation_continue_handler, consultation_summary_handler, consultation_individual_handler,

    consultation_final_handler, context_reference_handler, handle_casual_new_question, handle_tarot_related_question,

    extract_question_topic, exception_handler, emotional_support_handler, start_specific_spread_consultation, tool_result_handler

)

from .utils.timing import ensure_temporal_context

from .utils.helpers import convert_numpy_types

from .utils.translation import translate_korean_to_english_with_llm

# =================================================================

# 최적화된 그래프 생성 함수

# =================================================================

def create_optimized_tarot_graph():
    """🆕 최적화된 타로 그래프 - 기존 함수들 100% 재사용"""
    workflow = StateGraph(TarotState)
    workflow.add_node("state_classifier", state_classifier_node)
    workflow.add_node("supervisor_master", supervisor_master_node)
    workflow.add_node("unified_processor", unified_processor_node)
    workflow.add_node("unified_tool_handler", unified_tool_handler_node)
    workflow.add_edge(START, "state_classifier")
    workflow.add_conditional_edges(
        "state_classifier",
        state_router,
        {
            "consultation_direct": "unified_processor",
            "context_reference_direct": "unified_processor",
            "supervisor_master": "supervisor_master"
        }
    )
    workflow.add_edge("supervisor_master", "unified_processor")
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

# 메인 실행 함수

# =================================================================

def main():
    """🆕 최적화된 메인 실행 함수"""
    print("🔮 최적화된 타로 시스템을 초기화하는 중...")
    global rag_system
    try:
        initialize_rag_system()
        print("✅ RAG 시스템 초기화 성공!")
    except Exception as e:
        print(f"⚠️ RAG 시스템 초기화 실패: {e}")
        print("📝 기본 모드로 계속 진행합니다...")
        rag_system = None
    try:
        app = create_optimized_tarot_graph().compile()
        print("✅ 최적화된 타로 시스템 초기화 완료!")
        print("🚀 Fast Track 기능으로 멀티턴 성능 대폭 향상!")
        print("=" * 50)
    except Exception as e:
        print(f"❌ 그래프 초기화 실패: {e}")
        return
    current_state = {
        "messages": [AIMessage(content="🔮 안녕하세요! 타로 상담사입니다. 오늘은 어떤 도움이 필요하신가요?")],
        "user_intent": "unknown",
        "user_input": "",
        "consultation_data": None,
        "supervisor_decision": None
    }
    first_message = current_state["messages"][0]
    print(f"\n🔮 타로 상담사: {first_message.content}")
    while True:
        user_input = input("\n사용자: ").strip()
        if user_input.lower() in ['quit', 'exit', '종료', '끝', '그만', 'bye']:
            print("🔮 타로 상담이 도움이 되었기를 바랍니다. 좋은 하루 되세요! ✨")
            break
        if not user_input:
            print("💬 무엇이든 편하게 말씀해주세요!")
            continue
        current_state["messages"].append(HumanMessage(content=user_input))
        current_state["user_input"] = user_input
        import time
        start_time = time.time()
        try:
            result = app.invoke(current_state)
            current_state = result
            end_time = time.time()
            response_time = end_time - start_time
            messages = current_state.get("messages", [])
            if messages:
                last_message = messages[-1]
                if isinstance(last_message, AIMessage):
                    print(f"\n🔮 타로 상담사: {last_message.content}")
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
