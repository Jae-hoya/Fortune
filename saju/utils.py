"""
FortuneAI 유틸리티 함수들
UI, 쿼리 처리, 디스플레이 관련 모든 기능 통합
"""

import os
import sys
import time
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
# from langchain_teddynote.messages import stream_graph  # 더 이상 사용하지 않음


# ================================
# UI / 디스플레이 관련 함수들
# ================================

def print_banner():
    """시스템 배너 출력"""
    print("=" * 70)
    print("🔮 FortuneAI - LangGraph 사주 시스템 🔮")
    print("=" * 70)
    print("✨ Supervisor 패턴 기반 고성능 사주 계산기")
    print("🚀 LangGraph 멀티 워커 시스템")
    print("-" * 70)
    print("🏗️  시스템 구조:")
    print("  • Supervisor → SajuExpert / Search / GeneralAnswer")
    print("  • 사주계산: calculate_saju_tool")
    print("  • 통합검색: saju_retriever_tool + tavily_tool + duck_tool")
    print("  • 일반QA: general_qa_tool (Google Gemini)")
    print("-" * 70)
    print("📝 사용법:")
    print("  • 사주 계산: '1995년 8월 26일 오전 10시 15분 남자 사주'")
    print("  • 운세 상담: '1995년 8월 26일생 2024년 연애운'")
    print("  • 일반 검색: '사주에서 십신이란?'")
    print("  • 종료: 'quit' 또는 'exit'")
    print("  • 성능분석: '--debug' 또는 'debug:질문' (실행시간 분석)")
    print("=" * 70)


def print_system_info():
    """시스템 정보 출력"""
    print("\n🔧 시스템 정보:")
    print(f"  • 실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  • Python 버전: {sys.version.split()[0]}")
    print(f"  • 작업 디렉토리: {os.getcwd()}")
    print(f"  • 워커 노드: Supervisor, SajuExpert(manse+retriever), WebTool, GeneralQA")
    print(f"  • 출력: 상세 워크플로 표시 / debug명령어로 성능 분석 추가")
    print()


def format_response(response: str) -> str:
    """응답 포맷팅"""
    if not response:
        return "❌ 응답을 생성할 수 없습니다."
    
    # 응답 앞에 구분선 추가
    formatted = "\n" + "🎯 " + "=" * 55 + "\n"
    formatted += "📋 **FortuneAI 분석 결과**\n"
    formatted += "=" * 58 + "\n\n"
    formatted += response
    formatted += "\n\n" + "=" * 58
    
    return formatted


def print_help():
    """도움말 출력"""
    print("""
📚 **FortuneAI 사용 가이드**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔮 **사주 계산**: '1995년 8월 26일 오전 10시 15분 남자 사주'
📖 **사주 해석**: '사주에서 십신이란 무엇인가요?'
🌐 **일반 질문**: '2024년 갑진년의 특징은?'

🛠️  **명령어**:
  • new, clear      : 새로운 세션 시작
  • help, ?         : 도움말 보기
  • quit, exit      : 프로그램 종료
  • debug:질문      : 성능 분석 모드로 실행

🏗️  **워크플로 구조**:
  1. Supervisor: 질문 분석 후 적절한 에이전트로 라우팅
  2. SajuExpert: 사주 계산 전담 → calculate_saju_tool
  3. Search: 통합 검색 → saju_retriever_tool + tavily_tool + duck_tool
  4. GeneralAnswer: 비사주 질문 → general_qa_tool (Google Gemini)

🎯 **출력 방식**:
  • 기본: 모든 노드의 상세한 실행 과정과 툴 정보 표시
  • debug: 추가로 성능 분석 및 실행 시간 상세 정보 제공

🔧 **사용 가능한 툴**:
  • calculate_saju_tool: 사주팔자 계산
  • saju_retriever_tool: 사주 지식 벡터DB 검색
  • tavily_tool, duck_tool: 웹 검색
  • general_qa_tool: Google Gemini 기반 일반 QA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)


def print_node_header(node_name: str, is_debug: bool = False):
    """노드 헤더 출력 - 모드에 따라 다르게 표시"""
    if is_debug:
        # 디버그 모드: 상세한 설명
        print("\n" + "=" * 60)
        
        node_descriptions = {
            "Supervisor": "🎯 워크플로 관리자 - 적절한 에이전트로 라우팅",
            "SajuExpert": "🔮 사주 전문가 - 사주팔자 계산 전담",
            "Search": "🔍 통합 검색기 - RAG 검색 + 웹 검색",
            "GeneralAnswer": "💬 일반 QA - Google Gemini 모델 사용"
        }
        
        description = node_descriptions.get(node_name, "🔧 시스템 노드")
        print(f"🔄 Node: \033[1;36m{node_name}\033[0m")
        print(f"📝 {description}")
        print("- " * 30)
    else:
        # 기본 모드: 간단하고 스트리밍 친화적
        node_info = {
            "SajuExpert": ("🔮", "사주 전문가"),
            "Search": ("🔍", "통합 검색"),
            "GeneralAnswer": ("💬", "일반 상담")
        }
        
        icon, name = node_info.get(node_name, ("🔧", node_name))
        print(f"\n{icon} {name} 실시간 응답:")
        print("─" * 30)


def print_simple_node_info(node_name: str, current_time: str = None):
    """기본 모드: 간단한 노드 정보 표시 (시간 포함)"""
    node_info = {
        "Supervisor": "🎯 워크플로 관리",
        "SajuExpert": "🔮 사주 전문가",
        "Search": "🔍 통합 검색",
        "GeneralAnswer": "💬 일반 상담"
    }
    
    info = node_info.get(node_name, f"🔧 {node_name}")
    time_str = f" ({current_time})" if current_time else ""
    print(f"\n{info} 중...{time_str}")


def print_node_execution(node_name: str):
    """디버그 모드: 상세한 노드 실행 정보와 사용 툴 표시"""
    node_tool_info = {
        "Supervisor": ("🎯", "라우팅", "워크플로 관리"),
        "SajuExpert": ("🔮", "사주계산", "calculate_saju_tool"),
        "Search": ("🔍", "통합검색", "saju_retriever_tool + tavily_tool + duck_tool"),
        "GeneralAnswer": ("💬", "일반상담", "general_qa_tool (Google Gemini)")
    }
    
    icon, action, tools = node_tool_info.get(node_name, ("🔧", node_name, "unknown"))
    
    print(f"\n{icon} {action} 노드 실행")
    print(f"  🛠️  사용 툴: {tools}")
    print("─" * 40)


def print_completion(is_debug: bool = False):
    """완료 메시지 출력"""
    if is_debug:
        print("\n" + "=" * 60)
        print("✅ 디버그 모드 완료! (전체 워크플로 + 성능 분석)")
        print("📊 모든 노드의 상세 정보가 표시되었습니다")
        print("=" * 60)
    else:
        print("\n" + "─" * 30)
        print("✅ 스트리밍 완료!")
        print("═" * 40)


# ================================
# 쿼리 처리 관련 함수들
# ================================

def handle_debug_query(query: str, app, conversation_history: list, session_start_time: str, session_id: str) -> str:
    """성능 분석 쿼리 처리"""
    if not query.startswith("debug:"):
        return None
    
    actual_query = query[6:].strip()
    if not actual_query:
        return "❌ 성능 분석할 질문을 입력해주세요. 예: debug:1995년 8월 26일 사주"
    
    print(f"\n🔍 성능 분석 모드로 실행 중: '{actual_query}'")
    print("-" * 50)
    
    start_time = time.time()
    response = run_query_with_app(actual_query, app, conversation_history, session_start_time, session_id)
    execution_time = time.time() - start_time
    
    analysis_info = f"""
📊 **성능 분석 결과**
• 실행 시간: {execution_time:.2f}초
• 질문: {actual_query}
• 워크플로: Supervisor → 전문 에이전트 → 응답 생성

📋 **최종 응답**
{response}

⚡ **성능 정보**
• 총 처리 시간: {execution_time:.2f}초
• 메모리 사용: 체크포인터 활용한 상태 관리
"""
    return analysis_info


def run_query_with_app(query: str, app, conversation_history: list, session_start_time: str, session_id: str) -> str:
    """상세 스트리밍 모드: 모든 노드 + 상세 정보 + 툴 추적"""
    print(f"🔍 쿼리 실행: {query}")
    
    # 새로운 사용자 메시지를 히스토리에 추가
    conversation_history.append(HumanMessage(content=query))
    
    # 현재 상태 설정 (세션 정보 유지, 현재 시간만 갱신)
    current_state = {
        "question": query,
        "messages": conversation_history.copy(),
        "next": "",
        "session_start_time": session_start_time,  # 세션 시작 시간 (고정)
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 현재 쿼리 시간
        "session_id": session_id  # 세션 ID (고정)
    }
    
    # 설정 생성 (Checkpointer용)
    config = {
        "configurable": {
            "thread_id": session_id
        }
    }
    
    print("🚀 AI 워크플로 실행 중...")
    
    # 동기 스트림 처리 (stream_mode="updates")
    final_response = ""
    
    for chunk in app.stream(current_state, config=config, stream_mode="updates"):
        # chunk는 dictionary 형태 (key: 노드, value: 노드의 상태 값)
        for node, value in chunk.items():
            if node:
                print_node_header(node, is_debug=True)
                print_node_execution(node)
                print()
            
            # final_answer가 있으면 출력하고 저장
            if "final_answer" in value and value["final_answer"]:
                final_answer = value["final_answer"]
                print(final_answer)
                final_response = final_answer
            
            # messages가 있으면 마지막 메시지 출력
            elif "messages" in value and value["messages"]:
                last_message = value["messages"][-1]
                if hasattr(last_message, 'content') and last_message.content:
                    print(last_message.content)
                    if not final_response:  # final_answer가 없으면 마지막 메시지를 응답으로 사용
                        final_response = last_message.content
    
    print_completion(is_debug=False)
    
    # 최종 응답이 없으면 기본 메시지
    if not final_response:
        final_response = "응답을 생성하지 못했습니다."
    
    conversation_history.append(AIMessage(content=final_response))
    return final_response


def get_node_tools(node_name: str) -> str:
    """노드별 사용 툴 반환"""
    node_tools = {
        "Supervisor": "워크플로 관리",
        "SajuExpert": "calculate_saju_tool",
        "Search": "saju_retriever_tool + tavily_tool + duck_tool",
        "GeneralAnswer": "general_qa_tool (Google Gemini)"
    }
    return node_tools.get(node_name, "unknown")



 