"""
FortuneAI 유틸리티 함수들
UI, 쿼리 처리, 디스플레이 관련 모든 기능 통합
"""

import os
import sys
import time
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage


# ================================
# UI / 디스플레이 관련 함수들
# ================================

def print_banner():
    """시스템 배너 출력"""
    print("=" * 70)
    print("🔮 FortuneAI - LangGraph 사주 시스템 🔮")
    print("=" * 70)
    print("✨ Supervisor 패턴 기반 고성능 사주 계산기")
    print("🎯 98점 전문가 검증 완료")
    print("🚀 LangGraph 멀티 워커 시스템")
    print("-" * 70)
    print("🏗️  시스템 구조:")
    print("  • Supervisor → SajuExpert(manse + retriever) / WebTool / GeneralQA")
    print("  • 사주계산: calculate_saju_tool")
    print("  • RAG검색: saju_retriever_tool") 
    print("  • 웹검색: tavily_tool, duck_tool")
    print("  • 일반QA: general_qa_tool (Google Gemini)")
    print("-" * 70)
    print("📝 사용법:")
    print("  • 사주 계산: '1995년 8월 26일 오전 10시 15분 남자 사주'")
    print("  • 운세 상담: '1995년 8월 26일생 2024년 연애운'")
    print("  • 일반 검색: '사주에서 십신이란?'")
    print("  • 종료: 'quit' 또는 'exit'")
    print("  • 디버그: '--debug' 또는 'debug:질문' (상세 개발자 모드)")
    print("=" * 70)


def print_system_info():
    """시스템 정보 출력"""
    print("\n🔧 시스템 정보:")
    print(f"  • 실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  • Python 버전: {sys.version.split()[0]}")
    print(f"  • 작업 디렉토리: {os.getcwd()}")
    print(f"  • 워커 노드: Supervisor, SajuExpert(manse+retriever), WebTool, GeneralQA")
    print(f"  • 모드: 기본(주요 노드만) / 디버그(전체 노드 + 성능 분석)")
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
  • debug:질문      : 디버그 모드로 실행

🏗️  **워크플로 구조**:
  1. Supervisor: 질문 분석 후 적절한 에이전트로 라우팅
  2. SajuExpert: 사주 관련 → manse(계산) + retriever(RAG검색)
  3. WebTool: 일반 사주 개념 → tavily_tool, duck_tool
  4. GeneralQA: 비사주 질문 → general_qa_tool (Google Gemini)

🎯 **모드 설명**:
  • 기본 모드: 주요 작업 노드만 깔끔하게 표시 (사용자 친화적)
  • 디버그 모드: 모든 노드 + 성능 분석 (개발자용)

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
            "SajuExpert": "🔮 사주 전문가 - 만세력 계산 + RAG 검색",
            "manse": "📅 만세력 계산기 - 사주팔자 계산 툴 사용",
            "retriever": "🔍 RAG 검색기 - 사주 지식 벡터DB 검색",
            "WebTool": "🌐 웹 검색기 - Tavily/DuckDuckGo 검색 툴 사용",
            "GeneralQA": "💬 일반 QA - Google Gemini 모델 사용"
        }
        
        description = node_descriptions.get(node_name, "🔧 시스템 노드")
        print(f"🔄 Node: \033[1;36m{node_name}\033[0m")
        print(f"📝 {description}")
        print("- " * 30)
    else:
        # 기본 모드: 간단하고 스트리밍 친화적
        node_info = {
            "SajuExpert": ("🔮", "사주 전문가"),
            "manse": ("📅", "만세력 계산"),
            "retriever": ("🔍", "지식 검색"), 
            "WebTool": ("🌐", "웹 검색"),
            "GeneralQA": ("💬", "일반 상담")
        }
        
        icon, name = node_info.get(node_name, ("🔧", node_name))
        print(f"\n{icon} {name} 실시간 응답:")
        print("─" * 30)


def print_simple_node_info(node_name: str):
    """기본 모드: 간단한 노드 정보 표시"""
    node_info = {
        "SajuExpert": "🔮 사주 전문가",
        "manse": "📅 만세력 계산", 
        "retriever": "🔍 지식 검색",
        "WebTool": "🌐 웹 검색",
        "GeneralQA": "💬 일반 상담"
    }
    
    info = node_info.get(node_name, f"🔧 {node_name}")
    print(f"\n{info} 중...")


def print_node_execution(node_name: str):
    """디버그 모드: 상세한 노드 실행 정보와 사용 툴 표시"""
    node_tool_info = {
        "Supervisor": ("🎯", "라우팅", "워크플로 관리"),
        "SajuExpert": ("🔮", "사주분석", "manse + retriever 서브그래프"),
        "manse": ("📅", "만세력계산", "calculate_saju_tool"),
        "retriever": ("🔍", "지식검색", "saju_retriever_tool"),
        "WebTool": ("🌐", "웹검색", "tavily_tool + duck_tool"),
        "GeneralQA": ("💬", "일반상담", "general_qa_tool (Google Gemini)")
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

def handle_debug_query(query: str, app, conversation_history: list) -> str:
    """디버그 쿼리 처리"""
    if not query.startswith("debug:"):
        return None
    
    actual_query = query[6:].strip()
    if not actual_query:
        return "❌ 디버그할 질문을 입력해주세요. 예: debug:1995년 8월 26일 사주"
    
    print(f"\n🔍 디버그 모드로 실행 중: '{actual_query}'")
    print("-" * 50)
    
    start_time = time.time()
    response = run_query_with_debug(actual_query, app, conversation_history)
    execution_time = time.time() - start_time
    
    debug_info = f"""
🔍 **디버그 분석 결과**
• 실행 시간: {execution_time:.2f}초
• 질문: {actual_query}
• 노드 경로: Supervisor → 전문 에이전트 → 응답 생성

📋 **최종 응답**
{response}

⚡ **성능 정보**
• 총 처리 시간: {execution_time:.2f}초
• 메모리 사용: 체크포인터 활용한 상태 관리
"""
    return debug_info


def run_query_with_app(query: str, app, conversation_history: list) -> str:
    """기본 모드: 향상된 스트리밍 사용"""
    # 향상된 스트리밍 함수를 호출
    return run_query_with_streaming(query, app, conversation_history)


def get_node_tools(node_name: str) -> str:
    """노드별 사용 툴 반환"""
    node_tools = {
        "Supervisor": "워크플로 관리",
        "SajuExpert": "manse + retriever 서브그래프",
        "manse": "calculate_saju_tool",
        "retriever": "saju_retriever_tool",
        "WebTool": "tavily_tool + duck_tool",
        "GeneralQA": "general_qa_tool (Google Gemini)"
    }
    return node_tools.get(node_name, "unknown")


def run_query_with_streaming(query: str, app, conversation_history: list) -> str:
    """기본 모드: 깔끔한 스트리밍 (주요 노드만)"""
    print(f"🔍 쿼리 실행: {query}")
    
    # 새로운 사용자 메시지를 히스토리에 추가
    conversation_history.append(HumanMessage(content=query))
    
    current_state = {
        "messages": conversation_history.copy(),
        "next": ""
    }
    
    # 설정 생성 (Checkpointer용)
    config = {
        "configurable": {
            "thread_id": f"thread_{int(time.time())}"
        }
    }
    
    try:
        print("🚀 AI 분석 시작...")
        
        # 기본 모드: 주요 작업 노드만 간단하게 표시
        final_response = ""
        prev_node = ""
        node_sequence = []
        displayed_content = []
        
        # 주요 작업 노드만 필터링 (Supervisor는 제외)
        work_nodes = ["SajuExpert", "manse", "retriever", "WebTool", "GeneralQA"]
        
        for chunk_msg, metadata in app.stream(current_state, config=config, stream_mode="messages"):
            curr_node = metadata.get("langgraph_node", "")
            
            # 주요 작업 노드만 표시
            if curr_node in work_nodes and curr_node != prev_node:
                print_simple_node_info(curr_node)
                node_sequence.append(curr_node)
                prev_node = curr_node
            
            # 토큰별로 실시간 출력
            if chunk_msg.content:
                print(chunk_msg.content, end="", flush=True)
                displayed_content.append(chunk_msg.content)
        
        # 간단한 완료 정보
        print(f"\n\n✅ 완료! (경로: {' → '.join(node_sequence)})")
        
        # 최종 응답 획득
        if displayed_content:
            final_response = "".join(displayed_content)
            conversation_history.append(AIMessage(content=final_response))
            return final_response
        else:
            print("❌ 응답 생성 실패")
            return "응답을 생성하지 못했습니다."
            
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        return f"오류가 발생했습니다: {str(e)}"


def run_query_with_debug(query: str, app, conversation_history: list) -> str:
    """디버그 모드: 모든 노드 + 상세 정보 + 툴 추적"""
    print(f"🔍 쿼리 실행 (디버그): {query}")
    
    # 새로운 사용자 메시지를 히스토리에 추가
    conversation_history.append(HumanMessage(content=query))
    
    # 현재 상태 설정
    current_state = {
        "messages": conversation_history.copy(),
        "next": ""
    }
    
    # 설정 생성 (Checkpointer용)
    config = {
        "configurable": {
            "thread_id": f"thread_{int(time.time())}"
        }
    }
    
    try:
        print("🚀 워크플로 실행 중 (전체 노드 + 툴 추적)...")
        
        # 디버그 모드: 모든 노드와 상세 정보 표시
        final_response = ""
        prev_node = ""
        displayed_content = []
        node_sequence = []
        tool_usage = {}  # 노드별 툴 사용 기록
        
        for chunk_msg, metadata in app.stream(current_state, config=config, stream_mode="messages"):
            curr_node = metadata.get("langgraph_node", "")
            
            # 새로운 노드 진입 시 상세 정보 출력
            if curr_node and curr_node != prev_node:
                print_node_header(curr_node, is_debug=True)
                print_node_execution(curr_node)  # 툴 정보도 함께 출력
                node_sequence.append(curr_node)
                tool_usage[curr_node] = get_node_tools(curr_node)
                print("💬 상세 응답:")
                prev_node = curr_node
            
            # 토큰별로 실시간 출력
            if chunk_msg.content:
                print(chunk_msg.content, end="", flush=True)
                displayed_content.append(chunk_msg.content)
        
        # 디버그 정보 요약
        print(f"\n\n📊 워크플로 분석 결과:")
        print(f"🎯 실행된 노드: {' → '.join(node_sequence)}")
        print(f"🛠️  사용된 툴:")
        for node, tools in tool_usage.items():
            print(f"   • {node}: {tools}")
        
        print_completion(is_debug=True)
        
        # 최종 응답 획득
        if displayed_content:
            final_response = "".join(displayed_content)
        else:
            result = app.invoke(current_state, config=config)
            messages = result.get("messages", [])
            if messages:
                final_response = messages[-1].content
            else:
                final_response = "응답을 생성하지 못했습니다."
        
        conversation_history.append(AIMessage(content=final_response))
        return final_response
            
    except Exception as e:
        print(f"❌ 디버그 모드 오류 발생: {str(e)}")
        return f"오류가 발생했습니다: {str(e)}" 