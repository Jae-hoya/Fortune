"""
FortuneAI LangGraph 시스템 메인 실행 파일
Supervisor 패턴 기반 사주 계산기
"""

import os
import sys
import time
import uuid
from datetime import datetime

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage, AIMessage
from graph import create_workflow
from langchain_core.runnables import RunnableConfig

# utils.py에서 함수들 import
from utils import (
    print_banner, print_system_info, format_response, print_help,
    handle_debug_query, run_query_with_app
)

def main():
    """메인 실행 함수"""
    print_banner()
    print_system_info()
    
    # ✨ 시스템 시작 시 AgentState 구조 초기화
    print("🔧 시스템 초기화 중...")
    print("  - SajuExpert 에이전트 로딩...")
    print("  - Search 에이전트 로딩...")
    print("  - GeneralAnswer 에이전트 로딩...")
    print("✅ 시스템 초기화 완료!")
    
    # ✨ 워크플로도 미리 생성
    print("⚙️ 워크플로 생성 중...")
    app = create_workflow()
    print("✅ 워크플로 준비 완료!")
    
    # 세션 및 대화 히스토리 관리
    conversation_history = []
    session_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session_id = f"session_{int(time.time())}"
    query_count = 0
    
    print(f"🕐 세션 시작: {session_start_time}")
    print(f"🆔 세션 ID: {session_id}")
    
    print("💬 질문을 입력해주세요 (종료: quit/exit, 도움말: help):")
    
    while True:
        try:
            # 사용자 입력 받기
            user_input = input("\n🤔 질문: ").strip()
            
            # 종료 명령 처리
            if user_input.lower() in ['quit', 'exit', '종료', 'q']:
                print("\n👋 FortuneAI를 이용해주셔서 감사합니다!")
                print("🌟 좋은 하루 되세요! 🌟")
                break
            
            # 새 세션 시작 명령 처리
            if user_input.lower() in ['new', 'clear']:
                session_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                session_id = f"session_{int(time.time())}"
                query_count = 0
                conversation_history = []  # 대화 히스토리 초기화
                print(f"\n🔄 새로운 대화를 시작합니다.")
                print(f"🕐 세션 시작: {session_start_time}")
                print(f"🆔 세션 ID: {session_id}")
                
                # 환영 메시지 생성
                welcome_response = run_query_with_app("안녕하세요! FortuneAI입니다. 무엇을 도와드릴까요?", app, conversation_history, session_start_time, session_id)
                print(f"🔮 FortuneAI: {welcome_response}")
                print("-" * 60)
                continue
            
            # 도움말 명령 처리
            if user_input.lower() in ['help', 'h', '도움말', '?']:
                print_help()
                continue
            
            # 빈 입력 처리
            if not user_input:
                print("❓ 질문을 입력해주세요.")
                continue
            
            query_count += 1
            print(f"\n⏳ 분석 중... (질문 #{query_count})")
            
            # 성능 분석 모드 처리
            analysis_response = handle_debug_query(user_input, app, conversation_history, session_start_time, session_id)
            if analysis_response:
                print(analysis_response)
                continue
            
            # 일반 쿼리 실행 - 상세 스트리밍 표시
            start_time = time.time()
            response = run_query_with_app(user_input, app, conversation_history, session_start_time, session_id)
            execution_time = time.time() - start_time
            
            # 실행 시간 표시
            print(f"\n⏱️  실행 시간: {execution_time:.2f}초")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  사용자가 중단했습니다.")
            print("👋 FortuneAI를 이용해주셔서 감사합니다!")
            break
            
        except Exception as e:
            print(f"\n❌ 오류 발생: {str(e)}")
            print("🔧 시스템을 다시 시도해보세요.")
            continue

if __name__ == "__main__":
    # 명령행 인자 처리
    if len(sys.argv) > 1:
        conversation_history = []  # 명령행 모드에서도 히스토리 초기화
        session_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session_id = f"session_{int(time.time())}"
        
        # --debug 플래그 확인
        is_debug = '--debug' in sys.argv
        if is_debug:
            sys.argv.remove('--debug')
        
        if len(sys.argv) > 1:
            query = " ".join(sys.argv[1:])
            
            # 시스템 초기화
            print("🔧 시스템 초기화 중...")
            from nodes import get_node_manager
            get_node_manager()
            print("⚙️ 워크플로 생성 중...")
            app = create_workflow()
            print(f"🕐 세션 시작: {session_start_time}")
            print(f"🆔 세션 ID: {session_id}")
            
            if is_debug:
                # 성능 분석 모드
                result = handle_debug_query(f"debug:{query}", app, conversation_history, session_start_time, session_id)
                print(result)
            else:
                # 기본 모드 - 상세 스트리밍 출력
                response = run_query_with_app(query, app, conversation_history, session_start_time, session_id)
                # 상세 스트리밍이 이미 완료됨
        else:
            print("❌ 질문을 입력해주세요.")
            print("사용법: python main.py [--debug] '질문'")
    else:
        # 대화형 모드
        main() 