import os
import sys
from datetime import datetime

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage, AIMessage
from graph import create_workflow
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import stream_graph, random_uuid, invoke_graph


def run_saju_analysis(messages, thread_id=None, use_stream=True):
    graph = create_workflow()
    if not graph:
        return "그래프 생성에 실패했습니다."
    if thread_id is None:
        thread_id = random_uuid()
    config = RunnableConfig(recursion_limit=20, configurable={"thread_id": thread_id})
    inputs = {"messages": messages}
    if use_stream:
        return stream_graph(graph, inputs, config)
    else:
        return invoke_graph(graph, inputs, config)


def main():
    print("사주 에이전틱 RAG 시스템 (병렬 구조 버전)을 시작합니다... ")
    print("생년월일, 태이난 시각, 성별을 입력해 주세요.")
    print("윤달에 태어나신 경우, 윤달이라고 작성해주세요.")
    example_questions = [
        "1996년 12월 13일 남자, 10시 30분 출생 운세봐줘.",
        "대운과 세운, 조심해야 할것들 알려줘",
        "금전운알려줘",
        "정관이 뭐야? 상세히 설명해줘",
        "사주의 개념에 대해서 알려줘"
    ]
    print("\n사용 가능한 예시 질문:")
    for i, question in enumerate(example_questions, 1):
        print(f"{i}. {question}")
    print("\n질문을 입력하세요 (종료하려면 'quit' 입력):")
    chat_history = []
    thread_id = random_uuid()
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
        if not user_input:
            continue
        chat_history.append(HumanMessage(content=user_input))
        try:
            print("\n분석을 시작합니다...")
            result = run_saju_analysis(chat_history, thread_id=thread_id, use_stream=True)
            print("\n분석 완료!")
            if hasattr(result, '__iter__') and not isinstance(result, str):
                last_ai_msg = None
                for msg in result:
                    if hasattr(msg, 'content'):
                        last_ai_msg = msg
                if last_ai_msg:
                    chat_history.append(AIMessage(content=last_ai_msg.content))
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main() 