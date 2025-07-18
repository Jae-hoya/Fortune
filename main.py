"""
FortuneAI LangGraph 시스템 메인 실행 파일
Supervisor 패턴 기반 사주 계산기
"""

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
        user_input = input("\n질문: ").strip()
        if user_input.lower() in ['quit', 'exit', '종료']:
            print("시스템을 종료합니다.")
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