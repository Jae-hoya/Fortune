import asyncio
import os
import signal
import sys
import time
import traceback
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.messages.ai import AIMessageChunk
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel
from websockets.exceptions import ConnectionClosed


# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 1) Simple: basicConfig with filename
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="app.log",  # ← writes all logs to this file
    filemode="a",  # ← "w" to overwrite each run, "a" to append
)

logger = logging.getLogger(__name__)


# 전역 변수
memory = None
compiled_graph = None
session_store: Dict[str, Dict] = {}
debug_mode = True  # 디버깅 모드


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # Startup
    global memory, compiled_graph

    debug_log("🔧 FortuneAI 시스템 초기화 시작...")

    # 1단계: 모듈 임포트 확인
    debug_log("1️⃣ 단계 1: 모듈 임포트 확인")
    create_workflow_func, import_success = safe_import_modules()

    if not import_success:
        debug_log("❌ 모듈 임포트 실패로 시스템 초기화 중단", "ERROR")
        yield
        return

    # 2단계: 워크플로 생성
    debug_log("2️⃣ 단계 2: 워크플로 생성")
    try:
        compiled_graph = create_workflow_func()
        debug_log(f"✅ 워크플로 생성 성공: {type(app)}")
    except Exception as e:
        debug_log(f"❌ 워크플로 생성 실패: {e}", "ERROR")
        debug_log(f"❌ 상세 오류: {traceback.format_exc()}", "ERROR")
        yield
        return

    # 3단계: 메모리 초기화
    debug_log("3️⃣ 단계 3: 메모리 초기화")
    try:
        memory = MemorySaver()
        debug_log(f"✅ 메모리 초기화 성공: {type(memory)}")
    except Exception as e:
        debug_log(f"❌ 메모리 초기화 실패: {e}", "ERROR")
        yield
        return

    debug_log("✅ FortuneAI 시스템 초기화 완료!")

    yield

    # Shutdown (if needed)
    debug_log("🛑 FortuneAI 시스템 종료")


# FastAPI 앱 초기화
app = FastAPI(
    title="FortuneAI API",
    description="Supervisor 패턴 기반 사주 계산기 웹 서비스",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def debug_log(message: str, level: str = "INFO"):
    """디버깅 로그 출력"""
    if debug_mode:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] [{level}] {message}")


def safe_import_modules():
    """안전한 모듈 임포트"""
    # try:
    debug_log("📦 모듈 임포트 시작...")

    # langgraph_system.graph 임포트 시도

    from Fortune.graph import create_workflow

    debug_log("✅ langgraph_system.graph 임포트 성공")
    return create_workflow, True
    # except ImportError as e:
    #     debug_log(f"❌ langgraph_system.graph 임포트 실패: {e}", "ERROR")
    #     return None, False
    # except Exception as e:
    #     debug_log(f"❌ langgraph_system.graph 예상치 못한 오류: {e}", "ERROR")
    #     return None, False

    # except Exception as e:
    #     debug_log(f"❌ 전체 모듈 임포트 실패: {e}", "ERROR")
    #     return None, False


def initialize_session(session_id: str) -> Dict:
    """새 세션 초기화"""
    session_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    session_data = {
        "messages": [],
        "next": "",
        "session_id": session_id,
        "session_start_time": session_start_time,
        "query_count": 0,
        "conversation_history": [],
        "is_active": True,
        "last_activity": datetime.now(),
    }

    session_store[session_id] = session_data
    debug_log(f"🆔 새 세션 생성: {session_id}")

    return session_data


def get_or_create_session(session_id: str) -> Dict:
    """세션 가져오기 또는 생성"""
    if session_id not in session_store:
        return initialize_session(session_id)

    session_store[session_id]["last_activity"] = datetime.now()
    return session_store[session_id]


def generate_fallback_response(user_input: str, error_msg: str = None) -> str:
    """폴백 응답 생성"""
    base_responses = [
        f"안녕하세요! '{user_input}'에 대한 질문을 받았습니다.",
        f"'{user_input}'에 대해 분석 중입니다. 잠시만 기다려주세요.",
        f"질문 '{user_input}'을 처리하고 있습니다.",
        f"'{user_input}'에 대한 답변을 준비하고 있습니다.",
    ]

    import random

    response = random.choice(base_responses)

    if error_msg:
        response += f"\n\n(시스템 상태: {error_msg})"

    return response


@app.websocket("/ws/chat/saju/{session_id}")
async def chat_websocket_saju(websocket: WebSocket, session_id: str):
    """
    WebSocket 연결을 통해 클라이언트와 실시간 채팅 메시지를 주고받는 엔드포인트입니다.

    Args:
        websocket (WebSocket): FastAPI WebSocket 객체
        session_id (str): 클라이언트 세션 식별자

    동작:
        - 클라이언트 연결 요청 수락
        - 세션 초기화 및 환영 메시지 전송
        - 메시지 수신 및 처리 루프 실행 (타임아웃 및 오류 처리 포함)
        - 연결 종료 시 세션 정리 및 로그 기록
    """
    debug_log(f"🔌 WebSocket 연결 요청: {session_id}")

    try:
        await websocket.accept()
        debug_log(f"✅ WebSocket 연결 성공: {session_id}")

        # 세션 초기화
        session_data = get_or_create_session(session_id)

        # 환영 메시지 전송
        # welcome_msg = f"🔮 FortuneAI에 오신 것을 환영합니다!"
        # await websocket.send_text(welcome_msg)
        debug_log(f"📤 환영 메시지 전송 완료")

        # 메인 루프 (연결 상태 확인 포함)
        while True:
            try:
                debug_log("👂 사용자 입력 대기 중...")

                # 연결 상태 확인
                if websocket.client_state.name != "CONNECTED":
                    debug_log(
                        f"🔌 WebSocket 연결 상태 변경: {websocket.client_state.name}",
                        "WARN",
                    )
                    break

                # 타임아웃과 함께 메시지 수신 (무한 대기 방지)
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                    user_input = data.strip()
                    debug_log(f"📝 사용자 입력 수신: {user_input}")

                except asyncio.TimeoutError:
                    # 1초마다 연결 상태 확인
                    continue

                if not user_input:
                    debug_log(websocket, "❓ 질문을 입력해주세요.")
                    continue

                # 쿼리 처리 시작
                session_data["query_count"] += 1
                debug_log(f"🔄 쿼리 #{session_data['query_count']} 처리 시작")

                # 처리 중 메시지
                processing_msg = f"⏳ 분석 중... (질문 #{session_data['query_count']})"
                debug_log(websocket, processing_msg)
                debug_log("📤 처리 중 메시지 전송 완료")

                # 응답 생성 (타임아웃 포함)
                start_time = time.time()

                session_data["messages"].append(HumanMessage(content=user_input))

                # 네임스페이스 문자열을 보기 좋은 형식으로 변환하는 포맷팅 함수
                def format_namespace(namespace):
                    return (
                        namespace[-1].split(":")[0]
                        if len(namespace) > 0
                        else "parent graph"
                    )

                sent_message_ids = set()

                def parse_namespace_info(info: tuple) -> tuple[str, str]:
                    if len(info) > 1:
                        namespace, node_name = info
                        return node_name.split(":")[0], namespace.split(":")[0]
                    return info[0].split(":")[0], "parent graph"

                kind = None

                async for event in compiled_graph.astream_events(
                    session_data,
                    config={"configurable": {"thread_id": session_id}},
                    version="v2",
                    subgraphs=True,
                ):
                    kind = event["event"]
                    with open("log.json", "a") as f:
                        logger.info(event)

                    # 이벤트 종류와 태그 정보 추출
                    if kind == "on_chat_model_start":
                        print(f"\n========= on_chat_model_start =========\n")

                    # 채팅 모델 스트림 이벤트 및 최종 노드 태그 필터링
                    elif kind == "on_chat_model_stream":
                        if (
                            "SajuExpert" in event["metadata"]["langgraph_checkpoint_ns"]
                            and "retriever"
                            in event["metadata"]["langgraph_checkpoint_ns"]
                        ) or (
                            "GeneralQA" in event["metadata"]["langgraph_checkpoint_ns"]
                        ):
                            # 이벤트 데이터 추출
                            data = event["data"]

                            # 토큰 단위의 스트리밍 출력
                            if data["chunk"].content:
                                await websocket.send_text(data["chunk"].content)
                                # print(data["chunk"].content, end="", flush=True)

                    elif kind == "on_tool_start":
                        print(f"\n========= tool_start =========\n")
                        data = event["data"]
                        if "input" in data:
                            tool_msg = data["input"]
                            print(tool_msg)

                    elif kind == "on_tool_end":
                        print(f"\n========= tool_end =========\n")
                        data = event["data"]
                        if "output" in data:
                            tool_msg = data["output"]
                            print(tool_msg.content)
                execution_time = time.time() - start_time

                debug_log(f"✅ 응답 생성 완료 ({execution_time:.2f}초)")

                # 완료 정보 전송
                completion_msg = f"✅ 완료 (실행시간: {execution_time:.2f}초)"

                debug_log("📤 완료 메시지 전송 완료")
                debug_log(f"🔍 완료 메시지: {completion_msg}")

            except WebSocketDisconnect:
                debug_log("🔌 WebSocket 연결 끊어짐 (사용자 측)", "WARN")
                break
            except ConnectionClosed:
                debug_log("🔌 WebSocket 연결 닫힘 (ConnectionClosed)", "WARN")
                break
            except Exception as e:
                debug_log(f"❌ WebSocket 처리 오류: {str(e)}", "ERROR")
                debug_log(f"❌ 상세 오류: {traceback.format_exc()}", "ERROR")

                try:
                    error_msg = f"❌ 처리 중 오류가 발생했습니다: {str(e)}"
                    debug_log(websocket, error_msg)
                    debug_log("📤 오류 메시지 전송 완료")
                except:
                    debug_log("❌ 오류 메시지 전송 실패", "ERROR")
                    break

    except Exception as e:
        debug_log(f"❌ WebSocket 연결 실패: {str(e)}", "ERROR")
        debug_log(f"❌ 상세 오류: {traceback.format_exc()}", "ERROR")
    finally:
        # 세션 정리
        if session_id in session_store:
            session_store[session_id]["is_active"] = False
        debug_log(f"🔌 WebSocket 연결 종료: {session_id}")

@app.websocket("/ws/chat/tarot/{session_id}")
async def chat_websocket_tarot(websocket: WebSocket, session_id: str):
    """
    WebSocket 연결을 통해 클라이언트와 실시간 채팅 메시지를 주고받는 엔드포인트입니다.

    Args:
        websocket (WebSocket): FastAPI WebSocket 객체
        session_id (str): 클라이언트 세션 식별자

    동작:
        - 클라이언트 연결 요청 수락
        - 세션 초기화 및 환영 메시지 전송
        - 메시지 수신 및 처리 루프 실행 (타임아웃 및 오류 처리 포함)
        - 연결 종료 시 세션 정리 및 로그 기록
    """
    debug_log(f"🔌 WebSocket 연결 요청: {session_id}")

    try:
        await websocket.accept()
        debug_log(f"✅ WebSocket 연결 성공: {session_id}")

        # 세션 초기화
        session_data = get_or_create_session(session_id)

        # 환영 메시지 전송
        # welcome_msg = f"🔮 FortuneAI에 오신 것을 환영합니다!"
        # await websocket.send_text(welcome_msg)
        debug_log(f"📤 환영 메시지 전송 완료")

        # 메인 루프 (연결 상태 확인 포함)
        while True:
            try:
                debug_log("👂 사용자 입력 대기 중...")

                # 연결 상태 확인
                if websocket.client_state.name != "CONNECTED":
                    debug_log(
                        f"🔌 WebSocket 연결 상태 변경: {websocket.client_state.name}",
                        "WARN",
                    )
                    break

                # 타임아웃과 함께 메시지 수신 (무한 대기 방지)
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                    user_input = data.strip()
                    debug_log(f"📝 사용자 입력 수신: {user_input}")

                except asyncio.TimeoutError:
                    # 1초마다 연결 상태 확인
                    continue

                if not user_input:
                    debug_log(websocket, "❓ 질문을 입력해주세요.")
                    continue

                # 쿼리 처리 시작
                session_data["query_count"] += 1
                debug_log(f"🔄 쿼리 #{session_data['query_count']} 처리 시작")

                # 처리 중 메시지
                processing_msg = f"⏳ 분석 중... (질문 #{session_data['query_count']})"
                debug_log(websocket, processing_msg)
                debug_log("📤 처리 중 메시지 전송 완료")

                # 응답 생성 (타임아웃 포함)
                start_time = time.time()

                session_data["messages"].append(HumanMessage(content=user_input))

                # 네임스페이스 문자열을 보기 좋은 형식으로 변환하는 포맷팅 함수
                def format_namespace(namespace):
                    return (
                        namespace[-1].split(":")[0]
                        if len(namespace) > 0
                        else "parent graph"
                    )

                sent_message_ids = set()

                def parse_namespace_info(info: tuple) -> tuple[str, str]:
                    if len(info) > 1:
                        namespace, node_name = info
                        return node_name.split(":")[0], namespace.split(":")[0]
                    return info[0].split(":")[0], "parent graph"

                kind = None

                async for event in compiled_graph.astream_events(
                    session_data,
                    config={"configurable": {"thread_id": session_id}},
                    version="v2",
                    subgraphs=True,
                ):
                    kind = event["event"]
                    with open("log.json", "a") as f:
                        logger.info(event)

                    # 이벤트 종류와 태그 정보 추출
                    if kind == "on_chat_model_start":
                        print(f"\n========= on_chat_model_start =========\n")

                    # 채팅 모델 스트림 이벤트 및 최종 노드 태그 필터링
                    elif kind == "on_chat_model_stream":
                        if (
                            "SajuExpert" in event["metadata"]["langgraph_checkpoint_ns"]
                            and "retriever"
                            in event["metadata"]["langgraph_checkpoint_ns"]
                        ) or (
                            "GeneralQA" in event["metadata"]["langgraph_checkpoint_ns"]
                        ):
                            # 이벤트 데이터 추출
                            data = event["data"]

                            # 토큰 단위의 스트리밍 출력
                            if data["chunk"].content:
                                await websocket.send_text(data["chunk"].content)
                                # print(data["chunk"].content, end="", flush=True)

                    elif kind == "on_tool_start":
                        print(f"\n========= tool_start =========\n")
                        data = event["data"]
                        if "input" in data:
                            tool_msg = data["input"]
                            print(tool_msg)

                    elif kind == "on_tool_end":
                        print(f"\n========= tool_end =========\n")
                        data = event["data"]
                        if "output" in data:
                            tool_msg = data["output"]
                            print(tool_msg.content)
                execution_time = time.time() - start_time

                debug_log(f"✅ 응답 생성 완료 ({execution_time:.2f}초)")

                # 완료 정보 전송
                completion_msg = f"✅ 완료 (실행시간: {execution_time:.2f}초)"

                debug_log("📤 완료 메시지 전송 완료")
                debug_log(f"🔍 완료 메시지: {completion_msg}")

            except WebSocketDisconnect:
                debug_log("🔌 WebSocket 연결 끊어짐 (사용자 측)", "WARN")
                break
            except ConnectionClosed:
                debug_log("🔌 WebSocket 연결 닫힘 (ConnectionClosed)", "WARN")
                break
            except Exception as e:
                debug_log(f"❌ WebSocket 처리 오류: {str(e)}", "ERROR")
                debug_log(f"❌ 상세 오류: {traceback.format_exc()}", "ERROR")

                try:
                    error_msg = f"❌ 처리 중 오류가 발생했습니다: {str(e)}"
                    debug_log(websocket, error_msg)
                    debug_log("📤 오류 메시지 전송 완료")
                except:
                    debug_log("❌ 오류 메시지 전송 실패", "ERROR")
                    break

    except Exception as e:
        debug_log(f"❌ WebSocket 연결 실패: {str(e)}", "ERROR")
        debug_log(f"❌ 상세 오류: {traceback.format_exc()}", "ERROR")
    finally:
        # 세션 정리
        if session_id in session_store:
            session_store[session_id]["is_active"] = False
        debug_log(f"🔌 WebSocket 연결 종료: {session_id}")




# 디버깅용 엔드포인트들
@app.get("/api/debug/system-status")
async def system_status():
    """시스템 상태 확인"""
    return {
        "timestamp": datetime.now().isoformat(),
        "system_components": {
            "memory": memory is not None,
            "compiled_graph": compiled_graph is not None,
        },
        "sessions": {
            "total": len(session_store),
            "active": len([s for s in session_store.values() if s["is_active"]]),
        },
        "debug_mode": debug_mode,
    }



@app.get("/api/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_loaded": {
            "compiled_graph": compiled_graph is not None,
            "memory": memory is not None,
        },
    }


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "🔮 FortuneAI API Server (Debug Mode)",
        "version": "1.0.0",
        "debug_mode": debug_mode,
        "status": "running",
    }


# 신호 핸들러 (Ctrl+C 처리)
def signal_handler(signum, frame):
    debug_log("🛑 종료 신호 수신 (Ctrl+C)", "WARN")
    sys.exit(0)


if __name__ == "__main__":
    import uvicorn

    # 신호 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    debug_log("🚀 FortuneAI FastAPI 서버 시작...")

    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,  # 디버깅 시 reload 비활성화
            log_level="info",
        )
    except KeyboardInterrupt:
        debug_log("🛑 서버 종료", "WARN")
    except Exception as e:
        debug_log(f"❌ 서버 실행 오류: {e}", "ERROR")
