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

from langchain_core._api.beta_decorator import warn_beta

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages.ai import AIMessageChunk
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel
from websockets.exceptions import ConnectionClosed


# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

# 1) Simple: basicConfig with filename
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="app.log",  # ← writes all logs to this file
    filemode="a",  # ← "w" to overwrite each run, "a" to append
)

logger = logging.getLogger(__name__)


# 전역 변수 - Saju 시스템
memory = None
compiled_graph = None
session_store: Dict[str, Dict] = {}
debug_mode = True  # 디버깅 모드

# 전역 변수 - Tarot 시스템
rag_system = None
tarot_compiled_graph = None
tarot_session_store: Dict[str, Dict] = {}

# 전역 함수 변수 - Tarot
initialize_rag_system = None
create_optimized_tarot_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # Startup
    global memory, compiled_graph, rag_system, tarot_compiled_graph

    debug_log("🔧 FortuneAI 시스템 초기화 시작...")

    # 1단계: 모듈 임포트 확인
    debug_log("1️⃣ 단계 1: 모듈 임포트 확인")
    create_workflow_func, import_success = safe_import_modules()
    tarot_import_success = safe_import_tarot_modules()

    if not import_success:
        debug_log("❌ Saju 모듈 임포트 실패로 시스템 초기화 중단", "ERROR")
        yield
        return

    if not tarot_import_success:
        debug_log("❌ Tarot 모듈 임포트 실패로 시스템 초기화 중단", "ERROR")
        yield
        return

    # 2단계: 메모리 초기화
    debug_log("2️⃣ 단계 2: 메모리 초기화")
    try:
        memory = MemorySaver()
        debug_log(f"✅ 메모리 초기화 성공: {type(memory)}")
    except Exception as e:
        debug_log(f"❌ 메모리 초기화 실패: {e}", "ERROR")
        yield
        return

    # 3단계: Saju 워크플로 생성
    debug_log("3️⃣ 단계 3: Saju 워크플로 생성")
    try:
        if create_workflow_func:
            compiled_graph = create_workflow_func()
            debug_log(f"✅ Saju 워크플로 생성 성공: {type(compiled_graph)}")
        else:
            debug_log("❌ create_workflow_func가 로드되지 않음", "ERROR")
            yield
            return
    except Exception as e:
        debug_log(f"❌ Saju 워크플로 생성 실패: {e}", "ERROR")
        debug_log(f"❌ 상세 오류: {traceback.format_exc()}", "ERROR")
        yield
        return

    # 4단계: Tarot 시스템 초기화
    debug_log("4️⃣ 단계 4: Tarot 시스템 초기화")
    try:
        if initialize_rag_system and create_optimized_tarot_graph:
            rag_system = initialize_rag_system()
            tarot_compiled_graph = create_optimized_tarot_graph().compile(
                checkpointer=memory
            )
            debug_log(f"✅ Tarot 시스템 초기화 성공: {type(tarot_compiled_graph)}")
        else:
            debug_log("❌ Tarot 함수들이 로드되지 않음", "ERROR")
            yield
            return
    except Exception as e:
        debug_log(f"❌ Tarot 시스템 초기화 실패: {e}", "ERROR")
        debug_log(f"❌ 상세 오류: {traceback.format_exc()}", "ERROR")
        yield
        return

    debug_log("✅ FortuneAI 시스템 초기화 완료!")

    yield

    # Shutdown (if needed)
    debug_log("🛑 FortuneAI 시스템 종료")


# FastAPI 앱 초기화
app = FastAPI(
    title="FortuneAI API",
    description="Supervisor 패턴 기반 사주 계산기 및 타로 상담 웹 서비스",
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
        logger.info(f"[{timestamp}] [{level}] {message}")


def safe_import_modules():
    """안전한 모듈 임포트 - Saju"""
    debug_log("📦 Saju 모듈 임포트 시작...")

    try:
        from Fortune.graph import create_workflow

        debug_log("✅ graph 임포트 성공")
        return create_workflow, True
    except ImportError as e:
        debug_log(f"❌ graph 임포트 실패: {e}", "ERROR")
        return None, False
    except Exception as e:
        debug_log(f"❌ graph 예상치 못한 오류: {e}", "ERROR")
        return None, False


def safe_import_tarot_modules():
    """안전한 모듈 임포트 - Tarot"""
    global initialize_rag_system, create_optimized_tarot_graph
    debug_log("📦 Tarot 모듈 임포트 시작...")

    try:
        from Fortune.parser.tarot_agent.agent import create_optimized_tarot_graph
        from Fortune.parser.tarot_agent.utils.tools import initialize_rag_system

        debug_log("✅ tarot_agent 모듈 임포트 성공")
        return True
    except ImportError as e:
        debug_log(f"❌ tarot_agent 모듈 임포트 실패: {e}", "ERROR")
        return False
    except Exception as e:
        debug_log(f"❌ tarot_agent 예상치 못한 오류: {e}", "ERROR")
        return False


def initialize_session(session_id: str) -> Dict:
    """새 세션 초기화 - Saju"""
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
    debug_log(f"🆔 새 Saju 세션 생성: {session_id}")

    return session_data


def initialize_tarot_session(session_id: str) -> Dict:
    """새 세션 초기화 - Tarot"""
    session_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    session_data = {
        "messages": [
            AIMessage(
                content="🔮 안녕하세요! 타로 상담사입니다. 오늘은 어떤 도움이 필요하신가요?"
            )
        ],
        "user_intent": "unknown",
        "user_input": "",
        "consultation_data": None,
        "supervisor_decision": None,
        "session_id": session_id,
        "session_start_time": session_start_time,
        "query_count": 0,
        "conversation_history": [],
        "is_active": True,
        "last_activity": datetime.now(),
    }

    tarot_session_store[session_id] = session_data
    debug_log(f"🆔 새 Tarot 세션 생성: {session_id}")

    return session_data


def get_or_create_session(session_id: str) -> Dict:
    """세션 가져오기 또는 생성 - Saju"""
    if session_id not in session_store:
        return initialize_session(session_id)

    session_store[session_id]["last_activity"] = datetime.now()
    return session_store[session_id]


def get_or_create_tarot_session(session_id: str) -> Dict:
    """세션 가져오기 또는 생성 - Tarot"""
    if session_id not in tarot_session_store:
        return initialize_tarot_session(session_id)

    tarot_session_store[session_id]["last_activity"] = datetime.now()
    return tarot_session_store[session_id]


def generate_fallback_response(user_input: str, error_msg: Optional[str] = None) -> str:
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

# import asyncio
# from fastapi import WebSocket

@app.websocket("/ws/chat/saju/{session_id}")
async def chat_websocket_saju(websocket: WebSocket, session_id: str):
    debug_log(f"🔌 Saju WebSocket 연결 요청: {session_id}")

    try:
        await websocket.accept()
        debug_log(f"✅ Saju WebSocket 연결 성공: {session_id}")

        session_data = get_or_create_session(session_id)
        message_queue = asyncio.Queue()

        # 메시지 수신 태스크
        async def receive_messages():
            while True:
                try:
                    data = await websocket.receive_text()
                    user_input = data.strip()
                    if user_input:
                        await message_queue.put(user_input)
                        debug_log(f"📝 사용자 입력 큐에 추가: {user_input}")
                except Exception as e:
                    debug_log(f"❌ 메시지 수신 오류: {e}", "ERROR")
                    break

        # 메시지 처리 태스크
        async def process_messages():
            while True:
                user_input = await message_queue.get()
                session_data["query_count"] += 1
                session_data["messages"].append(HumanMessage(content=user_input))
                debug_log(f"🔄 쿼리 #{session_data['query_count']} 처리 시작")

                try:
                    async for event in compiled_graph.astream_events(
                        session_data,
                        config={"configurable": {"thread_id": session_id}},
                        version="v2",
                        subgraphs=True,
                    ):
                        kind = event["event"]
                        if kind == "on_chat_model_stream":
                            if (
                                "manse" in event["metadata"]["langgraph_checkpoint_ns"]
                                and "agent" in event["metadata"]["langgraph_checkpoint_ns"]
                            ) or (
                                "GeneralQA" in event["metadata"]["langgraph_checkpoint_ns"]
                            ):
                                data = event["data"]
                                if data["chunk"].content:
                                    await websocket.send_json({
                                        "type": "stream",
                                        "content": str(data["chunk"].content)
                                    })
                    await websocket.send_json({
                        "type": "complete",
                        "content": f"✅ 완료 (질문 #{session_data['query_count']})"
                    })
                    

                
                except Exception as e:
                    debug_log(f"❌ LangGraph 처리 오류: {e}", "ERROR")
                    await websocket.send_json({
                        "type": "error",
                        "content": f"❌ 처리 중 오류가 발생했습니다: {str(e)}"
                    })

        # 두 태스크를 동시에 실행
        receive_task = asyncio.create_task(receive_messages())
        process_task = asyncio.create_task(process_messages())

        # WebSocket 연결이 끊길 때까지 대기
        await receive_task
        process_task.cancel()
        debug_log("🔌 WebSocket 연결 종료 (receive_task 종료)")

    except Exception as e:
        debug_log(f"❌ WebSocket 연결 실패: {str(e)}", "ERROR")
        debug_log(f"❌ 상세 오류: {traceback.format_exc()}", "ERROR")
    finally:
        if session_id in session_store:
            session_store[session_id]["is_active"] = False
        debug_log(f"🔌 Saju WebSocket 연결 종료: {session_id}")


@app.websocket("/ws/chat/tarot/{session_id}")
async def chat_websocket_tarot(websocket: WebSocket, session_id: str):
    debug_log(f"🔌 Tarot WebSocket 연결 요청: {session_id}")

    try:
        await websocket.accept()
        debug_log(f"✅ Tarot WebSocket 연결 성공: {session_id}")

        session_data = get_or_create_tarot_session(session_id)
        message_queue = asyncio.Queue()

        # 메시지 수신 태스크
        async def receive_messages():
            while True:
                try:
                    data = await websocket.receive_text()
                    user_input = data.strip()
                    if user_input:
                        await message_queue.put(user_input)
                        debug_log(f"📝 사용자 입력 큐에 추가: {user_input}")
                except Exception as e:
                    debug_log(f"❌ 메시지 수신 오류: {e}", "ERROR")
                    break

        # 메시지 처리 태스크
        
        async def process_messages():
            while True:
                user_input = await message_queue.get()
                session_data["query_count"] += 1
                session_data["messages"].append(HumanMessage(content=user_input))
                debug_log(f"🔄 쿼리 #{session_data['query_count']} 처리 시작")

                try:
                    config = {"configurable": {"thread_id": session_id}}
                    async for event in tarot_compiled_graph.astream_events(
                        session_data,
                        config=config,
                        version="v2",
                        subgraphs=True,
                    ):
                        kind = event["event"]
                        
                        debug_log(event)
                        if kind == "on_chat_model_stream":
                            try:
                                if event['metadata']['final_response'] == "yes":
                                        data = event.get("data", {})
                                        chunk = data.get("chunk", None)
                                        content = getattr(chunk, "content", None)
                                    
                                        if content:
                                            await websocket.send_json({"type": "stream", "content": str(content)})
                            except Exception as e:
                                continue
                    # 쿼리 처리 후 최종 state를 프론트로 전송
                    state = await tarot_compiled_graph.aget_state(config)
                    state_dict = state.values if hasattr(state, "values") else state
                    send_state = dict(state_dict)
                    send_state.pop("messages", None)
                    
                    await websocket.send_json({
                        "type": "final_state",
                        "state": send_state
                    })
                    await websocket.send_json({
                        "type": "complete",
                        "content": f"✅ 완료 (질문 #{session_data['query_count']})"
                    })
                except Exception as e:
                    debug_log(f"❌ LangGraph 처리 오류: {e}", "ERROR")
                    await websocket.send_json({
                        "type": "error",
                        "content": f"❌ 처리 중 오류가 발생했습니다: {str(e)}"
                    })
        # 두 태스크를 동시에 실행
        receive_task = asyncio.create_task(receive_messages())
        process_task = asyncio.create_task(process_messages())

        # WebSocket 연결이 끊길 때까지 대기
        await receive_task
        process_task.cancel()
        debug_log("🔌 WebSocket 연결 종료 (receive_task 종료)")

    except Exception as e:
        debug_log(f"❌ WebSocket 연결 실패: {str(e)}", "ERROR")
        debug_log(f"❌ 상세 오류: {traceback.format_exc()}", "ERROR")
    finally:
        if session_id in tarot_session_store:
            tarot_session_store[session_id]["is_active"] = False
        debug_log(f"🔌 Tarot WebSocket 연결 종료: {session_id}")


# 디버깅용 엔드포인트들
@app.get("/api/debug/system-status")
async def system_status():
    """시스템 상태 확인"""
    return {
        "timestamp": datetime.now().isoformat(),
        "system_components": {
            "memory": memory is not None,
            "compiled_graph": compiled_graph is not None,
            "rag_system": rag_system is not None,
            "tarot_compiled_graph": tarot_compiled_graph is not None,
        },
        "sessions": {
            "saju_total": len(session_store),
            "saju_active": len([s for s in session_store.values() if s["is_active"]]),
            "tarot_total": len(tarot_session_store),
            "tarot_active": len(
                [s for s in tarot_session_store.values() if s["is_active"]]
            ),
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
            "rag_system": rag_system is not None,
            "tarot_compiled_graph": tarot_compiled_graph is not None,
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
        "endpoints": {
            "saju": "/ws/chat/saju/{session_id}",
            "tarot": "/ws/chat/tarot/{session_id}",
        },
    }


# 신호 핸들러 (Ctrl+C 처리)
def signal_handler(signum, frame):
    debug_log("🛑 종료 신호 수신 (Ctrl+C)", "WARN")
    sys.exit(0)


if __name__ == "__main__":
    
    from Fortune.parser.tarot_agent.agent import create_optimized_tarot_graph
    app = create_optimized_tarot_graph()

    # import uvicorn

    # # 신호 핸들러 등록
    # signal.signal(signal.SIGINT, signal_handler)
    # signal.signal(signal.SIGTERM, signal_handler)

    # debug_log("🚀 FortuneAI FastAPI 서버 시작...")

    # try:
    #     uvicorn.run(
    #         "main:app",
    #         host="0.0.0.0",
    #         port=8000,
    #         reload=False,  # 디버깅 시 reload 비활성화
    #         log_level="info",
    #     )
    # except KeyboardInterrupt:
    #     debug_log("🛑 서버 종료", "WARN")
    # except Exception as e:
    #     debug_log(f"❌ 서버 실행 오류: {e}", "ERROR")
