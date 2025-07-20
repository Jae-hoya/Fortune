# 🔮 타로 에이전트 질문 횟수 제한 시스템 구현 계획

**작성일**: 2024.12.28  
**버전**: v1.0  
**구현 방식**: 데이터베이스 + 미들웨어  

---

## 📋 목차

1. [현재 시스템 분석](#현재-시스템-분석)
2. [구현 아키텍처](#구현-아키텍처)
3. [데이터베이스 설계](#데이터베이스-설계)
4. [미들웨어 구현](#미들웨어-구현)
5. [FastAPI 통합](#fastapi-통합)
6. [구현 단계](#구현-단계)
7. [테스트 계획](#테스트-계획)

---

## 🎯 현재 시스템 분석

### 🏗️ 기존 아키텍처
```
parsing/parser/tarot_agent/
├── agent.py                 # 메인 에이전트 (콘솔 기반)
├── utils/
│   ├── state.py            # TarotState 정의
│   ├── nodes.py            # 핸들러 함수들 (2327줄)
│   ├── helpers.py          # 유틸리티 함수들 (1500줄)
│   ├── analysis.py         # 분석 모듈
│   ├── timing.py           # 시간 예측 모듈
│   ├── web_search.py       # 웹 검색 모듈
│   ├── translation.py      # 번역 모듈
│   └── tools.py            # RAG 도구
├── tarot_rag_system.py     # RAG 시스템
└── embedding.py            # 임베딩 처리
```

### 🔍 현재 상태 관리
```python
class TarotState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_intent: Literal["card_info", "spread_info", "consultation", "general", "simple_card", "unknown"]
    user_input: str
    consultation_data: Optional[Dict[str, Any]]
    supervisor_decision: Optional[Dict[str, Any]]
    routing_decision: Optional[str]
    target_handler: Optional[str]
    needs_llm: Optional[bool]
    session_memory: Optional[Dict[str, Any]]
    conversation_memory: Optional[Dict[str, Any]]
    temporal_context: Optional[Dict[str, Any]]
    search_results: Optional[Dict[str, Any]]
    search_decision: Optional[Dict[str, Any]]
```

### 📊 현재 시스템 특징
- **LangGraph 기반**: 상태 기반 워크플로우 관리
- **Fast Track 시스템**: 멀티턴 대화 최적화
- **RAG 시스템**: FAISS + BM25 + FlashRank 하이브리드 검색
- **메타데이터 지원**: FastAPI 통합을 위한 메타데이터 시스템
- **콘솔 기반 실행**: 현재는 CLI 환경에서만 동작

---

## 🏛️ 구현 아키텍처

### 전체 시스템 구조
```mermaid
graph TB
    subgraph "🌐 FastAPI 서버"
        API[FastAPI 앱]
        MIDDLEWARE[질문 횟수 제한 미들웨어]
        ENDPOINTS[타로 상담 엔드포인트]
    end
    
    subgraph "🗄️ 데이터베이스"
        USERS[users 테이블]
        SUBSCRIPTIONS[subscriptions 테이블]
        USAGE[usage_logs 테이블]
        SESSIONS[sessions 테이블]
    end
    
    subgraph "🔮 타로 에이전트"
        AGENT[타로 에이전트]
        NODES[노드 함수들]
        RAG[RAG 시스템]
    end
    
    subgraph "🔐 인증 시스템"
        AUTH[사용자 인증]
        JWT[JWT 토큰]
        RATE_LIMITER[요금제 체크]
    end
    
    API --> MIDDLEWARE
    MIDDLEWARE --> AUTH
    AUTH --> RATE_LIMITER
    RATE_LIMITER --> USERS
    RATE_LIMITER --> SUBSCRIPTIONS
    RATE_LIMITER --> USAGE
    MIDDLEWARE --> ENDPOINTS
    ENDPOINTS --> AGENT
    AGENT --> NODES
    NODES --> RAG
    
    style MIDDLEWARE fill:#ffeb3b
    style RATE_LIMITER fill:#ff9800
    style AGENT fill:#4caf50
```

### 🔄 요청 처리 플로우
```mermaid
sequenceDiagram
    participant U as 사용자
    participant API as FastAPI
    participant MW as 미들웨어
    participant DB as 데이터베이스
    participant TA as 타로 에이전트
    
    U->>API: 타로 상담 요청
    API->>MW: 미들웨어 호출
    MW->>DB: 사용자 정보 조회
    DB-->>MW: 사용자 정보 반환
    MW->>DB: 사용량 체크
    DB-->>MW: 사용량 정보 반환
    
    alt 제한 초과
        MW-->>API: 429 Too Many Requests
        API-->>U: 제한 초과 메시지
    else 제한 내
        MW->>TA: 타로 에이전트 호출
        TA-->>MW: 상담 결과 반환
        MW->>DB: 사용량 기록
        MW-->>API: 상담 결과 반환
        API-->>U: 상담 결과 전송
    end
```

---

## 🗄️ 데이터베이스 설계

### 1. users 테이블
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    last_login TIMESTAMP,
    
    -- 인덱스
    INDEX idx_users_username (username),
    INDEX idx_users_email (email)
);
```

### 2. subscriptions 테이블
```sql
CREATE TABLE subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    plan_type VARCHAR(20) NOT NULL, -- 'free', 'basic', 'premium', 'enterprise'
    question_limit INTEGER NOT NULL, -- 월간 질문 제한 수
    current_usage INTEGER DEFAULT 0, -- 현재 사용량
    reset_date DATE NOT NULL, -- 리셋 날짜 (매월 1일)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP, -- 유료 플랜 만료일
    is_active BOOLEAN DEFAULT TRUE,
    
    -- 외래키
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    
    -- 인덱스
    INDEX idx_subscriptions_user_id (user_id),
    INDEX idx_subscriptions_plan_type (plan_type),
    INDEX idx_subscriptions_reset_date (reset_date)
);
```

### 3. usage_logs 테이블
```sql
CREATE TABLE usage_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    session_id UUID,
    question_text TEXT NOT NULL,
    response_handler VARCHAR(50), -- 어떤 핸들러가 처리했는지
    processing_time FLOAT, -- 처리 시간 (초)
    tokens_used INTEGER, -- 사용된 토큰 수
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT,
    
    -- 외래키
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    
    -- 인덱스
    INDEX idx_usage_logs_user_id (user_id),
    INDEX idx_usage_logs_created_at (created_at),
    INDEX idx_usage_logs_session_id (session_id)
);
```

### 4. sessions 테이블
```sql
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    session_data JSONB, -- 상담 세션 데이터
    consultation_type VARCHAR(50), -- 'simple_card', 'consultation', 'card_info' 등
    questions_count INTEGER DEFAULT 0, -- 세션 내 질문 수
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP, -- 세션 만료 시간
    is_active BOOLEAN DEFAULT TRUE,
    
    -- 외래키
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    
    -- 인덱스
    INDEX idx_sessions_user_id (user_id),
    INDEX idx_sessions_created_at (created_at),
    INDEX idx_sessions_expires_at (expires_at)
);
```

### 5. plan_configs 테이블
```sql
CREATE TABLE plan_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plan_type VARCHAR(20) UNIQUE NOT NULL,
    question_limit INTEGER NOT NULL,
    price_monthly DECIMAL(10,2) DEFAULT 0.00,
    features JSONB, -- 플랜별 기능 설정
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 기본 데이터 삽입
    INDEX idx_plan_configs_plan_type (plan_type)
);

-- 기본 플랜 데이터 삽입
INSERT INTO plan_configs (plan_type, question_limit, price_monthly, features) VALUES
('free', 10, 0.00, '{"web_search": false, "advanced_analysis": false}'),
('basic', 50, 9.99, '{"web_search": true, "advanced_analysis": false}'),
('premium', 200, 19.99, '{"web_search": true, "advanced_analysis": true}'),
('enterprise', -1, 49.99, '{"web_search": true, "advanced_analysis": true, "unlimited": true}');
```

---

## 🔧 미들웨어 구현

### 1. 질문 횟수 제한 미들웨어
```python
# parsing/parser/api/middleware/rate_limiter.py
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import asyncio
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class QuestionRateLimiter:
    """질문 횟수 제한 미들웨어"""
    
    def __init__(self, db_session):
        self.db = db_session
    
    async def check_rate_limit(self, user_id: str) -> dict:
        """사용자의 질문 제한 체크"""
        try:
            # 1. 사용자 구독 정보 조회
            subscription = await self.get_user_subscription(user_id)
            if not subscription:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="사용자 구독 정보를 찾을 수 없습니다."
                )
            
            # 2. 월간 리셋 체크
            await self.check_monthly_reset(subscription)
            
            # 3. 사용량 체크
            if subscription['current_usage'] >= subscription['question_limit']:
                if subscription['plan_type'] == 'free':
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail={
                            "message": "무료 플랜의 월간 질문 제한에 도달했습니다.",
                            "current_usage": subscription['current_usage'],
                            "limit": subscription['question_limit'],
                            "reset_date": subscription['reset_date'],
                            "upgrade_available": True
                        }
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail={
                            "message": f"{subscription['plan_type']} 플랜의 월간 질문 제한에 도달했습니다.",
                            "current_usage": subscription['current_usage'],
                            "limit": subscription['question_limit'],
                            "reset_date": subscription['reset_date']
                        }
                    )
            
            # 4. 사용 가능한 질문 수 반환
            remaining = subscription['question_limit'] - subscription['current_usage']
            return {
                "allowed": True,
                "remaining": remaining,
                "limit": subscription['question_limit'],
                "current_usage": subscription['current_usage'],
                "plan_type": subscription['plan_type'],
                "reset_date": subscription['reset_date']
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="질문 제한 확인 중 오류가 발생했습니다."
            )
    
    async def increment_usage(self, user_id: str, question_text: str, 
                            response_handler: str, processing_time: float) -> None:
        """사용량 증가 및 로그 기록"""
        try:
            # 1. 구독 정보 업데이트
            await self.db.execute("""
                UPDATE subscriptions 
                SET current_usage = current_usage + 1, 
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = %s AND is_active = TRUE
            """, (user_id,))
            
            # 2. 사용 로그 기록
            await self.db.execute("""
                INSERT INTO usage_logs 
                (user_id, question_text, response_handler, processing_time, created_at)
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
            """, (user_id, question_text, response_handler, processing_time))
            
            await self.db.commit()
            
        except Exception as e:
            logger.error(f"Usage increment error: {e}")
            await self.db.rollback()
            raise
    
    async def get_user_subscription(self, user_id: str) -> Optional[dict]:
        """사용자 구독 정보 조회"""
        result = await self.db.fetch_one("""
            SELECT s.*, p.features
            FROM subscriptions s
            JOIN plan_configs p ON s.plan_type = p.plan_type
            WHERE s.user_id = %s AND s.is_active = TRUE
        """, (user_id,))
        
        return dict(result) if result else None
    
    async def check_monthly_reset(self, subscription: dict) -> None:
        """월간 리셋 체크 및 실행"""
        today = datetime.now().date()
        reset_date = subscription['reset_date']
        
        if today >= reset_date:
            # 다음 달 리셋 날짜 계산
            next_reset = reset_date.replace(month=reset_date.month + 1) if reset_date.month < 12 else reset_date.replace(year=reset_date.year + 1, month=1)
            
            # 사용량 리셋
            await self.db.execute("""
                UPDATE subscriptions 
                SET current_usage = 0, 
                    reset_date = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = %s AND is_active = TRUE
            """, (next_reset, subscription['user_id']))
            
            await self.db.commit()
            logger.info(f"Monthly reset completed for user {subscription['user_id']}")
```

### 2. FastAPI 미들웨어 통합
```python
# parsing/parser/api/middleware/__init__.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
import time
import json
from .rate_limiter import QuestionRateLimiter
from ..auth import get_current_user
from ..database import get_db_session

class TarotRateLimitMiddleware(BaseHTTPMiddleware):
    """타로 상담 질문 제한 미들웨어"""
    
    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.protected_paths = [
            "/api/v1/tarot/consultation",
            "/api/v1/tarot/simple-card",
            "/api/v1/tarot/card-info",
            "/api/v1/tarot/spread-info"
        ]
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # 보호된 경로인지 확인
        if request.url.path in self.protected_paths:
            try:
                # 사용자 인증 확인
                user = await get_current_user(request)
                if not user:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "인증이 필요합니다."}
                    )
                
                # 질문 제한 체크
                db = await get_db_session()
                rate_limiter = QuestionRateLimiter(db)
                rate_check = await rate_limiter.check_rate_limit(user['id'])
                
                # 요청 처리
                response = await call_next(request)
                
                # 성공적인 응답인 경우 사용량 증가
                if response.status_code == 200:
                    # 요청 본문에서 질문 텍스트 추출
                    if request.method == "POST":
                        body = await request.json()
                        question_text = body.get('user_input', '')
                        
                        # 처리 시간 계산
                        processing_time = time.time() - start_time
                        
                        # 응답에서 핸들러 정보 추출 (메타데이터 활용)
                        response_data = json.loads(response.body)
                        handler = response_data.get('responses', [{}])[0].get('handler', 'unknown')
                        
                        # 사용량 증가
                        await rate_limiter.increment_usage(
                            user['id'], question_text, handler, processing_time
                        )
                
                # 응답 헤더에 사용량 정보 추가
                response.headers["X-RateLimit-Limit"] = str(rate_check['limit'])
                response.headers["X-RateLimit-Remaining"] = str(rate_check['remaining'])
                response.headers["X-RateLimit-Reset"] = str(rate_check['reset_date'])
                response.headers["X-Plan-Type"] = rate_check['plan_type']
                
                return response
                
            except HTTPException as e:
                return JSONResponse(
                    status_code=e.status_code,
                    content=e.detail
                )
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"detail": "서버 내부 오류가 발생했습니다."}
                )
        
        # 보호되지 않은 경로는 그대로 처리
        return await call_next(request)
```

---

## 🌐 FastAPI 통합

### 1. 메인 FastAPI 앱
```python
# parsing/parser/api/main.py
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
import asyncio
import sys
import os

# 타로 에이전트 임포트
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tarot_agent.agent import create_optimized_tarot_graph
from tarot_agent.utils.tools import initialize_rag_system

from .middleware import TarotRateLimitMiddleware
from .auth import get_current_user
from .database import init_database

app = FastAPI(
    title="🔮 타로 에이전트 API",
    description="LangGraph 기반 지능형 타로 상담 시스템",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 질문 제한 미들웨어 추가
app.add_middleware(TarotRateLimitMiddleware)

# 타로 에이전트 초기화
tarot_agent = None

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    global tarot_agent
    
    # 데이터베이스 초기화
    await init_database()
    
    # RAG 시스템 초기화
    try:
        initialize_rag_system()
        print("✅ RAG 시스템 초기화 성공!")
    except Exception as e:
        print(f"⚠️ RAG 시스템 초기화 실패: {e}")
    
    # 타로 에이전트 초기화
    try:
        tarot_agent = create_optimized_tarot_graph().compile()
        print("✅ 타로 에이전트 초기화 성공!")
    except Exception as e:
        print(f"❌ 타로 에이전트 초기화 실패: {e}")

# 요청/응답 모델
class TarotRequest(BaseModel):
    user_input: str
    session_id: Optional[str] = None

class TarotResponse(BaseModel):
    success: bool
    responses: list
    total_final_responses: int
    session_id: Optional[str] = None
    usage_info: Optional[dict] = None

# 보호된 엔드포인트들
@app.post("/api/v1/tarot/consultation", response_model=TarotResponse)
async def tarot_consultation(
    request: TarotRequest,
    current_user: dict = Depends(get_current_user)
):
    """타로 상담 엔드포인트"""
    try:
        # 타로 에이전트 실행
        initial_state = {
            "messages": [],
            "user_input": request.user_input,
            "user_intent": "unknown",
            "consultation_data": None,
            "supervisor_decision": None,
            "session_memory": {},
            "conversation_memory": {}
        }
        
        result = tarot_agent.invoke(initial_state)
        
        # 최종 응답만 필터링 (메타데이터 활용)
        final_responses = []
        for message in result.get("messages", []):
            if hasattr(message, 'metadata') and message.metadata.get("final_response") == "yes":
                final_responses.append({
                    "content": message.content,
                    "handler": message.metadata.get("handler"),
                    "timestamp": datetime.now().isoformat()
                })
        
        return TarotResponse(
            success=True,
            responses=final_responses,
            total_final_responses=len(final_responses),
            session_id=request.session_id
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"타로 상담 처리 중 오류가 발생했습니다: {str(e)}"
        )

@app.get("/api/v1/user/usage")
async def get_user_usage(current_user: dict = Depends(get_current_user)):
    """사용자 사용량 조회"""
    try:
        db = await get_db_session()
        
        # 구독 정보 조회
        subscription = await db.fetch_one("""
            SELECT s.*, p.features
            FROM subscriptions s
            JOIN plan_configs p ON s.plan_type = p.plan_type
            WHERE s.user_id = %s AND s.is_active = TRUE
        """, (current_user['id'],))
        
        if not subscription:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="구독 정보를 찾을 수 없습니다."
            )
        
        # 최근 사용 로그 조회
        recent_logs = await db.fetch_all("""
            SELECT question_text, response_handler, processing_time, created_at
            FROM usage_logs
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 10
        """, (current_user['id'],))
        
        return {
            "subscription": dict(subscription),
            "recent_usage": [dict(log) for log in recent_logs],
            "usage_percentage": (subscription['current_usage'] / subscription['question_limit']) * 100
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"사용량 조회 중 오류가 발생했습니다: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2. 데이터베이스 연결
```python
# parsing/parser/api/database.py
import asyncpg
import os
from typing import Optional

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/tarot_db")

class Database:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
    
    async def connect(self):
        """데이터베이스 연결 풀 생성"""
        self.pool = await asyncpg.create_pool(DATABASE_URL)
    
    async def disconnect(self):
        """연결 풀 종료"""
        if self.pool:
            await self.pool.close()
    
    async def execute(self, query: str, *args):
        """쿼리 실행"""
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)
    
    async def fetch_one(self, query: str, *args):
        """단일 결과 조회"""
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)
    
    async def fetch_all(self, query: str, *args):
        """다중 결과 조회"""
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)

# 전역 데이터베이스 인스턴스
db = Database()

async def init_database():
    """데이터베이스 초기화"""
    await db.connect()

async def get_db_session():
    """데이터베이스 세션 반환"""
    return db
```

---

## 🚀 구현 단계

### Phase 1: 기본 인프라 구축 (1주)
1. **데이터베이스 설계 및 구축**
   - PostgreSQL 설치 및 설정
   - 테이블 스키마 생성
   - 기본 데이터 삽입 (플랜 설정)

2. **FastAPI 프로젝트 구조 생성**
   ```
   parsing/parser/api/
   ├── __init__.py
   ├── main.py              # 메인 FastAPI 앱
   ├── auth.py              # 인증 시스템
   ├── database.py          # 데이터베이스 연결
   ├── models.py            # Pydantic 모델
   ├── middleware/
   │   ├── __init__.py
   │   └── rate_limiter.py  # 질문 제한 미들웨어
   └── routers/
       ├── __init__.py
       ├── tarot.py         # 타로 관련 엔드포인트
       └── user.py          # 사용자 관련 엔드포인트
   ```

### Phase 2: 미들웨어 구현 (1주)
1. **질문 제한 미들웨어 개발**
   - `QuestionRateLimiter` 클래스 구현
   - 사용량 체크 로직 구현
   - 월간 리셋 로직 구현

2. **FastAPI 미들웨어 통합**
   - `TarotRateLimitMiddleware` 구현
   - 보호된 경로 설정
   - 에러 처리 로직 구현

### Phase 3: 타로 에이전트 통합 (1주)
1. **타로 에이전트 FastAPI 연동**
   - 기존 콘솔 기반 에이전트를 API로 변환
   - 메타데이터 기반 응답 필터링
   - 세션 관리 구현

2. **엔드포인트 구현**
   - `/api/v1/tarot/consultation` - 타로 상담
   - `/api/v1/tarot/simple-card` - 간단한 카드 뽑기
   - `/api/v1/tarot/card-info` - 카드 정보 조회
   - `/api/v1/tarot/spread-info` - 스프레드 정보 조회

### Phase 4: 인증 시스템 구현 (1주)
1. **사용자 인증 시스템**
   - JWT 기반 인증 구현
   - 회원가입/로그인 엔드포인트
   - 패스워드 해싱 및 검증

2. **사용자 관리 기능**
   - 사용량 조회 엔드포인트
   - 구독 정보 조회
   - 플랜 변경 기능

### Phase 5: 테스트 및 최적화 (1주)
1. **단위 테스트 작성**
   - 미들웨어 테스트
   - 데이터베이스 로직 테스트
   - API 엔드포인트 테스트

2. **성능 최적화**
   - 데이터베이스 쿼리 최적화
   - 캐싱 시스템 구현
   - 비동기 처리 최적화

---

## 🧪 테스트 계획

### 1. 단위 테스트
```python
# tests/test_rate_limiter.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from api.middleware.rate_limiter import QuestionRateLimiter

@pytest.mark.asyncio
async def test_check_rate_limit_success():
    """정상적인 질문 제한 체크 테스트"""
    # Mock 데이터베이스
    mock_db = AsyncMock()
    mock_db.fetch_one.return_value = {
        'user_id': 'test-user',
        'plan_type': 'free',
        'question_limit': 10,
        'current_usage': 5,
        'reset_date': '2024-01-01'
    }
    
    rate_limiter = QuestionRateLimiter(mock_db)
    result = await rate_limiter.check_rate_limit('test-user')
    
    assert result['allowed'] == True
    assert result['remaining'] == 5
    assert result['limit'] == 10

@pytest.mark.asyncio
async def test_check_rate_limit_exceeded():
    """질문 제한 초과 테스트"""
    mock_db = AsyncMock()
    mock_db.fetch_one.return_value = {
        'user_id': 'test-user',
        'plan_type': 'free',
        'question_limit': 10,
        'current_usage': 10,
        'reset_date': '2024-01-01'
    }
    
    rate_limiter = QuestionRateLimiter(mock_db)
    
    with pytest.raises(HTTPException) as exc_info:
        await rate_limiter.check_rate_limit('test-user')
    
    assert exc_info.value.status_code == 429
```

### 2. 통합 테스트
```python
# tests/test_api_integration.py
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_tarot_consultation_with_valid_user():
    """유효한 사용자의 타로 상담 테스트"""
    # 로그인
    login_response = client.post("/api/v1/auth/login", json={
        "username": "test_user",
        "password": "test_password"
    })
    token = login_response.json()["access_token"]
    
    # 타로 상담 요청
    response = client.post(
        "/api/v1/tarot/consultation",
        json={"user_input": "오늘 운세가 어떨까요?"},
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 200
    assert response.json()["success"] == True
    assert "responses" in response.json()

def test_tarot_consultation_rate_limit():
    """질문 제한 테스트"""
    # 로그인
    login_response = client.post("/api/v1/auth/login", json={
        "username": "limited_user",
        "password": "test_password"
    })
    token = login_response.json()["access_token"]
    
    # 제한 횟수만큼 요청
    for i in range(10):  # 무료 플랜 제한
        response = client.post(
            "/api/v1/tarot/consultation",
            json={"user_input": f"질문 {i+1}"},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
    
    # 제한 초과 요청
    response = client.post(
        "/api/v1/tarot/consultation",
        json={"user_input": "제한 초과 질문"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 429
```

### 3. 성능 테스트
```python
# tests/test_performance.py
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

async def test_concurrent_requests():
    """동시 요청 처리 성능 테스트"""
    async def make_request():
        # 타로 상담 요청 시뮬레이션
        start_time = time.time()
        # ... API 호출 로직
        end_time = time.time()
        return end_time - start_time
    
    # 100개 동시 요청
    tasks = [make_request() for _ in range(100)]
    response_times = await asyncio.gather(*tasks)
    
    avg_response_time = sum(response_times) / len(response_times)
    assert avg_response_time < 5.0  # 평균 응답 시간 5초 이내
```

---

## 📊 모니터링 및 관리

### 1. 사용량 대시보드
```python
# api/routers/admin.py
@app.get("/api/v1/admin/usage-stats")
async def get_usage_stats(current_user: dict = Depends(get_admin_user)):
    """관리자용 사용량 통계"""
    db = await get_db_session()
    
    # 플랜별 사용자 수
    plan_stats = await db.fetch_all("""
        SELECT plan_type, COUNT(*) as user_count
        FROM subscriptions
        WHERE is_active = TRUE
        GROUP BY plan_type
    """)
    
    # 일별 질문 수
    daily_questions = await db.fetch_all("""
        SELECT DATE(created_at) as date, COUNT(*) as question_count
        FROM usage_logs
        WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY DATE(created_at)
        ORDER BY date DESC
    """)
    
    return {
        "plan_statistics": [dict(stat) for stat in plan_stats],
        "daily_questions": [dict(stat) for stat in daily_questions]
    }
```

### 2. 알림 시스템
```python
# api/utils/notifications.py
async def send_usage_warning(user_id: str, usage_percentage: float):
    """사용량 경고 알림"""
    if usage_percentage >= 80:
        # 이메일 또는 푸시 알림 발송
        message = f"질문 사용량이 {usage_percentage:.1f}%에 도달했습니다."
        await send_notification(user_id, message)

async def send_limit_exceeded_notification(user_id: str):
    """제한 초과 알림"""
    message = "월간 질문 제한에 도달했습니다. 플랜 업그레이드를 고려해보세요."
    await send_notification(user_id, message)
```

---

---

## 🎯 **간단한 해결책** (추천)

복잡한 데이터베이스 없이 **파일 1개 + 코드 몇 줄**로 해결하는 방법:

### 1. `limit_checker.py` 파일 생성

```python
# parsing/parser/limit_checker.py
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any

class SimpleTurnLimiter:
    """간단한 멀티턴 횟수 제한기"""
    
    def __init__(self, limit: int = 10):
        self.limit = limit
        self.data_file = "user_turns.json"
        self.data = self.load_data()
    
    def load_data(self) -> Dict[str, Any]:
        """데이터 로드"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_data(self):
        """데이터 저장"""
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def check_and_increment(self, user_id: str = "default_user") -> bool:
        """횟수 체크 및 증가"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        if user_id not in self.data:
            self.data[user_id] = {}
        
        user_data = self.data[user_id]
        
        # 날짜가 바뀌면 리셋
        if user_data.get("date") != today:
            user_data["date"] = today
            user_data["count"] = 0
        
        # 제한 체크
        if user_data["count"] >= self.limit:
            return False  # 제한 초과
        
        # 횟수 증가
        user_data["count"] += 1
        self.save_data()
        return True  # 사용 가능
    
    def get_remaining(self, user_id: str = "default_user") -> int:
        """남은 횟수 반환"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        if user_id not in self.data:
            return self.limit
        
        user_data = self.data[user_id]
        
        # 날짜가 바뀌면 리셋
        if user_data.get("date") != today:
            return self.limit
        
        return max(0, self.limit - user_data.get("count", 0))

# 전역 인스턴스
limiter = SimpleTurnLimiter(limit=10)

def check_limit(user_id: str = "default_user") -> bool:
    """제한 체크 함수"""
    return limiter.check_and_increment(user_id)

def get_remaining_count(user_id: str = "default_user") -> int:
    """남은 횟수 조회"""
    return limiter.get_remaining(user_id)
```

### 2. 기존 `agent.py`에 2줄만 추가

```python
# parsing/parser/tarot_agent/agent.py의 main() 함수 수정

# 기존 코드 상단에 추가
USE_LIMIT = True  # 요금제 사용자는 False로 설정

def main():
    # ... 기존 초기화 코드 ...
    
    while True:
        user_input = input("\n사용자: ").strip()
        
        if user_input.lower() in ['quit', 'exit', '종료', '끝', '그만', 'bye']:
            print("🔮 타로 상담이 도움이 되었기를 바랍니다. 좋은 하루 되세요! ✨")
            break
            
        if not user_input:
            print("💬 무엇이든 편하게 말씀해주세요!")
            continue
        
        # ✨ 여기에 제한 체크 추가 (2줄만!)
        if USE_LIMIT:
            from ..limit_checker import check_limit, get_remaining_count
            if not check_limit():
                remaining = get_remaining_count()
                print(f"🚫 일일 질문 제한 10회에 도달했습니다. 내일 다시 이용해주세요!")
                print(f"💎 무제한 이용을 원하시면 프리미엄 플랜을 이용해주세요.")
                continue
            else:
                remaining = get_remaining_count()
                print(f"📊 남은 질문 횟수: {remaining}회")
        
        # ... 기존 처리 코드 계속 ...
```

### 3. 사용법

**무료 사용자 (제한 있음):**
```python
USE_LIMIT = True  # 이렇게 설정
```

**요금제 사용자 (제한 없음):**
```python
USE_LIMIT = False  # 이렇게 설정
```

### 4. 동작 방식

1. **`user_turns.json`** 파일에 사용자별 일일 사용량 저장
2. **매일 자동 리셋** (날짜 체크)
3. **10회 제한** 도달 시 사용 차단
4. **요금제 사용자**는 `USE_LIMIT = False`로 설정해서 체크 안함

### 5. 생성되는 데이터 파일 예시

```json
{
  "default_user": {
    "date": "2024-12-28",
    "count": 7
  },
  "user123": {
    "date": "2024-12-28", 
    "count": 3
  }
}
```

**장점**: 파일 1개 + 코드 몇 줄로 간단하게 멀티턴 횟수 제한 완료! 🎯

---

## 🎯 결론

### 💡 **추천 방법**: 간단한 해결책
- **구현 시간**: 10분
- **필요 파일**: 1개 (`limit_checker.py`)
- **코드 수정**: 기존 파일에 몇 줄 추가
- **데이터베이스**: 불필요 (JSON 파일 사용)

### 🏗️ **확장 방법**: 복잡한 해결책  
- **구현 시간**: 5주
- **필요 시스템**: 데이터베이스 + FastAPI + 미들웨어
- **장점**: 확장성, 모니터링, 다양한 요금제
- **단점**: 복잡성, 개발 시간

**결론**: 일단 간단한 해결책으로 시작하고, 나중에 필요하면 확장하는 것을 추천합니다! 🚀 