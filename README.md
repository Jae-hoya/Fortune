# 🔮 타로 에이전트 시스템 (Tarot Agent System)

LangGraph 기반의 지능형 타로 상담 시스템입니다. 고급 RAG(Retrieval-Augmented Generation) 기술과 과학적 분석을 활용하여 개인화된 타로 상담을 제공합니다.

## 📋 목차

- [시스템 개요](#시스템-개요)
- [주요 기능](#주요-기능)
- [LangGraph 아키텍처](#langgraph-아키텍처)
- [입력 및 흐름 제어](#입력-및-흐름-제어)
- [설치 및 실행](#설치-및-실행)
- [사용 방법](#사용-방법)
- [노드 함수 상세](#노드-함수-상세)
- [고급 기능](#고급-기능)
- [개발 및 확장](#개발-및-확장)
- [문제 해결](#문제-해결)
- [최신 변경사항](#최신-변경사항)
- [기여](#기여)
- [라이선스](#라이선스)

## 🎯 시스템 개요

타로 에이전트 시스템은 다음과 같은 핵심 기술을 기반으로 구축되었습니다:

### 🏗️ 기술 스택
- **LangGraph StateGraph**: 복잡한 상담 워크플로우 관리
- **RAG 시스템**: FAISS + BM25 + FlashRank 리랭킹
- **과학적 분석**: 하이퍼기하분포, 원소 균형, 수비학 (로컬 계산)
- **실시간 번역**: 한영 번역 지원

### 🎪 시스템 특징
- **감정 분석**: 사용자 감정 상태 파악 및 공감적 응답
- **과학적 분석**: 하이퍼기하분포 기반 확률 계산 (비용 무료)
- **시간 예측**: 카드 메타데이터와 현재 날짜 기반 타이밍 예측
- **최적화된 성능**: Fast Track 시스템으로 응답 시간 단축


## 🌟 주요 기능

### 1. 🔍 지능형 상담 분류
- **상태 기반 분류**: 진행 중인 상담과 새로운 질문 자동 구분
- **의도 파악**: 카드 정보, 스프레드 정보, 상담 요청 등 자동 분류
- **Fast Track**: 단순 후속 질문은 LLM 없이 빠른 처리 (2-5초 단축)

### 2. 🎴 다양한 상담 모드
- **간단한 카드 뽑기**: 질문에 대한 즉석 카드 해석
- **전문 스프레드 상담**: 다양한 타로 스프레드 활용
- **개인화 상담**: 사용자 상황에 맞춤형 조언

### 3. 🧠 과학적 분석 기능 (비용 무료)
- **확률 분석**: 하이퍼기하분포 기반 성공 확률 계산
- **원소 균형 분석**: 4원소(불, 물, 공기, 흙) 균형 평가
- **수비학 분석**: 카드 번호 기반 수비학적 의미 해석
- **시너지 분석**: 카드 조합의 상호작용 효과 분석

### 4. ⏰ 시간 예측 시스템
- **카드별 타이밍**: 각 카드의 메타데이터 기반 시기 예측
- **현재 날짜 통합**: 실제 달력과 연계한 구체적 날짜 제시
- **계절별 조언**: 현재 계절과 시기를 고려한 조언

## 🏛️ LangGraph 아키텍처

### 상태 관리 (TarotState)
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
```

### 타로 랭그래프 구조도
```
                              START
                                │
                                ▼
                        ┌─────────────────┐
                        │  state_classifier │
                        │   (상태 분류기)    │
                        └─────────────────┘
                                │
                    ┌───────────┼───────────┐
                    │           │           │
                    ▼           ▼           ▼
        consultation_direct  context_      supervisor_
                           reference_direct  master
                    │           │           │
                    │           │           ▼
                    │           │    ┌─────────────────┐
                    │           │    │ supervisor_master │
                    │           │    │  (복합 분석기)    │
                    │           │    └─────────────────┘
                    │           │           │
                    └───────────┼───────────┘
                                ▼
                        ┌─────────────────┐
                        │ unified_processor │
                        │   (통합 처리기)    │
                        └─────────────────┘
                                │
                        ┌───────┼───────┐
                        │       │       │
                        ▼       │       ▼
                     tools      │      END
                        │       │
                        ▼       │
                ┌─────────────────┐   │
                │unified_tool_    │   │
                │   handler       │   │
                │ (도구 실행기)    │   │
                └─────────────────┘   │
                        │             │
                        └─────────────┘
                        │
                        ▼
                      END
```

📋 주요 노드별 기능
🎯 1단계: 상태 분류 (state_classifier)
Fast Track 감지: 상담 진행중/완료 후 추가질문
Complex Analysis: 새로운 세션이나 복잡한 상황

🧠 2단계: 라우팅 결정
consultation_direct: 상담 진행중 → 바로 처리
context_reference_direct: 추가 질문 → 맥락 참조
supervisor_master: 복합 분석 필요 → LLM 판단

⚙️ 3단계: 통합 처리 (unified_processor)
핸들러 맵핑:
consultation_handler - 타로 상담 시작
card_info_handler - 카드 정보 질문
spread_info_handler - 스프레드 정보 질문
simple_card_handler - 간단한 카드 뽑기
general_handler - 일반 대화/날짜 질문
context_reference_handler - 추가 질문 처리

🔧 4단계: 도구 실행 (선택적)
RAG 검색: search_tarot_cards, search_tarot_spreads
과학적 분석: 확률계산, 원소균형, 수비학
번역: 한영/영한 번역


## 🎯 1. 입력 및 흐름 제어

사용자 입력
    ↓
state_classifier_node (상태 분류기)
    ↓
state_router (상태 라우팅)
    ↓
라우팅 결정 (CONSULTATION_ACTIVE/FOLLOWUP_QUESTION/NEW_SESSION)

### 📥 **1단계: 사용자 입력 처리**
```
사용자 메시지 → TarotState.messages에 추가
```

### 🔍 **2단계: state_classifier_node (상태 분류기)**
```
┌─────────────────────────────────────┐
│         상태 기반 빠른 분류             │
├─────────────────────────────────────┤
│ Step 1: 명확한 상태 체크 (LLM 없이)    │
│ • consultation_data.status 확인      │
│   - "spread_selection" → Fast Track │
│   - "card_selection" → Fast Track   │
│   - "summary_shown" → Fast Track    │
│   - "completed" → 추가질문 체크       │
├─────────────────────────────────────┤
│ Step 2: 복잡한 경우만 LLM 사용        │
│ • routing_decision 결정:            │
│   - CONSULTATION_ACTIVE             │
│   - FOLLOWUP_QUESTION               │
│   - NEW_SESSION                     │
└─────────────────────────────────────┘
```

### 🔀 **3단계: state_router (상태 라우팅)**
```
routing_decision 값에 따른 분기:

┌─ CONSULTATION_ACTIVE ──→ consultation_direct
│
├─ FOLLOWUP_QUESTION ──→ context_reference_direct  
│
└─ NEW_SESSION ──→ supervisor_master
```

### 🧠 **4단계: supervisor_master_node (복합 분석)**
```
NEW_SESSION인 경우에만 실행:

┌─────────────────────────────────────┐
│        supervisor_llm_node          │
│ • 대화 맥락 분석 (LLM 사용)           │
│ • Follow-up vs New Topic 판단       │
│ • 타로 상담 키워드 체크               │
├─────────────────────────────────────┤
│      classify_intent_node           │
│ • 트리거 기반 의도 분류               │
│ • "타로 봐줘" → consultation         │
│ • "카드 뽑아" → simple_card         │
│ • 카드/스프레드 질문 → 해당 intent   │
├─────────────────────────────────────┤
│    determine_target_handler         │
│ • 최종 핸들러 결정                   │
│ • target_handler 설정               │
└─────────────────────────────────────┘
```

### ⚡ **Fast Track vs Full Analysis**
```
🚀 Fast Track (LLM 호출 최소화):
• 상담 진행중 → 바로 해당 핸들러로
• 간단한 추가질문 → 패턴 매칭으로 처리
• 2-5초 응답시간 단축

🧠 Full Analysis (LLM 활용):
• 새로운 세션
• 복잡한 맥락 판단 필요
• 의도가 불분명한 경우
```
## 🎯 2. 핵심 처리 및 분석

┌─────────────────────────────────────────┐
│        Unified Processor (통합 처리기)        │
├─────────────────────────────────────────┤
│           TarotState (통합 상태 관리)          │
├─────────────────────────────────────────┤
│  메시지     │  사용자 의도  │  상담 데이터  │
│  상담 데이터  │  분석 결과   │  세션 메모리  │
├─────────────────────────────────────────┤
│            핵심 기능 모듈 (Utils)            │
├─────────────────────────────────────────┤
│ analysis.py: 과학적 분석 (비용 무료)        │
│ • 하이퍼기하분포 확률 계산                    │
│ • 원소 균형 분석                           │
│ • 수비학 분석                             │
│ • 카드 조합 시너지 분석                      │
├─────────────────────────────────────────┤
│ helpers.py: LLM 기반 분석 (비용 발생)       │
│ • 감정 분석 및 공감 메시지                    │
│ • 키워드 추출                             │
│ • 고민 판단                               │
├─────────────────────────────────────────┤
│ translation.py: 번역 (비용 발생)           │
│ • 한영/영한 번역                          │
│ • 타로 용어 번역                          │
├─────────────────────────────────────────┤
│ timing.py: 시간 분석 (비용 무료)            │
│ • 현재 시간/날짜 정보                       │
│ • 계절/요일 분석                          │
├─────────────────────────────────────────┤
│ RAG 시스템: 카드/스프레드 DB (비용 무료)      │
│ • FAISS 벡터 검색                         │
│ • 하이브리드 검색                          │
└─────────────────────────────────────────┘

## 🎯 3. 최종 답변 생성

핸들러 함수 매핑 (LLM 기반)
    ↓
• consultation_handler: 타로 상담 시작
• card_info_handler: 카드 정보 질문  
• spread_info_handler: 스프레드 정보
• simple_card_handler: 간단한 카드 뽑기
• general_handler: 일반 대화/날짜 질문
• context_reference_handler: 추가 질문
    ↓
processor_router (도구 호출 체크)
    ↓
필요시 unified_tool_handler 실행
    ↓
최종 응답 생성

## 핵심 특징
Fast Track 시스템
-상담 진행중: LLM 호출 없이 바로 처리
-추가 질문: 패턴 매칭으로 즉시 라우팅

스마트 라우팅
-State 계층 통한 데이터 관리
-필요시에만 Tool 호출
-사용자 맥락 및 상담 상태 추적
-실시간 검색 및 RAG 연동


## 🚀 설치 및 실행

### 1. 환경 설정
```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정
`.env` 파일을 생성하고 다음 내용을 추가하세요:

```env
# OpenAI API 키 (필수)
OPENAI_API_KEY=your_openai_api_key
```

### 3. 실행
```bash
# 메인 에이전트 실행
python -m parsing.parser.tarot_agent.agent

# 또는 직접 실행
cd parsing/parser/tarot_agent
python agent.py
```

## 📱 사용 방법

### 기본 사용법
```
🔮 타로 상담사: 안녕하세요! 타로 상담사입니다. 오늘은 어떤 도움이 필요하신가요?

사용자: 오늘 면접 잘 볼 수 있을까요?

🔮 타로 상담사: [카드 한 장 뽑기 또는 전문 상담 제안]
```

### 다양한 질문 유형

#### 1. 간단한 질문
```
사용자: 짬뽕 vs 짜장면 뭐가 좋을까?
→ 카드 한 장으로 즉석 답변
```

#### 2. 카드 정보 질문
```
사용자: 연인 카드 의미가 뭐야?
→ RAG 시스템으로 상세 정보 제공
```

#### 3. 스프레드 정보 질문
```
사용자: 켈틱 크로스 스프레드 어떻게 써?
→ 스프레드 사용법과 특징 설명
```

#### 4. 전문 상담 요청
```
사용자: 취업 관련해서 타로 봐줘
→ 상황 분석 → 스프레드 추천 → 카드 선택 → 종합 해석
```

### 상담 진행 과정

1. **상황 분석**: 사용자의 고민과 상황 파악
2. **스프레드 추천**: 상황에 맞는 최적의 스프레드 제안
3. **카드 선택**: 사용자가 직접 카드 번호 선택
4. **종합 해석**: 
   - 개별 카드 해석
   - 과학적 분석 (확률, 원소, 수비학)
   - 시기 예측
   - 실용적 조언

## 🔧 노드 함수 상세

### 분류 및 라우팅 노드
- `state_classifier_node`: 상태 기반 빠른 분류
- `supervisor_master_node`: 복잡한 상황 관리
- `classify_intent_node`: 사용자 의도 분류
- `state_router`: 상태 기반 라우팅
- `processor_router`: 도구 호출 체크

### 핸들러 노드
- `card_info_handler`: 카드 정보 검색 및 설명
- `spread_info_handler`: 스프레드 정보 제공
- `simple_card_handler`: 간단한 카드 뽑기
- `consultation_handler`: 전문 상담 시작
- `general_handler`: 일반적인 질문 처리
- `context_reference_handler`: 추가 질문 처리

### 상담 진행 노드
- `consultation_flow_handler`: 상담 흐름 관리
- `consultation_continue_handler`: 상담 진행
- `consultation_individual_handler`: 개별 카드 해석
- `consultation_summary_handler`: 상담 요약
- `consultation_final_handler`: 상담 마무리

### 분석 노드
- `emotion_analyzer_node`: 감정 분석
- `situation_analyzer_node`: 상황 분석
- `spread_recommender_node`: 스프레드 추천
- `card_count_inferrer_node`: 필요한 카드 수 추론
- `spread_extractor_node`: 스프레드 추출
- `status_determiner_node`: 상태 결정

### 도구 및 검색 노드
- `unified_tool_handler_node`: 통합 도구 관리

## 🎯 고급 기능

### 1. 과학적 분석 시스템 (비용 무료)
```python
# 하이퍼기하분포 기반 정확한 확률 계산
def calculate_card_draw_probability(deck_size=78, cards_of_interest=1, 
                                  cards_drawn=3, exact_matches=1):
    from scipy.stats import hypergeom
    rv = hypergeom(deck_size, cards_of_interest, cards_drawn)
    return {
        "exact_probability": float(rv.pmf(exact_matches)),
        "at_least_one_probability": float(1 - rv.pmf(0)),
        "expected_value": float(rv.mean()),
        "variance": float(rv.var())
    }
```

### 2. 원소 균형 분석
```python
def analyze_elemental_balance(selected_cards):
    # 4원소(불, 물, 공기, 흙) 균형 분석
    elements = {"Fire": 0, "Water": 0, "Air": 0, "Earth": 0}
    # 카드별 원소 계산 및 균형 점수 산출
    return {
        "elements": elements,
        "balance_score": float,
        "dominant_element": str,
        "interpretation": str
    }
```

### 3. 수비학 분석
```python
def calculate_numerological_significance(selected_cards):
    # 카드 번호 기반 수비학 계산
    total = sum(card.get("number", 0) for card in selected_cards)
    reduced_value = reduce_to_single_digit(total)
    return {
        "total_value": total,
        "reduced_value": reduced_value,
        "is_master_number": bool,
        "meaning": str
    }
```

### 4. 시간 예측 시스템
```python
# 카드별 메타데이터 기반 시기 예측
def predict_timing_from_card_metadata(card_info):
    # 카드의 원소, 랭크, 방향을 고려한 시기 예측
    return {
        "days_min": int,
        "days_max": int,
        "time_frame": str,
        "confidence": str
    }
```

### 5. 감정 지원 시스템 (LLM 기반)
```python
# 사용자 감정 분석 및 공감적 응답
def analyze_emotion_and_empathy(user_input):
    # 감정 키워드 분석, 강도 측정, 공감 메시지 생성
    return {
        "primary_emotion": str,
        "emotion_intensity": str,
        "empathy_tone": str,
        "comfort_message": str,
        "response_style": str
    }
```

## 💰 비용 최적화

### 무료 기능 (로컬 계산)
- **과학적 분석**: 하이퍼기하분포, 원소 균형, 수비학, 시너지 분석
- **시간 분석**: 날짜/시간 정보, 계절 분석
- **RAG 검색**: FAISS 벡터 검색, 하이브리드 검색

### 유료 기능 (LLM API 호출)
- **감정 분석**: 사용자 감정 상태 파악
- **번역**: 한영/영한 번역
- **LLM 응답**: 최종 답변 생성


## 🛠️ 개발 

### 프로젝트 구조
```
parsing/parser/tarot_agent/
├── agent.py                 # 메인 에이전트 실행
├── utils/
│   ├── state.py            # 상태 정의
│   ├── nodes.py            # 노드 함수들 
│   ├── helpers.py          # 헬퍼 함수들 
│   ├── tools.py            # RAG 도구들
│   ├── analysis.py         # 과학적 분석 함수들 (무료)
│   ├── timing.py           # 시간 예측 함수들 (무료)
│   └── translation.py      # 번역 함수들 (유료)
└── README.md              # 이 파일
```

## 📊 성능 최적화

### Fast Track 시스템
- **CONSULTATION_ACTIVE**: 진행 중인 상담은 LLM 없이 빠른 처리
- **FOLLOWUP_QUESTION**: 단순 후속 질문 패턴 매칭
- **캐싱**: 번역 결과 및 검색 결과 캐싱

### 메모리 관리
- **NumPy 타입 변환**: 메모리 효율적인 데이터 타입 사용
- **결과 캐싱**: 반복 계산 최소화

