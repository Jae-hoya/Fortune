# 🔮 타로 에이전트 시스템 (Tarot Agent System)

LangGraph 기반의 지능형 타로 상담 시스템입니다. 고급 RAG(Retrieval-Augmented Generation) 기술과 멀티모달 AI를 활용하여 개인화된 타로 상담을 제공합니다.

## 📋 목차

- [시스템 개요](#시스템-개요)
- [주요 기능](#주요-기능)
- [LangGraph 아키텍처](#langgraph-아키텍처)
- [설치 및 실행](#설치-및-실행)
- [사용 방법](#사용-방법)
- [노드 함수 상세](#노드-함수-상세)
- [고급 기능](#고급-기능)
- [개발 및 확장](#개발-및-확장)
- [문제 해결](#문제-해결)
- [기여](#기여)
- [라이선스](#라이선스)

## 🎯 시스템 개요

타로 에이전트 시스템은 다음과 같은 핵심 기술을 기반으로 구축되었습니다:

### 🏗️ 기술 스택
- **LangGraph StateGraph**: 복잡한 상담 워크플로우 관리
- **RAG 시스템**: FAISS + BM25 + FlashRank 리랭킹
- **다중 LLM 지원**: OpenAI GPT-4o, GPT-4o-mini
- **웹 검색 통합**: Tavily Search + DuckDuckGo 백업
- **실시간 번역**: 한영 번역 지원

### 🎪 시스템 특징
- **감정 분석**: 사용자 감정 상태 파악 및 공감적 응답
- **확률 계산**: 하이퍼기하분포 기반 성공 확률 분석
- **시간 예측**: 카드 메타데이터와 현재 날짜 기반 타이밍 예측
- **최적화된 성능**: Fast Track 시스템으로 응답 시간 단축

## 🌟 주요 기능

### 1. 🔍 지능형 상담 분류
- **상태 기반 분류**: 진행 중인 상담과 새로운 질문 자동 구분
- **의도 파악**: 카드 정보, 스프레드 정보, 상담 요청 등 자동 분류
- **Fast Track**: 단순 후속 질문은 LLM 없이 빠른 처리

### 2. 🎴 다양한 상담 모드
- **간단한 카드 뽑기**: 질문에 대한 즉석 카드 해석
- **전문 스프레드 상담**: 다양한 타로 스프레드 활용
- **개인화 상담**: 사용자 상황에 맞춤형 조언

### 3. 🧠 고급 분석 기능
- **확률 분석**: 하이퍼기하분포 기반 성공 확률 계산
- **원소 균형 분석**: 4원소(불, 물, 공기, 흙) 균형 평가
- **수비학 분석**: 카드 번호 기반 수비학적 의미 해석
- **시너지 분석**: 카드 조합의 상호작용 효과 분석

### 4. 🌐 웹 검색 통합
- **한국 정보 우선**: 한국 사이트 우선 검색으로 현지화된 정보 제공
- **실시간 정보**: 최신 트렌드와 시장 상황 반영
- **이중 백업**: Tavily + DuckDuckGo 이중 검색 시스템

### 5. ⏰ 시간 예측 시스템
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
    # ... 추가 필드들
```

### 워크플로우 구조
```
START → state_classifier → [라우팅 결정]
                         ↓
                    supervisor_master (복잡한 경우)
                         ↓
                    unified_processor → [의도별 핸들러]
                         ↓
                    unified_tool_handler → END
```

### 핵심 노드들
1. **state_classifier_node**: 상태 기반 빠른 분류
2. **supervisor_master_node**: 복잡한 상황 관리
3. **unified_processor_node**: 통합 처리기
4. **unified_tool_handler_node**: 도구 실행 관리

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

# Tavily API 키 (선택사항 - 더 나은 웹 검색을 위해)
TAVILY_API_KEY=your_tavily_api_key
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
   - 카드 조합 분석
   - 확률 및 시기 예측
   - 실용적 조언

## 🔧 노드 함수 상세

### 분류 및 라우팅 노드
- `state_classifier_node`: 상태 기반 빠른 분류
- `supervisor_master_node`: 복잡한 상황 관리
- `classify_intent_node`: 사용자 의도 분류

### 핸들러 노드
- `card_info_handler`: 카드 정보 검색 및 설명
- `spread_info_handler`: 스프레드 정보 제공
- `simple_card_handler`: 간단한 카드 뽑기
- `consultation_handler`: 전문 상담 시작
- `general_handler`: 일반적인 질문 처리

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

### 도구 및 검색 노드
- `web_search_decider_node`: 웹 검색 필요성 판단
- `web_searcher_node`: 웹 검색 실행
- `unified_tool_handler_node`: 통합 도구 관리

## 🎯 고급 기능

### 1. 확률 계산 시스템
```python
# 하이퍼기하분포 기반 정확한 확률 계산
def calculate_card_draw_probability(deck_size=78, cards_of_interest=1, 
                                  cards_drawn=3, exact_matches=1):
    # 특정 카드가 뽑힐 확률을 수학적으로 계산
    return {
        "exact_probability": float,
        "at_least_one_probability": float,
        "expected_value": float,
        "variance": float
    }
```

### 2. 시간 예측 시스템
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

### 3. 감정 지원 시스템
```python
# 사용자 감정 분석 및 공감적 응답
def analyze_emotion_and_empathy(user_input):
    # 감정 키워드 분석, 강도 측정, 공감 메시지 생성
    return {
        "primary_emotion": str,
        "intensity": float,
        "empathy_message": str
    }
```

### 4. 웹 검색 통합
```python
# 한국 정보 우선 검색
def perform_web_search(query, search_type="general"):
    # 1순위: 한국 사이트 우선 검색
    # 2순위: 일반 검색
    # 백업: DuckDuckGo 검색
    return {
        "results": list,
        "source": str,
        "success": bool
    }
```

## 🛠️ 개발 및 확장

### 프로젝트 구조
```
parsing/parser/tarot_agent/
├── agent.py                 # 메인 에이전트 실행
├── utils/
│   ├── state.py            # 상태 정의
│   ├── nodes.py            # 노드 함수들 
│   ├── helpers.py          # 헬퍼 함수들 
│   ├── tools.py            # RAG 도구들
│   ├── analysis.py         # 확률 및 분석 함수들
│   ├── timing.py           # 시간 예측 함수들
│   ├── web_search.py       # 웹 검색 함수들
│   └── translation.py      # 번역 함수들
└── README.md              # 이 파일
```

### 새로운 노드 추가
```python
def my_custom_node(state: TarotState) -> TarotState:
    """커스텀 노드 함수"""
    user_input = state["user_input"]
    
    # 로직 처리
    result = process_custom_logic(user_input)
    
    # 상태 업데이트
    return {
        "messages": [AIMessage(content=result)],
        "custom_field": "custom_value"
    }

# 그래프에 노드 추가
workflow.add_node("my_custom_node", my_custom_node)
```

### 새로운 핸들러 추가
```python
def custom_handler(state: TarotState) -> TarotState:
    """커스텀 핸들러"""
    # 특정 상황에 대한 처리 로직
    return {"messages": [AIMessage(content="Custom response")]}
```

## 🐛 문제 해결

### 자주 발생하는 문제들

#### 1. RAG 시스템 초기화 실패
```
⚠️ RAG 시스템 초기화 실패: [Error Message]
```
**해결책**: 
- FAISS 인덱스 파일 경로 확인
- 필요한 의존성 설치: `pip install faiss-cpu`

#### 2. OpenAI API 키 오류
```
❌ OpenAI API 키가 설정되지 않았습니다.
```
**해결책**: 
- `.env` 파일에 `OPENAI_API_KEY` 설정
- 환경 변수 로드 확인

#### 3. 웹 검색 실패
```
❌ 모든 웹 검색 도구 실패
```
**해결책**: 
- 인터넷 연결 확인
- Tavily API 키 설정 (선택사항)
- DuckDuckGo 의존성 확인

#### 4. 메모리 부족
```
Memory Error during processing
```
**해결책**: 
- 배치 크기 조정
- 캐시 클리어
- 시스템 메모리 확인

### 디버깅 모드
```python
# 상세한 로그 출력을 위한 디버그 모드
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📊 성능 최적화

### Fast Track 시스템
- **CONSULTATION_ACTIVE**: 진행 중인 상담은 LLM 없이 빠른 처리
- **FOLLOWUP_QUESTION**: 단순 후속 질문 패턴 매칭
- **캐싱**: 번역 결과 및 검색 결과 캐싱

### 메모리 관리
- **NumPy 타입 변환**: 메모리 효율적인 데이터 타입 사용
- **결과 캐싱**: 반복 계산 최소화
- **가비지 컬렉션**: 적절한 메모리 해제

## 🔗 참고 자료

### 기술 문서
- [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/)
- [OpenAI API 문서](https://platform.openai.com/docs)
- [FAISS 문서](https://faiss.ai/)



**🔮 타로 에이전트 시스템으로 더 나은 상담 경험을 만들어보세요!** 