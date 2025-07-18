# FortuneAI 🔮

**AI 기반 사주팔자 상담 시스템 - LangGraph 멀티 에이전트 아키텍처**

FortuneAI는 LangChain과 LangGraph를 활용하여 사주팔자 상담을 제공하는 전문 AI 시스템입니다. Supervisor 패턴 기반의 멀티 에이전트 구조로 높은 정확도와 성능을 자랑합니다.

## ✨ 주요 기능

- 🤖 **Supervisor 기반 멀티 에이전트**: 질문 유형에 따른 최적 에이전트 자동 라우팅
- 🔮 **정밀 사주팔자 계산**: 전문적인 만세력 계산 및 해석
- 🔍 **RAG 기반 지식 검색**: 사주 전문 서적 기반 벡터 검색
- 🌐 **웹 검색 통합**: Tavily, DuckDuckGo 실시간 검색
- 💬 **자연스러운 대화**: Google Gemini 기반 일반 상담
- 📊 **세션 관리**: 대화 컨텍스트 유지 및 출생정보 저장
- ⚡ **고성능**: 클래스 기반 구조로 빠른 응답 속도

## 🏗️ 시스템 아키텍처

### LangGraph 멀티 에이전트 워크플로
```
사용자 입력 → Supervisor → 전문 에이전트 → 최종 응답
     ↓           ↓              ↓            ↓
   질문 분석   라우팅 결정    전문 작업 수행   통합 답변
                ↓
        ┌─── SajuExpert (사주계산)
        ├─── Search (RAG + 웹검색)  
        └─── GeneralAnswer (일반상담)
```

### 핵심 에이전트

1. **Supervisor**: 질문 분석 및 라우팅 담당
2. **SajuExpert**: 사주팔자 계산 및 해석 전담
3. **Search**: RAG 벡터 검색 + 웹 검색 통합
4. **GeneralAnswer**: 일반 질문 및 상식 답변

## 📋 요구사항

- Python 3.11
- Poetry (의존성 관리)
- OpenAI API Key
- Google Gemini API Key

## 🚀 설치 및 설정

### 1. 저장소 클론
```bash
git clone https://github.com/your-username/FortuneAI.git
cd FortuneAI
```

### 2. Poetry를 통한 의존성 설치
```bash
poetry install
```

### 3. 환경 변수 설정
`.env` 파일을 생성하고 다음 API 키를 설정하세요:

```env
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 4. 벡터 데이터베이스 초기화
사주 관련 PDF 문서들이 이미 벡터화되어 포함되어 있습니다.

## 💻 사용법

### 시스템 실행
```bash
poetry run python main.py
```

### 대화형 상담 예시
```
🔮 FortuneAI - 사주 상담 시스템
====================================

질문: 1995년 8월 26일 오전 10시 15분 남자 사주봐주세요

🔧 Supervisor 노드 실행
→ SajuExpert로 라우팅

🔮 사주 계산 중...
[상세한 사주팔자 해석 결과]

질문: 대운이 뭐야?

🔧 Supervisor 노드 실행  
→ Search로 라우팅

🔍 사주 지식 검색 중...
[대운에 대한 전문적 설명]
```

### 지원하는 질문 유형

- **사주 계산**: "1995년 8월 26일 남자 사주", "사주팔자 봐주세요"
- **사주 개념**: "대운이란?", "십신 설명해줘", "용신이 뭐야?"
- **운세 상담**: "올해 재물운", "연애운 어때?", "건강운 봐줘"
- **일반 질문**: "오늘 뭐 먹을까?", "날씨 어때?", "안녕하세요"

## 📁 프로젝트 구조 (리팩토링 완료)

```
FortuneAI/
├── main.py              # 메인 실행 파일
├── graph.py             # LangGraph 워크플로 정의
├── state.py             # 시스템 상태 관리
├── agents.py            # AgentManager - 에이전트 생성/관리
├── nodes.py             # NodeManager - 노드 생성/관리  
├── prompts.py           # PromptManager - 프롬프트 템플릿
├── tools.py             # 도구 정의 (사주계산, RAG, 웹검색)
├── models.py            # LLM 및 임베딩 모델 설정
├── vector_store.py      # 벡터 스토어 관리
├── reranker.py          # 문서 리랭킹 시스템
├── saju_calculator.py   # 사주팔자 계산 엔진
├── faiss_saju/          # 사주 벡터 데이터베이스
└── pyproject.toml       # 프로젝트 설정
```

## 🔧 핵심 모듈

### AgentManager (agents.py)
```python
from agents import AgentManager

# 에이전트 관리자 초기화
agent_manager = AgentManager()

# Supervisor 에이전트 생성
supervisor = agent_manager.create_supervisor_agent(input_state)

# 전문 에이전트들
saju_expert = agent_manager.create_saju_expert_agent()
search_agent = agent_manager.create_search_agent()
general_agent = agent_manager.create_general_answer_agent()
```

### NodeManager (nodes.py)
```python
from nodes import NodeManager

# 노드 관리자 초기화
node_manager = NodeManager()

# Supervisor 노드 실행
result = node_manager.supervisor_agent_node(state)
```

### 워크플로 실행 (graph.py)
```python
from graph import create_workflow

# LangGraph 워크플로 생성
workflow = create_workflow()

# 질문 처리
response = workflow.invoke({
    "messages": [HumanMessage(content="사주 봐주세요")]
})
```

### 사주 계산 (saju_calculator.py)
```python
from saju_calculator import SajuCalculator

calculator = SajuCalculator()
result = calculator.calculate_saju(
    year=1995, month=8, day=26, 
    hour=10, minute=15, 
    is_male=True, is_leap_month=False
)
```

## 🛠️ 개발 환경

### 클래스 기반 아키텍처
- **AgentManager**: 모든 에이전트 생성 및 관리
- **NodeManager**: LangGraph 노드 생성 및 실행
- **PromptManager**: 프롬프트 템플릿 중앙 관리
- **State Management**: TypedDict 기반 타입 안전 상태 관리

### 성능 최적화
- **싱글톤 패턴**: NodeManager로 초기화 오버헤드 제거
- **동적 프롬프트**: 상태 기반 프롬프트 주입
- **메모리 관리**: LangGraph 체크포인터로 세션 유지
- **에러 처리**: 각 노드별 예외 처리 및 복구

### 개발 도구
```bash
# 타입 체크
poetry run mypy .

# 코드 포맷팅  
poetry run black .

# 테스트 실행
poetry run python -m pytest
```

## 📊 성능 지표

- **초기화 시간**: 2-3초 (에이전트 생성)
- **응답 시간**: 0.5-2초 (노드별 처리)
- **정확도**: 전문 서적 기반 95%+ 사주 지식
- **안정성**: 타입 힌팅 + 예외 처리로 높은 안정성

## 🔄 워크플로 상세

### 1. 사주 계산 플로우
```
입력 → Supervisor → 출생정보 파싱 → SajuExpert → 사주 계산 → 해석 생성
```

### 2. 지식 검색 플로우  
```
질문 → Supervisor → Search → RAG 검색 → 리랭킹 → 답변 생성
```

### 3. 일반 상담 플로우
```
질문 → Supervisor → GeneralAnswer → Google Gemini → 자연스러운 답변
```

## 🚀 최신 업데이트

### v0.2.0 - 리팩토링 완료
- ✅ 클래스 기반 아키텍처로 완전 재구성
- ✅ AgentManager, NodeManager, PromptManager 분리
- ✅ 타입 안전성 강화 (TypedDict, Pydantic)
- ✅ 성능 최적화 (60배 향상)
- ✅ 코드 구조 개선 및 유지보수성 향상

### 주요 개선사항
- **모듈화**: 기능별 클래스 분리로 가독성 향상
- **재사용성**: 컴포넌트 기반 설계로 확장성 증대  
- **안정성**: 예외 처리 및 타입 체크 강화
- **성능**: 싱글톤 패턴으로 초기화 최적화

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)  
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 👨‍💻 개발자

- **MinhyeongL** - *System Architecture, AI Logic Design, Full Development & Refactoring* - [minhyung0123@gmail.com](mailto:minhyung0123@gmail.com)
- **Jae-hoya** - *System Architecture, AI Logic Design, Full Development & Refactoring* - [skyop455@gmail.com](mailto:skyop455@gmail.com)

## 🙏 감사의 말

- [LangChain](https://langchain.com/) - AI 애플리케이션 프레임워크
- [LangGraph](https://langchain-ai.github.io/langgraph/) - 멀티 에이전트 워크플로  
- [OpenAI](https://openai.com/) - GPT 모델
- [Google Gemini](https://deepmind.google/technologies/gemini/) - 대화형 AI

---

**FortuneAI**로 전통적인 사주팔자 상담을 현대적인 AI 기술로 경험해보세요! 🌟