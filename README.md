# FortuneAI 🔮

**AI 기반 사주팔자 상담 시스템**

FortuneAI는 LangChain과 RAG(Retrieval-Augmented Generation) 기술을 활용하여 사주팔자와 타로 관련 질문에 대해 전문적이고 자연스러운 상담을 제공하는 AI 시스템입니다.

## ✨ 주요 기능

- 🤖 **AI 사주 상담사**: 자연스러운 대화를 통한 사주팔자 분석
- 🔍 **하이브리드 검색**: 시맨틱 검색과 BM25를 결합한 고도화된 문서 검색
- 📚 **전문 지식 기반**: 사주팔자 및 타로 관련 전문 서적 데이터베이스
- 💬 **대화 기록 관리**: 세션별 대화 컨텍스트 유지
- 🎯 **리랭킹 시스템**: FlashRank를 활용한 검색 결과 최적화
- 🌐 **다중 LLM 지원**: OpenAI GPT, Google Gemini 모델 지원
- 🔀 **LangGraph 워크플로**: Supervisor 패턴 기반 멀티 에이전트 시스템

## 🏗️ 시스템 아키텍처

### 단일 에이전트 시스템
```
사용자 질문 → 하이브리드 검색 → 리랭킹 → LLM 생성 → 응답
     ↓              ↓           ↓        ↓        ↓
   입력 처리    벡터DB + BM25   FlashRank  GPT/Gemini  자연어 답변
```

### LangGraph 멀티 에이전트 시스템
```
사용자 입력 → Supervisor → SajuAgent/RagAgent/WebAgent → ResultGenerator → 최종 응답
     ↓           ↓              ↓                              ↓            ↓
   질문 분류   라우팅 결정    전문 작업 수행                응답 생성      통합 답변
```

![LangGraph 워크플로](./langgraph_logic.png)

### 핵심 구성 요소

- **벡터 스토어**: ChromaDB 기반 임베딩 저장소
- **임베딩 모델**: BAAI/bge-m3 (다국어 지원)
- **검색 시스템**: EnsembleRetriever (시맨틱 80% + BM25 20%)
- **리랭커**: FlashRank 기반 검색 결과 재정렬
- **LLM**: OpenAI GPT-4.1-mini / Google Gemini-2.0-flash
- **워크플로 엔진**: LangGraph 기반 멀티 에이전트 시스템

## 📋 요구사항

- Python 3.11
- Poetry (의존성 관리)
- CUDA (선택사항, GPU 가속)

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
```

### 4. 벡터 데이터베이스 초기화
```bash
poetry run python doc_parse.ipynb  # 문서 파싱 및 벡터화
```

## 💻 사용법

### 단일 에이전트 시스템 실행
```bash
poetry run python main.py
```

### LangGraph 멀티 에이전트 시스템 실행 (권장)
```bash
poetry run python main_langgraph.py
```

### 대화형 상담 시작
```
🔮 FortuneAI - LangGraph 사주 시스템 🔮
============================================================
✨ Supervisor 패턴 기반 고성능 사주 계산기
🎯 98점 전문가 검증 완료
🚀 LangGraph 멀티 워커 시스템
------------------------------------------------------------
📝 사용법:
  • 사주 계산: '1995년 8월 26일 오전 10시 15분 남자 사주'
  • 운세 상담: '1995년 8월 26일생 2024년 연애운'
  • 일반 검색: '사주에서 십신이란?'
  • 종료: 'quit' 또는 'exit'
  • 디버그: 'debug:질문' (상세 실행 정보)
  • 상세모드: 'verbose:질문' (노드별 상세 로깅)
============================================================

🤔 질문: 1992년 8월 15일 오전 10시에 태어났습니다. 사주를 봐주세요.
🔮 [AI 상담사의 답변]
```

### 명령어
- `new`: 새로운 대화 세션 시작
- `help`: 사용 가이드 표시
- `debug:질문`: 디버그 모드로 실행
- `verbose:질문`: 상세 모드로 실행
- `q` 또는 `exit`: 프로그램 종료

## 📁 프로젝트 구조

```
FortuneAI/
├── main.py              # 단일 에이전트 시스템 애플리케이션
├── main_langgraph.py    # LangGraph 멀티 에이전트 시스템 (권장)
├── rag.py               # RAG 파이프라인 구현
├── models.py            # LLM 및 임베딩 모델 설정
├── search.py            # 하이브리드 검색 시스템
├── vector_store.py      # 벡터 스토어 관리
├── reranker.py          # 문서 리랭킹 시스템
├── prompts.py           # 프롬프트 템플릿
├── tools.py             # LangGraph 도구 모음 (사주계산, RAG, 웹검색)
├── saju_calculator.py   # 정밀 사주팔자 계산 엔진
├── langgraph_system/    # LangGraph 멀티 에이전트 시스템
│   ├── __init__.py      # 패키지 초기화
│   ├── state.py         # 워크플로 상태 정의
│   ├── agents.py        # AgentManager (Supervisor, 전문 에이전트들)
│   ├── nodes.py         # NodeManager (싱글톤, 노드 생성)
│   └── graph.py         # 워크플로 그래프 구성
├── files/               # 사주팔자 관련 PDF 문서
├── saju_vectordb/       # 사주 벡터 데이터베이스
├── tarot_db/            # 타로 벡터 데이터베이스
├── output_md/           # 파싱된 마크다운 파일
├── langgraph_logic.png  # LangGraph 워크플로 다이어그램
└── pyproject.toml       # 프로젝트 설정 및 의존성
```

## 🔧 주요 모듈

### LangGraph 멀티 에이전트 시스템 (권장)
```python
from main_langgraph import run_query_with_app, app

# LangGraph 워크플로로 사주 상담
response = await run_query_with_app(app, "1992년 8월 15일 오전 10시 사주")

# 디버그 모드
response = await run_query_with_app(app, "debug:질문")

# 상세 모드
response = await run_query_with_app(app, "verbose:질문")
```

### LangGraph 시스템 구성
```python
from langgraph_system import (
    AgentManager, NodeManager, create_workflow, State
)

# 에이전트 관리자 초기화
agent_manager = AgentManager()
supervisor = agent_manager.create_supervisor_agent()

# 노드 관리자 (싱글톤)
node_manager = NodeManager()
workflow = create_workflow()
```

### RagPipeline (단일 에이전트 시스템)
```python
from rag import RagPipeline, retrieve_and_generate

# 파이프라인 초기화
pipeline = RagPipeline(retriever, llm, reranker)

# 질의응답 실행
answer = retrieve_and_generate("질문", pipeline, session_id)
```

### 사주 계산 엔진
```python
from saju_calculator import SajuCalculator
from tools import ToolManager

# 정밀 사주팔자 계산
calculator = SajuCalculator()
saju_data = calculator.calculate_saju(
    birth_date="1992-08-15", 
    birth_time="10:00",
    is_male=True
)

# 도구 관리자
tool_manager = ToolManager()
saju_tools = tool_manager.calendar_tools
rag_tools = tool_manager.rag_tools
```

### 하이브리드 검색
```python
from search import create_hybrid_retriever

# 시맨틱 검색 + BM25 결합
retriever = create_hybrid_retriever(
    vectorstore=vectorstore,
    documents=all_docs,
    weights=[0.8, 0.2],  # 시맨틱 80%, BM25 20%
    top_k=20
)
```

## 📊 성능 최적화

### 단일 에이전트 시스템
- **임베딩 정규화**: 검색 정확도 향상
- **하이브리드 검색**: 다양한 검색 방식 결합
- **리랭킹**: 검색 결과 품질 개선
- **대화 기록 관리**: 컨텍스트 길이 최적화
- **GPU 가속**: CUDA 지원으로 처리 속도 향상

### LangGraph 멀티 에이전트 시스템 (신규)
- **Manager 패턴**: 전문화된 에이전트 역할 분담
- **Supervisor 라우팅**: 질문 유형별 최적 워크플로 선택
- **싱글톤 노드 관리**: 6-10초 초기화 → 0.1초 응답 (성능 60배 향상)
- **Pydantic 타입 안전성**: 구조화된 출력으로 안정성 확보
- **조건부 워크플로**: 생년월일시 → 사주계산 → 해석 순차 처리
- **무한 루프 방지**: 애매한 입력에 대한 조기 종료 로직

## 🛠️ 개발 환경

### Jupyter Notebook 지원
- `test.ipynb`: 기능 테스트 및 실험
- `langgraph.ipynb`: LangGraph 실험
- `doc_parse.ipynb`: 문서 파싱 및 전처리

### 개발 도구
```bash
# 개발 의존성 설치
poetry install --with dev

# 타입 체크
poetry run mypy .

# 테스트 실행
poetry run python test.ipynb
```

## 📚 데이터 소스

프로젝트에 포함된 전문 서적:
- Four Pillar Unveil Your Destiny (사주팔자 완전 가이드)
- The Complete Guide to the Tarot (타로 완전 가이드)
- Simple Chinese Astrology (중국 점성술)
- Chinese Horoscope (중국 운세)

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 👨‍💻 개발자

- **MinhyeongL** - *Initial work* - [minhyung0123@gmail.com](mailto:minhyung0123@gmail.com)

## 🙏 감사의 말

- [LangChain](https://langchain.com/) - RAG 파이프라인 프레임워크
- [ChromaDB](https://www.trychroma.com/) - 벡터 데이터베이스
- [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) - 리랭킹 시스템
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) - 임베딩 모델

---

**FortuneAI**로 AI의 힘을 빌려 전통적인 사주팔자 상담을 현대적으로 경험해보세요! 🌟