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

## 🏗️ 시스템 아키텍처

```
사용자 질문 → 하이브리드 검색 → 리랭킹 → LLM 생성 → 응답
     ↓              ↓           ↓        ↓        ↓
   입력 처리    벡터DB + BM25   FlashRank  GPT/Gemini  자연어 답변
```

### 핵심 구성 요소

- **벡터 스토어**: ChromaDB 기반 임베딩 저장소
- **임베딩 모델**: BAAI/bge-m3 (다국어 지원)
- **검색 시스템**: EnsembleRetriever (시맨틱 80% + BM25 20%)
- **리랭커**: FlashRank 기반 검색 결과 재정렬
- **LLM**: OpenAI GPT-4o-mini / Google Gemini-2.0-flash

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

### 기본 실행
```bash
poetry run python main.py
```

### 대화형 상담 시작
```
✨ FortuneAI 사주 상담사 ✨
• 자연스러운 대화로 사주팔자를 분석해 드립니다.
• 원하는 내용을 자유롭게 질문해 보세요.
• 새 대화를 시작하려면 'new'를 입력하세요.
• 종료하려면 'q' 또는 'exit'를 입력하세요.

🙋 1985년 3월 15일 오전 10시에 태어났는데 올해 운세가 어떨까요?
🔮 [AI 상담사의 답변]
```

### 명령어
- `new`: 새로운 대화 세션 시작
- `q` 또는 `exit`: 프로그램 종료

## 📁 프로젝트 구조

```
FortuneAI/
├── main.py              # 메인 애플리케이션
├── rag.py               # RAG 파이프라인 구현
├── models.py            # LLM 및 임베딩 모델 설정
├── search.py            # 하이브리드 검색 시스템
├── vector_store.py      # 벡터 스토어 관리
├── reranker.py          # 문서 리랭킹 시스템
├── prompts.py           # 프롬프트 템플릿
├── files/               # 사주팔자 관련 PDF 문서
├── saju_vectordb/       # 사주 벡터 데이터베이스
├── tarot_db/            # 타로 벡터 데이터베이스
├── output_md/           # 파싱된 마크다운 파일
└── pyproject.toml       # 프로젝트 설정 및 의존성
```

## 🔧 주요 모듈

### RagPipeline
```python
from rag import RagPipeline, retrieve_and_generate

# 파이프라인 초기화
pipeline = RagPipeline(retriever, llm, reranker)

# 질의응답 실행
answer = retrieve_and_generate("질문", pipeline, session_id)
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

- **임베딩 정규화**: 검색 정확도 향상
- **하이브리드 검색**: 다양한 검색 방식 결합
- **리랭킹**: 검색 결과 품질 개선
- **대화 기록 관리**: 컨텍스트 길이 최적화
- **GPU 가속**: CUDA 지원으로 처리 속도 향상

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

- **TeddyNote** - *Initial work* - [teddylee777@gmail.com](mailto:teddylee777@gmail.com)

## 🙏 감사의 말

- [LangChain](https://langchain.com/) - RAG 파이프라인 프레임워크
- [ChromaDB](https://www.trychroma.com/) - 벡터 데이터베이스
- [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) - 리랭킹 시스템
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) - 임베딩 모델

---

**FortuneAI**로 AI의 힘을 빌려 전통적인 사주팔자 상담을 현대적으로 경험해보세요! 🌟