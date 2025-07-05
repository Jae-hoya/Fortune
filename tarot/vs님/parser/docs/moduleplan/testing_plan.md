# 타로 LangGraph 모듈화 테스트 계획

## 📋 개요

이 문서는 `tarot_langgraph.py` 파일을 LangGraph 표준 구조에 맞게 모듈화한 후 기능 검증을 위한 테스트 계획을 설명합니다. 모듈화된 코드가 기존 코드와 동일하게 작동하는지 확인하기 위한 체계적인 테스트 전략을 제시합니다.

## 🎯 테스트 목표

1. **기능 동일성 검증**: 모듈화된 코드가 기존 코드와 동일한 기능을 제공하는지 확인
2. **성능 유지 확인**: 응답 시간, 메모리 사용량 등이 기존과 동일하거나 개선되었는지 확인
3. **오류 발견 및 수정**: 모듈화 과정에서 발생할 수 있는 오류 식별 및 수정
4. **코드 품질 향상**: 모듈화된 코드의 가독성, 유지보수성 향상 여부 확인

## 🔍 테스트 범위

### 1. 모듈 단위 테스트

각 모듈의 핵심 기능을 독립적으로 테스트합니다:

- **state.py**: `TarotState` 클래스 정의 및 기능 테스트
- **tools.py**: `search_tarot_cards`, `search_tarot_spreads` 등의 도구 함수 테스트
- **web_search.py**: 웹 검색 관련 함수 테스트
- **analysis.py**: 카드 분석 관련 함수 테스트
- **timing.py**: 시간/타이밍 관련 함수 테스트
- **translation.py**: 번역 관련 함수 테스트
- **helpers.py**: 기타 헬퍼 함수 테스트
- **nodes.py**: 주요 노드 함수 테스트

### 2. 통합 테스트

여러 모듈이 함께 작동하는 통합 기능을 테스트합니다:

- **그래프 생성**: 그래프가 올바르게 생성되는지 확인
- **노드 연결**: 노드 간 연결이 올바르게 작동하는지 확인
- **상태 흐름**: 상태가 노드 간에 올바르게 전달되는지 확인
- **도구 호출**: 도구 호출이 올바르게 작동하는지 확인

### 3. 엔드투엔드 테스트

전체 시스템의 엔드투엔드 기능을 테스트합니다:

- **사용자 시나리오**: 다양한 사용자 시나리오에 대한 테스트
- **에러 처리**: 예외 상황에서의 동작 테스트
- **성능 측정**: 응답 시간, 메모리 사용량 등 성능 지표 측정

## 🛠️ 테스트 방법

### 1. 자동화된 단위 테스트

`unittest` 또는 `pytest` 프레임워크를 사용하여 자동화된 단위 테스트를 작성합니다:

```python
# 예시: state.py 테스트
import unittest
from parsing.parser.tarot_agent.utils.state import TarotState

class TestTarotState(unittest.TestCase):
    def test_tarot_state_creation(self):
        # 기본 상태 생성 테스트
        state = TarotState(
            messages=[],
            user_intent="unknown",
            user_input="테스트 입력"
        )
        self.assertEqual(state["user_intent"], "unknown")
        self.assertEqual(state["user_input"], "테스트 입력")
```

### 2. 수동 테스트 스크립트

자동화하기 어려운 기능에 대해 수동 테스트 스크립트를 작성합니다:

```python
# 예시: 웹 검색 기능 테스트 스크립트
from parsing.parser.tarot_agent.utils.web_search import perform_web_search

# 웹 검색 테스트
result = perform_web_search("2025년 취업 전망")
print(f"검색 성공 여부: {result['success']}")
print(f"검색 결과 수: {len(result['results'])}")
print(f"첫 번째 결과: {result['results'][0] if result['results'] else 'No results'}")
```

### 3. 비교 테스트

기존 코드와 모듈화된 코드의 결과를 비교하는 테스트를 수행합니다:

```python
# 예시: 카드 분석 결과 비교 테스트
import sys
import os

# 기존 코드 경로 추가
sys.path.append("parsing/parser")
import tarot_langgraph as old_code

# 모듈화된 코드 임포트
from parsing.parser.tarot_agent.utils.analysis import analyze_card_combination_synergy

# 테스트 데이터
test_cards = [
    {"name": "The Fool", "number": 0, "arcana": "major"},
    {"name": "The Magician", "number": 1, "arcana": "major"},
    {"name": "The High Priestess", "number": 2, "arcana": "major"}
]

# 기존 코드 결과
old_result = old_code.analyze_card_combination_synergy(test_cards)

# 모듈화된 코드 결과
new_result = analyze_card_combination_synergy(test_cards)

# 결과 비교
print(f"결과 일치 여부: {old_result == new_result}")
if old_result != new_result:
    print("차이점:")
    print(f"기존 결과: {old_result}")
    print(f"새 결과: {new_result}")
```

### 4. 성능 테스트

기존 코드와 모듈화된 코드의 성능을 비교하는 테스트를 수행합니다:

```python
# 예시: 응답 시간 비교 테스트
import time
import sys
import os

# 기존 코드 경로 추가
sys.path.append("parsing/parser")
import tarot_langgraph as old_code

# 모듈화된 코드 임포트
from parsing.parser.tarot_agent.agent import create_optimized_tarot_graph

# 테스트 데이터
test_input = "직장에서 승진할 수 있을까요?"

# 기존 코드 성능 측정
start_time = time.time()
old_graph = old_code.create_optimized_tarot_graph()
old_app = old_graph.compile()
old_state = {
    "messages": [],
    "user_intent": "unknown",
    "user_input": test_input
}
old_result = old_app.invoke(old_state)
old_time = time.time() - start_time

# 모듈화된 코드 성능 측정
start_time = time.time()
new_graph = create_optimized_tarot_graph()
new_app = new_graph.compile()
new_state = {
    "messages": [],
    "user_intent": "unknown",
    "user_input": test_input
}
new_result = new_app.invoke(new_state)
new_time = time.time() - start_time

# 결과 비교
print(f"기존 코드 응답 시간: {old_time:.4f}초")
print(f"모듈화된 코드 응답 시간: {new_time:.4f}초")
print(f"성능 차이: {((new_time - old_time) / old_time) * 100:.2f}%")
```

## 📊 테스트 시나리오

### 1. 기본 기능 테스트

기본적인 타로 상담 기능을 테스트합니다:

1. **카드 정보 조회**: "태양 카드의 의미가 무엇인가요?"
2. **스프레드 정보 조회**: "셀틱 크로스 스프레드에 대해 알려주세요."
3. **간단한 카드 뽑기**: "오늘의 운세를 봐주세요."

### 2. 상담 기능 테스트

타로 상담 기능을 테스트합니다:

1. **일반 상담**: "직장에서 승진할 수 있을까요?"
2. **연애 상담**: "현재 교제 중인 사람과의 관계가 어떻게 될까요?"
3. **진로 상담**: "프로그래머로 전향하는 것이 좋을까요?"

### 3. 웹 검색 통합 테스트

웹 검색 기능이 통합된 상담을 테스트합니다:

1. **시장 동향 질문**: "2025년 부동산 시장은 어떨까요?"
2. **취업 관련 질문**: "AI 개발자 취업 전망이 어떤가요?"
3. **투자 관련 질문**: "비트코인에 투자해도 될까요?"

### 4. 예외 처리 테스트

다양한 예외 상황에서의 동작을 테스트합니다:

1. **잘못된 입력**: "ㅁㄴㅇㄹ" (의미 없는 입력)
2. **빈 입력**: "" (빈 문자열)
3. **매우 긴 입력**: 매우 긴 문장 입력 (1000자 이상)
4. **특수 문자**: "!@#$%^&*()" (특수 문자만 포함)

## 📝 테스트 체크리스트

### 모듈 단위 테스트 체크리스트

- [ ] `state.py`: `TarotState` 클래스 정의 및 기능 테스트
- [ ] `tools.py`: `search_tarot_cards`, `search_tarot_spreads` 등의 도구 함수 테스트
- [ ] `web_search.py`: 웹 검색 관련 함수 테스트
- [ ] `analysis.py`: 카드 분석 관련 함수 테스트
- [ ] `timing.py`: 시간/타이밍 관련 함수 테스트
- [ ] `translation.py`: 번역 관련 함수 테스트
- [ ] `helpers.py`: 기타 헬퍼 함수 테스트
- [ ] `nodes.py`: 주요 노드 함수 테스트

### 통합 테스트 체크리스트

- [ ] 그래프 생성 테스트
- [ ] 노드 연결 테스트
- [ ] 상태 흐름 테스트
- [ ] 도구 호출 테스트

### 엔드투엔드 테스트 체크리스트

- [ ] 기본 기능 테스트 (카드 정보, 스프레드 정보, 간단한 카드 뽑기)
- [ ] 상담 기능 테스트 (일반, 연애, 진로)
- [ ] 웹 검색 통합 테스트 (시장 동향, 취업, 투자)
- [ ] 예외 처리 테스트 (잘못된 입력, 빈 입력, 매우 긴 입력, 특수 문자)

## 📈 테스트 결과 기록

테스트 결과를 기록하기 위한 템플릿:

```
# 테스트 결과 보고서

## 테스트 정보
- 테스트 날짜: YYYY-MM-DD
- 테스트 환경: [환경 정보]
- 테스트 대상: [테스트 대상 모듈/기능]

## 테스트 결과 요약
- 성공: X개
- 실패: Y개
- 성공률: Z%

## 상세 테스트 결과
1. [테스트 이름]
   - 상태: 성공/실패
   - 설명: [테스트 설명]
   - 예상 결과: [예상 결과]
   - 실제 결과: [실제 결과]
   - 비고: [추가 정보]

2. [테스트 이름]
   ...

## 발견된 문제점
1. [문제점 1]
   - 심각도: 높음/중간/낮음
   - 설명: [문제 설명]
   - 해결 방안: [해결 방안]

2. [문제점 2]
   ...

## 결론 및 권장사항
[테스트 결과에 대한 결론 및 권장사항]
```

## 🔄 테스트-수정 사이클

테스트 중 발견된 문제를 해결하기 위한 사이클:

1. **테스트 실행**: 테스트 스크립트 실행
2. **문제 식별**: 실패한 테스트 케이스 식별
3. **원인 분석**: 문제의 원인 분석
4. **코드 수정**: 문제 해결을 위한 코드 수정
5. **재테스트**: 수정된 코드 재테스트
6. **문서화**: 문제 및 해결 방법 문서화

## 📅 테스트 일정

| 단계 | 테스트 내용 | 예상 소요 시간 | 담당자 |
|------|------------|--------------|-------|
| 1단계 | 모듈 단위 테스트 | 2일 | [담당자] |
| 2단계 | 통합 테스트 | 1일 | [담당자] |
| 3단계 | 엔드투엔드 테스트 | 1일 | [담당자] |
| 4단계 | 성능 테스트 | 1일 | [담당자] |
| 5단계 | 문제 해결 및 재테스트 | 2일 | [담당자] |

## 🚀 테스트 자동화 계획

향후 테스트 자동화를 위한 계획:

1. **CI/CD 파이프라인 구축**: GitHub Actions 또는 Jenkins를 사용한 자동화된 테스트 파이프라인 구축
2. **테스트 커버리지 측정**: 코드 커버리지 도구를 사용한 테스트 커버리지 측정
3. **회귀 테스트 자동화**: 새로운 기능 추가 시 기존 기능이 손상되지 않도록 회귀 테스트 자동화
4. **성능 모니터링**: 지속적인 성능 모니터링 시스템 구축 