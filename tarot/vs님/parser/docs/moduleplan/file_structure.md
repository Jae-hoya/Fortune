# 타로 LangGraph 모듈화 파일 구조

## 최종 디렉토리 구조

```
parsing/parser/
├── tarot_agent/                # 메인 에이전트 디렉토리
│   ├── utils/                  # 그래프를 위한 유틸리티들
│   │   ├── __init__.py         # 유틸리티 패키지 초기화
│   │   ├── state.py            # 그래프의 상태 정의
│   │   ├── tools.py            # 그래프를 위한 도구들 (@tool)
│   │   ├── nodes.py            # 그래프를 위한 노드 함수들
│   │   ├── web_search.py       # 웹 검색 관련 함수들
│   │   ├── analysis.py         # 카드 분석 관련 함수들
│   │   ├── timing.py           # 시간/타이밍 관련 함수들
│   │   ├── translation.py      # 번역 관련 함수들
│   │   └── helpers.py          # 기타 헬퍼 함수들
│   ├── __init__.py             # 에이전트 패키지 초기화
│   └── agent.py                # 그래프 구성 코드 + 메인 실행
├── tarot_rag_system.py         # 기존 RAG 시스템 (변경 없음)
└── tarot_langgraph.py          # 기존 파일 (참조용으로 유지)
```

## 파일별 역할 및 내용

### 1. `tarot_agent/__init__.py`

에이전트 패키지 초기화 파일입니다. 패키지 버전 및 기본 임포트를 정의합니다.

```python
"""
타로 LangGraph 에이전트 패키지
"""

__version__ = "1.0.0"

# 기본 임포트
from .agent import create_optimized_tarot_graph
```

### 2. `tarot_agent/utils/__init__.py`

유틸리티 패키지 초기화 파일입니다. 유틸리티 함수들의 편리한 임포트를 제공합니다.

```python
"""
타로 LangGraph 유틸리티 패키지
"""

# 상태 임포트
from .state import TarotState

# 도구 임포트
from .tools import search_tarot_spreads, search_tarot_cards, initialize_rag_system
```

### 3. `tarot_agent/utils/state.py`

그래프의 상태를 정의하는 파일입니다. `TarotState` 클래스를 포함합니다.

```python
"""
타로 LangGraph 상태 정의 모듈
"""
from typing import Annotated, List, Dict, Any, Optional, Literal
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class TarotState(TypedDict):
    """최적화된 타로 상태"""
    # 기본 메시지 관리
    messages: Annotated[List[BaseMessage], add_messages]
    
    # 사용자 의도 (핵심!)
    user_intent: Literal["card_info", "spread_info", "consultation", "general", "simple_card", "unknown"]
    user_input: str
    
    # 상담 전용 데이터 (consultation일 때만 사용)
    consultation_data: Optional[Dict[str, Any]]
    
    # Supervisor 관련 필드
    supervisor_decision: Optional[Dict[str, Any]]
    
    # 라우팅 관련 (새로 추가)
    routing_decision: Optional[str]
    target_handler: Optional[str]
    needs_llm: Optional[bool]
    
    # 세션 메모리 (새로 추가)
    session_memory: Optional[Dict[str, Any]]
    conversation_memory: Optional[Dict[str, Any]]
    
    # 시간 맥락 정보 (새로 추가)
    temporal_context: Optional[Dict[str, Any]]
    search_timestamp: Optional[str]
    
    # 웹 검색 관련 필드 (새로 추가)
    search_results: Optional[Dict[str, Any]]
    search_decision: Optional[Dict[str, Any]]
```

### 4. `tarot_agent/utils/tools.py`

LLM이 직접 호출하는 도구들을 정의하는 파일입니다. `@tool` 데코레이터 함수와 RAG 시스템 초기화 함수를 포함합니다.

```python
"""
타로 LangGraph 도구 모듈
LLM이 직접 호출하는 @tool 데코레이터 함수들과 RAG 시스템 초기화 함수 포함
"""
from langchain_core.tools import tool
import sys
import os

# 상위 디렉토리를 path에 추가하여 tarot_rag_system.py를 임포트할 수 있도록 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from tarot_rag_system import TarotRAGSystem

# 전역 변수 (다른 모듈에서 임포트하여 사용)
rag_system = None

def initialize_rag_system():
    """RAG 시스템 초기화"""
    # 함수 내용...

@tool
def search_tarot_spreads(query: str) -> str:
    """타로 스프레드를 검색합니다."""
    # 함수 내용...

@tool  
def search_tarot_cards(query: str) -> str:
    """타로 카드의 의미를 검색합니다."""
    # 함수 내용...
```

### 5. `tarot_agent/utils/web_search.py`

웹 검색 관련 함수들을 포함하는 파일입니다.

```python
"""
웹 검색 관련 유틸리티 함수들
"""
import os
from typing import Dict, List, Any
from datetime import datetime
from langchain_openai import ChatOpenAI

# 웹 검색 관련 imports
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
# 함수 및 변수 정의...
```

### 6. `tarot_agent/utils/analysis.py`

카드 분석 관련 함수들을 포함하는 파일입니다.

```python
"""
카드 분석 관련 유틸리티 함수들
"""
from typing import Dict, List, Any
import numpy as np
import scipy.stats as stats
from scipy.stats import hypergeom
import math
from collections import Counter

# 함수 정의...
```

### 7. `tarot_agent/utils/timing.py`

시간/타이밍 관련 함수들을 포함하는 파일입니다.

```python
"""
시간/타이밍 관련 유틸리티 함수들
"""
from datetime import datetime, timedelta
import pytz
from typing import Dict, Any

# 함수 정의...
```

### 8. `tarot_agent/utils/translation.py`

번역 관련 함수들을 포함하는 파일입니다.

```python
"""
번역 관련 유틸리티 함수들
"""
from langchain_openai import ChatOpenAI
from typing import Dict, Any

# 함수 정의...
```

### 9. `tarot_agent/utils/helpers.py`

기타 헬퍼 함수들을 포함하는 파일입니다.

```python
"""
기타 헬퍼 유틸리티 함수들
"""
import numpy as np
import re
import random
from typing import List, Dict, Any

# 함수 정의...
```

### 10. `tarot_agent/agent.py`

그래프 구성 및 메인 실행 함수를 포함하는 파일입니다.

```python
"""
타로 LangGraph 에이전트 메인 모듈
그래프 구성 및 메인 실행 함수 포함
"""
from dotenv import load_dotenv
load_dotenv()

import os
import time
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

# 내부 모듈 임포트
from .utils.state import TarotState
from .utils.tools import initialize_rag_system, search_tarot_spreads, search_tarot_cards
from .utils.nodes import (
    state_classifier_node, supervisor_master_node, 
    unified_processor_node, unified_tool_handler_node
)

# 라우터 함수 및 그래프 구성 함수 정의...

def create_optimized_tarot_graph():
    """최적화된 타로 그래프 생성"""
    # 함수 내용...

def main():
    """메인 실행 함수"""
    # 함수 내용...

if __name__ == "__main__":
    main()
```

## 모듈 간 의존성 관리

### 순환 참조 방지

모듈 간 순환 참조를 방지하기 위해 다음과 같은 전략을 사용합니다:

1. **지연 임포트**: 함수 내부에서 필요한 모듈을 임포트하여 순환 참조 방지
2. **인자 전달**: 객체를 함수 인자로 전달하여 직접 임포트 회피
3. **중앙 집중식 임포트**: 핵심 모듈에서 필요한 모든 함수 임포트

### 전역 변수 관리

전역 변수는 적절한 모듈에 정의하고 필요한 곳에서 임포트하여 사용합니다:

1. **RAG 시스템**: `tools.py`에서 정의하고 다른 모듈에서 임포트
2. **검색 도구**: `web_search.py`에서 정의하고 필요한 곳에서 임포트

## 임포트 구조

### 상대 경로 임포트

모듈 간 임포트는 상대 경로를 사용하여 명확하게 표현합니다:

```python
# 같은 패키지 내 모듈 임포트
from .state import TarotState
from .tools import search_tarot_spreads

# 상위 패키지 모듈 임포트
from ..utils.web_search import perform_web_search
```

### 외부 임포트

외부 라이브러리 임포트는 각 모듈 상단에 명시적으로 선언합니다:

```python
# 표준 라이브러리
import os
import sys
import json
from typing import Dict, List, Any

# 외부 라이브러리
import numpy as np
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph
``` 