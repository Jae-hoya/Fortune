# 타로 에이전트 메타데이터 추가 계획

## 개요
FastAPI 연동을 위해 타로 에이전트의 각 핸들러에서 LLM 호출 시 메타데이터를 추가하여 최종 응답임을 식별할 수 있도록 합니다.

## 수정 대상 파일 및 위치

### 1. parsing/parser/tarot_agent/utils/nodes.py

이 파일에는 대부분의 핸들러 함수들이 정의되어 있으며, 총 **21개의 LLM 호출 지점**에서 메타데이터 추가가 필요합니다.

#### 1.1 card_info_handler (라인 85)
**수정 전:**
```python
response = llm_with_tools.invoke([HumanMessage(content=prompt)])
```
**수정 후:**
```python
response = llm_with_tools.invoke([HumanMessage(content=prompt)], {"metadata": {"final_response": "yes", "handler": "card_info_handler"}})
```

#### 1.2 spread_info_handler (라인 103)
**수정 전:**
```python
response = llm_with_tools.invoke([HumanMessage(content=prompt)])
```
**수정 후:**
```python
response = llm_with_tools.invoke([HumanMessage(content=prompt)], {"metadata": {"final_response": "yes", "handler": "spread_info_handler"}})
```

#### 1.3 simple_card_handler (라인 166)
**수정 전:**
```python
interpretation_response = llm.invoke([HumanMessage(content=interpretation_prompt)])
```
**수정 후:**
```python
interpretation_response = llm.invoke([HumanMessage(content=interpretation_prompt)], {"metadata": {"final_response": "yes", "handler": "simple_card_handler"}})
```

#### 1.4 general_handler (라인 312, 409)
**수정 전:**
```python
result = json.loads(response.content.strip())
# 라인 312
response = llm.invoke([HumanMessage(content=prompt)])

# 라인 409  
response = llm.invoke([HumanMessage(content=prompt)])
```
**수정 후:**
```python
# 라인 312 - 날짜 질문 판단용 (내부 로직이므로 메타데이터 불필요)
result = json.loads(response.content.strip())

# 라인 409 - 최종 응답
response = llm.invoke([HumanMessage(content=prompt)], {"metadata": {"final_response": "yes", "handler": "general_handler"}})
```

#### 1.5 consultation_summary_handler (라인 815, 1100)
**수정 전:**
```python
# 라인 815 - 개별 카드 해석
response = llm.invoke([HumanMessage(content=interpretation_prompt)])

# 라인 1100 - 종합 분석
comprehensive_response = llm.invoke([HumanMessage(content=analysis_prompt)])
```
**수정 후:**
```python
# 라인 815 - 개별 카드 해석 (내부 처리용이므로 메타데이터 불필요)
response = llm.invoke([HumanMessage(content=interpretation_prompt)])

# 라인 1100 - 종합 분석 (최종 응답)
comprehensive_response = llm.invoke([HumanMessage(content=analysis_prompt)], {"metadata": {"final_response": "yes", "handler": "consultation_summary_handler"}})
```

#### 1.6 consultation_individual_handler (라인 1274)
**수정 전:**
```python
advice_response = llm.invoke([HumanMessage(content=detailed_advice_prompt)])
```
**수정 후:**
```python
advice_response = llm.invoke([HumanMessage(content=detailed_advice_prompt)], {"metadata": {"final_response": "yes", "handler": "consultation_individual_handler"}})
```

#### 1.7 context_reference_handler (라인 1396)
**수정 전:**
```python
classification_response = llm.invoke([HumanMessage(content=classification_prompt)])
```
**수정 후:**
```python
# 이는 분류용이므로 메타데이터 불필요 (내부 로직)
classification_response = llm.invoke([HumanMessage(content=classification_prompt)])
```

#### 1.8 emotional_support_handler (라인 1475)
**수정 전:**
```python
response = llm.invoke([HumanMessage(content=prompt)])
```
**수정 후:**
```python
response = llm.invoke([HumanMessage(content=prompt)], {"metadata": {"final_response": "yes", "handler": "emotional_support_handler"}})
```

#### 1.9 tool_result_handler (라인 1610)
**수정 전:**
```python
response = llm.invoke([HumanMessage(content=prompt)])
```
**수정 후:**
```python
response = llm.invoke([HumanMessage(content=prompt)], {"metadata": {"final_response": "yes", "handler": "tool_result_handler"}})
```

#### 1.10 supervisor_llm_node (라인 1841)
**수정 전:**
```python
response = llm.invoke([HumanMessage(content=prompt)])
```
**수정 후:**
```python
# 내부 라우팅 로직이므로 메타데이터 불필요
response = llm.invoke([HumanMessage(content=prompt)])
```

#### 1.11 spread_recommender_node (라인 2058)
**수정 전:**
```python
response = llm.invoke([HumanMessage(content=recommendation_prompt)])
```
**수정 후:**
```python
response = llm.invoke([HumanMessage(content=recommendation_prompt)], {"metadata": {"final_response": "yes", "handler": "spread_recommender_node"}})
```

#### 1.12 spread_extractor_node (라인 2105, 2128)
**수정 전:**
```python
# 라인 2105
response = llm.invoke([HumanMessage(content=extract_prompt)])

# 라인 2128  
response = llm.invoke([HumanMessage(content=default_prompt)])
```
**수정 후:**
```python
# 라인 2105 - 내부 처리용이므로 메타데이터 불필요
response = llm.invoke([HumanMessage(content=extract_prompt)])

# 라인 2128 - 내부 처리용이므로 메타데이터 불필요
response = llm.invoke([HumanMessage(content=default_prompt)])
```

#### 1.13 situation_analyzer_node (라인 2164)
**수정 전:**
```python
response = llm.invoke([HumanMessage(content=prompt)])
```
**수정 후:**
```python
response = llm.invoke([HumanMessage(content=prompt)], {"metadata": {"final_response": "yes", "handler": "situation_analyzer_node"}})
```

#### 1.14 card_count_inferrer_node (라인 2186)
**수정 전:**
```python
card_count_response = llm.invoke([HumanMessage(content=card_count_prompt)])
```
**수정 후:**
```python
# 내부 처리용이므로 메타데이터 불필요
card_count_response = llm.invoke([HumanMessage(content=card_count_prompt)])
```

#### 1.15 status_determiner_node (라인 2218)
**수정 전:**
```python
status_response = llm.invoke([HumanMessage(content=status_prompt)])
```
**수정 후:**
```python
# 내부 처리용이므로 메타데이터 불필요
status_response = llm.invoke([HumanMessage(content=status_prompt)])
```

### 2. parsing/parser/tarot_agent/utils/helpers.py

이 파일에는 **6개의 LLM 호출 지점**이 있습니다.

#### 2.1 analyze_emotion_and_empathy (라인 242)
**수정 전:**
```python
response = llm.invoke([HumanMessage(content=emotion_prompt)])
```
**수정 후:**
```python
# 내부 감정 분석용이므로 메타데이터 불필요
response = llm.invoke([HumanMessage(content=emotion_prompt)])
```

#### 2.2 extract_concern_keywords (라인 362)
**수정 전:**
```python
response = llm.invoke([HumanMessage(content=prompt)])
```
**수정 후:**
```python
# 내부 키워드 추출용이므로 메타데이터 불필요
response = llm.invoke([HumanMessage(content=prompt)])
```

#### 2.3 check_if_has_specific_concern (라인 430)
**수정 전:**
```python
response = llm.invoke([HumanMessage(content=prompt)])
```
**수정 후:**
```python
# 내부 판단 로직이므로 메타데이터 불필요
response = llm.invoke([HumanMessage(content=prompt)])
```

#### 2.4 handle_casual_new_question (라인 814)
**수정 전:**
```python
response = llm.invoke([HumanMessage(content=casual_prompt)])
```
**수정 후:**
```python
response = llm.invoke([HumanMessage(content=casual_prompt)], {"metadata": {"final_response": "yes", "handler": "handle_casual_new_question"}})
```

#### 2.5 handle_tarot_related_question (라인 876)
**수정 전:**
```python
response = llm.invoke([HumanMessage(content=prompt)])
```
**수정 후:**
```python
response = llm.invoke([HumanMessage(content=prompt)], {"metadata": {"final_response": "yes", "handler": "handle_tarot_related_question"}})
```

#### 2.6 integrate_search_results_with_tarot (라인 965)
**수정 전:**
```python
response = llm.invoke([HumanMessage(content=prompt)])
```
**수정 후:**
```python
# 내부 통합 처리용이므로 메타데이터 불필요
response = llm.invoke([HumanMessage(content=prompt)])
```

### 3. parsing/parser/tarot_agent/utils/web_search.py

이 파일에는 **3개의 LLM 호출 지점**이 있습니다.

#### 3.1 decide_web_search_need_with_llm (라인 224)
**수정 전:**
```python
response = llm.invoke([HumanMessage(content=prompt)])
```
**수정 후:**
```python
# 내부 판단 로직이므로 메타데이터 불필요
response = llm.invoke([HumanMessage(content=prompt)])
```

#### 3.2 filter_korean_results (라인 270)
**수정 전:**
```python
response = llm.invoke([HumanMessage(content=prompt)])
```
**수정 후:**
```python
# 내부 필터링 로직이므로 메타데이터 불필요
response = llm.invoke([HumanMessage(content=prompt)])
```

#### 3.3 integrate_search_results_with_tarot (라인 453)
**수정 전:**
```python
response = llm.invoke([HumanMessage(content=prompt)])
```
**수정 후:**
```python
# 내부 통합 처리용이므로 메타데이터 불필요
response = llm.invoke([HumanMessage(content=prompt)])
```

### 4. parsing/parser/tarot_agent/utils/translation.py

이 파일에는 **2개의 LLM 호출 지점**이 있습니다.

#### 4.1 translate_text_with_llm (라인 46)
**수정 전:**
```python
response = llm.invoke([HumanMessage(content=prompt)])
```
**수정 후:**
```python
# 내부 번역 처리용이므로 메타데이터 불필요
response = llm.invoke([HumanMessage(content=prompt)])
```

#### 4.2 translate_korean_to_english_with_llm (라인 167)
**수정 전:**
```python
response = llm.invoke([HumanMessage(content=translation_prompt)])
```
**수정 후:**
```python
# 내부 번역 처리용이므로 메타데이터 불필요
response = llm.invoke([HumanMessage(content=translation_prompt)])
```

### 5. parsing/parser/tarot_agent/utils/analysis.py

이 파일에는 **1개의 LLM 호출 지점**이 있습니다.

#### 5.1 analyze_emotion_and_empathy (라인 334)
**수정 전:**
```python
response = llm.invoke([HumanMessage(content=emotion_prompt)])
```
**수정 후:**
```python
# 내부 감정 분석용이므로 메타데이터 불필요
response = llm.invoke([HumanMessage(content=emotion_prompt)])
```

## 메타데이터 추가 기준

### 최종 응답에 메타데이터 추가가 필요한 경우:
- 사용자에게 직접 반환되는 최종 응답을 생성하는 LLM 호출
- 핸들러의 마지막 단계에서 실행되는 응답 생성

### 메타데이터 추가가 불필요한 경우:
- 내부 로직 처리용 LLM 호출 (분류, 판단, 번역, 키워드 추출 등)
- 중간 단계의 데이터 처리용 LLM 호출
- 다른 함수에서 사용될 데이터를 생성하는 LLM 호출

## 메타데이터 구조

```python
{
    "metadata": {
        "final_response": "yes",
        "handler": "핸들러명"
    }
}
```

## ✅ 최종 확인 결과 (2024년 기준)

### 실제 적용된 메타데이터 개수:

#### **nodes.py: 10개**
1. `card_info_handler` (라인 87) ✅
2. `spread_info_handler` (라인 105) ✅
3. `simple_card_handler` (라인 166) ✅
4. `general_handler` (라인 312) ✅
5. `consultation_summary_handler` (라인 1100) ✅
6. `consultation_individual_handler` (라인 1274) ✅
7. `emotional_support_handler` (라인 1475) ✅
8. `tool_result_handler` (라인 1610) ✅
9. `spread_recommender_node` (라인 2058) ✅
10. `situation_analyzer_node` (라인 2164) ✅

#### **helpers.py: 2개**
11. `handle_casual_new_question` (라인 814) ✅
12. `handle_tarot_related_question` (라인 876) ✅

### **총 메타데이터 적용: 12개** 🎯

## FastAPI 응답 예시

### 메타데이터가 있는 응답 (최종 사용자 응답)
```python
# LangChain AIMessage 객체
{
    "content": "🔮 타로 카드가 말하는 당신의 연애 운세는...",
    "metadata": {
        "final_response": "yes",
        "handler": "consultation_summary_handler"
    }
}
```

### 메타데이터가 없는 응답 (내부 처리용)
```python
# LangChain AIMessage 객체
{
    "content": "사용자 의도: consultation, 신뢰도: 0.85",
    "metadata": {}  # 또는 메타데이터 없음
}
```

### FastAPI에서 활용 방법
```python
@app.post("/tarot-consultation")
async def tarot_consultation(request: TarotRequest):
    # 타로 에이전트 실행
    result = await tarot_agent.run(request.user_input)
    
    # 최종 응답만 필터링
    final_messages = []
    for message in result.get("messages", []):
        if hasattr(message, 'metadata') and message.metadata.get("final_response") == "yes":
            final_messages.append({
                "content": message.content,
                "handler": message.metadata.get("handler"),
                "timestamp": datetime.now().isoformat()
            })
    
    return {
        "success": True,
        "responses": final_messages,
        "total_final_responses": len(final_messages)
    }
```

### 예상 API 응답
```json
{
    "success": true,
    "responses": [
        {
            "content": "🔮 타로 카드가 말하는 당신의 연애 운세는 매우 긍정적입니다...",
            "handler": "consultation_summary_handler",
            "timestamp": "2024-01-15T10:30:00"
        }
    ],
    "total_final_responses": 1
}
```

## 총 수정 대상

- **최종 응답용 LLM 호출**: 12개 ✅
- **내부 처리용 LLM 호출**: 21개 (메타데이터 불필요)
- **총 LLM 호출 지점**: 33개

## FastAPI 연동 시 활용 방안

1. 메타데이터가 있는 응답만 사용자에게 반환
2. 핸들러별 응답 타입 구분 가능
3. 로깅 및 모니터링 시 응답 유형 식별 가능
4. API 응답 구조화 시 메타데이터 활용 가능
5. 응답 품질 관리 및 A/B 테스트 가능

----

## 🎯 **타로 에이전트 메타데이터 응답 시나리오**

### 📋 **시나리오 1: 간단한 카드 질문**

**사용자 입력:** "The Fool 카드가 뭐야?"

**내부 처리 과정:**
1. `classify_intent_node` → 내부 분류 (메타데이터 없음)
2. `card_info_handler` → **최종 응답 생성** ✅

**FastAPI가 받는 응답:**
```python
[
    # 내부 처리 메시지들 (메타데이터 없음)
    AIMessage(content="사용자 의도: card_info, 신뢰도: 0.95"),
    
    # 최종 응답 (메타데이터 있음) ✅
    AIMessage(
        content="🃏 **The Fool (바보)** 카드에 대해 알려드릴게요!\n\n**기본 의미:**\n새로운 시작, 순수함, 모험정신...",
        metadata={"final_response": "yes", "handler": "card_info_handler"}
    )
]
```

**사용자가 받는 최종 응답:**
```json
{
    "success": true,
    "responses": [
        {
            "content": "🃏 **The Fool (바보)** 카드에 대해 알려드릴게요!\n\n**기본 의미:**\n새로운 시작, 순수함, 모험정신...",
            "handler": "card_info_handler",
            "timestamp": "2024-01-15T10:30:00"
        }
    ],
    "total_final_responses": 1
}
```

---

### 📋 **시나리오 2: 본격적인 타로 상담**

**사용자 입력:** "연애 고민이 있어서 타로 봐줘"

**내부 처리 과정:**
1. `classify_intent_node` → 내부 분류 (메타데이터 없음)
2. `emotion_analyzer_node` → 내부 감정 분석 (메타데이터 없음)
3. `web_search_decider_node` → 내부 웹검색 판단 (메타데이터 없음)
4. `spread_recommender_node` → **스프레드 추천 응답** ✅
5. 사용자가 "1번"을 선택
6. `consultation_summary_handler` → **종합 분석 응답** ✅

**FastAPI가 받는 응답 (1단계):**
```python
[
    # 내부 처리들 (메타데이터 없음)
    AIMessage(content="의도: consultation, 감정: 불안"),
    AIMessage(content="웹검색 불필요"),
    
    # 스프레드 추천 (메타데이터 있음) ✅
    AIMessage(
        content="💕 **연애 고민을 위한 타로 스프레드를 추천해드릴게요!**\n\n1️⃣ **연인 관계 스프레드** (3장)...",
        metadata={"final_response": "yes", "handler": "spread_recommender_node"}
    )
]
```

**FastAPI가 받는 응답 (2단계):**
```python
[
    # 내부 카드 해석들 (메타데이터 없음)
    AIMessage(content="개별 카드 해석: The Lovers 정방향..."),
    AIMessage(content="시기 분석: 2-4주 후..."),
    
    # 종합 분석 (메타데이터 있음) ✅
    AIMessage(
        content="🔮 **타로가 말하는 당신의 연애 운세**\n\n📋 **뽑힌 카드:**\n1. 과거: The Lovers (연인) ⬆️\n2. 현재: Two of Cups (컵 2) ⬆️\n3. 미래: Ten of Cups (컵 10) ⬆️\n\n💫 **종합 해석:**\n카드들이 매우 긍정적인 메시지를 전하고 있습니다...",
        metadata={"final_response": "yes", "handler": "consultation_summary_handler"}
    )
]
```

**사용자가 받는 최종 응답들:**
```json
{
    "success": true,
    "responses": [
        {
            "content": "💕 **연애 고민을 위한 타로 스프레드를 추천해드릴게요!**\n\n1️⃣ **연인 관계 스프레드** (3장)...",
            "handler": "spread_recommender_node",
            "timestamp": "2024-01-15T10:30:00"
        },
        {
            "content": "🔮 **타로가 말하는 당신의 연애 운세**\n\n📋 **뽑힌 카드:**\n1. 과거: The Lovers (연인) ⬆️...",
            "handler": "consultation_summary_handler", 
            "timestamp": "2024-01-15T10:32:15"
        }
    ],
    "total_final_responses": 2
}
```

---

### 📋 **시나리오 3: 추가 질문**

**사용자 입력:** "그 카드들이 언제 현실화될까?"

**내부 처리 과정:**
1. `context_reference_handler` → 내부 분류 (메타데이터 없음)
2. `handle_tarot_related_question` → **추가 질문 응답** ✅

**FastAPI가 받는 응답:**
```python
[
    # 내부 분류 (메타데이터 없음)
    AIMessage(content="추가 질문 유형: timing"),
    
    # 추가 질문 답변 (메타데이터 있음) ✅
    AIMessage(
        content="⏰ **시기에 대해 더 자세히 설명해드릴게요!**\n\n방금 뽑힌 카드들을 보면:\n- The Lovers: 2-3주 후 감정적 변화\n- Two of Cups: 현재부터 한 달 내 만남의 기회\n- Ten of Cups: 3-6개월 후 안정적 관계 발전...",
        metadata={"final_response": "yes", "handler": "handle_tarot_related_question"}
    )
]
```

---

### 📋 **시나리오 4: 일상적 질문**

**사용자 입력:** "오늘 뭐 입을까?"

**내부 처리 과정:**
1. `classify_intent_node` → 내부 분류 (메타데이터 없음)
2. `handle_casual_new_question` → **일상 질문 응답** ✅

**FastAPI가 받는 응답:**
```python
[
    # 내부 분류 (메타데이터 없음)
    AIMessage(content="의도: casual_question"),
    
    # 일상 질문 응답 (메타데이터 있음) ✅
    AIMessage(
        content="👗 오늘 옷차림에 대한 고민이시군요! 날씨와 기분, 하루 일정을 고려해서 편안하면서도 자신감을 줄 수 있는 스타일이 좋을 것 같아요.\n\n만약 카드 한 장을 뽑아 당신의 스타일 감각을 더 깊이 알아보길 원하신다면 '네'라고 답해 주세요. 그리고 본격적인 타로 상담을 원하신다면 '타로 봐줘'라고 말씀해 주세요!",
        metadata={"final_response": "yes", "handler": "handle_casual_new_question"}
    )
]
```

---

### 🔧 **FastAPI 처리 로직**

```python
@app.post("/tarot-consultation")
async def tarot_consultation(request: TarotRequest):
    # 타로 에이전트 실행
    result = await tarot_agent.run(request.user_input)
    
    # 메타데이터 기반 필터링
    final_responses = []
    internal_logs = []
    
    for message in result.get("messages", []):
        if hasattr(message, 'metadata') and message.metadata.get("final_response") == "yes":
            # 사용자에게 보여줄 최종 응답
            final_responses.append({
                "content": message.content,
                "handler": message.metadata.get("handler"),
                "timestamp": datetime.now().isoformat()
            })
        else:
            # 내부 로그 (디버깅용)
            internal_logs.append({
                "content": message.content,
                "type": "internal_processing"
            })
    
    return {
        "success": True,
        "responses": final_responses,  # 사용자가 보는 부분
        "total_final_responses": len(final_responses),
        "internal_logs": internal_logs if DEBUG_MODE else []  # 개발 시에만
    }
```

### 🎯 **핵심 포인트:**

1. **내부 처리 메시지들**: 메타데이터 없음 → FastAPI에서 필터링
2. **최종 사용자 응답**: 메타데이터 있음 → 사용자에게 전달
3. **핸들러별 구분**: `handler` 필드로 어떤 기능인지 식별
4. **다단계 상담**: 여러 번의 최종 응답이 순차적으로 전달 가능

이렇게 하면 사용자는 깔끔한 타로 상담 응답만 받고, 개발자는 내부 처리 과정을 모니터링할 수 있습니다! 🔮✨