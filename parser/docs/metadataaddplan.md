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