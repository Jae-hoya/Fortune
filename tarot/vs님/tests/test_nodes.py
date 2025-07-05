from dotenv import load_dotenv
load_dotenv()

import pytest
from parsing.parser.tarot_agent.utils.nodes import (
    state_classifier_node,
    unified_processor_node,
    unified_tool_handler_node
)
from langchain_core.messages import HumanMessage, AIMessage

def test_state_classifier_node():
    state = {"consultation_data": {"status": "spread_selection"}}
    result = state_classifier_node(state)
    assert isinstance(result, dict)
    assert "routing_decision" in result

def test_unified_tool_handler_node():
    state = {
        "messages": [
            HumanMessage(content="타로 카드 검색 테스트"),
            AIMessage(content="AI 응답 예시")  # 최소 1개 AIMessage 포함
        ]
    }
    result = unified_tool_handler_node(state)
    assert isinstance(result, dict)

def test_unified_processor_node():
    state = {"target_handler": "unknown_handler"}
    result = unified_processor_node(state)
    assert isinstance(result, dict)
    assert "messages" in result    