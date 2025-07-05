import pytest
from parsing.parser.tarot_agent import agent

def test_create_optimized_tarot_graph():
    graph = agent.create_optimized_tarot_graph()
    assert hasattr(graph, "add_node") and hasattr(graph, "add_edge")

# main 함수는 실제 실행이므로, 최소한 예외 없이 동작하는지만 확인

def test_main_runs(monkeypatch):
    monkeypatch.setattr(agent, "create_optimized_tarot_graph", lambda: None)
    try:
        agent.main()
    except Exception as e:
        pytest.fail(f"main 함수 실행 중 예외 발생: {e}") 