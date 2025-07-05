import pytest
from parsing.parser.tarot_agent.utils import web_search

class DummySearchTool:
    def invoke(self, params):
        return [{"title": "Result1", "content": "내용1", "url": "url1"}]

def dummy_translate(text):
    return f"EN({text})"

def setup_module(module):
    web_search.SEARCH_TOOLS["tavily"] = DummySearchTool()
    web_search.SEARCH_TOOLS["duckduckgo_results"] = DummySearchTool()
    web_search.translate_korean_to_english_with_llm = dummy_translate

def teardown_module(module):
    web_search.SEARCH_TOOLS["tavily"] = None
    web_search.SEARCH_TOOLS["duckduckgo_results"] = None

def test_perform_web_search_success():
    result = web_search.perform_web_search("테스트")
    assert result["success"] and result["results"]

def test_perform_web_search_fail():
    web_search.SEARCH_TOOLS["tavily"] = None
    web_search.SEARCH_TOOLS["duckduckgo_results"] = None
    result = web_search.perform_web_search("테스트")
    assert not result["success"]
    web_search.SEARCH_TOOLS["tavily"] = DummySearchTool()
    web_search.SEARCH_TOOLS["duckduckgo_results"] = DummySearchTool()

def test_decide_web_search_need_with_llm(monkeypatch):
    class DummyLLM:
        def invoke(self, prompt):
            class DummyResp:
                content = '{"needs_search": true, "reason": "테스트", "search_query": "test"}'
            return DummyResp()
    monkeypatch.setattr(web_search, "ChatOpenAI", lambda **kwargs: DummyLLM())
    result = web_search.decide_web_search_need_with_llm("취업 질문", "")
    assert result["needs_search"] is True

def test_integrate_search_results_with_tarot(monkeypatch):
    class DummyLLM:
        def invoke(self, prompt):
            class DummyResp:
                content = "통합 결과"
            return DummyResp()
    monkeypatch.setattr(web_search, "ChatOpenAI", lambda **kwargs: DummyLLM())
    tarot_cards = [{"name": "카드1", "meaning": "의미1"}]
    search_results = {"success": True, "results": [{"title": "Result1", "content": "내용1", "url": "url1"}]}
    result = web_search.integrate_search_results_with_tarot(tarot_cards, search_results, "고민")
    assert "통합 결과" in result

def test_format_search_results_for_display():
    search_results = {"results": [{"title": "Result1", "content": "내용1", "url": "url1"}]}
    result = web_search.format_search_results_for_display(search_results)
    assert "Result1" in result 