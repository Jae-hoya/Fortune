import pytest
from parsing.parser.tarot_agent.utils import tools
import types

class DummyRAG:
    def search_spreads(self, query, final_k=5):
        return [{"title": "Spread1"}, {"title": "Spread2"}]
    def search_cards(self, query, final_k=5):
        return [{"title": "Card1"}, {"title": "Card2"}]

def dummy_translate(text):
    return f"EN({text})"

def setup_module(module):
    tools.rag_system = DummyRAG()
    tools.translate_korean_to_english_with_llm = dummy_translate

def teardown_module(module):
    tools.rag_system = None


def test_search_tarot_spreads_success():
    result = tools.search_tarot_spreads("사랑")
    assert "Spread1" in result and "Spread2" in result

def test_search_tarot_spreads_no_rag():
    tools.rag_system = None
    result = tools.search_tarot_spreads("사랑")
    assert "초기화" in result
    tools.rag_system = DummyRAG()

def test_search_tarot_cards_success():
    result = tools.search_tarot_cards("행운")
    assert "Card1" in result and "Card2" in result

def test_search_tarot_cards_no_rag():
    tools.rag_system = None
    result = tools.search_tarot_cards("행운")
    assert "초기화" in result
    tools.rag_system = DummyRAG() 