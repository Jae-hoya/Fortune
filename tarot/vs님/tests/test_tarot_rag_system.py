import os
import pytest
from parsing.parser.tarot_rag_system import TarotRAGSystem

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # parsing/tests
PARSER_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "parser"))

card_faiss_path = os.path.join(PARSER_DIR, "tarot_card_faiss_index")
spread_faiss_path = os.path.join(PARSER_DIR, "tarot_spread_faiss_index")

def setup_module(module):
    global rag
    rag = TarotRAGSystem(
        card_faiss_path=card_faiss_path,
        spread_faiss_path=spread_faiss_path,
        embedding_model_name="BAAI/bge-m3",
        reranker_model_name="ms-marco-MiniLM-L-12-v2",
        semantic_weight=0.8,
        keyword_weight=0.2
    )

def test_auto_search():
    test_queries = [
        "What does The Fool card mean?",
        "How to do a Celtic Cross spread?",
        "Ace of Cups meaning in love",
        "Three card spread for relationships",
        "Death card interpretation",
        "Past present future reading"
    ]
    for query in test_queries:
        results = rag.search_auto(query, final_k=2, show_details=False)
        assert (results["cards"] or results["spreads"]), f"No results for query: {query}"
        for doc, score in results["cards"]:
            assert hasattr(doc, 'page_content')
            assert isinstance(score, float)
        for doc, score in results["spreads"]:
            assert hasattr(doc, 'page_content')
            assert isinstance(score, float)

def test_card_only_search():
    card_results = rag.search_cards("What does Strength card represent?", final_k=3, show_details=False)
    assert len(card_results) > 0, "No card results found"
    for doc, score in card_results:
        assert hasattr(doc, 'page_content')
        assert isinstance(score, float)

def test_spread_only_search():
    spread_results = rag.search_spreads("chakra energy spread", final_k=3, show_details=False)
    assert len(spread_results) > 0, "No spread results found"
    for doc, score in spread_results:
        assert hasattr(doc, 'page_content')
        assert isinstance(score, float) 