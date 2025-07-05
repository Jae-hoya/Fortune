from dotenv import load_dotenv
load_dotenv()

import pytest
from parsing.parser.tarot_agent.utils.analysis import (
    calculate_card_draw_probability,
    calculate_success_probability_from_cards,
    analyze_card_combination_synergy,
    analyze_elemental_balance
)

def test_calculate_card_draw_probability():
    result = calculate_card_draw_probability(deck_size=78, cards_of_interest=2, cards_drawn=5, exact_matches=1)
    assert isinstance(result, dict)
    assert 0 <= result["exact_probability"] <= 1
    assert 0 <= result["at_least_one_probability"] <= 1
    assert result["distribution_type"] == "hypergeometric"

def test_calculate_success_probability_from_cards():
    cards = [
        {"name": "The Fool", "orientation": "upright"},
        {"name": "The Magician", "orientation": "upright"}
    ]
    result = calculate_success_probability_from_cards(cards)
    assert isinstance(result, dict)
    assert "success_probability" in result
    assert "confidence" in result

def test_analyze_card_combination_synergy():
    cards = [
        {"name": "The Fool"},
        {"name": "The Magician"},
        {"name": "The Star"}
    ]
    result = analyze_card_combination_synergy(cards)
    assert isinstance(result, dict)
    assert "synergy_score" in result
    assert "combinations" in result
    assert "warnings" in result

def test_analyze_elemental_balance():
    cards = [
        {"name": "Ace of Cups"},
        {"name": "King of Swords"},
        {"name": "Queen of Wands"},
        {"name": "Ten of Pentacles"}
    ]
    result = analyze_elemental_balance(cards)
    assert isinstance(result, dict)
    assert "elements" in result
    assert "dominant_element" in result
    assert "missing_elements" in result
    assert "balance_score" in result 