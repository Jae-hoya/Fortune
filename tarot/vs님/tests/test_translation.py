from dotenv import load_dotenv
load_dotenv()


import pytest
from parsing.parser.tarot_agent.utils.translation import (
    translate_text_with_llm,
    translate_card_info,
    translate_korean_to_english_with_llm
)

def test_translate_text_with_llm():
    result = translate_text_with_llm("The Sun", text_type="spread_name")
    assert isinstance(result, str)
    assert len(result) > 0

def test_translate_card_info():
    info = translate_card_info("The Fool", "upright")
    assert isinstance(info, dict)
    assert info["name"] == "바보"
    assert info["direction"] == "정방향"
    info2 = translate_card_info("Ace of Cups", "reversed")
    assert isinstance(info2, dict)
    assert info2["direction"] == "역방향"

def test_translate_korean_to_english_with_llm():
    result = translate_korean_to_english_with_llm("연인")
    assert isinstance(result, str)
    assert len(result) > 0 