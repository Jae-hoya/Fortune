from dotenv import load_dotenv
load_dotenv()

import numpy as np
from parsing.parser.tarot_agent.utils.helpers import (
    convert_numpy_types, parse_card_numbers, get_default_spreads, extract_suit_from_name, is_major_arcana
)

def test_convert_numpy_types():
    obj = {"a": np.float32(1.5), "b": np.int32(2), "c": np.array([1,2,3])}
    result = convert_numpy_types(obj)
    assert result == {"a": 1.5, "b": 2, "c": [1,2,3]}

def test_parse_card_numbers():
    assert parse_card_numbers("1,2,3", 3) == [1,2,3]
    assert parse_card_numbers("1,1,2", 3) is None  # 중복
    assert parse_card_numbers("1,2", 3) is None  # 개수 부족

def test_get_default_spreads():
    spreads = get_default_spreads()
    assert isinstance(spreads, list)
    assert any("three card spread" in s["normalized_name"] for s in spreads)

def test_extract_suit_from_name():
    assert extract_suit_from_name("Ace of Cups") == "Cups"
    assert extract_suit_from_name("King of Swords") == "Swords"
    assert extract_suit_from_name("The Fool") == ""

def test_is_major_arcana():
    assert is_major_arcana("The Fool")
    assert not is_major_arcana("Ace of Cups") 