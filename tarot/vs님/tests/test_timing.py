from dotenv import load_dotenv
load_dotenv()

import pytest
from parsing.parser.tarot_agent.utils.timing import (
    get_current_context,
    get_weekday_korean,
    get_season,
    calculate_days_until_target,
    get_time_period_description,
    ensure_temporal_context
)

def test_get_current_context():
    ctx = get_current_context()
    assert isinstance(ctx, dict)
    assert "current_date" in ctx
    assert "weekday_kr" in ctx

def test_get_weekday_korean():
    assert get_weekday_korean(0) == "월요일"
    assert get_weekday_korean(6) == "일요일"

def test_get_season():
    assert get_season(1) == "겨울"
    assert get_season(4) == "봄"
    assert get_season(7) == "여름"
    assert get_season(10) == "가을"

def test_calculate_days_until_target():
    days = calculate_days_until_target(12, 31)
    assert isinstance(days, int)
    assert days >= 0

def test_get_time_period_description():
    assert "일 이내" in get_time_period_description(3)
    assert "약 1주 후" in get_time_period_description(10)
    assert "약 1개월 후" in get_time_period_description(40)
    assert "약 1년 후" in get_time_period_description(400)

def test_ensure_temporal_context():
    state = {}
    result = ensure_temporal_context(state)
    assert "temporal_context" in result 