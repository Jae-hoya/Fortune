from dotenv import load_dotenv
load_dotenv()

from parsing.parser.tarot_agent.utils.state import TarotState

def test_tarot_state_creation():
    state = TarotState(
        messages=[],
        user_intent="unknown",
        user_input="테스트 입력"
    )
    assert state["user_intent"] == "unknown"
    assert state["user_input"] == "테스트 입력" 