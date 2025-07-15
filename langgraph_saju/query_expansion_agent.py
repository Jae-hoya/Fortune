from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import BaseTool
from langchain_core.callbacks.manager import CallbackManagerForToolRun
from query_expansion import SajuQueryExpander
from typing import Optional, Type, Dict, Any
from pydantic import BaseModel, Field

class QueryExpansionInput(BaseModel):
    """Query Expansion 입력 스키마"""
    query: str = Field(description="확장할 한글 사주 질문")

class QueryExpansionTool(BaseTool):
    """한글 사주 질문을 영어로 확장하는 도구"""
    
    name: str = "query_expansion_tool"
    description: str = """
    한글로 된 사주 질문을 영어로 확장하여 더 나은 검색을 위한 쿼리로 변환합니다.
    사주 전문 용어를 포함한 한글 질문에 사용하세요.
    예: '정관이 뭐야?' -> 'What is Zheng Guan Direct Officer in Four Pillars of Destiny?'
    """
    args_schema: Type[BaseModel] = QueryExpansionInput
    
    # Pydantic 모델 설정
    model_config = {"extra": "allow"}
    
    def __init__(self):
        super().__init__()
        # private 변수로 expander 저장
        self._expander = SajuQueryExpander()
    
    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """한글 쿼리를 영어로 확장"""
        try:
            # 쿼리 확장 실행
            expanded_query = self._expander.expand_query(query)
            
            # 결과 포맷팅
            result = f"""
원본 한글 질문: {query}
확장된 영어 질문: {expanded_query}

이 확장된 영어 질문을 사용해서 문서 검색을 수행하세요.
"""
            return result
        except Exception as e:
            return f"쿼리 확장 중 오류가 발생했습니다: {str(e)}"

# =============================================================================
# LangGraph 노드 함수들
# =============================================================================

def query_expansion_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Query Expansion 노드 함수
    한글 사주 질문을 영어로 확장하여 state에 추가
    """
    # state에서 messages 추출
    messages = state.get("messages", [])
    if not messages:
        return state
    
    # 마지막 사용자 메시지 추출
    last_message = messages[-1]
    if hasattr(last_message, 'content'):
        user_query = last_message.content
    else:
        user_query = str(last_message)
    
    # Query Expansion 실행
    try:
        expander = SajuQueryExpander()
        expanded_query = expander.expand_query(user_query)
        
        # 결과 메시지 생성
        expansion_result = f"""
[Query Expansion 결과]
원본 질문: {user_query}
확장된 영어 질문: {expanded_query}

검색을 위해 확장된 쿼리를 사용합니다.
"""
        
        # state 업데이트
        new_state = state.copy()
        new_state["expanded_query"] = expanded_query
        new_state["original_query"] = user_query
        new_state["expansion_result"] = expansion_result
        
        return new_state
        
    except Exception as e:
        # 오류 발생 시 원본 쿼리 유지
        error_message = f"Query Expansion 오류: {str(e)}"
        new_state = state.copy()
        new_state["expanded_query"] = user_query  # 원본 유지
        new_state["original_query"] = user_query
        new_state["expansion_result"] = error_message
        
        return new_state

# =============================================================================
# Agent 생성 함수들 (기존 유지)
# =============================================================================

def create_query_expansion_agent(model_name: str = "gpt-4o-mini", temperature: float = 0):
    """Query Expansion Agent 생성"""
    
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    tools = [QueryExpansionTool()]
    
    # 시스템 프롬프트 
    system_prompt = """
당신은 한글 사주 질문을 영어로 확장하는 전문가입니다.

역할:
1. 사용자의 한글 사주 질문을 분석합니다
2. query_expansion_tool을 사용해서 영어로 확장합니다
3. 확장된 결과를 사용자에게 전달합니다

처리 방식:
- 한글 사주 질문이 들어오면 query_expansion_tool을 사용하세요
- 확장된 영어 질문을 명확하게 제시하세요
- 생년월일이 포함된 질문과 개념적 질문을 구분해서 처리하세요

항상 한국어로 응답하세요.
"""
    
    agent = create_react_agent(llm, tools, prompt=system_prompt)
    return agent

# 편의 함수
def get_query_expansion_agent():
    """Query Expansion Agent 반환"""
    return create_query_expansion_agent()

def get_query_expansion_node():
    """Query Expansion 노드 함수 반환"""
    return query_expansion_node

# # 테스트 함수
# def test_query_expansion_node():
#     """Query Expansion 노드 테스트"""
#     print("=== Query Expansion Node 테스트 ===")
    
#     test_states = [
#         {"messages": [("user", "정관이 뭐야?")]},
#         {"messages": [("user", "오행에서 화의 특징은?")]},
#         {"messages": [("user", "1995년 3월 28일 남자 사주")]},
#     ]
    
#     for i, state in enumerate(test_states, 1):
#         print(f"\n테스트 {i}: {state['messages'][0][1]}")
#         print("-" * 50)
        
#         result = query_expansion_node(state)
#         print(f"확장된 쿼리: {result.get('expanded_query', 'N/A')}")
#         print(f"결과 메시지: {result.get('expansion_result', 'N/A')[:100]}...")
#         print("=" * 70)

# def test_query_expansion_agent():
#     """Query Expansion Agent 테스트"""
#     print("=== Query Expansion Agent 테스트 ===")
    
#     agent = get_query_expansion_agent()
    
#     test_queries = [
#         "정관이 뭐야?",
#         "오행에서 화의 특징은?",
#         "1995년 3월 28일 남자 사주",
#         "대운 계산하는 방법"
#     ]
    
#     for query in test_queries:
#         print(f"\n테스트 질문: {query}")
#         print("-" * 50)
        
#         response = agent.invoke({"messages": [("user", query)]})
#         print("응답:")
#         print(response["messages"][-1].content)
#         print("=" * 70)

# # if __name__ == "__main__":
# #     print("노드 테스트:")
# #     test_query_expansion_node()
# #     print("\n" + "="*80 + "\n")
# #     print("Agent 테스트:")
# #     test_query_expansion_agent() 