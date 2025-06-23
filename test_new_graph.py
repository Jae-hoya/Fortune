"""
새로운 LangGraph 구조 테스트 스크립트 - 통합된 nodes.py 사용
"""

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph_system.graph import create_workflow

def test_new_graph():
    """새로운 그래프 구조 테스트"""
    print("🔧 그래프 생성 중...")
    print("  - NodeManager 초기화...")
    print("  - SajuExpert 서브그래프 생성...")
    print("  - 노드들 통합 로딩...")
    
    try:
        # 워크플로 생성
        app = create_workflow()
        print("✅ 그래프 생성 완료!")
        
        # 테스트 쿼리
        test_query = "1995년 3월 28일 남자, 12시 30분 출생 운세봐줘"
        
        print(f"\n🔍 테스트 쿼리: {test_query}")
        
        # 초기 상태 설정
        initial_state = {
            "messages": [HumanMessage(content=test_query)],
            "next": ""
        }
        
        # 설정 생성
        config = RunnableConfig(
            recursion_limit=20, 
            configurable={"thread_id": "test_123"}
        )
        
        print("🚀 실행 중...")
        print("  - Supervisor 시작...")
        print("  - SajuExpert 서브그래프 실행...")
        print("  - Manse -> Retriever 순차 처리...")
        
        result = app.invoke(initial_state, config=config)
        
        # 결과 확인
        messages = result.get("messages", [])
        if messages:
            print(f"\n✅ 성공! 메시지 수: {len(messages)}")
            print(f"📋 최종 응답:\n{messages[-1].content}")
        else:
            print("❌ 응답이 없습니다.")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_new_graph() 