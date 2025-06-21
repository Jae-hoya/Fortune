manse_6.py : 만세툴
manse_7.py : 윤달 적용 만세툴

saju_agentic_rag4.ipynb: 각 노드별 실험이 있는 ipynb (Query_expansin)
saju_agentic_rag5.ipynb: 각 노드별 실험이 있는, 바리게이트 노드가 추가된 ipynb(Query_expansin, Barricade)
saju_agentic_rag6.ipynb: 노드만 있는 ipynb! (실험x, Query_expansin)



saju_agentic_rag.py : 랭그래프를 py화 한것
saju_agentic_rag2.py: chat_history 리스트를 직접 만들어서 관리. supervisor 프롬프트 약간 수정
saju_agentic_rag3.py:  chat_history 리스트를 집적 만들어 관리. Supervisor 프롬프트: 라우팅 규칙을 훨씬 더 명확하고 엄격하게 변경했습니다.
예를 들어, "ManseTool 다음에는 반드시 RetrieverTool을 호출해야 한다" 와 같은 강력한 규칙을 추가하여 에이전트의 행동을 더 정교하게 제어해야 한다.


- ipynb에서 발생하지 않던 오류들 발생
    -> saju_agentic_rag코드에선 자꾸 웹툴로 빠지는 문제.
    -> 계속 질문을 하면 처음에는 생년월일 및 만세력, 사주오행을 올바르게 가져오지만, 나중에는 만세력, 사주오행을 제대로 반영하지 못하는 문제.