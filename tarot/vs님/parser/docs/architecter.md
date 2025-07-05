graph TB
    subgraph "📱 사용자 입력"
        UI[사용자 메시지]
    end
    
    subgraph "🚀 Layer 1: Fast Track (1-2초)"
        UI --> SC[state_classifier_node<br/>📊 상태 분류]
        SC --> |CONSULTATION_ACTIVE| FT1[Fast Track 1<br/>상담 진행 중]
        SC --> |FOLLOWUP_QUESTION| FT2[Fast Track 2<br/>추가 질문]
        
        subgraph "Fast Track 1 처리"
            FT1 --> CCH[consultation_continue_handler<br/>스프레드 선택 응답]
            FT1 --> CSH[consultation_summary_handler<br/>카드 해석 & 고급 분석]
            FT1 --> CFH[consultation_final_handler<br/>개별 해석]
        end
        
        subgraph "Fast Track 2 처리"
            FT2 --> CRH[context_reference_handler<br/>이전 답변 참조 설명]
        end
        
        CCH --> UP1[Unified Processor]
        CSH --> UP1
        CFH --> UP1
        CRH --> UP1
    end
    
    subgraph "🧠 Layer 2: Full Analysis (3-5초)"
        SC --> |NEW_SESSION| SM[supervisor_master_node<br/>🎯 마스터 분석기]
        
        subgraph "Full Analysis 내부"
            SM --> SLN[supervisor_llm_node<br/>🤖 LLM 의도 파악]
            SLN --> |route_to_intent| CIN[classify_intent_node<br/>🔍 정밀 의도 분류]
            CIN --> ITH[Intent-based Handler<br/>의도별 처리]
            
            subgraph "의도별 분기"
                ITH --> |card_info| CIH[Card Info Handler<br/>카드 정보 RAG 검색]
                ITH --> |spread_info| SIH[Spread Info Handler<br/>스프레드 정보 RAG 검색]
                ITH --> |consultation| CH[Consultation Handler<br/>신규 상담 시작]
                ITH --> |general| GH[General Handler<br/>일반 질문 + 웹 검색]
            end
        end
        
        subgraph "리팩토링된 상담 플로우"
            CH --> EAN[emotion_analyzer_node<br/>😊 감정 분석 LLM]
            EAN --> WSDN[web_search_decider_node<br/>🔍 웹 검색 판단 LLM]
            WSDN --> WSN[web_searcher_node<br/>🌐 웹 검색 실행]
            WSN --> SRN[spread_recommender_node<br/>🎯 스프레드 추천 LLM]
        end
        
        CIH --> UP2[Unified Processor]
        SIH --> UP2
        SRN --> UP2
        GH --> UP2
    end
    
    subgraph "🔧 Layer 3: Tool Processing (1-3초)"
        UP1 --> PR{processor_router<br/>도구 호출 체크}
        UP2 --> PR
        
        PR --> |tool_calls 감지| UTH[unified_tool_handler<br/>🛠️ 통합 도구 처리]
        PR --> |완료| END1[응답 완료]
        
        subgraph "도구 실행 시스템"
            UTH --> TN[ToolNode<br/>실제 도구 실행]
            TN --> |search_tarot_cards| STC[🃏 타로 카드 RAG 검색]
            TN --> |search_tarot_spreads| STS[🔮 스프레드 RAG 검색]
            
            STC --> RAG[(🧠 RAG 시스템<br/>FAISS 벡터 DB<br/>7개 타로책 지식)]
            STS --> RAG
            
            TN --> TRH[tool_result_handler<br/>📝 결과를 사용자 친화적으로 변환]
            TRH --> END2[최종 응답]
        end
        
        subgraph "웹 검색 시스템"
            WSN --> WS1[Tavily Search<br/>1순위 검색]
            WSN --> WS2[DuckDuckGo Search<br/>2순위 백업]
            WS1 --> |실패시| WS2
        end
    end
    
    subgraph "🎨 성능 최적화 요소"
        CACHE[🗄️ 스프레드 캐싱]
        PARALLEL[⚡ 병렬 처리]
        FALLBACK[🛡️ 우아한 실패 처리]
        MONITOR[📊 성능 모니터링]
    end
    
    %% 성능 메트릭 표시
    SC -.-> |"패턴 매칭<br/>0.1초"| MONITOR
    SLN -.-> |"LLM 호출<br/>1-2초"| MONITOR
    RAG -.-> |"벡터 검색<br/>0.3초"| MONITOR
    
    %% 스타일링
    classDef fastTrack fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    classDef fullAnalysis fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef toolLayer fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef optimization fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef llmNode fill:#ffebee,stroke:#d32f2f,stroke-width:2px
    
    class SC,FT1,FT2,CCH,CSH,CFH,CRH,UP1 fastTrack
    class SM,SLN,CIN,ITH,CIH,SIH,CH,GH,EAN,WSDN,SRN,UP2 fullAnalysis
    class UTH,TN,STC,STS,RAG,TRH,WSN,WS1,WS2 toolLayer
    class CACHE,PARALLEL,FALLBACK,MONITOR optimization
    class SLN,EAN,WSDN,SRN,CIN llmNode