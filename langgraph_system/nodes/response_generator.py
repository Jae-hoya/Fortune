"""
응답 생성 노드
모든 워커 결과를 종합하여 최종 사용자 응답을 생성
"""

import time
from typing import Dict, Any, List, Optional

from ..state import SupervisorState
from models import get_openai_llm

class ResponseGenerator:
    """최종 응답 생성기"""
    
    def __init__(self):
        self.llm = get_openai_llm()
    
    def __call__(self, state: SupervisorState) -> SupervisorState:
        """응답 생성 노드 실행"""
        start_time = time.time()
        
        try:
            # 최종 응답 생성
            final_response = self._generate_comprehensive_response(state)
            
            # 상태 업데이트
            return {
                **state,
                "final_response": final_response,
                "response_generated": True,
                "total_execution_time": time.time() - start_time
            }
            
        except Exception as e:
            # 오류 발생시 기본 응답 생성
            fallback_response = self._generate_fallback_response(state, str(e))
            
            return {
                **state,
                "final_response": fallback_response,
                "response_generated": True,
                "response_error": str(e),
                "total_execution_time": time.time() - start_time
            }
    
    def _generate_comprehensive_response(self, state: SupervisorState) -> str:
        """종합적인 응답 생성"""
        user_query = state["user_query"]
        question_type = state.get("question_type", "")
        worker_results = state.get("worker_results", {})
        
        # 워커별 결과 수집
        saju_result = self._extract_worker_result(worker_results, "saju")
        rag_result = self._extract_worker_result(worker_results, "rag")
        web_result = self._extract_worker_result(worker_results, "web")
        
        # 질문 유형에 따른 응답 생성
        if question_type == "saju_calculation":
            return self._generate_saju_response(user_query, saju_result, rag_result)
        elif question_type == "fortune_consultation":
            return self._generate_fortune_response(user_query, saju_result, rag_result, web_result)
        elif question_type == "general_search":
            return self._generate_general_response(user_query, web_result, rag_result)
        else:
            return self._generate_mixed_response(user_query, saju_result, rag_result, web_result)
    
    def _extract_worker_result(self, worker_results: Dict[str, Any], worker_name: str) -> Optional[Dict[str, Any]]:
        """워커 결과 추출"""
        if worker_name not in worker_results:
            return None
        
        worker_result = worker_results[worker_name]
        if not worker_result.get("success", False):
            return None
        
        return worker_result.get("result")
    
    def _generate_saju_response(self, user_query: str, saju_result: Optional[Dict], rag_result: Optional[Dict]) -> str:
        """사주 계산 전용 응답 생성"""
        if not saju_result:
            return "죄송합니다. 사주 계산에 필요한 생년월일시 정보가 부족합니다. 정확한 생년월일시를 다시 알려주세요."
        
        # 기본 사주 정보
        response_parts = []
        
        # 1. 사주팔자 표시
        if "saju_chart" in saju_result:
            response_parts.append("🔮 **사주팔자**")
            chart = saju_result["saju_chart"]
            chart_text = f"""
년주: {chart.get('year_pillar', '')}
월주: {chart.get('month_pillar', '')}
일주: {chart.get('day_pillar', '')}
시주: {chart.get('hour_pillar', '')}
일간: {chart.get('day_master', '')}
"""
            response_parts.append(chart_text)
            response_parts.append("")
        
        # 2. 포맷팅된 텍스트 (있는 경우)
        if "formatted_text" in saju_result:
            response_parts.append("📊 **사주 분석**")
            response_parts.append(saju_result["formatted_text"])
            response_parts.append("")
        
        # 3. 분석 데이터 (있는 경우)
        if "analysis" in saju_result:
            analysis = saju_result["analysis"]
            if "day_master_strength" in analysis:
                response_parts.append("💪 **일간 강약**")
                response_parts.append(f"일간 강도: {analysis['day_master_strength']}")
                response_parts.append("")
        
        # 4. RAG 보완 정보
        if rag_result and rag_result.get("relevant_passages"):
            response_parts.append("📚 **전문 해석**")
            for passage in rag_result["relevant_passages"][:3]:
                response_parts.append(f"• {passage}")
            response_parts.append("")
        
        # 응답이 비어있으면 기본 메시지
        if not response_parts:
            response_parts.append("🔮 **사주 계산 완료**")
            response_parts.append("사주 계산이 완료되었습니다. 상세한 해석을 원하시면 추가 질문을 해주세요.")
        
        return "\n".join(response_parts)
    
    def _generate_fortune_response(self, user_query: str, saju_result: Optional[Dict], 
                                 rag_result: Optional[Dict], web_result: Optional[Dict]) -> str:
        """운세 상담 응답 생성"""
        response_parts = []
        
        # 1. 사주 기반 분석 (있는 경우)
        if saju_result:
            response_parts.append("🔮 **개인 사주 기반 분석**")
            if "analysis_summary" in saju_result:
                response_parts.append(saju_result["analysis_summary"])
            response_parts.append("")
        
        # 2. 전문 지식 기반 해석
        if rag_result and rag_result.get("relevant_passages"):
            response_parts.append("📚 **전문 운세 해석**")
            for passage in rag_result["relevant_passages"][:3]:
                response_parts.append(f"• {passage}")
            response_parts.append("")
        
        # 3. 최신 운세 정보
        if web_result and web_result.get("filtered_content"):
            response_parts.append("🌐 **최신 운세 정보**")
            for content in web_result["filtered_content"][:3]:
                response_parts.append(f"• {content}")
            response_parts.append("")
        
        # 4. 종합 조언
        response_parts.append("💡 **종합 조언**")
        advice = self._generate_personalized_advice(user_query, saju_result, rag_result, web_result)
        response_parts.append(advice)
        
        return "\n".join(response_parts)
    
    def _generate_general_response(self, user_query: str, web_result: Optional[Dict], 
                                 rag_result: Optional[Dict]) -> str:
        """일반 검색 응답 생성"""
        response_parts = []
        has_useful_content = False
        
        # 1. 웹 검색 결과 확인 및 처리
        if web_result and web_result.get("filtered_content"):
            # 유용한 웹 검색 결과가 있는지 확인
            useful_content = [content for content in web_result["filtered_content"][:5] 
                            if content and len(content.strip()) > 10]
            if useful_content:
                response_parts.append("🔍 **검색 결과**")
                for content in useful_content:
                    response_parts.append(f"• {content}")
                response_parts.append("")
                has_useful_content = True
        
        # 2. RAG 결과 확인 및 처리 (한국어 사주 관련 내용만)
        if rag_result and rag_result.get("relevant_passages"):
            # 한국어 사주 관련 내용인지 엄격하게 확인
            korean_passages = []
            for passage in rag_result["relevant_passages"][:3]:
                if passage:
                    # 영어 내용 제외 (Tiger, Rat, Dragon 등 서양 점성술 키워드)
                    english_keywords = ["Tiger", "Rat", "Dragon", "Snake", "Horse", "Rabbit", "Monkey", "Rooster", "Dog", "Pig", "Ox", "A.M.", "P.M."]
                    has_english = any(keyword in passage for keyword in english_keywords)
                    
                    # 한국어 사주 키워드 확인
                    korean_keywords = ["오행", "십신", "사주", "팔자", "명리", "대운", "천간", "지지", "갑을병정", "자축인묘"]
                    has_korean = any(keyword in passage for keyword in korean_keywords)
                    
                    # 영어 내용이 없고 한국어 키워드가 있는 경우만 포함
                    if not has_english and has_korean:
                        korean_passages.append(passage)
            
            if korean_passages:
                response_parts.append("📚 **관련 전문 지식**")
                for passage in korean_passages:
                    response_parts.append(f"• {passage}")
                response_parts.append("")
                has_useful_content = True
        
        # 3. 참고 링크
        if web_result and web_result.get("source_urls"):
            useful_urls = [url for url in web_result["source_urls"][:3] if url and url.startswith("http")]
            if useful_urls:
                response_parts.append("🔗 **참고 링크**")
                for url in useful_urls:
                    response_parts.append(f"• {url}")
                response_parts.append("")
                has_useful_content = True
        
        # 4. 검색 결과가 없거나 부족한 경우 기본 지식으로 답변
        if not has_useful_content:
            return self._generate_basic_saju_knowledge(user_query)
        
        return "\n".join(response_parts)
    
    def _generate_mixed_response(self, user_query: str, saju_result: Optional[Dict], 
                               rag_result: Optional[Dict], web_result: Optional[Dict]) -> str:
        """복합 질문 응답 생성"""
        response_parts = []
        
        # 사용 가능한 모든 정보 종합
        if saju_result:
            response_parts.append("🔮 **사주 분석**")
            if "analysis_summary" in saju_result:
                response_parts.append(saju_result["analysis_summary"])
            response_parts.append("")
        
        if rag_result and rag_result.get("relevant_passages"):
            response_parts.append("📚 **전문 지식**")
            for passage in rag_result["relevant_passages"][:2]:
                response_parts.append(f"• {passage}")
            response_parts.append("")
        
        if web_result and web_result.get("filtered_content"):
            response_parts.append("🌐 **최신 정보**")
            for content in web_result["filtered_content"][:2]:
                response_parts.append(f"• {content}")
            response_parts.append("")
        
        # 종합 결론
        response_parts.append("💡 **종합 결론**")
        conclusion = self._generate_comprehensive_conclusion(user_query, saju_result, rag_result, web_result)
        response_parts.append(conclusion)
        
        return "\n".join(response_parts)
    
    def _generate_personalized_advice(self, user_query: str, saju_result: Optional[Dict], 
                                    rag_result: Optional[Dict], web_result: Optional[Dict]) -> str:
        """개인화된 조언 생성"""
        # LLM을 사용한 개인화된 조언 생성
        context_parts = []
        
        if saju_result:
            context_parts.append(f"사주 분석: {saju_result.get('analysis_summary', '')}")
        
        if rag_result:
            context_parts.append(f"전문 지식: {' '.join(rag_result.get('relevant_passages', [])[:2])}")
        
        if web_result:
            context_parts.append(f"최신 정보: {' '.join(web_result.get('filtered_content', [])[:2])}")
        
        context = " | ".join(context_parts)
        
        prompt = f"""
사용자 질문: {user_query}
수집된 정보: {context}

위 정보를 바탕으로 사용자에게 도움이 되는 개인화된 조언을 3-4문장으로 작성해주세요.
따뜻하고 격려적인 톤으로, 구체적이고 실용적인 조언을 포함해주세요.
"""
        
        try:
            advice = self.llm.invoke(prompt).content
            return advice.strip()
        except:
            return "현재 상황을 종합해보면, 긍정적인 마음가짐을 유지하며 차근차근 나아가시는 것이 좋겠습니다. 어려움이 있더라도 이는 성장의 기회가 될 수 있으니 너무 걱정하지 마세요."
    
    def _generate_comprehensive_conclusion(self, user_query: str, saju_result: Optional[Dict], 
                                         rag_result: Optional[Dict], web_result: Optional[Dict]) -> str:
        """종합 결론 생성"""
        # 간단한 결론 생성 로직
        if saju_result and rag_result and web_result:
            return "개인 사주, 전문 지식, 최신 정보를 종합한 결과입니다. 다양한 관점에서 균형잡힌 시각을 가지시기 바랍니다."
        elif saju_result and rag_result:
            return "개인 사주와 전문 지식을 바탕으로 한 분석입니다. 참고하시되 너무 맹신하지는 마세요."
        elif saju_result:
            return "개인 사주를 바탕으로 한 분석입니다. 하나의 참고 자료로 활용해주세요."
        else:
            return "수집된 정보를 바탕으로 한 답변입니다. 추가 질문이 있으시면 언제든 말씀해주세요."
    
    def _generate_basic_saju_knowledge(self, user_query: str) -> str:
        """기본 사주 지식 제공"""
        query_lower = user_query.lower()
        
        # 오행 관련 질문
        if "오행" in user_query:
            return """
🌟 **오행(五行)이란?**

오행은 동양철학의 핵심 개념으로, 우주의 모든 현상을 다섯 가지 기본 요소로 설명하는 이론입니다.

**🔥 화(火) - 불**
• 특성: 뜨겁고 위로 올라가는 성질
• 계절: 여름
• 방향: 남쪽
• 감정: 기쁨

**🌱 목(木) - 나무**
• 특성: 생장하고 뻗어나가는 성질
• 계절: 봄
• 방향: 동쪽
• 감정: 분노

**🌍 토(土) - 흙**
• 특성: 중앙에서 포용하는 성질
• 계절: 늦여름(환절기)
• 방향: 중앙
• 감정: 사려

**⚪ 금(金) - 쇠**
• 특성: 수렴하고 단단한 성질
• 계절: 가을
• 방향: 서쪽
• 감정: 슬픔

**💧 수(水) - 물**
• 특성: 아래로 흐르고 스며드는 성질
• 계절: 겨울
• 방향: 북쪽
• 감정: 두려움

**상생(相生)**: 목→화→토→금→수→목 (서로 돕는 관계)
**상극(相剋)**: 목→토→수→화→금→목 (서로 견제하는 관계)
"""
        
        # 십신 관련 질문
        elif "십신" in user_query:
            return """
🎭 **십신(十神)이란?**

십신은 사주팔자에서 일간(본인)을 중심으로 다른 글자들과의 관계를 나타내는 10가지 신(神)입니다.

**👑 비견(比肩)**: 나와 같은 오행
• 의미: 형제, 친구, 동료
• 특성: 독립심, 자존심

**🤝 겁재(劫財)**: 나와 같은 오행이지만 음양이 다름
• 의미: 경쟁자, 라이벌
• 특성: 경쟁심, 투쟁심

**🍽️ 식신(食神)**: 내가 생하는 오행 (같은 음양)
• 의미: 자식, 재능, 표현력
• 특성: 창조력, 예술성

**💰 상관(傷官)**: 내가 생하는 오행 (다른 음양)
• 의미: 기술, 재능, 반항심
• 특성: 개성, 창의력

**💎 편재(偏財)**: 내가 극하는 오행 (다른 음양)
• 의미: 유동적 재물, 인연
• 특성: 사교성, 활동력

**🏦 정재(正財)**: 내가 극하는 오행 (같은 음양)
• 의미: 고정적 재물, 아내
• 특성: 성실함, 책임감

**⚔️ 편관(偏官, 칠살)**: 나를 극하는 오행 (다른 음양)
• 의미: 권력, 압박, 시련
• 특성: 추진력, 결단력

**👮 정관(正官)**: 나를 극하는 오행 (같은 음양)
• 의미: 직업, 명예, 남편
• 특성: 질서, 책임감

**🎓 편인(偏印)**: 나를 생하는 오행 (다른 음양)
• 의미: 학문, 종교, 특수 기술
• 특성: 직감력, 신비성

**👩‍🏫 정인(正印)**: 나를 생하는 오행 (같은 음양)
• 의미: 학문, 명예, 어머니
• 특성: 학습력, 포용력
"""
        
        # 합충형해 관련 질문
        elif any(keyword in user_query for keyword in ["합충", "형해", "합", "충", "형", "해"]):
            return """
🔄 **합충형해(合沖刑害)란?**

사주팔자에서 지지(地支) 간의 특수한 관계를 나타내는 네 가지 작용입니다.

**💕 합(合) - 화합**
• **삼합**: 같은 오행으로 모이는 관계
  - 신자진(申子辰) 수국 합
  - 인오술(寅午戌) 화국 합
  - 사유축(巳酉丑) 금국 합
  - 해묘미(亥卯未) 목국 합

• **육합**: 두 지지가 만나 화합하는 관계
  - 자축(子丑), 인해(寅亥), 묘술(卯戌)
  - 진유(辰酉), 사신(巳申), 오미(午未)

**⚡ 충(沖) - 대립**
• 정반대 방향의 지지가 충돌하는 관계
  - 자오충(子午沖), 축미충(丑未沖)
  - 인신충(寅申沖), 묘유충(卯酉沖)
  - 진술충(辰戌沖), 사해충(巳亥沖)

**⚖️ 형(刑) - 형벌**
• 서로 해를 끼치는 관계
  - 자묘형(子卯刑): 무례지형
  - 인사신형(寅巳申刑): 무은지형
  - 축술미형(丑戌未刑): 지세지형
  - 진오유해형(辰午酉亥刑): 자형

**☠️ 해(害) - 해로움**
• 서로 손해를 주는 관계
  - 자미해(子未害), 축오해(丑午害)
  - 인사해(寅巳害), 묘진해(卯辰害)
  - 신해해(申亥害), 유술해(酉戌害)

**영향**:
• 합: 화합, 협력, 인연
• 충: 변화, 이동, 갈등
• 형: 형벌, 질병, 사고
• 해: 손해, 방해, 질투
"""
        
        # 일반적인 사주 질문
        else:
            return f"""
🔮 **사주명리학 기본 개념**

**질문**: {user_query}

사주명리학은 동양의 전통 철학으로, 개인의 출생 시간을 바탕으로 운명을 해석하는 학문입니다.

**🏗️ 기본 구조**
• **사주팔자**: 년월일시의 8글자 (천간 4개 + 지지 4개)
• **천간**: 갑을병정무기경신임계 (10개)
• **지지**: 자축인묘진사오미신유술해 (12개)

**🌟 핵심 이론**
• **오행**: 목화토금수의 다섯 기운
• **십신**: 일간 중심의 10가지 관계
• **대운**: 10년 단위의 운세 변화
• **세운**: 매년의 운세

**💡 활용 분야**
• 성격 분석 및 적성 파악
• 인간관계 및 궁합 분석
• 직업 선택 및 사업 운세
• 건강 관리 및 주의사항

더 구체적인 질문이 있으시면 생년월일시와 함께 문의해 주세요!
"""

    def _generate_fallback_response(self, state: SupervisorState, error_message: str) -> str:
        """오류 발생시 대체 응답 생성"""
        user_query = state["user_query"]
        
        return f"""
죄송합니다. 응답 생성 중 오류가 발생했습니다.

**질문**: {user_query}

**상황**: 시스템 처리 중 일시적인 문제가 발생했습니다.

**해결 방법**:
• 잠시 후 다시 시도해주세요
• 질문을 더 구체적으로 다시 작성해보세요
• 생년월일시가 필요한 질문의 경우 정확한 정보를 포함해주세요

**기술적 정보**: {error_message}

언제든 다시 질문해주시면 최선을 다해 도움드리겠습니다! 🙏
"""

# 노드 함수로 래핑
def response_generator_node(state: SupervisorState) -> SupervisorState:
    """응답 생성 노드 실행 함수"""
    generator = ResponseGenerator()
    return generator(state) 