import numpy as np

from typing import List, Dict, Any

from langchain_openai import ChatOpenAI

from langchain_core.messages import HumanMessage

def calculate_card_draw_probability(deck_size: int = 78, cards_of_interest: int = 1, 
                                 cards_drawn: int = 3, exact_matches: int = 1) -> dict:
    """하이퍼기하분포를 이용한 정확한 카드 확률 계산"""
    try:
        from scipy.stats import hypergeom
        rv = hypergeom(deck_size, cards_of_interest, cards_drawn)
        exact_prob = rv.pmf(exact_matches)
        at_least_one = 1 - rv.pmf(0)
        mean = rv.mean()
        variance = rv.var()
        return {
            "exact_probability": float(exact_prob),
            "at_least_one_probability": float(at_least_one),
            "expected_value": float(mean),
            "variance": float(variance),
            "distribution_type": "hypergeometric"
        }
    except Exception as e:
        return {
            "error": str(e),
            "exact_probability": 0.0,
            "at_least_one_probability": 0.0
        }
def calculate_success_probability_from_cards(selected_cards: List[Dict]) -> dict:
    """선택된 카드들을 기반으로 성공 확률 계산"""
    if not selected_cards:
        return {"success_probability": 0.5, "confidence": "low", "factors": []}
    total_weight = 0
    positive_factors = []
    negative_factors = []
    success_weights = {
        "The Fool": 0.6, "The Magician": 0.9, "The High Priestess": 0.7, "The Empress": 0.8, "The Emperor": 0.8, "The Hierophant": 0.7, "The Lovers": 0.8, "The Chariot": 0.9, "Strength": 0.8, "The Hermit": 0.6, "Wheel of Fortune": 0.7, "Justice": 0.8, "The Hanged Man": 0.4, "Death": 0.5, "Temperance": 0.8, "The Devil": 0.3, "The Tower": 0.2, "The Star": 0.9, "The Moon": 0.4, "The Sun": 0.95, "Judgement": 0.7, "The World": 0.95,
        "Ace": 0.8, "Two": 0.6, "Three": 0.7, "Four": 0.7, "Five": 0.3, "Six": 0.8, "Seven": 0.5, "Eight": 0.7, "Nine": 0.8, "Ten": 0.6, "Page": 0.6, "Knight": 0.7, "Queen": 0.8, "King": 0.8
    }
    suit_modifiers = {"Wands": 0.1, "Cups": 0.05, "Swords": -0.05, "Pentacles": 0.08}
    for card in selected_cards:
        card_name = card.get("name", "")
        orientation = card.get("orientation", "upright")
        weight = 0.5
        if card_name in success_weights:
            weight = success_weights[card_name]
        else:
            for rank in success_weights:
                if rank in card_name and rank not in ["Ace", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"]:
                    weight = success_weights[rank]
                    break
            for suit, modifier in suit_modifiers.items():
                if suit in card_name:
                    weight += modifier
                    break
        if orientation == "reversed":
            if weight > 0.5:
                weight = 1.0 - weight
            else:
                weight = min(0.8, weight + 0.2)
        total_weight += weight
        if weight >= 0.7:
            positive_factors.append(f"{card_name} ({orientation}) - 강한 긍정 에너지")
        elif weight >= 0.6:
            positive_factors.append(f"{card_name} ({orientation}) - 긍정적 영향")
        elif weight <= 0.3:
            negative_factors.append(f"{card_name} ({orientation}) - 주의 필요")
        elif weight <= 0.4:
            negative_factors.append(f"{card_name} ({orientation}) - 도전 요소")
    avg_probability = total_weight / len(selected_cards) if selected_cards else 0.5
    if len(positive_factors) >= 2:
        confidence = "high"
    elif len(positive_factors) >= 1 and len(negative_factors) <= 1:
        confidence = "medium"
    else:
        confidence = "low"
    return {
        "success_probability": round(avg_probability, 3),
        "confidence": confidence,
        "positive_factors": positive_factors,
        "negative_factors": negative_factors,
        "total_cards_analyzed": len(selected_cards)
    }
def analyze_card_combination_synergy(selected_cards: List[Dict]) -> dict:
    """카드 조합의 시너지 효과 분석"""
    if len(selected_cards) < 2:
        return {"synergy_score": 0.5, "combinations": [], "warnings": []}
    synergy_score = 0.5
    combinations = []
    warnings = []
    elements = {"Wands": 0, "Cups": 0, "Swords": 0, "Pentacles": 0}
    major_count = 0
    for card in selected_cards:
        card_name = card.get("name", "")
        if any(major in card_name for major in ["The", "Fool", "Magician", "Priestess"]):
            major_count += 1
        else:
            for element in elements:
                if element in card_name:
                    elements[element] += 1
                    break
    active_elements = sum(1 for count in elements.values() if count > 0)
    if active_elements >= 3:
        synergy_score += 0.1
        combinations.append("다양한 원소의 균형잡힌 조합")
    elif active_elements == 2:
        synergy_score += 0.05
        combinations.append("두 원소의 조화로운 결합")
    if major_count >= 2:
        synergy_score += 0.15
        combinations.append("강력한 Major Arcana 에너지")
    elif major_count == 1:
        synergy_score += 0.05
        combinations.append("Major Arcana의 지도력")
    card_names = [card.get("name", "") for card in selected_cards]
    success_combinations = [
        (["The Magician", "The Star"], 0.2, "의지력과 희망의 완벽한 조합"),
        (["The Sun", "The World"], 0.25, "성공과 완성의 최고 조합"),
        (["Ace of", "The Fool"], 0.15, "새로운 시작"),
        (["Queen", "King"], 0.1, "성숙한 리더십의 조화")
    ]
    for combo_cards, bonus, description in success_combinations:
        if all(any(combo_card in card_name for card_name in card_names) 
               for combo_card in combo_cards):
            synergy_score += bonus
            combinations.append(description)
    warning_combinations = [
        (["The Tower", "Death"], "급격한 변화와 파괴의 이중 충격"),
        (["The Devil", "The Moon"], "혼란과 속박의 위험한 조합"),
        (["Five of", "Seven of"], "갈등과 도전이 겹치는 어려운 시기")
    ]
    for combo_cards, warning in warning_combinations:
         if all(any(combo_card in card_name for card_name in card_names) 
                for combo_card in combo_cards):
             synergy_score -= 0.1
             warnings.append(warning)
    return {
        "synergy_score": round(max(0.1, min(1.0, synergy_score)), 3),
        "combinations": combinations,
        "warnings": warnings,
        "element_distribution": elements,
        "major_arcana_count": major_count
    }
def analyze_elemental_balance(selected_cards: List[Dict]) -> dict:
    """카드의 원소 균형 분석"""
    elements = {
        "Fire": {"count": 0, "cards": [], "keywords": ["열정", "행동", "창조", "에너지"]},
        "Water": {"count": 0, "cards": [], "keywords": ["감정", "직감", "관계", "치유"]},
        "Air": {"count": 0, "cards": [], "keywords": ["사고", "소통", "갈등", "변화"]},
        "Earth": {"count": 0, "cards": [], "keywords": ["물질", "안정", "실용", "성장"]}
    }
    element_mapping = {
        "Wands": "Fire",
        "Cups": "Water", 
        "Swords": "Air",
        "Pentacles": "Earth"
    }
    for card in selected_cards:
        card_name = card.get("name", "")
        for suit, elem in element_mapping.items():
            if suit in card_name:
                elements[elem]["count"] += 1
                elements[elem]["cards"].append(card_name)
    dominant = None
    max_count = 0
    for elem, data in elements.items():
        if data["count"] > max_count:
            dominant = elem
            max_count = data["count"]
    missing = [elem for elem, data in elements.items() if data["count"] == 0]
    balance_score = 0.5
    if max_count >= 2:
        balance_score += 0.2
    if len(missing) == 0:
        balance_score += 0.1
    elif len(missing) >= 2:
        balance_score -= 0.1
    return {
        "elements": elements,
        "dominant_element": dominant,
        "missing_elements": missing,
        "balance_score": round(max(0.1, min(1.0, balance_score)), 3)
    }
def generate_elemental_interpretation(elements: dict, dominant: str, missing: list) -> str:
    """원소 분석 결과 해석 생성"""
    interpretations = []
    if dominant and elements[dominant]["count"] > 0:
        element_meanings = {
            "Fire": "강한 행동력과 열정이 지배적입니다. 적극적으로 추진하되 성급함을 주의하세요.",
            "Water": "감정과 직감이 중요한 역할을 합니다. 관계와 내면의 소리에 귀 기울이세요.",
            "Air": "사고와 소통이 핵심입니다. 명확한 계획과 의사소통이 성공의 열쇠입니다.",
            "Earth": "실용적이고 안정적인 접근이 필요합니다. 차근차근 기반을 다지세요."
        }
        interpretations.append(element_meanings.get(dominant, ""))
    if missing:
        missing_advice = {
            "Fire": "더 적극적이고 열정적인 행동이 필요합니다.",
            "Water": "감정적 측면과 직감을 더 고려해보세요.",
            "Air": "논리적 사고와 소통을 강화하세요.",
            "Earth": "현실적이고 실용적인 계획이 부족합니다."
        }
        for elem in missing:
            interpretations.append(missing_advice.get(elem, ""))
    return " ".join(interpretations)
def calculate_numerological_significance(selected_cards: List[Dict]) -> dict:
    """카드의 수비학적 의미 분석"""
    if not selected_cards:
        return {"total_value": 0, "reduced_value": 0, "meaning": ""}
    numerology_values = {
        "The Fool": 0, "The Magician": 1, "The High Priestess": 2, "The Empress": 3,
        "The Emperor": 4, "The Hierophant": 5, "The Lovers": 6, "The Chariot": 7,
        "Strength": 8, "The Hermit": 9, "Wheel of Fortune": 10, "Justice": 11,
        "The Hanged Man": 12, "Death": 13, "Temperance": 14, "The Devil": 15,
        "The Tower": 16, "The Star": 17, "The Moon": 18, "The Sun": 19,
        "Judgement": 20, "The World": 21,
        "Ace": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5,
        "Six": 6, "Seven": 7, "Eight": 8, "Nine": 9, "Ten": 10,
        "Page": 11, "Knight": 12, "Queen": 13, "King": 14
    }
    total_value = 0
    card_values = []
    for card in selected_cards:
        card_name = card.get("name", "")
        value = 0
        if card_name in numerology_values:
            value = numerology_values[card_name]
        else:
            for rank, num_value in numerology_values.items():
                if rank in card_name:
                    value = num_value
                    break
        total_value += value
        card_values.append({"card": card_name, "value": value})
    reduced_value = total_value
    while reduced_value > 9 and reduced_value not in [11, 22, 33]:
        reduced_value = sum(int(digit) for digit in str(reduced_value))
    numerology_meanings = {
        0: "무한한 가능성과 새로운 시작",
        1: "리더십과 독립성, 새로운 시작",
        2: "협력과 균형, 파트너십",
        3: "창조성과 표현, 소통",
        4: "안정성과 질서, 근면",
        5: "자유와 모험, 변화",
        6: "책임과 보살핌, 조화",
        7: "영성과 내면 탐구, 완벽",
        8: "물질적 성공과 권력, 성취",
        9: "완성과 지혜, 봉사",
        11: "직감과 영감, 마스터 넘버",
        22: "마스터 빌더, 큰 꿈의 실현",
        33: "마스터 교사, 무조건적 사랑"
    }
    return {
        "total_value": total_value,
        "reduced_value": reduced_value,
        "meaning": numerology_meanings.get(reduced_value, "특별한 의미"),
        "card_values": card_values,
        "is_master_number": reduced_value in [11, 22, 33]
    }
def generate_integrated_analysis(selected_cards: List[Dict]) -> dict:
    """확률, 원소, 수비학을 통합한 종합 분석"""
    success_analysis = calculate_success_probability_from_cards(selected_cards)
    synergy_analysis = analyze_card_combination_synergy(selected_cards)
    elemental_analysis = analyze_elemental_balance(selected_cards)
    numerology_analysis = calculate_numerological_significance(selected_cards)
    integrated_score = (
        success_analysis.get("success_probability", 0.5) * 0.4 +
        synergy_analysis.get("synergy_score", 0.5) * 0.3 +
        elemental_analysis.get("balance_score", 0.5) * 0.2 +
        min(1.0, numerology_analysis.get("reduced_value", 5) / 9) * 0.1
    )
    interpretation = []
    success_prob = success_analysis.get("success_probability", 0.5)
    if success_prob >= 0.7:
        interpretation.append("🌟 높은 성공 가능성을 보여줍니다")
    elif success_prob >= 0.6:
        interpretation.append("✨ 긍정적인 결과가 예상됩니다")
    elif success_prob <= 0.4:
        interpretation.append("⚠️ 신중한 접근이 필요합니다")
    if elemental_analysis.get("balance_score", 0) >= 0.7:
        interpretation.append("🔮 원소들이 조화롭게 균형을 이룹니다")
    elif elemental_analysis.get("dominant_element"):
        dominant = elemental_analysis["dominant_element"]
        interpretation.append(f"🔥 {dominant} 원소의 강한 영향을 받습니다")
    if numerology_analysis.get("is_master_number"):
        interpretation.append(f"✨ 마스터 넘버 {numerology_analysis['reduced_value']}의 특별한 에너지")
    return {
        "integrated_score": round(integrated_score, 3),
        "success_analysis": success_analysis,
        "synergy_analysis": synergy_analysis,
        "elemental_analysis": elemental_analysis,
        "numerology_analysis": numerology_analysis,
        "interpretation": " | ".join(interpretation),
        "recommendation": generate_integrated_recommendation(integrated_score, success_analysis, elemental_analysis)
    }
def generate_integrated_recommendation(score: float, success_analysis: dict, elemental_analysis: dict) -> str:
    """통합 분석 기반 추천사항 생성"""
    recommendations = []
    if score >= 0.7:
        recommendations.append("적극적으로 추진하세요")
    elif score >= 0.6:
        recommendations.append("신중하되 긍정적으로 접근하세요")
    elif score >= 0.5:
        recommendations.append("균형잡힌 접근이 필요합니다")
    else:
        recommendations.append("충분한 준비와 대안을 마련하세요")
    dominant = elemental_analysis.get("dominant_element")
    if dominant == "Fire":
        recommendations.append("열정을 조절하며 계획적으로 행동하세요")
    elif dominant == "Water":
        recommendations.append("직감을 믿되 현실적 판단도 함께 하세요")
    elif dominant == "Air":
        recommendations.append("소통과 정보 수집에 집중하세요")
    elif dominant == "Earth":
        recommendations.append("안정적이고 실용적인 방법을 선택하세요")
    return " | ".join(recommendations)
def analyze_emotion_and_empathy(user_input: str) -> Dict[str, Any]:
    """사용자 입력에서 감정 상태 분석 및 공감 톤 결정"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    emotion_prompt = f"""
    다음 사용자 입력에서 감정을 분석하고 적절한 공감 방식을 제안해주세요:
    "{user_input}"
    다음 JSON 형식으로 답변해주세요:
    {{
        "primary_emotion": "불안/슬픔/분노/혼란/기대/걱정/좌절/기타",
        "emotion_intensity": "낮음/보통/높음/매우높음",
        "empathy_tone": "gentle/supportive/encouraging/understanding",
        "comfort_message": "적절한 위로 메시지 (한 문장)",
        "response_style": "formal/casual/warm/professional"
    }}
    """
    try:
        response = llm.invoke([HumanMessage(content=emotion_prompt)])
        import json
        try:
            emotion_data = json.loads(response.content)
            return emotion_data
        except:
            return {
                "primary_emotion": "혼란",
                "emotion_intensity": "보통",
                "empathy_tone": "supportive",
                "comfort_message": "마음이 복잡하시겠어요. 함께 답을 찾아보겠습니다.",
                "response_style": "warm"
            }
    except Exception as e:
        print(f"🔍 감정 분석 오류: {e}")
        return {
            "primary_emotion": "혼란",
            "emotion_intensity": "보통", 
            "empathy_tone": "supportive",
            "comfort_message": "마음이 복잡하시겠어요. 함께 답을 찾아보겠습니다.",
            "response_style": "warm"
        }
def generate_empathy_message(emotional_analysis: Dict, user_concern: str) -> str:
    """감정 상태에 따른 공감 메시지 생성"""
    emotion = emotional_analysis.get('primary_emotion', '혼란')
    intensity = emotional_analysis.get('emotion_intensity', '보통')
    empathy_templates = {
        "불안": {
            "매우높음": "지금 정말 많이 불안하시겠어요. 그런 마음 충분히 이해합니다. 🤗 함께 차근차근 풀어보아요.",
            "높음": "많이 불안하시겠어요. 그런 마음이 드는 게 당연합니다. 타로가 좋은 방향을 제시해줄 거예요.",
            "보통": "걱정이 많으시군요. 마음이 복잡하실 텐데, 함께 해답을 찾아보아요.",
            "낮음": "약간의 불안감이 느껴지시는군요. 차근차근 살펴보겠습니다."
        },
        "슬픔": {
            "매우높음": "정말 많이 힘드시겠어요. 혼자가 아니니까 괜찮습니다. 💙 시간이 걸리더라도 함께 이겨내요.",
            "높음": "정말 힘드시겠어요. 마음이 아프시겠지만, 위로가 되는 답을 찾아드릴게요.",
            "보통": "마음이 무거우시겠어요. 슬픈 마음이 조금이라도 가벼워질 수 있도록 도와드릴게요.",
            "낮음": "조금 속상하시는 것 같아요. 함께 이야기해보면서 마음을 정리해보아요."
        },
        "걱정": {
            "매우높음": "정말 많이 걱정되시겠어요. 그런 마음이 드는 게 당연합니다. 함께 불안을 덜어보아요.",
            "높음": "많이 걱정되시는군요. 미래에 대한 두려움이 크시겠어요. 희망적인 답을 찾아보겠습니다.",
            "보통": "걱정이 되시는 상황이군요. 타로를 통해 안심할 수 있는 답을 찾아보아요.",
            "낮음": "조금 걱정되시는 것 같아요. 함께 살펴보면 마음이 편해질 거예요."
        }
    }
    emotion_messages = empathy_templates.get(emotion, empathy_templates.get("걱정", {}))
    message = emotion_messages.get(intensity, "마음이 복잡하시겠어요. 함께 답을 찾아보겠습니다.")
    return message
