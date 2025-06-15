from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# --- 사주 관련 클래스 및 계산기 ---
@dataclass
class SajuPillar:
    heavenly_stem: str
    earthly_branch: str
    def __str__(self):
        return f"{self.heavenly_stem}{self.earthly_branch}"

@dataclass
class SajuChart:
    year_pillar: SajuPillar
    month_pillar: SajuPillar
    day_pillar: SajuPillar
    hour_pillar: SajuPillar
    birth_info: Dict
    def get_day_master(self) -> str:
        return self.day_pillar.heavenly_stem

class SajuCalculator:
    """사주팔자 계산기 - 개선된 버전"""
    def __init__(self):
        self.heavenly_stems = [
            "갑", "을", "병", "정", "무", "기", "경", "신", "임", "계"
        ]
        self.earthly_branches = [
            "자", "축", "인", "묘", "진", "사", "오", "미", "신", "유", "술", "해"
        ]
        self.five_elements = {
            "갑": "목", "을": "목",
            "병": "화", "정": "화", 
            "무": "토", "기": "토",
            "경": "금", "신": "금",
            "임": "수", "계": "수",
            "자": "수", "축": "토", "인": "목", "묘": "목",
            "진": "토", "사": "화", "오": "화", "미": "토",
            "신": "금", "유": "금", "술": "토", "해": "수"
        }
        self.ten_gods_mapping = {
            "목": {
                "목": ["비견", "겁재"], "화": ["식신", "상관"], 
                "토": ["편재", "정재"], "금": ["편관", "정관"], 
                "수": ["편인", "정인"]
            },
            "화": {
                "화": ["비견", "겁재"], "토": ["식신", "상관"],
                "금": ["편재", "정재"], "수": ["편관", "정관"],
                "목": ["편인", "정인"]
            },
            "토": {
                "토": ["비견", "겁재"], "금": ["식신", "상관"],
                "수": ["편재", "정재"], "목": ["편관", "정관"],
                "화": ["편인", "정인"]
            },
            "금": {
                "금": ["비견", "겁재"], "수": ["식신", "상관"],
                "목": ["편재", "정재"], "화": ["편관", "정관"],
                "토": ["편인", "정인"]
            },
            "수": {
                "수": ["비견", "겁재"], "목": ["식신", "상관"],
                "화": ["편재", "정재"], "토": ["편관", "정관"],
                "금": ["편인", "정인"]
            }
        }
        self.hidden_stems = {
            "자": [("계", 100)],
            "축": [("기", 60), ("계", 30), ("신", 10)],
            "인": [("갑", 60), ("병", 30), ("무", 10)],
            "묘": [("을", 100)],
            "진": [("무", 60), ("을", 30), ("계", 10)],
            "사": [("병", 60), ("무", 30), ("경", 10)],
            "오": [("정", 70), ("기", 30)],
            "미": [("기", 60), ("정", 30), ("을", 10)],
            "신": [("경", 60), ("임", 30), ("무", 10)],
            "유": [("신", 100)],
            "술": [("무", 60), ("신", 30), ("정", 10)],
            "해": [("임", 70), ("갑", 30)]
        }
        self.solar_terms_2024 = {
            "입춘": (2, 4, 16, 27),
            "우수": (2, 19, 12, 13),
            "경칩": (3, 5, 22, 23),
            "춘분": (3, 20, 15, 6),
            "청명": (4, 4, 21, 2),
            "곡우": (4, 20, 3, 20),
            "입하": (5, 5, 8, 10),
            "소만": (5, 20, 20, 59),
            "망종": (6, 5, 12, 10),
            "하지": (6, 21, 4, 51),
            "소서": (7, 6, 22, 20),
            "대서": (7, 22, 15, 44),
            "입추": (8, 7, 9, 9),
            "처서": (8, 22, 23, 55),
            "백로": (9, 7, 12, 11),
            "추분": (9, 22, 20, 44),
            "한로": (10, 8, 3, 0),
            "상강": (10, 23, 6, 15),
            "입동": (11, 7, 6, 20),
            "소설": (11, 22, 3, 56),
            "대설": (12, 7, 0, 17),
            "동지": (12, 21, 17, 21)
        }

    def calculate_saju(self, year: int, month: int, day: int, hour: int, minute: int = 0, is_male: bool = True, timezone: str = "Asia/Seoul") -> SajuChart:
        birth_datetime = datetime(year, month, day, hour, minute)
        if timezone == "Asia/Seoul":
            birth_datetime = birth_datetime - timedelta(minutes=5, seconds=32)
        base_date = datetime(1900, 1, 1)
        days_diff = (birth_datetime.date() - base_date.date()).days
        year_pillar = self._calculate_year_pillar(year)
        month_pillar = self._calculate_month_pillar_improved(year, month, day)
        day_pillar = self._calculate_day_pillar(days_diff)
        hour_pillar = self._calculate_hour_pillar_improved(day_pillar.heavenly_stem, hour, minute)
        birth_info = {
            "year": year, "month": month, "day": day, 
            "hour": hour, "minute": minute,
            "is_male": is_male, "timezone": timezone,
            "birth_datetime": birth_datetime
        }
        return SajuChart(year_pillar, month_pillar, day_pillar, hour_pillar, birth_info)

    def _calculate_year_pillar(self, year: int) -> SajuPillar:
        base_year = 1984
        year_diff = year - base_year
        stem_index = year_diff % 10
        branch_index = year_diff % 12
        return SajuPillar(
            self.heavenly_stems[stem_index],
            self.earthly_branches[branch_index]
        )

    def _calculate_month_pillar_improved(self, year: int, month: int, day: int) -> SajuPillar:
        month_branch_index = self._get_month_branch_by_solar_terms(year, month, day)
        year_stem_index = (year - 1984) % 10
        if year_stem_index in [0, 5]:
            month_stem_base = 2
        elif year_stem_index in [1, 6]:
            month_stem_base = 4
        elif year_stem_index in [2, 7]:
            month_stem_base = 6
        elif year_stem_index in [3, 8]:
            month_stem_base = 8
        else:
            month_stem_base = 0
        month_offset = (month_branch_index - 2) % 12
        month_stem_index = (month_stem_base + month_offset) % 10
        return SajuPillar(
            self.heavenly_stems[month_stem_index],
            self.earthly_branches[month_branch_index]
        )

    def _get_month_branch_by_solar_terms(self, year: int, month: int, day: int) -> int:
        if year == 2024:
            if month == 1:
                return 0
            elif month == 2:
                if day < 4:
                    return 0
                else:
                    return 2
            elif month == 3:
                if day < 5:
                    return 2
                else:
                    return 3
            elif month == 4:
                if day < 4:
                    return 3
                else:
                    return 4
            elif month == 5:
                if day < 5:
                    return 4
                else:
                    return 5
            elif month == 6:
                if day < 5:
                    return 5
                else:
                    return 6
            elif month == 7:
                if day < 6:
                    return 6
                else:
                    return 7
            elif month == 8:
                if day < 7:
                    return 7
                else:
                    return 8
            elif month == 9:
                if day < 7:
                    return 8
                else:
                    return 9
            elif month == 10:
                if day < 8:
                    return 9
                else:
                    return 10
            elif month == 11:
                if day < 7:
                    return 10
                else:
                    return 11
            else:
                if day < 7:
                    return 11
                else:
                    return 0
        else:
            if month == 1:
                return 0
            elif month == 2:
                return 2
            elif month == 3:
                return 3
            elif month == 4:
                return 4
            elif month == 5:
                return 5
            elif month == 6:
                return 6
            elif month == 7:
                return 7
            elif month == 8:
                return 8
            elif month == 9:
                return 9
            elif month == 10:
                return 10
            elif month == 11:
                return 11
            else:
                return 0

    def _calculate_day_pillar(self, days_diff: int) -> SajuPillar:
        target_date = datetime(1995, 8, 26)
        base_date = datetime(1900, 1, 1)
        target_days = (target_date - base_date).days
        target_stem = 5
        target_branch = 1
        base_stem = (target_stem - target_days) % 10
        base_branch = (target_branch - target_days) % 12
        stem_index = (base_stem + days_diff) % 10
        branch_index = (base_branch + days_diff) % 12
        return SajuPillar(
            self.heavenly_stems[stem_index],
            self.earthly_branches[branch_index]
        )

    def _calculate_hour_pillar_improved(self, day_stem: str, hour: int, minute: int = 0) -> SajuPillar:
        hour_branches = [
            "자", "축", "인", "묘", "진", "사", 
            "오", "미", "신", "유", "술", "해"
        ]
        total_minutes = hour * 60 + minute - 32
        if total_minutes >= 23 * 60 or total_minutes < 1 * 60:
            branch_idx = 0
        elif total_minutes < 3 * 60:
            branch_idx = 1
        elif total_minutes < 5 * 60:
            branch_idx = 2
        elif total_minutes < 7 * 60:
            branch_idx = 3
        elif total_minutes < 9 * 60:
            branch_idx = 4
        elif total_minutes < 11 * 60:
            branch_idx = 5
        elif total_minutes < 13 * 60:
            branch_idx = 6
        elif total_minutes < 15 * 60:
            branch_idx = 7
        elif total_minutes < 17 * 60:
            branch_idx = 8
        elif total_minutes < 19 * 60:
            branch_idx = 9
        elif total_minutes < 21 * 60:
            branch_idx = 10
        else:
            branch_idx = 11
        hour_branch = hour_branches[branch_idx]
        day_stem_idx = self.heavenly_stems.index(day_stem)
        if day_stem_idx in [0, 5]:
            hour_stem_base = 0
        elif day_stem_idx in [1, 6]:
            hour_stem_base = 2
        elif day_stem_idx in [2, 7]:
            hour_stem_base = 4
        elif day_stem_idx in [3, 8]:
            hour_stem_base = 6
        else:
            hour_stem_base = 8
        hour_stem_idx = (hour_stem_base + branch_idx) % 10
        return SajuPillar(
            self.heavenly_stems[hour_stem_idx],
            hour_branch
        )

    def analyze_ten_gods(self, saju_chart: SajuChart) -> Dict[str, List[str]]:
        day_master = saju_chart.get_day_master()
        day_master_element = self.five_elements[day_master]
        ten_gods = {
            "년주": [], "월주": [], "일주": [], "시주": []
        }
        pillars = [
            ("년주", saju_chart.year_pillar),
            ("월주", saju_chart.month_pillar), 
            ("일주", saju_chart.day_pillar),
            ("시주", saju_chart.hour_pillar)
        ]
        for pillar_name, pillar in pillars:
            stem_element = self.five_elements[pillar.heavenly_stem]
            if pillar.heavenly_stem != day_master:
                god_types = self.ten_gods_mapping[day_master_element][stem_element]
                stem_idx = self.heavenly_stems.index(pillar.heavenly_stem)
                day_idx = self.heavenly_stems.index(day_master)
                if (stem_idx % 2) == (day_idx % 2):
                    ten_gods[pillar_name].append(f"천간:{god_types[0]}")
                else:
                    ten_gods[pillar_name].append(f"천간:{god_types[1]}")
            hidden_stems = self.hidden_stems[pillar.earthly_branch]
            for hidden_stem, strength in hidden_stems:
                if hidden_stem != day_master:
                    hidden_element = self.five_elements[hidden_stem]
                    god_types = self.ten_gods_mapping[day_master_element][hidden_element]
                    hidden_idx = self.heavenly_stems.index(hidden_stem)
                    day_idx = self.heavenly_stems.index(day_master)
                    if (hidden_idx % 2) == (day_idx % 2):
                        ten_gods[pillar_name].append(f"지지:{god_types[0]}({strength}%)")
                    else:
                        ten_gods[pillar_name].append(f"지지:{god_types[1]}({strength}%)")
        return ten_gods

    def calculate_great_fortune_improved(self, saju_chart: SajuChart) -> List[Dict]:
        birth_info = saju_chart.birth_info
        year = birth_info["year"]
        month = birth_info["month"]
        day = birth_info["day"]
        is_male = birth_info["is_male"]
        year_stem = saju_chart.year_pillar.heavenly_stem
        year_stem_idx = self.heavenly_stems.index(year_stem)
        is_yang_year = (year_stem_idx % 2 == 0)
        if (is_yang_year and is_male) or (not is_yang_year and not is_male):
            direction = 1
        else:
            direction = -1
        start_age = self._calculate_precise_start_age(year, month, day, direction)
        month_stem_idx = self.heavenly_stems.index(saju_chart.month_pillar.heavenly_stem)
        month_branch_idx = self.earthly_branches.index(saju_chart.month_pillar.earthly_branch)
        great_fortunes = []
        for i in range(8):
            age = start_age + (i * 10)
            stem_idx = (month_stem_idx + (direction * (i + 1))) % 10
            branch_idx = (month_branch_idx + (direction * (i + 1))) % 12
            great_fortunes.append({
                "age": age,
                "pillar": f"{self.heavenly_stems[stem_idx]}{self.earthly_branches[branch_idx]}",
                "years": f"{year + age}년 ~ {year + age + 9}년"
            })
        return great_fortunes

    def _calculate_precise_start_age(self, year: int, month: int, day: int, direction: int) -> int:
        base_age = 6
        if day > 15:
            adjustment = 1 if direction == 1 else -1
        else:
            adjustment = 0
        return max(1, base_age + adjustment)

    def get_element_strength(self, saju_chart: SajuChart) -> Dict[str, int]:
        elements = {"목": 0, "화": 0, "토": 0, "금": 0, "수": 0}
        pillars = [saju_chart.year_pillar, saju_chart.month_pillar, 
                  saju_chart.day_pillar, saju_chart.hour_pillar]
        for pillar in pillars:
            stem_element = self.five_elements[pillar.heavenly_stem]
            elements[stem_element] += 20
            hidden_stems = self.hidden_stems[pillar.earthly_branch]
            for hidden_stem, strength in hidden_stems:
                hidden_element = self.five_elements[hidden_stem]
                elements[hidden_element] += int(strength * 0.15)
        return elements

def format_saju_analysis(saju_chart: SajuChart, calculator: SajuCalculator) -> str:
    analysis = []
    analysis.append("=== 사주팔자 ===")
    analysis.append(f"년주(年柱): {saju_chart.year_pillar}")
    analysis.append(f"월주(月柱): {saju_chart.month_pillar}")
    analysis.append(f"일주(日柱): {saju_chart.day_pillar}")
    analysis.append(f"시주(時柱): {saju_chart.hour_pillar}")
    analysis.append(f"일간(日干): {saju_chart.get_day_master()}")
    analysis.append("")
    elements = calculator.get_element_strength(saju_chart)
    analysis.append("=== 오행 강약 ===")
    for element, strength in elements.items():
        analysis.append(f"{element}: {strength}점")
    analysis.append("")
    ten_gods = calculator.analyze_ten_gods(saju_chart)
    analysis.append("=== 십신 분석 ===")
    for pillar_name, gods in ten_gods.items():
        if gods:
            analysis.append(f"{pillar_name}: {', '.join(gods)}")
    analysis.append("")
    great_fortunes = calculator.calculate_great_fortune_improved(saju_chart)
    analysis.append("=== 대운 (정밀 계산) ===")
    for gf in great_fortunes[:4]:
        analysis.append(f"{gf['age']}세: {gf['pillar']} ({gf['years']})")
    return "\n".join(analysis)

# --- LangChain Tool 등록 ---
from langchain_core.tools import tool

saju_calculator = SajuCalculator()

@tool("calculate_saju_chart")
def calculate_saju_tool(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int = 0,
    is_male: bool = True,
    timezone: str = "Asia/Seoul"
) -> str:
    """
    생년월일, 시간, 성별을 입력받아 사주팔자 해석을 반환합니다.
    """
    chart = saju_calculator.calculate_saju(
        year=year,
        month=month,
        day=day,
        hour=hour,
        minute=minute,
        is_male=is_male,
        timezone=timezone
    )
    return format_saju_analysis(chart, saju_calculator)

# 예시 사용
# saju_calc = SajuCalculator()
# chart = saju_calc.calculate_saju(1995, 3, 28, 12, 30, True, "Asia/Seoul")
# print(format_saju_analysis(chart, saju_calc)) 
