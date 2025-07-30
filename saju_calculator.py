from datetime import datetime, timedelta
from typing import Dict, List
from dataclasses import dataclass
import re 


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
    age: int
    korean_age: int
    current_datetime: str
    is_leap_month: bool  # 윤달 여부 추가

    def get_day_master(self) -> str:
        return self.day_pillar.heavenly_stem

class SajuCalculator:
    def __init__(self):
        self.heavenly_stems = ["갑", "을", "병", "정", "무", "기", "경", "신", "임", "계"]
        self.earthly_branches = ["자", "축", "인", "묘", "진", "사", "오", "미", "신", "유", "술", "해"]
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
            "목": {"목": ["비견", "겁재"], "화": ["식신", "상관"], "토": ["편재", "정재"], "금": ["편관", "정관"], "수": ["편인", "정인"]},
            "화": {"화": ["비견", "겁재"], "토": ["식신", "상관"], "금": ["편재", "정재"], "수": ["편관", "정관"], "목": ["편인", "정인"]},
            "토": {"토": ["비견", "겁재"], "금": ["식신", "상관"], "수": ["편재", "정재"], "목": ["편관", "정관"], "화": ["편인", "정인"]},
            "금": {"금": ["비견", "겁재"], "수": ["식신", "상관"], "목": ["편재", "정재"], "화": ["편관", "정관"], "토": ["편인", "정인"]},
            "수": {"수": ["비견", "겁재"], "목": ["식신", "상관"], "화": ["편재", "정재"], "토": ["편관", "정관"], "금": ["편인", "정인"]}
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
        # 일간 기준 anchor (무축일), 1900-01-01로부터 경과일
        self.DAY_PILLAR_BASE_STEM = 5  # 무
        self.DAY_PILLAR_BASE_BRANCH = 1  # 축
        self.DAY_PILLAR_BASE_DAYS = (datetime(1995, 8, 26) - datetime(1900, 1, 1)).days
        self.monthly_stems = ["병", "정", "무", "기", "경", "신", "임", "계", "갑", "을"]
        
        # 윤달 정보 (1900-2100년)
        self.leap_months = {
            1900: 8, 1903: 5, 1906: 4, 1909: 2, 1911: 6, 1914: 5, 1917: 2, 1919: 7,
            1922: 5, 1925: 4, 1928: 2, 1930: 6, 1933: 5, 1936: 3, 1938: 7, 1941: 6,
            1944: 4, 1947: 2, 1949: 7, 1952: 5, 1955: 3, 1957: 8, 1960: 6, 1963: 4,
            1966: 3, 1968: 7, 1971: 5, 1974: 4, 1976: 8, 1979: 6, 1982: 4, 1984: 10,
            1987: 6, 1990: 5, 1993: 3, 1995: 8, 1998: 5, 2001: 4, 2004: 2, 2006: 7,
            2009: 5, 2012: 4, 2014: 9, 2017: 6, 2020: 4, 2023: 2, 2025: 6, 2028: 5,
            2031: 3, 2033: 11, 2036: 6, 2039: 5, 2042: 2, 2044: 7, 2047: 5, 2050: 3,
            2052: 8, 2055: 6, 2058: 4, 2061: 3, 2063: 7, 2066: 5, 2069: 4, 2071: 8,
            2074: 6, 2077: 4, 2080: 3, 2082: 7, 2085: 5, 2088: 4, 2090: 8, 2093: 6,
            2096: 4, 2099: 2
        }

    def _is_leap_month(self, year: int, month: int) -> bool:
        return year in self.leap_months and self.leap_months[year] == month

    def _calculate_international_age(self, birthdate: datetime, now: datetime) -> int:
        age = now.year - birthdate.year
        if (now.month, now.day) < (birthdate.month, birthdate.day):
            age -= 1
        return age

    def _calculate_korean_age(self, birthdate: datetime, now: datetime) -> int:
        return now.year - birthdate.year + 1

    def calculate_saju(self, year: int, month: int, day: int, hour: int, minute: int = 0, is_male: bool = True, is_leap_month: bool = False) -> SajuChart:
        birth_datetime = datetime(year, month, day, hour, minute) - timedelta(minutes=32, seconds=1)
        base_date = datetime(1900, 1, 1)
        days_diff = (birth_datetime.date() - base_date.date()).days
        now = datetime.now()
        age = self._calculate_international_age(birth_datetime, now)
        korean_age = self._calculate_korean_age(birth_datetime, now)
        year_pillar = self._calculate_year_pillar(year)
        month_pillar = self._calculate_month_pillar_improved(year, month, day, is_leap_month)
        day_pillar = self._calculate_day_pillar(days_diff)
        hour_pillar = self._calculate_hour_pillar_improved(day_pillar.heavenly_stem, hour, minute)
        birth_info = {
            "year": year, "month": month, "day": day, 
            "hour": hour, "minute": minute,
            "is_male": is_male,
            "birth_datetime": birth_datetime,
            "is_leap_month": is_leap_month
        }
        return SajuChart(
            year_pillar, month_pillar, day_pillar, hour_pillar, 
            birth_info,
            age=age,
            korean_age=korean_age,
            current_datetime=now.strftime("%Y-%m-%d %H:%M:%S"),
            is_leap_month=is_leap_month
        )

    def _calculate_year_pillar(self, year: int) -> SajuPillar:
        base_year = 1984
        year_diff = year - base_year
        stem_index = year_diff % 10
        branch_index = year_diff % 12
        return SajuPillar(self.heavenly_stems[stem_index], self.earthly_branches[branch_index])

    def _calculate_month_pillar_improved(self, year: int, month: int, day: int, is_leap_month: bool = False) -> SajuPillar:
        month_branch_index = self._get_month_branch_by_solar_terms(year, month, day, is_leap_month)    
        year_stem_index = (year - 1984) % 10
        month_stem_base_table = [2, 4, 6, 8, 0, 2, 4, 6, 8, 0]
        month_stem_base = month_stem_base_table[year_stem_index]
        month_stem_index = (month_stem_base + ((month_branch_index + 12 - 2) % 12)) % 10
        month_stem = self.heavenly_stems[month_stem_index]
        return SajuPillar(month_stem, self.earthly_branches[month_branch_index])

    def _get_month_branch_by_solar_terms(self, year: int, month: int, day: int, is_leap_month: bool = False) -> int:
        if is_leap_month:
            month += 1
            if month > 12:
                month = 1
                year += 1
        solar_terms = [
            (2, 4, 2),   # 입춘: 2월 4일 → 인(2)
            (3, 6, 3),   # 경칩: 3월 6일 → 묘(3)
            (4, 5, 4),   # 청명: 4월 5일 → 진(4)
            (5, 6, 5),   # 입하: 5월 6일 → 사(5)
            (6, 6, 6),   # 망종: 6월 6일 → 오(6)
            (7, 7, 7),   # 소서: 7월 7일 → 미(7)
            (8, 8, 8),   # 입추: 8월 8일 → 신(8)
            (9, 8, 9),   # 백로: 9월 8일 → 유(9)
            (10, 8, 10), # 한로: 10월 8일 → 술(10)
            (11, 7, 11), # 입동: 11월 7일 → 해(11)
            (12, 7, 0),  # 대설: 12월 7일 → 자(0)
            (1, 6, 1),   # 소한: 1월 6일 → 축(1)
        ]
        m, d = month, day
        for i in range(len(solar_terms)):
            sm, sd, idx = solar_terms[i]
            if (m, d) < (sm, sd):
                return solar_terms[i-1][2] if i > 0 else solar_terms[-1][2]
        return solar_terms[-2][2]

    def _calculate_day_pillar(self, days_diff: int) -> SajuPillar:
        base_stem = (self.DAY_PILLAR_BASE_STEM - self.DAY_PILLAR_BASE_DAYS) % 10
        base_branch = (self.DAY_PILLAR_BASE_BRANCH - self.DAY_PILLAR_BASE_DAYS) % 12
        stem_index = (base_stem + days_diff) % 10
        branch_index = (base_branch + days_diff) % 12
        return SajuPillar(self.heavenly_stems[stem_index], self.earthly_branches[branch_index])

    def _calculate_hour_pillar_improved(self, day_stem: str, hour: int, minute: int = 0) -> SajuPillar:
        hour_branches = ["자", "축", "인", "묘", "진", "사", "오", "미", "신", "유", "술", "해"]
        total_minutes = hour * 60 + minute - 32
        if total_minutes >= 23 * 60 or total_minutes < 1 * 60: branch_idx = 0
        elif total_minutes < 3 * 60: branch_idx = 1
        elif total_minutes < 5 * 60: branch_idx = 2
        elif total_minutes < 7 * 60: branch_idx = 3
        elif total_minutes < 9 * 60: branch_idx = 4
        elif total_minutes < 11 * 60: branch_idx = 5
        elif total_minutes < 13 * 60: branch_idx = 6
        elif total_minutes < 15 * 60: branch_idx = 7
        elif total_minutes < 17 * 60: branch_idx = 8
        elif total_minutes < 19 * 60: branch_idx = 9
        elif total_minutes < 21 * 60: branch_idx = 10
        else: branch_idx = 11
        hour_branch = hour_branches[branch_idx]
        day_stem_idx = self.heavenly_stems.index(day_stem)
        if day_stem_idx in [0, 5]: hour_stem_base = 0
        elif day_stem_idx in [1, 6]: hour_stem_base = 2
        elif day_stem_idx in [2, 7]: hour_stem_base = 4
        elif day_stem_idx in [3, 8]: hour_stem_base = 6
        else: hour_stem_base = 8
        hour_stem_idx = (hour_stem_base + branch_idx) % 10
        return SajuPillar(self.heavenly_stems[hour_stem_idx], hour_branch)

    def analyze_ten_gods(self, saju_chart: SajuChart) -> Dict[str, List[str]]:
        day_master = saju_chart.get_day_master()
        day_master_element = self.five_elements[day_master]
        ten_gods = {"년주": [], "월주": [], "일주": [], "시주": []}
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
        # 8점 만점 방식: 4기둥(년,월,일,시)의 천간+지지 8글자만 카운트
        elements = {"목": 0, "화": 0, "토": 0, "금": 0, "수": 0}
        # 4기둥의 천간/지지 8글자 추출
        pillars = [
            saju_chart.year_pillar.heavenly_stem, saju_chart.year_pillar.earthly_branch,
            saju_chart.month_pillar.heavenly_stem, saju_chart.month_pillar.earthly_branch,
            saju_chart.day_pillar.heavenly_stem, saju_chart.day_pillar.earthly_branch,
            saju_chart.hour_pillar.heavenly_stem, saju_chart.hour_pillar.earthly_branch,
        ]
        # 오행 매핑
        wuxing_map = {
            '목': ['갑', '을', '인', '묘'],
            '화': ['병', '정', '사', '오'],
            '토': ['무', '기', '진', '술', '축', '미'],
            '금': ['경', '신', '신', '유'],
            '수': ['임', '계', '자', '해'],
        }
        char2wuxing = {}
        for k, v in wuxing_map.items():
            for ch in v:
                char2wuxing[ch] = k
        for ch in pillars:
            element = char2wuxing.get(ch)
            if element:
                elements[element] += 1
            else:
                raise ValueError(f"오행 매핑표에 없는 글자: {ch}")
        return elements

def format_saju_analysis(saju_chart: SajuChart, calculator: SajuCalculator) -> str:
    analysis = []
    analysis.append("=== 사주팔자 ===")
    analysis.append(f"년주(年柱): {saju_chart.year_pillar}")
    analysis.append(f"월주(月柱): {saju_chart.month_pillar}")
    analysis.append(f"일주(日柱): {saju_chart.day_pillar}")
    analysis.append(f"시주(時柱): {saju_chart.hour_pillar}")
    analysis.append(f"일간(日干): {saju_chart.get_day_master()}")
    analysis.append(f"현재 나이: {saju_chart.age}세 / 한국식 나이: {saju_chart.korean_age}세")
    analysis.append(f"기준 시점: {saju_chart.current_datetime}")
    if saju_chart.is_leap_month:
        analysis.append("⚠️ 윤달 출생자입니다 (월간 계산이 조정되었습니다)")
    analysis.append("")
    elements = calculator.get_element_strength(saju_chart)
    analysis.append("=== 오행 강약 (8점 만점) ===")
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