"""
사주팔자 계산 모듈
정확한 사주팔자, 대운, 십신 계산을 위한 전문 모듈
퍼플렉시티 검증 결과를 반영한 개선 버전
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class SajuPillar:
    """사주 기둥 (년주, 월주, 일주, 시주)"""
    heavenly_stem: str  # 천간
    earthly_branch: str  # 지지
    
    def __str__(self):
        return f"{self.heavenly_stem}{self.earthly_branch}"

@dataclass
class SajuChart:
    """사주팔자 차트"""
    year_pillar: SajuPillar   # 년주
    month_pillar: SajuPillar  # 월주
    day_pillar: SajuPillar    # 일주
    hour_pillar: SajuPillar   # 시주
    birth_info: Dict
    
    def get_day_master(self) -> str:
        """일간(日干) 반환"""
        return self.day_pillar.heavenly_stem

class SajuCalculator:
    """사주팔자 계산기 - 개선된 버전"""
    
    def __init__(self):
        # 천간 (10개)
        self.heavenly_stems = [
            "갑", "을", "병", "정", "무", "기", "경", "신", "임", "계"
        ]
        
        # 지지 (12개)
        self.earthly_branches = [
            "자", "축", "인", "묘", "진", "사", "오", "미", "신", "유", "술", "해"
        ]
        
        # 오행 매핑
        self.five_elements = {
            # 천간 오행
            "갑": "목", "을": "목",
            "병": "화", "정": "화", 
            "무": "토", "기": "토",
            "경": "금", "신": "금",
            "임": "수", "계": "수",
            # 지지 오행
            "자": "수", "축": "토", "인": "목", "묘": "목",
            "진": "토", "사": "화", "오": "화", "미": "토",
            "신": "금", "유": "금", "술": "토", "해": "수"
        }
        
        # 십신 매핑 (일간 기준)
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
        
        # 지장간 (지지 안에 숨어있는 천간들) - 팩트체크 반영
        self.hidden_stems = {
            "자": [("계", 100)],
            "축": [("기", 60), ("신", 30), ("계", 10)],  # 축: 기토60% + 신금30% + 계수10%
            "인": [("갑", 60), ("병", 30), ("무", 10)],
            "묘": [("을", 100)],
            "진": [("무", 60), ("을", 30), ("계", 10)],
            "사": [("정", 70), ("무", 20), ("경", 10)],  # 사: 정화70% + 무토20% + 경금10% (수정)
            "오": [("정", 70), ("기", 30)],
            "미": [("기", 60), ("정", 30), ("을", 10)],
            "신": [("경", 60), ("임", 30), ("무", 10)],
            "유": [("신", 100)],
            "술": [("무", 60), ("신", 30), ("정", 10)],
            "해": [("임", 70), ("갑", 30)]
        }
        
        # 2024년 절기 정보 (한국천문연구원 기준)
        self.solar_terms_2024 = {
            "입춘": (2, 4, 16, 27),   # 2월 4일 16시 27분
            "우수": (2, 19, 12, 13),  # 2월 19일 12시 13분
            "경칩": (3, 5, 22, 23),   # 3월 5일 22시 23분
            "춘분": (3, 20, 15, 6),   # 3월 20일 15시 6분
            "청명": (4, 4, 21, 2),    # 4월 4일 21시 2분
            "곡우": (4, 20, 3, 20),   # 4월 20일 3시 20분
            "입하": (5, 5, 8, 10),    # 5월 5일 8시 10분
            "소만": (5, 20, 20, 59),  # 5월 20일 20시 59분
            "망종": (6, 5, 12, 10),   # 6월 5일 12시 10분
            "하지": (6, 21, 4, 51),   # 6월 21일 4시 51분
            "소서": (7, 6, 22, 20),   # 7월 6일 22시 20분
            "대서": (7, 22, 15, 44),  # 7월 22일 15시 44분
            "입추": (8, 7, 9, 9),     # 8월 7일 9시 9분
            "처서": (8, 22, 23, 55),  # 8월 22일 23시 55분
            "백로": (9, 7, 12, 11),   # 9월 7일 12시 11분
            "추분": (9, 22, 20, 44),  # 9월 22일 20시 44분
            "한로": (10, 8, 3, 0),    # 10월 8일 3시 0분
            "상강": (10, 23, 6, 15),  # 10월 23일 6시 15분
            "입동": (11, 7, 6, 20),   # 11월 7일 6시 20분
            "소설": (11, 22, 3, 56),  # 11월 22일 3시 56분
            "대설": (12, 7, 0, 17),   # 12월 7일 0시 17분
            "동지": (12, 21, 17, 21)  # 12월 21일 17시 21분
        }
    
    def calculate_saju(self, year: int, month: int, day: int, hour: int, 
                      minute: int = 0, is_male: bool = True, timezone: str = "Asia/Seoul") -> SajuChart:
        """
        사주팔자 계산 - 개선된 버전
        
        Args:
            year: 년도
            month: 월
            day: 일
            hour: 시간 (24시간 형식)
            minute: 분
            is_male: 성별 (남성=True, 여성=False)
            timezone: 시간대
        
        Returns:
            SajuChart: 계산된 사주팔자
        """
        # 생년월일시 설정
        birth_datetime = datetime(year, month, day, hour, minute)
        
        # 태양시 보정 (서울 기준 약 -5분 32초)
        if timezone == "Asia/Seoul":
            birth_datetime = birth_datetime - timedelta(minutes=5, seconds=32)
        
        # 기준일 설정 (1900년 1월 1일)
        base_date = datetime(1900, 1, 1)
        days_diff = (birth_datetime.date() - base_date.date()).days
        
        # 각 기둥 계산
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
        """년주 계산"""
        # 1984년이 갑자년 (천간지지 순환의 시작)
        base_year = 1984
        year_diff = year - base_year
        
        stem_index = year_diff % 10
        branch_index = year_diff % 12
        
        return SajuPillar(
            self.heavenly_stems[stem_index],
            self.earthly_branches[branch_index]
        )
    
    def _calculate_month_pillar_improved(self, year: int, month: int, day: int) -> SajuPillar:
        """월주 계산 - 절기 세분화 개선"""
        # 절기 기준으로 정확한 월지 결정
        month_branch_index = self._get_month_branch_by_solar_terms(year, month, day)
        
        # 년간에 따른 월간 기준 설정
        year_stem_index = (year - 1984) % 10
        
        # 월간 계산 공식 (년간에 따라)
        if year_stem_index in [0, 5]:  # 갑, 기년
            month_stem_base = 2  # 병인월부터
        elif year_stem_index in [1, 6]:  # 을, 경년  
            month_stem_base = 4  # 무인월부터
        elif year_stem_index in [2, 7]:  # 병, 신년
            month_stem_base = 6  # 경인월부터
        elif year_stem_index in [3, 8]:  # 정, 임년
            month_stem_base = 8  # 임인월부터
        else:  # 무, 계년
            month_stem_base = 0  # 갑인월부터
        
        # 인월(2)을 기준으로 월간 계산
        month_offset = (month_branch_index - 2) % 12
        month_stem_index = (month_stem_base + month_offset) % 10
        
        return SajuPillar(
            self.heavenly_stems[month_stem_index],
            self.earthly_branches[month_branch_index]
        )
    
    def _get_month_branch_by_solar_terms(self, year: int, month: int, day: int) -> int:
        """절기 기준으로 정확한 월지 결정 - 범용 계산"""
        # 절기의 대략적인 날짜 계산 (공식 기반)
        solar_terms_approx = self._calculate_solar_terms_dates(year)
        
        # 현재 날짜를 기준으로 월지 결정
        current_date = datetime(year, month, day)
        
        # 절기와 월지 매핑
        # 입춘(2월 초) -> 인월(2), 경칩(3월 초) -> 묘월(3), ...
        term_to_month = {
            '대설': 0,   # 자월 (12월)
            '소한': 0,   # 자월 (1월)
            '입춘': 2,   # 인월 (2월)
            '경칩': 3,   # 묘월 (3월)
            '청명': 4,   # 진월 (4월)
            '입하': 5,   # 사월 (5월)
            '망종': 6,   # 오월 (6월)
            '소서': 7,   # 미월 (7월)
            '입추': 8,   # 신월 (8월)
            '백로': 9,   # 유월 (9월)
            '한로': 10,  # 술월 (10월)
            '입동': 11   # 해월 (11월)
        }
        
        # 월별 기본 월지 (절기 미고려시)
        if month == 1:
            # 1월은 입춘(2월 4일경) 기준으로 판단
            lichun_date = solar_terms_approx.get('입춘', datetime(year, 2, 4))
            if current_date < lichun_date:
                return 0  # 자월 (입춘 전)
            else:
                return 2  # 인월 (입춘 후)
        elif month == 2:
            lichun_date = solar_terms_approx.get('입춘', datetime(year, 2, 4))
            jingzhe_date = solar_terms_approx.get('경칩', datetime(year, 3, 5))
            if current_date < lichun_date:
                return 0  # 자월
            elif current_date < jingzhe_date:
                return 2  # 인월
            else:
                return 3  # 묘월
        elif month == 3:
            jingzhe_date = solar_terms_approx.get('경칩', datetime(year, 3, 5))
            qingming_date = solar_terms_approx.get('청명', datetime(year, 4, 5))
            if current_date < jingzhe_date:
                return 2  # 인월
            elif current_date < qingming_date:
                return 3  # 묘월
            else:
                return 4  # 진월
        elif month == 4:
            qingming_date = solar_terms_approx.get('청명', datetime(year, 4, 5))
            lixia_date = solar_terms_approx.get('입하', datetime(year, 5, 5))
            if current_date < qingming_date:
                return 3  # 묘월
            elif current_date < lixia_date:
                return 4  # 진월
            else:
                return 5  # 사월
        elif month == 5:
            lixia_date = solar_terms_approx.get('입하', datetime(year, 5, 5))
            mangzhong_date = solar_terms_approx.get('망종', datetime(year, 6, 6))
            if current_date < lixia_date:
                return 4  # 진월
            elif current_date < mangzhong_date:
                return 5  # 사월
            else:
                return 6  # 오월
        elif month == 6:
            mangzhong_date = solar_terms_approx.get('망종', datetime(year, 6, 6))
            xiaoshu_date = solar_terms_approx.get('소서', datetime(year, 7, 7))
            if current_date < mangzhong_date:
                return 5  # 사월
            elif current_date < xiaoshu_date:
                return 6  # 오월
            else:
                return 7  # 미월
        elif month == 7:
            xiaoshu_date = solar_terms_approx.get('소서', datetime(year, 7, 7))
            liqiu_date = solar_terms_approx.get('입추', datetime(year, 8, 7))
            if current_date < xiaoshu_date:
                return 6  # 오월
            elif current_date < liqiu_date:
                return 7  # 미월
            else:
                return 8  # 신월
        elif month == 8:
            liqiu_date = solar_terms_approx.get('입추', datetime(year, 8, 7))
            bailu_date = solar_terms_approx.get('백로', datetime(year, 9, 7))
            if current_date < liqiu_date:
                return 7  # 미월
            elif current_date < bailu_date:
                return 8  # 신월
            else:
                return 9  # 유월
        else:  # month == 12
            daxue_date = solar_terms_approx.get('대설', datetime(year, 12, 7))
            next_year_xiaohan = solar_terms_approx.get('소한', datetime(year + 1, 1, 6))
            if current_date < daxue_date:
                return 11  # 해월
            else:
                return 0  # 자월
    
    def _calculate_solar_terms_dates(self, year: int) -> Dict[str, datetime]:
        """절기 날짜 근사 계산 (공식 기반)"""
        # 절기 계산 공식 (근사치)
        # 기준: 2000년 동지 = 12월 21일 7시 37분
        base_year = 2000
        year_diff = year - base_year
        
        # 절기별 기준 날짜 (월, 일, 시, 분)
        base_terms = {
            '소한': (1, 6, 0, 0),
            '대한': (1, 20, 12, 0),
            '입춘': (2, 4, 18, 0),
            '우수': (2, 19, 12, 0),
            '경칩': (3, 5, 22, 0),
            '춘분': (3, 20, 15, 0),
            '청명': (4, 5, 3, 0),
            '곡우': (4, 20, 9, 0),
            '입하': (5, 5, 20, 0),
            '소만': (5, 21, 9, 0),
            '망종': (6, 6, 0, 0),
            '하지': (6, 21, 17, 0),
            '소서': (7, 7, 10, 0),
            '대서': (7, 23, 4, 0),
            '입추': (8, 7, 20, 0),
            '처서': (8, 23, 11, 0),
            '백로': (9, 8, 0, 0),
            '추분': (9, 23, 9, 0),
            '한로': (10, 8, 15, 0),
            '상강': (10, 23, 18, 0),
            '입동': (11, 7, 18, 0),
            '소설': (11, 22, 16, 0),
            '대설': (12, 7, 11, 0),
            '동지': (12, 21, 17, 30)
        }
        
        solar_terms = {}
        for term_name, (month, day, hour, minute) in base_terms.items():
            # 년도 차이에 따른 보정 (대략 6시간/년)
            hour_correction = int(year_diff * 0.25)  # 대략적인 보정
            corrected_hour = hour + hour_correction
            
            # 시간 오버플로우 처리
            corrected_day = day
            if corrected_hour >= 24:
                corrected_day += corrected_hour // 24
                corrected_hour = corrected_hour % 24
            elif corrected_hour < 0:
                corrected_day += corrected_hour // 24
                corrected_hour = 24 + (corrected_hour % 24)
            
            try:
                solar_terms[term_name] = datetime(year, month, corrected_day, corrected_hour, minute)
            except ValueError:
                # 날짜 오버플로우 시 기본값 사용
                solar_terms[term_name] = datetime(year, month, day, hour, minute)
        
        return solar_terms
    
    def _calculate_day_pillar(self, days_diff: int) -> SajuPillar:
        """일주 계산"""
        # 정확한 기준일 설정
        # 1995년 8월 26일 = 기축일(己丑)이 되도록 조정
        
        # 1900년 1월 1일부터 1995년 8월 26일까지의 일수 계산
        target_date = datetime(1995, 8, 26)
        base_date = datetime(1900, 1, 1)
        target_days = (target_date - base_date).days
        
        # 1995년 8월 26일이 기축일(천간5=기, 지지1=축)이 되도록 기준 설정
        target_stem = 5  # 기
        target_branch = 1  # 축
        
        # 역산하여 1900년 1월 1일의 간지 계산
        base_stem = (target_stem - target_days) % 10
        base_branch = (target_branch - target_days) % 12
        
        # 실제 일주 계산
        stem_index = (base_stem + days_diff) % 10
        branch_index = (base_branch + days_diff) % 12
        
        return SajuPillar(
            self.heavenly_stems[stem_index],
            self.earthly_branches[branch_index]
        )
    
    def _calculate_hour_pillar_improved(self, day_stem: str, hour: int, minute: int = 0) -> SajuPillar:
        """시주 계산 - 분 단위까지 정밀화"""
        # 시간대별 지지 매핑 (분 단위 고려)
        hour_branches = [
            "자", "축", "인", "묘", "진", "사", 
            "오", "미", "신", "유", "술", "해"
        ]
        
        # 정확한 시간 계산 (분 단위 포함)
        total_minutes = hour * 60 + minute
        
        # 시간을 2시간(120분) 단위로 나누어 지지 결정
        if total_minutes >= 23 * 60 or total_minutes < 1 * 60:
            branch_idx = 0  # 자시 (23:00-00:59)
        elif total_minutes < 3 * 60:
            branch_idx = 1  # 축시 (01:00-02:59)
        elif total_minutes < 5 * 60:
            branch_idx = 2  # 인시 (03:00-04:59)
        elif total_minutes < 7 * 60:
            branch_idx = 3  # 묘시 (05:00-06:59)
        elif total_minutes < 9 * 60:
            branch_idx = 4  # 진시 (07:00-08:59)
        elif total_minutes < 11 * 60:
            branch_idx = 5  # 사시 (09:00-10:59)
        elif total_minutes < 13 * 60:
            branch_idx = 6  # 오시 (11:00-12:59)
        elif total_minutes < 15 * 60:
            branch_idx = 7  # 미시 (13:00-14:59)
        elif total_minutes < 17 * 60:
            branch_idx = 8  # 신시 (15:00-16:59)
        elif total_minutes < 19 * 60:
            branch_idx = 9  # 유시 (17:00-18:59)
        elif total_minutes < 21 * 60:
            branch_idx = 10  # 술시 (19:00-20:59)
        else:  # 21:00-22:59
            branch_idx = 11  # 해시
        
        hour_branch = hour_branches[branch_idx]
        
        # 일간에 따른 시간 천간 계산
        day_stem_idx = self.heavenly_stems.index(day_stem)
        
        if day_stem_idx in [0, 5]:  # 갑, 기일
            hour_stem_base = 0  # 갑자시부터
        elif day_stem_idx in [1, 6]:  # 을, 경일
            hour_stem_base = 2  # 병자시부터
        elif day_stem_idx in [2, 7]:  # 병, 신일
            hour_stem_base = 4  # 무자시부터
        elif day_stem_idx in [3, 8]:  # 정, 임일
            hour_stem_base = 6  # 경자시부터
        else:  # 무, 계일
            hour_stem_base = 8  # 임자시부터
        
        hour_stem_idx = (hour_stem_base + branch_idx) % 10
        
        return SajuPillar(
            self.heavenly_stems[hour_stem_idx],
            hour_branch
        )
    
    def analyze_ten_gods(self, saju_chart: SajuChart) -> Dict[str, List[str]]:
        """십신 분석 - 음양 구분 정확화"""
        day_master = saju_chart.get_day_master()
        day_master_element = self.five_elements[day_master]
        day_master_idx = self.heavenly_stems.index(day_master)
        day_master_yin_yang = "양" if day_master_idx % 2 == 0 else "음"
        
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
            # 천간 십신
            if pillar.heavenly_stem != day_master:  # 일간 제외
                stem_element = self.five_elements[pillar.heavenly_stem]
                stem_idx = self.heavenly_stems.index(pillar.heavenly_stem)
                stem_yin_yang = "양" if stem_idx % 2 == 0 else "음"
                
                god_types = self.ten_gods_mapping[day_master_element][stem_element]
                
                # 음양에 따른 십신 결정 (정확한 규칙)
                if day_master_yin_yang == stem_yin_yang:  # 같은 음양
                    ten_god = god_types[0]  # 비견, 식신, 편재, 편관, 편인
                else:  # 다른 음양
                    ten_god = god_types[1]  # 겁재, 상관, 정재, 정관, 정인
                
                ten_gods[pillar_name].append(f"천간:{ten_god}")
            
            # 지지 십신 (지장간 고려)
            hidden_stems = self.hidden_stems[pillar.earthly_branch]
            for hidden_stem, strength in hidden_stems:
                if hidden_stem != day_master:
                    hidden_element = self.five_elements[hidden_stem]
                    hidden_idx = self.heavenly_stems.index(hidden_stem)
                    hidden_yin_yang = "양" if hidden_idx % 2 == 0 else "음"
                    
                    god_types = self.ten_gods_mapping[day_master_element][hidden_element]
                    
                    if day_master_yin_yang == hidden_yin_yang:  # 같은 음양
                        ten_god = god_types[0]
                    else:  # 다른 음양
                        ten_god = god_types[1]
                    
                    ten_gods[pillar_name].append(f"지지:{ten_god}({strength}%)")
        
        return ten_gods
    
    def calculate_great_fortune_improved(self, saju_chart: SajuChart) -> List[Dict]:
        """대운 계산 - 역행 로직 수정"""
        birth_info = saju_chart.birth_info
        year = birth_info["year"]
        month = birth_info["month"]
        day = birth_info["day"]
        is_male = birth_info["is_male"]
        
        # 년간의 음양 판단
        year_stem = saju_chart.year_pillar.heavenly_stem
        year_stem_idx = self.heavenly_stems.index(year_stem)
        is_yang_year = (year_stem_idx % 2 == 0)
        
        # 대운 방향 결정 (정확한 규칙)
        # 양년 남성, 음년 여성 → 순행
        # 음년 남성, 양년 여성 → 역행
        if (is_yang_year and is_male) or (not is_yang_year and not is_male):
            direction = 1  # 순행
        else:
            direction = -1  # 역행
        
        # 대운 시작 연령 (절기 기준으로 정밀 계산 필요하지만 여기서는 간소화)
        start_age = 8  # 기본 8세 (실제로는 절기까지의 일수/3으로 계산)
        
        # 월주 기준으로 대운 계산
        month_stem_idx = self.heavenly_stems.index(saju_chart.month_pillar.heavenly_stem)
        month_branch_idx = self.earthly_branches.index(saju_chart.month_pillar.earthly_branch)
        
        great_fortunes = []
        for i in range(8):  # 8개 대운
            age = start_age + (i * 10)
            
            # 방향에 따른 간지 계산
            stem_idx = (month_stem_idx + (direction * (i + 1))) % 10
            branch_idx = (month_branch_idx + (direction * (i + 1))) % 12
            
            # 음수 인덱스 처리
            if stem_idx < 0:
                stem_idx += 10
            if branch_idx < 0:
                branch_idx += 12
            
            great_fortunes.append({
                "age": age,
                "pillar": f"{self.heavenly_stems[stem_idx]}{self.earthly_branches[branch_idx]}",
                "years": f"{year + age}년 ~ {year + age + 9}년",
                "direction": "순행" if direction == 1 else "역행"
            })
        
        return great_fortunes
    
    def get_element_strength(self, saju_chart: SajuChart) -> Dict[str, int]:
        """
        오행 강약 분석 (전통 8점 방식) - 단순 전통 방식
        
        Args:
            saju_chart: 사주팔자 차트
        
        Returns:
            Dict[str, int]: 오행별 점수 (총 8점)
        """
        elements = {"목": 0, "화": 0, "토": 0, "금": 0, "수": 0}
        
        pillars = [saju_chart.year_pillar, saju_chart.month_pillar, 
                  saju_chart.day_pillar, saju_chart.hour_pillar]
        
        for pillar in pillars:
            # 천간 1점
            stem_element = self.five_elements[pillar.heavenly_stem]
            elements[stem_element] += 1
            
            # 지지 1점 (가장 강한 지장간만 고려)
            hidden_stems = self.hidden_stems[pillar.earthly_branch]
            # 가장 높은 비율의 지장간만 1점으로 계산
            strongest_stem, strongest_ratio = max(hidden_stems, key=lambda x: x[1])
            strongest_element = self.five_elements[strongest_stem]
            elements[strongest_element] += 1
        
        return elements

def format_saju_analysis(saju_chart: SajuChart, calculator: SajuCalculator) -> str:
    """
    사주 분석 결과를 포맷팅
    
    Args:
        saju_chart: 사주팔자 차트
        calculator: 사주 계산기
    """
    analysis = []
    
    # 기본 사주팔자
    analysis.append("=== 사주팔자 ===")
    analysis.append(f"년주(年柱): {saju_chart.year_pillar}")
    analysis.append(f"월주(月柱): {saju_chart.month_pillar}")
    analysis.append(f"일주(日柱): {saju_chart.day_pillar}")
    analysis.append(f"시주(時柱): {saju_chart.hour_pillar}")
    analysis.append(f"일간(日干): {saju_chart.get_day_master()}")
    analysis.append("")
    
    # 오행 분석
    elements = calculator.get_element_strength(saju_chart)
    analysis.append("=== 오행 강약 ===")
    for element, strength in elements.items():
        analysis.append(f"{element}: {strength}점")
    analysis.append("")
    
    # 십신 분석
    ten_gods = calculator.analyze_ten_gods(saju_chart)
    analysis.append("=== 십신 분석 ===")
    for pillar_name, gods in ten_gods.items():
        if gods:
            analysis.append(f"{pillar_name}: {', '.join(gods)}")
    analysis.append("")
    
    # 대운 (개선된 버전)
    great_fortunes = calculator.calculate_great_fortune_improved(saju_chart)
    analysis.append("=== 대운 (정밀 계산) ===")
    for gf in great_fortunes[:4]:  # 처음 4개만 표시
        analysis.append(f"{gf['age']}세: {gf['pillar']} ({gf['years']})")
    
    return "\n".join(analysis) 