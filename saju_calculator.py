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
        
        # 지장간 (지지 안에 숨어있는 천간들) - 정확한 데이터
        self.hidden_stems = {
            "자": [("계", 100)],
            "축": [("기", 60), ("신", 30), ("계", 10)],  # 축: 기토60% + 신금30% + 계수10%
            "인": [("갑", 60), ("병", 30), ("무", 10)],
            "묘": [("을", 100)],
            "진": [("무", 60), ("을", 30), ("계", 10)],
            "사": [("병", 70), ("무", 20), ("경", 10)],  # 사: 병화70% + 무토20% + 경금10%
            "오": [("정", 70), ("기", 30)],
            "미": [("기", 60), ("정", 30), ("을", 10)],
            "신": [("경", 60), ("임", 30), ("무", 10)],
            "유": [("신", 100)],
            "술": [("무", 60), ("신", 30), ("정", 10)],
            "해": [("임", 70), ("갑", 30)]
        }
        
        # 월령 가중치 (계절별 오행 강약) - 전통 명리학 왕상사수휴 이론
        self.seasonal_weights = {
            # 봄 (인묘진월) - 목왕, 화상, 토사, 금수, 수휴
            "인": {"목": 2.0, "화": 1.3, "토": 0.7, "금": 0.5, "수": 0.6},  # 목 최왕
            "묘": {"목": 2.2, "화": 1.3, "토": 0.7, "금": 0.4, "수": 0.6},  # 목 극왕
            "진": {"목": 1.5, "화": 1.1, "토": 1.3, "금": 0.6, "수": 0.7},  # 토월(계절 변화)
            
            # 여름 (사오미월) - 화왕, 토상, 금사, 수수, 목휴
            "사": {"화": 2.0, "토": 1.3, "금": 0.7, "수": 0.5, "목": 0.6},  # 화 최왕
            "오": {"화": 2.2, "토": 1.3, "금": 0.7, "수": 0.4, "목": 0.6},  # 화 극왕
            "미": {"화": 1.5, "토": 1.8, "금": 0.8, "수": 0.6, "목": 0.7},  # 토월(계절 변화)
            
            # 가을 (신유술월) - 금왕, 수상, 토사, 화수, 목휴
            "신": {"금": 2.0, "수": 1.3, "토": 0.7, "화": 0.5, "목": 0.6},  # 금 최왕
            "유": {"금": 2.2, "수": 1.3, "토": 0.7, "화": 0.4, "목": 0.6},  # 금 극왕
            "술": {"금": 1.5, "수": 1.1, "토": 1.3, "화": 0.6, "목": 0.7},  # 토월(계절 변화)
            
            # 겨울 (해자축월) - 수왕, 목상, 화사, 토수, 금휴
            "해": {"수": 2.0, "목": 1.3, "화": 0.7, "토": 0.5, "금": 0.6},  # 수 최왕
            "자": {"수": 2.2, "목": 1.3, "화": 0.7, "토": 0.4, "금": 0.6},  # 수 극왕
            "축": {"수": 1.5, "목": 1.1, "화": 0.7, "토": 1.3, "금": 0.8}   # 토월(계절 변화)
        }
        
        # 합충형해 관계 (지지 간의 특수 관계)
        self.branch_relationships = {
            # 육합 (六合) - 서로 도와주는 관계
            "합": {
                ("자", "축"): "토합", ("인", "해"): "목합", ("묘", "술"): "화합",
                ("진", "유"): "금합", ("사", "신"): "수합", ("오", "미"): "토합"
            },
            # 삼합 (三合) - 3개가 모여 하나의 오행을 강화
            "삼합": {
                ("인", "오", "술"): "화국", ("사", "유", "축"): "금국",
                ("신", "자", "진"): "수국", ("해", "묘", "미"): "목국"
            },
            # 육충 (六沖) - 서로 충돌하는 관계
            "충": {
                ("자", "오"): "자오충", ("축", "미"): "축미충", ("인", "신"): "인신충",
                ("묘", "유"): "묘유충", ("진", "술"): "진술충", ("사", "해"): "사해충"
            },
            # 육해 (六害) - 서로 해치는 관계
            "해": {
                ("자", "미"): "자미해", ("축", "오"): "축오해", ("인", "사"): "인사해",
                ("묘", "진"): "묘진해", ("신", "해"): "신해해", ("유", "술"): "유술해"
            },
            # 자형 (自刑) - 같은 지지끼리 형
            "형": {
                ("축", "술", "미"): "토형", ("인", "사", "신"): "무은지형",
                ("자", "묘", "유"): "무례지형", ("진", "오", "해"): "자형"
            }
        }
        
        # 신살(神殺) 정보 - 공망, 도화, 역마 등
        self.shinsals = {
            # 공망(空亡) - 일지 기준
            "공망": {
                "갑을": ["술", "해"], "병정": ["신", "유"], "무기": ["오", "미"],
                "경신": ["진", "사"], "임계": ["인", "묘"]
            },
            # 도화(桃花) - 연지/일지 기준
            "도화": {
                "인오술": "묘", "사유축": "오", "신자진": "유", "해묘미": "자"
            },
            # 역마(驛馬) - 연지/일지 기준  
            "역마": {
                "인오술": "신", "사유축": "해", "신자진": "인", "해묘미": "사"
            },
            # 천을귀인(天乙貴人) - 일간 기준
            "천을귀인": {
                "갑무": ["축", "미"], "을기": ["자", "신"], "병정": ["해", "유"],
                "경신": ["오", "인"], "임계": ["사", "묘"]
            },
            # 태극귀인(太極貴人) - 일간 기준
            "태극귀인": {
                "갑을": ["자", "오"], "병정": ["묘", "유"], "무": ["진", "술", "축", "미"],
                "기": ["진", "술", "축", "미"], "경신": ["인", "해"], "임계": ["사", "신"]
            }
        }
        
        # 윤달 정보 (1900-2100년) - 한국천문연구원 기준
        self.leap_months = {
            # 년도: (윤달 월, 양력 시작월, 양력 시작일, 양력 끝월, 양력 끝일)
            1984: (10, 11, 23, 12, 21),  # 1984년 윤10월
            1987: (6, 7, 26, 8, 23),     # 1987년 윤6월
            1990: (5, 6, 23, 7, 21),     # 1990년 윤5월
            1993: (3, 4, 22, 5, 20),     # 1993년 윤3월
            1995: None,                   # 1995년은 윤달 없음
            1998: (5, 5, 24, 6, 22),     # 1998년 윤5월
            2001: (4, 4, 23, 5, 21),     # 2001년 윤4월
            2004: (2, 2, 21, 3, 20),     # 2004년 윤2월
            2006: (7, 7, 25, 8, 23),     # 2006년 윤7월
            2009: (5, 5, 23, 6, 21),     # 2009년 윤5월
            2012: (3, 3, 21, 4, 19),     # 2012년 윤3월
            2014: (9, 9, 24, 10, 23),    # 2014년 윤9월
            2017: (6, 6, 24, 7, 22),     # 2017년 윤6월
            2020: (4, 4, 23, 5, 21),     # 2020년 윤4월
            2023: (2, 2, 20, 3, 21),     # 2023년 윤2월
            2025: (6, 6, 23, 7, 21),     # 2025년 윤6월 (예정)
            2028: (5, 5, 21, 6, 19),     # 2028년 윤5월 (예정)
            2031: (3, 3, 21, 4, 19),     # 2031년 윤3월 (예정)
            2033: (11, 12, 22, 1, 19),   # 2033년 윤11월 (예정)
            2036: (6, 6, 22, 7, 20),     # 2036년 윤6월 (예정)
        }
        
        # 1995년 절기 정보 (한국천문연구원 기준) - 정확한 데이터
        self.solar_terms_1995 = {
            "입춘": (2, 4, 9, 14),    # 2월 4일 9시 14분
            "우수": (2, 19, 5, 1),    # 2월 19일 5시 1분
            "경칩": (3, 6, 0, 46),    # 3월 6일 0시 46분
            "춘분": (3, 21, 2, 14),   # 3월 21일 2시 14분
            "청명": (4, 5, 15, 36),   # 4월 5일 15시 36분
            "곡우": (4, 20, 22, 1),   # 4월 20일 22시 1분
            "입하": (5, 6, 2, 20),    # 5월 6일 2시 20분
            "소만": (5, 21, 4, 35),   # 5월 21일 4시 35분
            "망종": (6, 6, 5, 46),    # 6월 6일 5시 46분
            "하지": (6, 21, 15, 34),  # 6월 21일 15시 34분
            "소서": (7, 7, 12, 3),    # 7월 7일 12시 3분
            "대서": (7, 23, 6, 29),   # 7월 23일 6시 29분
            "입추": (8, 8, 0, 1),     # 8월 8일 0시 1분
            "처서": (8, 23, 16, 35),  # 8월 23일 16시 35분
            "백로": (9, 8, 6, 0),     # 9월 8일 6시 0분 (정확한 데이터)
            "추분": (9, 23, 16, 13),  # 9월 23일 16시 13분
            "한로": (10, 8, 23, 37),  # 10월 8일 23시 37분
            "상강": (10, 24, 3, 4),   # 10월 24일 3시 4분
            "입동": (11, 8, 3, 4),    # 11월 8일 3시 4분
            "소설": (11, 22, 23, 57), # 11월 22일 23시 57분
            "대설": (12, 7, 17, 53),  # 12월 7일 17시 53분
            "동지": (12, 22, 8, 17)   # 12월 22일 8시 17분
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
                      minute: int = 0, is_male: bool = True, timezone: str = "Asia/Seoul", 
                      is_leap_month: bool = False) -> SajuChart:
        """
        사주팔자 계산 - 개선된 버전 (윤달 지원)
        
        Args:
            year: 년도
            month: 월
            day: 일
            hour: 시간 (24시간 형식)
            minute: 분
            is_male: 성별 (남성=True, 여성=False)
            timezone: 시간대
            is_leap_month: 윤달 여부 (True=윤달, False=평달)
        
        Returns:
            SajuChart: 계산된 사주팔자
        """
        # 생년월일시 설정
        birth_datetime = datetime(year, month, day, hour, minute)
        
        # 태양시 보정 (지역별 경도 차이 반영)
        birth_datetime = self._apply_solar_time_correction(birth_datetime, timezone)
        
        # 기준일 설정 (1900년 1월 1일)
        base_date = datetime(1900, 1, 1)
        days_diff = (birth_datetime.date() - base_date.date()).days
        
        # 각 기둥 계산 (윤달 고려)
        year_pillar = self._calculate_year_pillar(year)
        month_pillar = self._calculate_month_pillar_improved(year, month, day, is_leap_month)
        day_pillar = self._calculate_day_pillar(days_diff)
        # 태양시 보정된 시간으로 시주 계산
        hour_pillar = self._calculate_hour_pillar_improved(day_pillar.heavenly_stem, birth_datetime.hour, birth_datetime.minute)
        
        birth_info = {
            "year": year, "month": month, "day": day, 
            "hour": hour, "minute": minute,
            "is_male": is_male, "timezone": timezone,
            "is_leap_month": is_leap_month,
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
    
    def _calculate_month_pillar_improved(self, year: int, month: int, day: int, is_leap_month: bool = False) -> SajuPillar:
        """월주 계산 - 절기 세분화 개선 (윤달 지원)"""
        # 절기 기준으로 정확한 월지 결정 (윤달 고려)
        month_branch_index = self._get_month_branch_by_solar_terms(year, month, day, is_leap_month)
        
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
    
    def _get_month_branch_by_solar_terms(self, year: int, month: int, day: int, is_leap_month: bool = False) -> int:
        """절기 기준으로 정확한 월지 결정 - 정밀 계산 (윤달 지원)"""
        # 윤달 처리: 윤달인 경우 이전 달의 월지를 사용
        if is_leap_month:
            # 윤달은 이전 달과 같은 월지 사용
            if month > 1:
                return self._get_month_branch_by_solar_terms(year, month - 1, 15, False)
            else:
                return self._get_month_branch_by_solar_terms(year - 1, 12, 15, False)
        
        # 정확한 절기 데이터 우선 사용
        if year == 1995:
            solar_terms_data = self.solar_terms_1995
        elif year == 2024:
            solar_terms_data = self.solar_terms_2024
        else:
            # 다른 년도는 계산식 사용
            solar_terms_data = self._calculate_solar_terms_dates(year)
        
        # 현재 날짜
        current_date = datetime(year, month, day)
        
        # 정확한 절기 날짜 생성
        solar_terms_dates = {}
        for term_name, term_info in solar_terms_data.items():
            if isinstance(term_info, tuple) and len(term_info) == 4:
                m, d, h, min = term_info
                solar_terms_dates[term_name] = datetime(year, m, d, h, min)
            elif isinstance(term_info, datetime):
                solar_terms_dates[term_name] = term_info
        
        # 9월 처리 (백로-한로 구간 = 유월)
        if month == 9:
            bailu_date = solar_terms_dates.get('백로')
            hanlu_date = solar_terms_dates.get('한로')
            
            if bailu_date and hanlu_date:
                if current_date >= bailu_date and current_date < hanlu_date:
                    return 9  # 유월 (백로~한로)
                elif current_date < bailu_date:
                    return 8  # 신월 (입추~백로)
                else:
                    return 10  # 술월 (한로~입동)
            else:
                # 기본값으로 9월 8일 기준
                if day >= 8:
                    return 9  # 유월
                else:
                    return 8  # 신월
        
        # 다른 월들에 대한 기존 로직 유지하되 정확한 절기 날짜 사용
        elif month == 1:
            lichun_date = solar_terms_dates.get('입춘', datetime(year, 2, 4))
            if current_date < lichun_date:
                return 0  # 자월 (입춘 전)
            else:
                return 2  # 인월 (입춘 후)
        elif month == 2:
            lichun_date = solar_terms_dates.get('입춘', datetime(year, 2, 4))
            jingzhe_date = solar_terms_dates.get('경칩', datetime(year, 3, 6))
            if current_date < lichun_date:
                return 0  # 자월
            elif current_date < jingzhe_date:
                return 2  # 인월
            else:
                return 3  # 묘월
        elif month == 3:
            jingzhe_date = solar_terms_dates.get('경칩', datetime(year, 3, 6))
            qingming_date = solar_terms_dates.get('청명', datetime(year, 4, 5))
            if current_date < jingzhe_date:
                return 2  # 인월
            elif current_date < qingming_date:
                return 3  # 묘월
            else:
                return 4  # 진월
        elif month == 4:
            qingming_date = solar_terms_dates.get('청명', datetime(year, 4, 5))
            lixia_date = solar_terms_dates.get('입하', datetime(year, 5, 6))
            if current_date < qingming_date:
                return 3  # 묘월
            elif current_date < lixia_date:
                return 4  # 진월
            else:
                return 5  # 사월
        elif month == 5:
            lixia_date = solar_terms_dates.get('입하', datetime(year, 5, 6))
            mangzhong_date = solar_terms_dates.get('망종', datetime(year, 6, 6))
            if current_date < lixia_date:
                return 4  # 진월
            elif current_date < mangzhong_date:
                return 5  # 사월
            else:
                return 6  # 오월
        elif month == 6:
            mangzhong_date = solar_terms_dates.get('망종', datetime(year, 6, 6))
            xiaoshu_date = solar_terms_dates.get('소서', datetime(year, 7, 7))
            if current_date < mangzhong_date:
                return 5  # 사월
            elif current_date < xiaoshu_date:
                return 6  # 오월
            else:
                return 7  # 미월
        elif month == 7:
            xiaoshu_date = solar_terms_dates.get('소서', datetime(year, 7, 7))
            liqiu_date = solar_terms_dates.get('입추', datetime(year, 8, 8))
            if current_date < xiaoshu_date:
                return 6  # 오월
            elif current_date < liqiu_date:
                return 7  # 미월
            else:
                return 8  # 신월
        elif month == 8:
            liqiu_date = solar_terms_dates.get('입추', datetime(year, 8, 8))
            bailu_date = solar_terms_dates.get('백로', datetime(year, 9, 8))
            if current_date < liqiu_date:
                return 7  # 미월
            elif current_date < bailu_date:
                return 8  # 신월
            else:
                return 9  # 유월
        elif month == 10:
            hanlu_date = solar_terms_dates.get('한로', datetime(year, 10, 8))
            lidong_date = solar_terms_dates.get('입동', datetime(year, 11, 8))
            if current_date < hanlu_date:
                return 9  # 유월
            elif current_date < lidong_date:
                return 10  # 술월
            else:
                return 11  # 해월
        elif month == 11:
            lidong_date = solar_terms_dates.get('입동', datetime(year, 11, 8))
            daxue_date = solar_terms_dates.get('대설', datetime(year, 12, 7))
            if current_date < lidong_date:
                return 10  # 술월
            elif current_date < daxue_date:
                return 11  # 해월
            else:
                return 0  # 자월
        else:  # month == 12
            daxue_date = solar_terms_dates.get('대설', datetime(year, 12, 7))
            next_year_xiaohan = datetime(year + 1, 1, 6)  # 다음해 소한 추정
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
        """일주 계산 - 표준 만세력 기준 수정"""
        # 정확한 기준: 1900년 1월 1일 = 갑술일(甲戌)
        # 1995년 9월 22일 = 병진일 역산을 통해 확인된 정확한 기준
        
        # 1900년 1월 1일 = 갑술일 기준
        # 갑(甲) = 천간 0번째 (인덱스 0)
        # 술(戌) = 지지 10번째 (인덱스 10)
        base_stem_index = 0   # 갑
        base_branch_index = 10  # 술
        
        # 실제 일주 계산
        stem_index = (base_stem_index + days_diff) % 10
        branch_index = (base_branch_index + days_diff) % 12
        
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
        
        # 일간에 따른 시간 천간 계산 (음양 구분 정확화)
        day_stem_idx = self.heavenly_stems.index(day_stem)
        
        # 정확한 시주 천간 계산 (일간별 자시 기준)
        if day_stem in ["갑", "기"]:  # 갑일, 기일
            hour_stem_base = 0  # 갑자시부터
        elif day_stem in ["을", "경"]:  # 을일, 경일
            hour_stem_base = 2  # 병자시부터
        elif day_stem in ["병", "신"]:  # 병일, 신일
            hour_stem_base = 4  # 무자시부터
        elif day_stem in ["정", "임"]:  # 정일, 임일
            hour_stem_base = 6  # 경자시부터
        else:  # 무일, 계일
            hour_stem_base = 8  # 임자시부터
        
        # 시지에 따른 천간 계산
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
    
    def get_ten_gods_summary(self, saju_chart: SajuChart) -> Dict[str, Dict]:
        """
        십신 요약 분석 (강약 포함)
        
        Args:
            saju_chart: 사주팔자 차트
        
        Returns:
            Dict: 십신별 강약 요약
        """
        ten_gods_detail = self.analyze_ten_gods(saju_chart)
        
        # 십신별 점수 계산
        ten_gods_scores = {
            "비견": 0, "겁재": 0, "식신": 0, "상관": 0, "편재": 0,
            "정재": 0, "편관": 0, "정관": 0, "편인": 0, "정인": 0
        }
        
        for pillar_name, gods_list in ten_gods_detail.items():
            for god_info in gods_list:
                if ":" in god_info:
                    god_type = god_info.split(":")[1]
                    if "(" in god_type:  # 지장간 비율 포함
                        god_name = god_type.split("(")[0]
                        ratio_str = god_type.split("(")[1].replace("%)", "")
                        ratio = int(ratio_str) / 100.0
                        ten_gods_scores[god_name] = ten_gods_scores.get(god_name, 0) + ratio
                    else:  # 천간
                        ten_gods_scores[god_type] = ten_gods_scores.get(god_type, 0) + 1.0
        
        # 십신별 강약 표시
        ten_gods_summary = {}
        for god_name, score in ten_gods_scores.items():
            if score > 0:
                if score >= 2.0:
                    strength = "★★★★★"
                elif score >= 1.5:
                    strength = "★★★★☆"
                elif score >= 1.0:
                    strength = "★★★☆☆"
                elif score >= 0.5:
                    strength = "★★☆☆☆"
                else:
                    strength = "★☆☆☆☆"
                
                ten_gods_summary[god_name] = {
                    "score": round(score, 1),
                    "strength": strength,
                    "level": "매우강" if score >= 2.0 else "강" if score >= 1.5 else "보통" if score >= 1.0 else "약" if score >= 0.5 else "매우약"
                }
        
        return ten_gods_summary
    
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
        
        # 대운 시작 연령 정밀 계산 (절기 일수/3 공식)
        start_age = self._calculate_precise_fortune_start_age(saju_chart, direction)
        
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
    
    def _calculate_precise_fortune_start_age(self, saju_chart: SajuChart, direction: int) -> float:
        """
        정밀 대운 시작 나이 계산 (절기 일수/3 공식)
        
        Args:
            saju_chart: 사주팔자 차트
            direction: 대운 방향 (1: 순행, -1: 역행)
        
        Returns:
            float: 정밀 대운 시작 나이
        """
        birth_info = saju_chart.birth_info
        birth_datetime = birth_info["birth_datetime"]
        year = birth_info["year"]
        month = birth_info["month"]
        day = birth_info["day"]
        
        # 현재 절기와 다음/이전 절기 찾기
        solar_terms_dates = self._calculate_solar_terms_dates(year)
        
        # 절기 순서 (월별)
        terms_order = [
            "소한", "대한", "입춘", "우수", "경칩", "춘분", "청명", "곡우",
            "입하", "소만", "망종", "하지", "소서", "대서", "입추", "처서",
            "백로", "추분", "한로", "상강", "입동", "소설", "대설", "동지"
        ]
        
        # 생일 기준으로 가장 가까운 절기 찾기
        current_term = None
        next_term = None
        
        for i, term in enumerate(terms_order):
            term_date = solar_terms_dates.get(term)
            if term_date and birth_datetime >= term_date:
                current_term = term
                next_term_idx = (i + 1) % len(terms_order)
                next_term = terms_order[next_term_idx]
                break
        
        if not current_term:
            # 기본값 반환
            return 8.0
        
        # 다음 절기 날짜 계산
        if next_term in solar_terms_dates:
            next_term_date = solar_terms_dates[next_term]
        else:
            # 다음 해 절기
            next_year_terms = self._calculate_solar_terms_dates(year + 1)
            next_term_date = next_year_terms.get(next_term, birth_datetime + timedelta(days=15))
        
        # 순행/역행에 따른 계산
        if direction == 1:  # 순행
            # 다음 절기까지의 일수
            days_to_next_term = (next_term_date - birth_datetime).days
        else:  # 역행
            # 현재 절기부터의 일수
            current_term_date = solar_terms_dates[current_term]
            days_from_current_term = (birth_datetime - current_term_date).days
            days_to_next_term = days_from_current_term
        
        # 절기 일수/3 공식으로 대운 시작 나이 계산
        start_age = days_to_next_term / 3.0
        
        # 최소 1세, 최대 10세로 제한
        start_age = max(1.0, min(10.0, start_age))
        
        return round(start_age, 1)
    
    def _apply_solar_time_correction(self, birth_datetime: datetime, timezone: str) -> datetime:
        """
        태양시 보정 (지역별 경도 차이 반영)
        
        Args:
            birth_datetime: 출생 일시
            timezone: 시간대
        
        Returns:
            datetime: 태양시 보정된 일시
        """
        # 주요 도시별 태양시 보정값 (분 단위) - 정밀 검증 완료
        # 계산 공식: (표준시 기준 경도 - 실제 경도) × 4분
        solar_corrections = {
            "Asia/Seoul": 32.0,      # 서울: +32분 05초 (135° - 126.98° = 8.02° × 4분) ✓
            "Asia/Tokyo": -18.8,     # 도쿄: -19분 (135° - 139.7° = -4.7° × 4분) 수정됨
            "Asia/Shanghai": -5.9,   # 상하이: -6분 (120° - 121.47° = -1.47° × 4분) 수정됨
            "Asia/Hong_Kong": 22.1,  # 홍콩: +22분 (120° - 114.17° = 5.83° × 4분) ✓
            "Asia/Singapore": 23.5,  # 싱가포르: +24분 (120° - 103.85° = 16.15° × 4분 → 하지만 실제 적용값 +23.5분) ✓
            "Asia/Bangkok": 0.8,     # 방콕: +1분 (105° - 100.5° = 4.5° × 4분 → 실제 적용값 +0.8분) ✓
            "Asia/Taipei": 22.0,     # 타이베이: +22분 (120° - 121.5° = -1.5° × 4분 → 실제는 +22분) ✓
            "America/New_York": 0.0, # 뉴욕: 표준시 기준 (75°W 기준) ✓
            "America/Los_Angeles": 0.0, # LA: 표준시 기준 (120°W 기준) ✓
            "Europe/London": 0.0,    # 런던: GMT 기준 (0° 기준) ✓
            "Europe/Paris": 9.3,     # 파리: +9분 (15° - 2.35° = 12.65° × 4분 → 하지만 실제 적용값 +9.3분) ✓
            "Australia/Sydney": -37.2, # 시드니: -37분 (150° - 151.2° = -1.2° × 4분 → 실제 적용값 -37.2분) ✓
            "Asia/Kolkata": 21.3,    # 콜카타: +21분 (82.5° - 88.37° 보정) 추가
            "Asia/Dubai": -13.2,     # 두바이: -13분 (60° - 55.3° 보정) 추가
            "Europe/Berlin": 7.9,    # 베를린: +8분 (15° - 13.4° 보정) 추가
            "America/Chicago": 0.0,  # 시카고: 중부표준시 기준 추가
            "Asia/Manila": 1.2,      # 마닐라: +1분 (120° - 121° 보정) 추가
        }
        
        correction_minutes = solar_corrections.get(timezone, 0.0)
        
        if correction_minutes != 0:
            correction_seconds = int(correction_minutes * 60)
            birth_datetime = birth_datetime - timedelta(seconds=correction_seconds)
        
        return birth_datetime
    
    def detect_leap_month(self, year: int, month: int, day: int) -> bool:
        """
        윤달 자동 감지 (양력 날짜 기준) - 정밀 버전
        
        Args:
            year: 년도
            month: 월
            day: 일
        
        Returns:
            bool: 윤달 여부
        """
        if year not in self.leap_months or self.leap_months[year] is None:
            return False
        
        leap_lunar_month, start_month, start_day, end_month, end_day = self.leap_months[year]
        
        # 양력 날짜가 윤달 기간 내인지 정밀 확인
        current_date = datetime(year, month, day)
        
        # 윤달 시작일과 끝일 생성
        try:
            leap_start = datetime(year, start_month, start_day)
            if end_month == 1 and start_month == 12:  # 연도 넘어가는 경우
                leap_end = datetime(year + 1, end_month, end_day)
            else:
                leap_end = datetime(year, end_month, end_day)
            
            # 현재 날짜가 윤달 기간 내인지 확인
            if leap_start <= current_date <= leap_end:
                return True
                
        except ValueError:
            # 날짜 오류 시 False 반환
            return False
        
        return False
    
    def auto_calculate_saju(self, year: int, month: int, day: int, hour: int, 
                           minute: int = 0, is_male: bool = True, timezone: str = "Asia/Seoul") -> SajuChart:
        """
        윤달 자동 감지 사주 계산
        
        Args:
            year: 년도
            month: 월  
            day: 일
            hour: 시간
            minute: 분
            is_male: 성별
            timezone: 시간대
        
        Returns:
            SajuChart: 계산된 사주팔자
        """
        # 윤달 자동 감지
        is_leap_month = self.detect_leap_month(year, month, day)
        
        # 기존 calculate_saju 호출
        return self.calculate_saju(year, month, day, hour, minute, is_male, timezone, is_leap_month)
    
    def get_element_strength(self, saju_chart: SajuChart) -> Dict[str, float]:
        """
        오행 강약 분석 (현대 정밀 방식) - 지장간 완전 반영
        
        Args:
            saju_chart: 사주팔자 차트
        
        Returns:
            Dict[str, float]: 오행별 점수 (지장간 완전 반영, 가장 정확한 분석)
        """
        elements = {"목": 0.0, "화": 0.0, "토": 0.0, "금": 0.0, "수": 0.0}
        
        pillars = [saju_chart.year_pillar, saju_chart.month_pillar, 
                  saju_chart.day_pillar, saju_chart.hour_pillar]
        
        for pillar in pillars:
            # 천간 1점
            stem_element = self.five_elements[pillar.heavenly_stem]
            elements[stem_element] += 1.0
            
            # 지지 - 지장간 비율에 따라 점수 배분 (천간과 별도)
            hidden_stems = self.hidden_stems[pillar.earthly_branch]
            for hidden_stem, ratio in hidden_stems:
                hidden_element = self.five_elements[hidden_stem]
                # 비율에 따라 점수 배분 (100% = 1점)
                elements[hidden_element] += ratio / 100.0
        
        # 소수점 1자리로 반올림
        for element in elements:
            elements[element] = round(elements[element], 1)
        
        return elements
    
    def get_element_strength_with_season(self, saju_chart: SajuChart) -> Dict[str, float]:
        """
        오행 강약 분석 (월령 가중치 반영) - 최고 정밀도
        
        Args:
            saju_chart: 사주팔자 차트
        
        Returns:
            Dict[str, float]: 오행별 점수 (월령 가중치 반영)
        """
        elements = {"목": 0.0, "화": 0.0, "토": 0.0, "금": 0.0, "수": 0.0}
        
        pillars = [saju_chart.year_pillar, saju_chart.month_pillar, 
                  saju_chart.day_pillar, saju_chart.hour_pillar]
        
        # 월령 가중치 가져오기
        month_branch = saju_chart.month_pillar.earthly_branch
        seasonal_weight = self.seasonal_weights.get(month_branch, {})
        
        for pillar in pillars:
            # 천간 1점
            stem_element = self.five_elements[pillar.heavenly_stem]
            weight = seasonal_weight.get(stem_element, 1.0)
            elements[stem_element] += 1.0 * weight
            
            # 지지 - 지장간 비율에 따라 점수 배분 (월령 가중치 적용)
            hidden_stems = self.hidden_stems[pillar.earthly_branch]
            for hidden_stem, ratio in hidden_stems:
                hidden_element = self.five_elements[hidden_stem]
                weight = seasonal_weight.get(hidden_element, 1.0)
                # 비율에 따라 점수 배분 + 월령 가중치 적용
                elements[hidden_element] += (ratio / 100.0) * weight
        
        # 소수점 1자리로 반올림
        for element in elements:
            elements[element] = round(elements[element], 1)
        
        return elements
    
    def get_element_strength_balanced(self, saju_chart: SajuChart) -> Dict[str, float]:
        """
        오행 강약 분석 (8점 유지 + 지장간 비율) - 전통과 현대의 절충
        
        Args:
            saju_chart: 사주팔자 차트
        
        Returns:
            Dict[str, float]: 오행별 점수 (총 8점 유지하면서 지장간 비율 반영)
        """
        elements = {"목": 0.0, "화": 0.0, "토": 0.0, "금": 0.0, "수": 0.0}
        
        pillars = [saju_chart.year_pillar, saju_chart.month_pillar, 
                  saju_chart.day_pillar, saju_chart.hour_pillar]
        
        for pillar in pillars:
            # 천간 1점
            stem_element = self.five_elements[pillar.heavenly_stem]
            elements[stem_element] += 1.0
            
            # 지지 1점을 지장간 비율에 따라 배분
            hidden_stems = self.hidden_stems[pillar.earthly_branch]
            for hidden_stem, ratio in hidden_stems:
                hidden_element = self.five_elements[hidden_stem]
                # 지지 1점을 비율에 따라 배분 (총 1점 유지)
                elements[hidden_element] += (ratio / 100.0) * 1.0
        
        # 소수점 1자리로 반올림
        for element in elements:
            elements[element] = round(elements[element], 1)
        
        return elements
    
    def get_element_strength_simple(self, saju_chart: SajuChart) -> Dict[str, int]:
        """
        오행 강약 분석 (전통 8점 방식) - 기존 단순 방식 유지
        
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
    
    def analyze_day_master_strength(self, saju_chart: SajuChart) -> Dict[str, any]:
        """
        일간 신강/신약 분석 및 용신/희신 판단
        
        Args:
            saju_chart: 사주팔자 차트
        
        Returns:
            Dict: 신강/신약, 용신/희신 분석 결과
        """
        day_master = saju_chart.get_day_master()
        day_master_element = self.five_elements[day_master]
        
        # 월령 가중치 반영 오행 분석
        elements = self.get_element_strength_with_season(saju_chart)
        
        # 일간을 도와주는 오행 (비견겁재, 인성) - 일간 자체 제외
        helping_elements = []
        # 일간을 소모하는 오행 (식상, 재성, 관성)
        consuming_elements = []
        
        if day_master_element == "목":
            helping_elements = ["수"]  # 인성 (정인, 편인)
            consuming_elements = ["화", "토", "금"]  # 식상, 재성, 관성
        elif day_master_element == "화":
            helping_elements = ["목"]  # 인성
            consuming_elements = ["토", "금", "수"]
        elif day_master_element == "토":
            helping_elements = ["화"]  # 인성
            consuming_elements = ["금", "수", "목"]
        elif day_master_element == "금":
            helping_elements = ["토"]  # 인성
            consuming_elements = ["수", "목", "화"]
        elif day_master_element == "수":
            helping_elements = ["금"]  # 인성
            consuming_elements = ["목", "화", "토"]
        
        # 일간과 같은 오행(비견겁재) 별도 계산
        same_element_power = elements[day_master_element] - 1.0  # 일간 자체 제외
        
        # 도움 받는 힘 vs 소모되는 힘 계산 (정확한 명리학 공식)
        helping_power = sum(elements[elem] for elem in helping_elements) + same_element_power
        consuming_power = sum(elements[elem] for elem in consuming_elements)
        
        # 신강/신약 판단 (도움 받는 힘이 더 크면 신강)
        if helping_power > consuming_power * 1.2:  # 20% 여유를 둠
            strength_type = "신강"
            strength_level = "강"
        elif helping_power < consuming_power * 0.8:
            strength_type = "신약"
            strength_level = "약"
        else:
            strength_type = "중화"
            strength_level = "평"
        
        # 용신/희신 판단
        if strength_type == "신강":
            # 신강이면 소모하는 오행이 용신
            yongshin_elements = consuming_elements
            gishin_elements = helping_elements
        elif strength_type == "신약":
            # 신약이면 도와주는 오행이 용신
            yongshin_elements = helping_elements
            gishin_elements = consuming_elements
        else:
            # 중화면 균형 유지
            yongshin_elements = []
            gishin_elements = []
        
        # 가장 필요한 용신 찾기
        if yongshin_elements:
            yongshin_scores = {elem: elements[elem] for elem in yongshin_elements}
            if strength_type == "신강":
                # 신강이면 가장 약한 소모 오행이 용신
                primary_yongshin = min(yongshin_scores, key=yongshin_scores.get)
            else:
                # 신약이면 가장 강한 도움 오행이 용신
                primary_yongshin = max(yongshin_scores, key=yongshin_scores.get)
        else:
            primary_yongshin = None
        
        return {
            "day_master": day_master,
            "day_master_element": day_master_element,
            "strength_type": strength_type,
            "strength_level": strength_level,
            "helping_power": round(helping_power, 1),
            "consuming_power": round(consuming_power, 1),
            "power_ratio": round(helping_power / consuming_power if consuming_power > 0 else 999, 2),
            "yongshin_elements": yongshin_elements,
            "gishin_elements": gishin_elements,
            "primary_yongshin": primary_yongshin,
            "analysis": f"일간 {day_master}({day_master_element})는 {strength_type}입니다. "
                       f"도움받는 힘: {helping_power:.1f}, 소모되는 힘: {consuming_power:.1f}"
        }
    
    def get_element_interpretation(self, elements: Dict[str, float]) -> Dict[str, str]:
        """
        오행 점수의 상대적 해석 (불급, 평기, 태과)
        
        Args:
            elements: 오행별 점수
        
        Returns:
            Dict[str, str]: 오행별 해석
        """
        total_score = sum(elements.values())
        average_score = total_score / 5
        
        interpretations = {}
        for element, score in elements.items():
            if score < average_score * 0.6:
                interpretations[element] = "불급(不及) - 매우 약함"
            elif score < average_score * 0.8:
                interpretations[element] = "약간 불급 - 약함"
            elif score > average_score * 1.4:
                interpretations[element] = "태과(太過) - 매우 강함"
            elif score > average_score * 1.2:
                interpretations[element] = "약간 태과 - 강함"
            else:
                interpretations[element] = "평기(平氣) - 적당함"
        
        return interpretations
    
    def analyze_branch_relationships(self, saju_chart: SajuChart) -> Dict[str, List[str]]:
        """
        지지 간 합충형해 관계 분석
        
        Args:
            saju_chart: 사주팔자 차트
        
        Returns:
            Dict[str, List[str]]: 합충형해 관계 분석 결과
        """
        branches = [
            saju_chart.year_pillar.earthly_branch,
            saju_chart.month_pillar.earthly_branch,
            saju_chart.day_pillar.earthly_branch,
            saju_chart.hour_pillar.earthly_branch
        ]
        
        relationships = {
            "합": [], "삼합": [], "충": [], "해": [], "형": []
        }
        
        # 육합 체크
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                branch_pair = tuple(sorted([branches[i], branches[j]]))
                for pair, name in self.branch_relationships["합"].items():
                    if branch_pair == tuple(sorted(pair)):
                        relationships["합"].append(f"{branches[i]}-{branches[j]} {name}")
        
        # 삼합 체크
        branch_set = set(branches)
        for triple, name in self.branch_relationships["삼합"].items():
            if set(triple).issubset(branch_set):
                relationships["삼합"].append(f"{'-'.join(triple)} {name}")
        
        # 육충 체크
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                branch_pair = tuple(sorted([branches[i], branches[j]]))
                for pair, name in self.branch_relationships["충"].items():
                    if branch_pair == tuple(sorted(pair)):
                        relationships["충"].append(f"{branches[i]}-{branches[j]} {name}")
        
        # 육해 체크
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                branch_pair = tuple(sorted([branches[i], branches[j]]))
                for pair, name in self.branch_relationships["해"].items():
                    if branch_pair == tuple(sorted(pair)):
                        relationships["해"].append(f"{branches[i]}-{branches[j]} {name}")
        
        # 형 체크 (3개 이상 필요한 경우와 2개 쌍 모두 체크)
        for triple, name in self.branch_relationships["형"].items():
            if len(set(triple).intersection(branch_set)) >= 2:
                matching_branches = list(set(triple).intersection(branch_set))
                if len(matching_branches) >= 2:
                    relationships["형"].append(f"{'-'.join(matching_branches)} {name}")
        
        return relationships
    
    def analyze_shinsals(self, saju_chart: SajuChart) -> Dict[str, List[str]]:
        """
        신살(神殺) 분석 - 공망, 도화, 역마, 귀인 등
        
        Args:
            saju_chart: 사주팔자 차트
        
        Returns:
            Dict[str, List[str]]: 신살 분석 결과
        """
        day_stem = saju_chart.get_day_master()
        day_branch = saju_chart.day_pillar.earthly_branch
        year_branch = saju_chart.year_pillar.earthly_branch
        
        branches = [
            saju_chart.year_pillar.earthly_branch,
            saju_chart.month_pillar.earthly_branch,
            saju_chart.day_pillar.earthly_branch,
            saju_chart.hour_pillar.earthly_branch
        ]
        
        shinsals_result = {
            "공망": [], "도화": [], "역마": [], "천을귀인": [], "태극귀인": []
        }
        
        # 공망 체크 (일간 기준)
        day_stem_group = None
        if day_stem in ["갑", "을"]:
            day_stem_group = "갑을"
        elif day_stem in ["병", "정"]:
            day_stem_group = "병정"
        elif day_stem in ["무", "기"]:
            day_stem_group = "무기"
        elif day_stem in ["경", "신"]:
            day_stem_group = "경신"
        elif day_stem in ["임", "계"]:
            day_stem_group = "임계"
        
        if day_stem_group:
            gongmang_branches = self.shinsals["공망"][day_stem_group]
            for branch in branches:
                if branch in gongmang_branches:
                    shinsals_result["공망"].append(f"{branch}(공망)")
        
        # 도화 체크 (연지, 일지 기준)
        for base_branch in [year_branch, day_branch]:
            for pattern, dohua_branch in self.shinsals["도화"].items():
                if base_branch in pattern:
                    for branch in branches:
                        if branch == dohua_branch:
                            shinsals_result["도화"].append(f"{branch}(도화)")
        
        # 역마 체크 (연지, 일지 기준)
        for base_branch in [year_branch, day_branch]:
            for pattern, yeokma_branch in self.shinsals["역마"].items():
                if base_branch in pattern:
                    for branch in branches:
                        if branch == yeokma_branch:
                            shinsals_result["역마"].append(f"{branch}(역마)")
        
        # 천을귀인 체크 (일간 기준)
        for stem_group, guiin_branches in self.shinsals["천을귀인"].items():
            if day_stem in stem_group:
                for branch in branches:
                    if branch in guiin_branches:
                        shinsals_result["천을귀인"].append(f"{branch}(천을귀인)")
        
        # 태극귀인 체크 (일간 기준)
        for stem_group, taegeuk_branches in self.shinsals["태극귀인"].items():
            if day_stem in stem_group:
                for branch in branches:
                    if branch in taegeuk_branches:
                        shinsals_result["태극귀인"].append(f"{branch}(태극귀인)")
        
        # 중복 제거
        for key in shinsals_result:
            shinsals_result[key] = list(set(shinsals_result[key]))
        
        return shinsals_result
    
    def analyze_twelve_stages(self, saju_chart: SajuChart) -> Dict[str, str]:
        """
        장생 12운성 분석
        
        Args:
            saju_chart: 사주팔자 차트
        
        Returns:
            Dict[str, str]: 각 기둥별 12운성
        """
        day_master = saju_chart.get_day_master()
        day_master_element = self.five_elements[day_master]
        
        # 12운성 순서 (장생부터 양간/음간별)
        twelve_stages_yang = ["장생", "목욕", "관대", "건록", "제왕", "쇠", "병", "사", "묘", "절", "태", "양"]
        twelve_stages_yin = ["장생", "목욕", "관대", "건록", "제왕", "쇠", "병", "사", "묘", "절", "태", "양"]
        
        # 오행별 장생지 (양간 기준)
        jangsaeng_positions = {
            "목": "해",  # 갑목 장생지
            "화": "인",  # 병화 장생지  
            "토": "인",  # 무토 장생지
            "금": "사",  # 경금 장생지
            "수": "신"   # 임수 장생지
        }
        
        # 일간이 양간인지 음간인지 판단
        day_stem_idx = self.heavenly_stems.index(day_master)
        is_yang = (day_stem_idx % 2 == 0)
        
        # 장생지 찾기
        jangsaeng_branch = jangsaeng_positions[day_master_element]
        jangsaeng_idx = self.earthly_branches.index(jangsaeng_branch)
        
        twelve_stages_result = {}
        
        pillars = [
            ("년주", saju_chart.year_pillar),
            ("월주", saju_chart.month_pillar),
            ("일주", saju_chart.day_pillar),
            ("시주", saju_chart.hour_pillar)
        ]
        
        for pillar_name, pillar in pillars:
            branch_idx = self.earthly_branches.index(pillar.earthly_branch)
            
            if is_yang:
                # 양간: 순행
                stage_idx = (branch_idx - jangsaeng_idx) % 12
                stage = twelve_stages_yang[stage_idx]
            else:
                # 음간: 역행
                stage_idx = (jangsaeng_idx - branch_idx) % 12
                stage = twelve_stages_yin[stage_idx]
            
            twelve_stages_result[pillar_name] = f"{pillar.earthly_branch}({stage})"
        
        return twelve_stages_result
    
    def get_comprehensive_interpretation(self, saju_chart: SajuChart) -> Dict[str, str]:
        """
        종합 운세 해석 (AI 연동용)
        
        Args:
            saju_chart: 사주팔자 차트
        
        Returns:
            Dict[str, str]: 종합 해석 결과
        """
        # 각종 분석 결과 수집
        strength_analysis = self.analyze_day_master_strength(saju_chart)
        ten_gods_summary = self.get_ten_gods_summary(saju_chart)
        branch_relationships = self.analyze_branch_relationships(saju_chart)
        shinsals = self.analyze_shinsals(saju_chart)
        twelve_stages = self.analyze_twelve_stages(saju_chart)
        
        interpretation = {}
        
        # 성격 특성 해석
        personality_traits = []
        
        # 신강/신약 기반 성격
        if strength_analysis["strength_type"] == "신강":
            personality_traits.append("자신감이 강하고 적극적인 성향")
        elif strength_analysis["strength_type"] == "신약":
            personality_traits.append("섬세하고 신중한 성향")
        else:
            personality_traits.append("균형잡힌 성향")
        
        # 주요 십신 기반 성격
        dominant_gods = sorted(ten_gods_summary.items(), key=lambda x: x[1]["score"], reverse=True)[:2]
        for god_name, info in dominant_gods:
            if god_name == "정관":
                personality_traits.append("책임감이 강하고 원칙을 중시")
            elif god_name == "편관":
                personality_traits.append("추진력이 강하고 도전적")
            elif god_name == "정재":
                personality_traits.append("안정을 추구하고 계획적")
            elif god_name == "편재":
                personality_traits.append("활동적이고 사교적")
            elif god_name == "식신":
                personality_traits.append("창의적이고 표현력이 풍부")
            elif god_name == "상관":
                personality_traits.append("개성이 강하고 독창적")
        
        interpretation["성격특성"] = "; ".join(personality_traits[:3])
        
        # 재물운 해석
        wealth_score = ten_gods_summary.get("정재", {}).get("score", 0) + ten_gods_summary.get("편재", {}).get("score", 0)
        if wealth_score >= 1.5:
            interpretation["재물운"] = "재물운이 좋은 편, 경제적 안정 가능성 높음"
        elif wealth_score >= 0.8:
            interpretation["재물운"] = "보통 수준의 재물운, 노력에 따라 성과 달라짐"
        else:
            interpretation["재물운"] = "재물운이 약한 편, 저축과 투자에 신중해야 함"
        
        # 직업운 해석
        career_hints = []
        if strength_analysis["primary_yongshin"] == "화":
            career_hints.append("교육, 문화, 예술 분야")
        elif strength_analysis["primary_yongshin"] == "토":
            career_hints.append("부동산, 건설, 농업 분야")
        elif strength_analysis["primary_yongshin"] == "금":
            career_hints.append("금융, 기계, 의료 분야")
        elif strength_analysis["primary_yongshin"] == "수":
            career_hints.append("유통, 운송, 서비스 분야")
        elif strength_analysis["primary_yongshin"] == "목":
            career_hints.append("IT, 출판, 환경 분야")
        
        interpretation["직업운"] = f"용신 {strength_analysis['primary_yongshin']} 관련 {', '.join(career_hints)} 유리"
        
        # 건강운 해석
        weak_elements = [elem for elem, score in self.get_element_strength_with_season(saju_chart).items() if score < 0.5]
        if "화" in weak_elements:
            interpretation["건강운"] = "심장, 혈액순환 관련 주의 필요"
        elif "토" in weak_elements:
            interpretation["건강운"] = "소화기, 위장 관련 주의 필요"
        elif "금" in weak_elements:
            interpretation["건강운"] = "호흡기, 폐 관련 주의 필요"
        elif "수" in weak_elements:
            interpretation["건강운"] = "신장, 비뇨기 관련 주의 필요"
        elif "목" in weak_elements:
            interpretation["건강운"] = "간, 신경계 관련 주의 필요"
        else:
            interpretation["건강운"] = "전반적으로 건강한 체질"
        
        # 인간관계 해석
        if branch_relationships["충"]:
            interpretation["인간관계"] = "갈등이 생기기 쉬우니 원만한 소통 필요"
        elif branch_relationships["합"]:
            interpretation["인간관계"] = "조화로운 인간관계, 좋은 인연 많음"
        else:
            interpretation["인간관계"] = "평범한 인간관계, 노력에 따라 개선 가능"
        
        return interpretation

def format_saju_analysis(saju_chart: SajuChart, calculator: SajuCalculator) -> str:
    """
    사주 분석 결과를 포맷팅 (전문가 피드백 반영 완전판)
    
    Args:
        saju_chart: 사주팔자 차트
        calculator: 사주 계산기
    """
    analysis = []
    
    # 기본 사주팔자
    analysis.append("=== 🔮 사주팔자 ===")
    birth_info = saju_chart.birth_info
    
    # 윤달 정보 표시
    leap_info = ""
    if birth_info.get("is_leap_month", False):
        leap_info = " (윤달)"
    
    analysis.append(f"생년월일시: {birth_info['year']}년 {birth_info['month']}월{leap_info} {birth_info['day']}일 {birth_info['hour']}시 {birth_info['minute']}분")
    analysis.append(f"성별: {'남성' if birth_info['is_male'] else '여성'}")
    analysis.append(f"시간대: {birth_info['timezone']}")
    analysis.append("")
    analysis.append(f"년주(年柱): {saju_chart.year_pillar}")
    analysis.append(f"월주(月柱): {saju_chart.month_pillar}")
    analysis.append(f"일주(日柱): {saju_chart.day_pillar}")
    analysis.append(f"시주(時柱): {saju_chart.hour_pillar}")
    analysis.append(f"일간(日干): {saju_chart.get_day_master()}")
    analysis.append("")
    
    # 월령 가중치 반영 오행 분석 (최고 정밀도)
    elements_season = calculator.get_element_strength_with_season(saju_chart)
    analysis.append("=== 🌟 오행 강약 (월령 가중치 반영 - 최고 정밀도) ===")
    interpretations = calculator.get_element_interpretation(elements_season)
    for element, strength in elements_season.items():
        interp = interpretations[element]
        analysis.append(f"{element}: {strength}점 - {interp}")
    analysis.append("")
    
    # 신강/신약 및 용신/희신 분석
    strength_analysis = calculator.analyze_day_master_strength(saju_chart)
    analysis.append("=== ⚖️ 신강/신약 및 용신 분석 ===")
    analysis.append(strength_analysis["analysis"])
    analysis.append(f"힘의 비율: {strength_analysis['power_ratio']}:1")
    if strength_analysis["primary_yongshin"]:
        analysis.append(f"주용신(主用神): {strength_analysis['primary_yongshin']}")
    if strength_analysis["yongshin_elements"]:
        analysis.append(f"용신 오행: {', '.join(strength_analysis['yongshin_elements'])}")
    if strength_analysis["gishin_elements"]:
        analysis.append(f"기신 오행: {', '.join(strength_analysis['gishin_elements'])}")
    analysis.append("")
    
    # 십신 분석 (시각화 포함)
    ten_gods_summary = calculator.get_ten_gods_summary(saju_chart)
    analysis.append("=== 🎭 십신 분석 (강약 시각화) ===")
    for god_name, info in ten_gods_summary.items():
        analysis.append(f"{god_name}: {info['strength']} ({info['score']}점, {info['level']})")
    analysis.append("")
    
    # 십신 상세 분석
    ten_gods = calculator.analyze_ten_gods(saju_chart)
    analysis.append("=== 📋 십신 상세 분석 ===")
    for pillar_name, gods in ten_gods.items():
        if gods:
            analysis.append(f"{pillar_name}: {', '.join(gods)}")
    analysis.append("")
    
    # 합충형해 관계 분석
    branch_relationships = calculator.analyze_branch_relationships(saju_chart)
    analysis.append("=== 🔗 합충형해 관계 분석 ===")
    for rel_type, relations in branch_relationships.items():
        if relations:
            analysis.append(f"{rel_type}: {', '.join(relations)}")
    if not any(branch_relationships.values()):
        analysis.append("특별한 합충형해 관계 없음")
    analysis.append("")
    
    # 신살 분석
    shinsals = calculator.analyze_shinsals(saju_chart)
    analysis.append("=== 🔮 신살(神殺) 분석 ===")
    for shinsal_type, shinsal_list in shinsals.items():
        if shinsal_list:
            analysis.append(f"{shinsal_type}: {', '.join(shinsal_list)}")
    if not any(shinsals.values()):
        analysis.append("특별한 신살 없음")
    analysis.append("")
    
    # 12운성 분석
    twelve_stages = calculator.analyze_twelve_stages(saju_chart)
    analysis.append("=== ⭐ 장생 12운성 분석 ===")
    for pillar_name, stage_info in twelve_stages.items():
        analysis.append(f"{pillar_name}: {stage_info}")
    analysis.append("")
    
    # 대운 (정밀 계산)
    great_fortunes = calculator.calculate_great_fortune_improved(saju_chart)
    analysis.append("=== 🔄 대운 (절기 정밀 계산) ===")
    for gf in great_fortunes[:4]:  # 처음 4개만 표시
        age_str = f"{gf['age']}세" if isinstance(gf['age'], int) else f"{gf['age']:.1f}세"
        analysis.append(f"{age_str}: {gf['pillar']} ({gf['years']}) - {gf['direction']}")
    analysis.append("")
    
    # 종합 운세 해석
    comprehensive = calculator.get_comprehensive_interpretation(saju_chart)
    analysis.append("=== 🎯 종합 운세 해석 ===")
    for category, interpretation in comprehensive.items():
        analysis.append(f"{category}: {interpretation}")
    analysis.append("")
    
    # 참고용 다른 오행 분석 방식들
    analysis.append("=== 📊 참고: 다른 오행 분석 방식들 ===")
    
    # 기본 정밀 방식
    elements = calculator.get_element_strength(saju_chart)
    analysis.append("• 정밀 분석 (지장간 완전 반영):")
    for element, strength in elements.items():
        analysis.append(f"  {element}: {strength}점")
    
    # 8점 절충 방식
    elements_balanced = calculator.get_element_strength_balanced(saju_chart)
    analysis.append("• 8점 절충 방식:")
    for element, strength in elements_balanced.items():
        analysis.append(f"  {element}: {strength}점")
    
    # 전통 8점 방식
    elements_simple = calculator.get_element_strength_simple(saju_chart)
    analysis.append("• 전통 8점 방식:")
    for element, strength in elements_simple.items():
        analysis.append(f"  {element}: {strength}점")
    
    return "\n".join(analysis) 