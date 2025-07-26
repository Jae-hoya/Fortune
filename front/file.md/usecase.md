# **상세 Use Case 문서**

이 문서는 **사주·타로 통합 채팅 플랫폼(웹)**의 기능 요구(PRD) 및 정보 구조(IA)를 기반으로 작성한 사용 사례(Use Case) 정의서입니다.

모든 예시 대화·입력·출력은 한국어 기준으로 서술합니다.

**1. Actor Definitions**

| **구분** | **명칭** | **책임** |
| --- | --- | --- |
| Primary | 게스트 사용자 | 회원가입 없이 운세를 체험하고 결과를 즉시 확인 |
| 〃 | 인증 사용자 | 소셜 로그인(OAuth) 후 운세를 저장·공유·프로필 관리 |
| Supporting | 웹 애플리케이션(UI·로컬 스토리지·PWA) | 화면 렌더링, 데이터 암호화 저장, 알림 트리거 |
| 〃 | 운세 엔진(LLM + LangGraph) | 사주·타로 해석, 자연어 응답 |
| 〃 | OAuth 공급자(카카오·구글) | 사용자의 인증·토큰 발행 |

**2. Use Case Scenarios**

| **UC ID** | **시나리오 이름** | **주 Actor** |
| --- | --- | --- |
| UC-1 | 게스트 타로 리딩 | 게스트 사용자 |
| UC-2 | 사주 리딩 & 운세 저장 | 인증 사용자 |
| UC-3 | 지인 생년월일 프로필 추가 | 인증 사용자 |
| UC-4 | 저장된 운세 열람·공유 | 인증 사용자 |
| UC-5 | 일일 운세 푸시 알림 설정 | 인증 사용자 |

**3. Main Steps (요약)**

1. 홈 진입 → 사주/타로 선택
2. 필수 입력(생년월일·스프레드) 후 채팅 시작
3. 운세 결과 표시 & 추가 액션(저장·공유·알림 설정)

**4. Exception Handling (요약)**

- 네트워크 오류: 로컬 캐시된 마지막 결과 표시 & 재시도 안내
- OAuth 실패: 팝업 재호출, 다른 로그인 방법 제안
- 입력 검증 실패: 생년월일 형식 오류, 과거/미래 날짜 경고
- 로컬 스토리지 용량 초과: 가장 오래된 기록 자동 삭제 + 알림

**5.**

**Comprehensive Actor Definitions**

| **Actor** | **상세 설명** | **협력 시스템** |
| --- | --- | --- |
| 게스트 사용자 | - 익명 세션 ID 자동 부여- 결과는 탭 종료 시 삭제 | 웹 애플리케이션 |
| 인증 사용자 | - Supabase Auth로 발급한 JWT 보유- 저장·알림·프로필 기능 사용 | OAuth 공급자, 웹 앱 |
| 웹 애플리케이션 | - Next.js + Tailwind UI- 사이드바 네비게이션·모달·카드 Flip | 운세 엔진 |
| 운세 엔진 | - LangChain 챗플로우(FSM)- Tarot 이미지·사주 데이터 로더 | OpenAI LLM |
| OAuth 공급자 | - OAuth 2.0 Authorization Code Flow 지원- Access·Refresh·ID 토큰 발급 | Supabase Auth |

**6.**

**Detailed Use Case Scenarios**

**UC-1  게스트 타로 리딩**

| **항목** | **내용** |
| --- | --- |
| 목표 | 로그인 없이 빠르게 타로 점 보기 |
| Pre-conditions | 웹 접속 가능, 브라우저 JavaScript 활성 |
| Main Flow | 1. 홈 → 타로 클릭 → /tarot2. 스프레드 선택 모달: 3장/6장/10장 중 택13. 카드 뽑기(드래그 또는 탭)4. 운세 엔진이 카드 해석 응답(스트리밍)5. 결과 요약 패널 표시 |
| Post-conditions | - 결과 데이터 세션 스토리지에 임시 보관- 탭 종료 시 자동 삭제 |
| Alternative | A1. 네트워크 끊김 → “네트워크 불안정, 재시도 하시겠어요?” 안내 |
| Exception | E1. 카드 데이터 로드 실패 → 기본 텍스트 fallback |

**UC-2  사주 리딩 & 운세 저장**

| **항목** | **내용** |
| --- | --- |
| 목표 | 자신의 사주 리딩 후 결과 저장 |
| Pre-conditions | 소셜 로그인 완료(JWT), 본인 생년월일 등록 |
| Main Flow | 1. /saju 페이지 진입2. 오늘의 운세 or 궁합 메뉴 선택3. 채팅 입력 → 운세 엔진 응답4. “저장” 버튼 클릭 → localStorage → readings:{uuid}에 AES 암호화 저장  5. “PDF로 내보내기” 클릭 시 client-side PDFKit 다운로드 |
| Post-conditions | - 저장 목록 /readings에 신규 카드 생성 |
| Alternative | A1. localStorage 용량 부족 → 가장 오래된 항목 삭제후 저장 |
| Exception | E1. 암호화 라이브러리 오류 → raw 저장 차단 & 오류 알림 |

**UC-3  지인 생년월일 프로필 추가**

| **항목** | **내용** |
| --- | --- |
| Pre | 인증 상태 |
| Flow | 1. /profile → “+ 프로필 추가” 클릭2. 모달에 이름·생년월일 입력3. 형식 검증 → 저장 |
| Post | - profiles 배열에 암호화된 새 객체 추가 |
| Edge | 생년월일 미래 날짜 → 오류 메시지 |

**UC-4  저장된 운세 열람·공유**

| … (구조 동일, 생략) |

**UC-5  일일 운세 푸시 알림 설정**

| … |

**7. Main Steps and Flow of Events (시퀀스 다이어그램 요약)**

User → WebApp : 스프레드 선택

WebApp → Engine : 카드 배열 전달

Engine → WebApp : 해석 스트림 응답

WebApp → User : 메시지 버블 실시간 렌더

**8. Alternative Flows & Edge Cases**

- 로컬 스토리지 사용 불가(사파리 프라이빗) → IndexedDB fallback
- 모바일 Landscape: 카드 Flip 대신 Fade 애니메이션
- 스크린리더: 카드 이미지 ALT → “은둔자(역방향)” 음성 출력

**9. Preconditions & Postconditions**

| **Use Case** | **Pre** | **Post** |
| --- | --- | --- |
| UC-1 | 게스트 세션 ID 발급 | 결과 세션 스토리지 임시 저장 |
| UC-2 | JWT 유효, 프로필 있음 | 암호화 결과 localStorage 저장 |
| … | … | … |

**10. Business Rules & Constraints**

1. 데이터베이스 미사용: 모든 개인 데이터는 클라이언트 암호화 후 로컬 저장
2. 생년월일 형식: YYYY-MM-DD, 1900–현재 연도만 허용
3. 타로 스프레드 제한: 한 세션당 최대 1회 선택, 변경 시 새 세션
4. 접근성: 포커스 인디케이터 필수(WCAG 2.4.7)

**11. Exception Handling Procedures**

| **코드** | **예외** | **처리** |
| --- | --- | --- |
| E401 | OAuth 토큰 만료 | 리프레시 → 실패 시 재로그인 팝업 |
| E501 | LLM 응답 지연 > 10s | “잠시만요…” 스켈레톤 UI & 취소 옵션 |
| E601 | localStorage QuotaExceeded | LRU 삭제 → 사용자에게 알림 |

**12. User Interface Considerations**

- 사이드바 탭 순서 고정, aria-current="page" 제공
- 모달: aria-modal="true", ESC 키 닫힘
- 포커스 링: 대비 3:1 이상 & 2px 두께
- 반응형: 3단계 브레이크포인트, 모바일 상단 탭바(선택·카드·결과)

**13. Data Requirements & Data Flow**

| **키** | **형식** | **설명** |
| --- | --- | --- |
| profiles | [Encrypted<Profile>] | 이름·생년월일 |
| readings:{uuid} | Encrypted<JSON> | 운세 결과·타임스탬프 |
| settings | JSON | 알림 여부, 테마 등 |

데이터 Flow: UI ↔ 로컬 암호화 ↔ 운세 엔진(API) (JWT 포함 헤더)

**14. Security & Privacy Considerations**

1. AES-256 + PBKDF2 Salt로 로컬 데이터 암호화; 키는 세션 메모리 유지(재로그인 시 재생성)
2. XSS 방지: 모든 LLM 응답 HTML 이스케이프
3. 토큰 보호: JWT는 Secure, SameSite=Strict 쿠키 사용
4. 개인 공유 링크: 24 h 만료 서명 URL(/tarot/result/:id?sig=)
5. 접근 제어: 프리미엄 상담은 결제 완료 Webhook 검증 후 접근 허용