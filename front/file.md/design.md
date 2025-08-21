# **UI / UX 디자인 가이드**

스타일: Modern | 컬러 스킴: Analogous (#0084ff 중심) | 무드 키워드: 신비로움 · 몽환적 · 안정적

**1. Design System Overview**

| **항목** | **가이드** |
| --- | --- |
| 디자인 원칙 | 1) 일관성: 전역 컴포넌트·톤 유지2) 가독성: 선명한 대비·명확한 계층 구조3) 심미성: 몽환적 그래디언트·반투명 글래스모피즘 패널4) 접근성: WCAG AA 색 대비, 키보드 내비 지원 |
| 타이포그래피 | - 타이틀: Pretendard Bold 32–40px- 본문: Pretendard Regular 16px- 캡션: 14px / 400 |
| 그리드 시스템 | Tailwind container, grid-cols-12• 모바일: 4 cols• 태블릿: 8 cols• 데스크톱 이상: 12 cols |
| 아이콘 | Lucide 24px 라인아이콘, 동일 스트로크 |
| 이미지 스타일 | 흐릿한 빛 번짐(Blur 8px) + 별빛 파티클 오버레이 |

**2. Color Palette for TailwindCSS**

| **토큰** | **50** | **100** | **200** | **300** | **400** | **500** | **600** | **700** | **800** | **900** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| primary (blue) | #e6f2ff | #cce4ff | #99c9ff | #66adff | #3392ff | #0084ff | #0076e6 | #0068cc | #005bb3 | #004d99 |
| secondary (cyan) | #e5fbff | #c0f5ff | #8aefff | #54e9ff | #1ee3ff | #00c8ff | #00b4e6 | #009fcc | #008ab3 | #007699 |
| accent (indigo) | #ecebff | #d2d0ff | #a5a0ff | #7871ff | #4b41ff | #3334ff | #2d2ee6 | #2728cc | #2121b3 | #1b1b99 |
| neutral (gray) | #fafbfc | #f2f4f6 | #e5e8ec | #cfd4da | #b9c0c8 | #a3acb6 | #8e98a3 | #78828e | #626c79 | #4c5664 |
| success | — | — | — | — | — | #12b76a | — | — | — | — |
| warning | — | — | — | — | — | #f79009 | — | — | — | — |
| error | — | — | — | — | — | #f04438 | — | — | — | — |

색상 선택 근거

- Analogous 스킴: Blue → Cyan → Indigo 를 연속적으로 배치하여 몽환적·신비로운 분위기 형성
- Neutral 회색군: 안정감을 주고, 원본 컨텐츠와 인터랙션 요소를 구분
- 상태 색상: 알림·피드백에만 제한적으로 사용해 주요 무드 컬러 집중 유지

**3. Page Implementations**

**3-1. 루트 페이지**

**/**

| **구분** | **내용** |
| --- | --- |
| 목적 | 첫 인상 제공 + 사주/타로 진입 선택 |
| 주요 컴포넌트 | Hero(배경 비디오+그라디언트)·CTA Buttons(“사주 보기”, “타로 보기”)·하단 특징 3컬럼 |
| 레이아웃 | grid-cols-1 md:grid-cols-12- Hero: 12cols- 특징: 모바일 1 × 3 / 데스크톱 3 × 1 |
| 대표 이미지 | https://picsum.photos/seed/astro/1200/800 |

**3-2. 사주 채팅**

**/saju**

| **항목** | **내용** |
| --- | --- |
| 키 컴포넌트 | ProfileSelector · ChatWindow · ExplanationPanel |
| 구조 | flex md:flex-row- 좌측 8/12: ChatWindow- 우측 4/12: ExplanationPanel(스크롤 고정) |
| UI 텍스트 | “생년월일을 선택하세요” → “오늘의 사주를 계산 중입니다…” |

**3-3. 타로 채팅**

**/tarot**

| **동일 부분** | **내용** |
| --- | --- |
| 스프레드 선택 모달 | 라디오 버튼(3장·6장·10장) → “선택 완료” |
| 카드 영역 | 그리드 3 × ? (선택 수에 따라)·Hover Glow |
| 대표 이미지 | https://picsum.photos/seed/tarot/800/600 |

**3-4. 저장 운세**

**/readings**

| **항목** | **내용** |
| --- | --- |
| 목적 | 로컬 저장된 결과 썸네일 & 리스트 |
| 컴포넌트 | ReadingCard(Glass) · FilterTabs(사주/타로) |
| 레이아웃 | grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 |

**3-5. 프로필**

**/profile**

**& 설정**

**/settings**

- 프로필: Form Card 2단 (본인/지인) + “+ 추가” FAB
- 설정: 토글 스위치(다크 모드, 알림) · 드롭다운(언어) · TTS 미리듣기 버튼

**3-6. 프리미엄**

**/premium**

- 전문가 카드 슬라이더 · 상세 모달 · 결제 CTA

**3-7. 소개**

**/about**

- 타임라인 스크롤애니메이션 · 팀 소개 카드

**4. Layout Components**

| **컴포넌트** | **적용 라우트** | **핵심 요소** | **Responsive** |
| --- | --- | --- | --- |
| Sidebar | 모든 페이지 | Logo, NavItem, “로그인/OAuth” 버튼 | • Desktop: 고정 72px→280px 토글• Mobile: 왼쪽 슬라이드 드로어 |
| Header | /, /saju, /tarot 등 | PageTitle, 테마 토글 | 높이 56px · Sticky |
| ChatWindow | /saju, /tarot | MessageBubble, InputBar, TypingIndicator | Mobile: 100% width; Desktop: 70% |
| Modal | 스프레드 선택, 프로필 추가 | Dimmer, Focus Trap | 중앙 90 vw / max-w-md |

**5. Interaction Patterns**

| **패턴** | **설명** |
| --- | --- |
| 카드 Flip | hover:rotateY-180 transition-transform duration-700 |
| 실시간 스트림 | 메시지 5–10자 단위 append, Skeleton bubble 표시 |
| 토글 | Tailwind peer 체크박스로 모션 슬라이드 |
| 모달 ESC | keydown[Esc] 닫힘, aria-modal 선언 |

**6. Breakpoints**

$breakpoints: (

'mobile': 320px,

'tablet': 768px,

'desktop': 1024px,

'wide': 1440px

);

| **레이아웃 변화** | **mobile** | **tablet** | **desktop** | **wide** |
| --- | --- | --- | --- | --- |
| Sidebar | 숨김 | 드로어 | 72px 고정 | 280px 확장 토글 |
| Grid cols | 4 | 8 | 12 | 12+max-w-7xl |

**부록 · 이미지 활용 예시**

| **컴포넌트** | **예시 URL** |
| --- | --- |
| Hero BG | https://picsum.photos/seed/night-sky/1600/900 |
| 프로필 | https://picsum.photos/seed/avatar/200/200 |
| 전문가 카드 | https://picsum.photos/seed/expert1/400/300 |

위 이미지는 실제 배포 시 자체 촬영·라이선스 이미지로 교체 권장.