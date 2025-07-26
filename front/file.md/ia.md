# **정보 구조(IA) 문서 - 수정본**

변경 사항 요약

- 타로: 카드 수·스프레드 선택이 채팅 페이지 내부 UI로 이동 → 별도 디렉터리 삭제
- URL, 사이트 맵, 페이지·컴포넌트 계층을 이에 맞게 단순화

**1. 사이트 맵 (Site Map)**

/

├─ /saju                 ─ 사주 채팅

│  └─ /saju/report/:id   ─ PDF 리포트

├─ /tarot                ─ 타로 채팅 (카드 수·스프레드 선택 포함)

│  └─ /tarot/result/:id  ─ 결과 요약(선택)

├─ /readings             ─ 저장된 운세 (로그인 필요)

├─ /profile              ─ 생년월일 관리

├─ /settings             ─ 개인 설정 · 접근성

├─ /premium              ─ 프리미엄 상담

└─ /about                ─ 서비스 소개

- /tarot는 단일 페이지로 모든 타로 기능 제공.

**2. 사용자 플로우 (User Flow)**

| **시나리오** | **단계** | **세부 흐름** |
| --- | --- | --- |
| 타로 이용 (비로그인) | ① 랜딩 → ② 타로 선택 → ③ /tarot 진입 → ④ 스프레드·장 수 선택(모달) → ⑤ 카드 뽑기·채팅 → ⑥ 결과 확인 |  |
| 타로 이용 (로그인) | ① 로그인(OAuth) → ② /tarot → ③ 스프레드 선택 → ④ 채팅 → ⑤ 저장 또는 공유 |  |
| 사주 이용 | 변경 없음 (기존 흐름 유지) |  |

**3. 내비게이션 구조 (Navigation Structure)**

| **위치** | **항목** | **설명** |
| --- | --- | --- |
| 사이드바 | 홈 / 사주 / 타로 / 내 운세 / 프로필 / 설정 / 프리미엄 / 소개 | 타로 클릭 시 /tarot로 이동 (하위 메뉴 없음) |
| 헤더 | 페이지 타이틀·다크모드·로그인 | /tarot의 헤더에는 현재 선택된 스프레드(예: “3장 스프레드”)를 보조 텍스트로 표시 |

**4. 페이지 계층 (Page Hierarchy)**

| **레벨** | **페이지** | **설명** |
| --- | --- | --- |
| L1 | /saju, /tarot, /readings, /profile, /settings, /premium, /about | 최상위 |
| L2 | /saju/report/:id, /tarot/result/:id | 상세 결과 |
| L3 | — | 스프레드별 디렉터리 제거됨 |

**5. 콘텐츠 조직 (Content Organization)**

| **영역** | **구성** | **비고** |
| --- | --- | --- |
| /tarot 메인 | • 스프레드 선택 모달 (3장·6장·10장 …)• 카드 선택 영역• 챗봇 대화창 | 동일 페이지에서 상태 전환 |
| 결과 패널 | • 카드 해석·요약• 저장/공유 버튼 | 필요 시 /tarot/result/:id로 이동 가능 |

**6. 상호작용 패턴 (Interaction Patterns)**

| **패턴** | **위치** | **설명** |
| --- | --- | --- |
| 스프레드 선택 모달 | /tarot 진입 시 또는 상단 버튼 클릭 | 모달에서 카드 수 선택 후 ChatWindow 상태 업데이트 |
| 카드 플립 | 카드 선택 시 | 선택된 카드가 3D Flip → 메시지 버블에 해석 출력 |
| 대화 | ChatWindow | 실시간 스트리밍 답변 |

**7. URL 구조 (URL Structure)**

| **URL** | **용도** | **설명** |
| --- | --- | --- |
| /tarot | 타로 채팅 · 스프레드 선택 | QueryString 없이 단일 경로 |
| /tarot/result/:id | 결과 공유·저장 링크 | 로그인 여부 관계없이 접근 가능 (noindex) |
| /saju/report/:id | 사주 PDF 리포트 | 개인 정보 보호를 위해 만료 토큰 포함 가능 |

**8. 컴포넌트 계층 (Component Hierarchy)**

<Layout>

├─ <Sidebar>

├─ <Header/>

└─ <MainArea>

├─ <TarotPage>            (/tarot)

│   ├─ <SpreadSelectModal/>

│   ├─ <CardDeck>

│   │   └─ <TarotCard/>

│   └─ <ChatWindow>

│       └─ <MessageBubble/>

└─ <SajuPage>             (/saju)

└─ ...

- SpreadSelectModal는 <TarotPage>가 처음 마운트될 때 한 번 호출.
- 카드 수·스프레드 상태는 React Context 또는 Zustand로 전역 관리 → CardDeck·ChatWindow에서 공유.

**반응형·접근성·SEO 메모**

| **항목** | **고려 사항** |
| --- | --- |
| 반응형 | /tarot 한 화면 내 전환이 많으므로 모바일 상단 고정 탭바(“선택 → 카드 → 결과”) 제공 |
| 접근성 | • 모달에 aria-modal="true"• 스프레드 선택 라디오 그룹에 role="radiogroup" |
| SEO | 타로는 단일 URL이므로 동적 메타 태그(Next.js metadata)로 스프레드명 삽입 |