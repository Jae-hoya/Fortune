이 프로젝트는 [`EasyNext`](https://github.com/easynext/easynext)를 사용해 생성된 [Next.js](https://nextjs.org) 프로젝트입니다.

## Getting Started

개발 서버를 실행합니다.<br/>
환경에 따른 명령어를 사용해주세요.

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

브라우저에서 [http://localhost:3000](http://localhost:3000)을 열어 결과를 확인할 수 있습니다.

`app/page.tsx` 파일을 수정하여 페이지를 편집할 수 있습니다. 파일을 수정하면 자동으로 페이지가 업데이트됩니다.

## WebSocket 설정

사주 채팅 기능을 사용하려면 WebSocket 서버가 필요합니다. 다음 환경 변수를 설정해주세요:

### 1. 환경 변수 파일 생성

프로젝트 루트에 `.env.local` 파일을 생성하고 다음 내용을 추가하세요:

```bash
# WebSocket 서버 URL
NEXT_PUBLIC_WEBSOCKET_URL=ws://localhost:8000
```

### 2. WebSocket 서버 실행

WebSocket 서버가 실행 중이어야 합니다. 서버는 다음 엔드포인트를 제공해야 합니다:

- `ws://localhost:8000/ws/saju` - 사주 채팅용 WebSocket 엔드포인트

서버는 다음과 같은 메시지 형식을 지원해야 합니다:

**클라이언트 → 서버:**
```json
{
  "message": "사용자 질문 내용"
}
```

**서버 → 클라이언트 (스트리밍):**
```json
{
  "type": "stream",
  "content": "스트리밍 텍스트 조각"
}
```

**서버 → 클라이언트 (완료):**
```json
{
  "type": "complete"
}
```

## 사주 채팅 기능

`/saju` 페이지에서는 WebSocket을 통한 실시간 사주 상담이 가능합니다:

- 실시간 텍스트 스트리밍
- 연결 상태 표시
- 에러 처리 및 재연결
- 반응형 UI
- **세션별 채팅 분리**: URL 파라미터 `session_id`로 각 세션을 구분

### 세션 사용 방법

```
/saju?session_id=user123
/saju?session_id=consultation_2024_01_15
/saju?session_id=premium_user_abc
```

- `session_id`가 없으면 `default` 세션으로 설정됩니다
- 각 세션은 독립적인 채팅 히스토리를 가집니다
- WebSocket 연결 시 `session_id`가 서버로 전송됩니다

## 기본 포함 라이브러리

- [Next.js](https://nextjs.org)
- [React](https://react.dev)
- [Tailwind CSS](https://tailwindcss.com)
- [TypeScript](https://www.typescriptlang.org)
- [ESLint](https://eslint.org)
- [Prettier](https://prettier.io)
- [Shadcn UI](https://ui.shadcn.com)
- [Lucide Icon](https://lucide.dev)
- [date-fns](https://date-fns.org)
- [react-use](https://github.com/streamich/react-use)
- [es-toolkit](https://github.com/toss/es-toolkit)
- [Zod](https://zod.dev)
- [React Query](https://tanstack.com/query/latest)
- [React Hook Form](https://react-hook-form.com)
- [TS Pattern](https://github.com/gvergnaud/ts-pattern)
- [Zustand](https://github.com/pmndrs/zustand) - 상태 관리

## 사용 가능한 명령어

한글버전 사용

```sh
easynext lang ko
```

최신버전으로 업데이트

```sh
npm i -g @easynext/cli@latest
# or
yarn add -g @easynext/cli@latest
# or
pnpm add -g @easynext/cli@latest
```

Supabase 설정

```sh
easynext supabase
```

Next-Auth 설정

```sh
easynext auth

# ID,PW 로그인
easynext auth idpw
# 카카오 로그인
easynext auth kakao
```

유용한 서비스 연동

```sh
# Google Analytics
easynext gtag

# Microsoft Clarity
easynext clarity

# ChannelIO
easynext channelio

# Sentry
easynext sentry

# Google Adsense
easynext adsense
```
