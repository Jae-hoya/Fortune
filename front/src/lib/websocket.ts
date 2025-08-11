// WebSocket 연결 관리를 위한 유틸리티

export interface WebSocketMessage {
  type: 'stream' | 'complete' | 'error' | 'json'
  content?: string
  message?: string
  session_id?: string
}

export class WebSocketManager {
  private ws: WebSocket | null = null
  private url: string
  private onMessage: (data: WebSocketMessage) => void
  private onOpen: () => void
  private onClose: () => void
  private onError: (error: Event) => void

  constructor(
    url: string,
    onMessage: (data: WebSocketMessage) => void,
    onOpen: () => void,
    onClose: () => void,
    onError: (error: Event) => void
  ) {
    this.url = url
    this.onMessage = onMessage
    this.onOpen = onOpen
    this.onClose = onClose
    this.onError = onError
  }

  connect(): void {
    try {
      console.log('WebSocket 연결 시도:', this.url)
      
      // 브라우저에서 WebSocket 지원 확인
      if (typeof WebSocket === 'undefined') {
        throw new Error('WebSocket이 지원되지 않는 환경입니다.')
      }

      this.ws = new WebSocket(this.url)
      
      this.ws.onopen = () => {
        console.log('WebSocket 연결 성공:', this.url)
        this.onOpen()
      }

      this.ws.onmessage = (event) => {
        try {
          console.log('WebSocket 메시지 수신:', event.data)
          
          // 먼저 JSON 파싱을 시도
          try {
            const parsedData = JSON.parse(event.data)
            console.log('JSON 파싱 성공:', parsedData)
            // JSON 데이터인 경우 type을 포함하여 전달
            const data: WebSocketMessage = {
              type: parsedData.type || 'json',
              content: parsedData.content || event.data,
              ...parsedData
            }
            this.onMessage(data)
          } catch (parseError) {
            // JSON 파싱 실패 시 텍스트 데이터로 처리
            console.log('텍스트 데이터로 처리:', event.data)
            const data: WebSocketMessage = { 
              type: 'stream', 
              content: event.data 
            }
            this.onMessage(data)
          }
        } catch (error) {
          console.error('WebSocket 메시지 처리 에러:', error, '원본 데이터:', event.data)
        }
      }

      this.ws.onclose = (event) => {
        console.log('WebSocket 연결 종료:', event.code, event.reason)
        this.onClose()
      }

      this.ws.onerror = (error) => {
        console.error('WebSocket 에러 발생:', error)
        console.error('WebSocket URL:', this.url)
        console.error('WebSocket 상태:', this.ws?.readyState)
        this.onError(error)
      }
    } catch (error) {
      console.error('WebSocket 연결 실패:', error)
      this.onError(error as Event)
    }
  }

  send(message: string | object): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      const messageToSend = typeof message === 'string' ? message : JSON.stringify(message)
      console.log('WebSocket 메시지 전송:', messageToSend)
      this.ws.send(messageToSend)
    } else {
      console.error('WebSocket이 연결되지 않았습니다. 상태:', this.ws?.readyState)
      console.error('WebSocket URL:', this.url)
    }
  }

  disconnect(): void {
    if (this.ws) {
      console.log('WebSocket 연결 해제')
      this.ws.close()
      this.ws = null
    }
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }

  getState(): number | null {
    return this.ws?.readyState || null
  }
}

// 환경 변수에서 WebSocket URL 가져오기
export const getWebSocketUrl = (type: string, sessionId: string): string => {
  // ws://localhost:8000/ws/chat/{type}/{sessionId} 형식으로 반환
  const url = `ws://localhost:8000/ws/chat/${type}/${encodeURIComponent(sessionId)}`;
  console.log('WebSocket URL 최종:', url);
  return url;
} 