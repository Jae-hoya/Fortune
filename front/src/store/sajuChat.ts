import { create } from "zustand"
import { WebSocketManager, getWebSocketUrl } from "@/lib/websocket"

export type ChatMessage = {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: number
}

interface ChatStoreType {
  messages: ChatMessage[]
  isLoading: boolean
  isConnected: boolean
  currentStreamingMessage: string
  wsManager: WebSocketManager | null
  currentSessionId: string | null
  // JSON 데이터 저장용
  lastJsonData: any | null
  addMessage: (role: "user" | "assistant", content: string) => void
  setIsLoading: (loading: boolean) => void
  setIsConnected: (connected: boolean) => void
  setCurrentStreamingMessage: (content: string) => void
  appendToStreamingMessage: (content: string) => void
  reset: () => void
  sendMessage: (content: string) => Promise<void>
  disconnect: () => void
  setCurrentSessionId: (sessionId: string) => void
  setLastJsonData: (data: any) => void
}

export const useSajuChatStore = create<ChatStoreType>((set, get) => ({
  messages: [],
  isLoading: false,
  isConnected: false,
  currentStreamingMessage: "",
  wsManager: null,
  currentSessionId: null,
  lastJsonData: null,
  addMessage: (role, content) =>
    set((state) => ({
      messages: [
        ...state.messages,
        {
          id: Math.random().toString(36).slice(2),
          role,
          content,
          timestamp: Date.now(),
        },
      ],
    })),
  setIsLoading: (loading) => set({ isLoading: loading }),
  setIsConnected: (connected) => set({ isConnected: connected }),
  setCurrentStreamingMessage: (content) => set({ currentStreamingMessage: content }),
  appendToStreamingMessage: (content) => 
    set((state) => ({ 
      currentStreamingMessage: state.currentStreamingMessage + content 
    })),
  reset: () => set({ messages: [], isLoading: false, currentStreamingMessage: "" }),
  disconnect: () => {
    const { wsManager } = get()
    if (wsManager) {
      wsManager.disconnect()
      set({ wsManager: null, isConnected: false })
    }
  },
  sendMessage: async (content: string) => {
    const { addMessage, setIsLoading, setCurrentStreamingMessage, appendToStreamingMessage, setIsConnected, wsManager, currentSessionId, setLastJsonData } = get()
    
    // sessionId가 없으면 에러
    if (!currentSessionId) {
      console.error('sessionId가 설정되지 않았습니다.')
      addMessage("assistant", "세션 ID가 설정되지 않았습니다. 페이지를 새로고침해주세요.")
      return
    }
    
    // 사용자 메시지 추가
    addMessage("user", content)
    setIsLoading(true)
    setCurrentStreamingMessage("")

    // --- ADD: fallback timeout for reused connection ---
    let loadingTimeout: NodeJS.Timeout | null = null
    const clearLoadingTimeout = () => {
      if (loadingTimeout) {
        clearTimeout(loadingTimeout)
        loadingTimeout = null
      }
    }

    try {
      // 같은 sessionId로 이미 연결되어 있는지 확인
      if (wsManager && wsManager.isConnected()) {
        console.log('기존 WebSocket 연결 재사용:', currentSessionId)
        const messageData = { message: content, session_id: currentSessionId }
        wsManager.send(messageData)
        // Fallback: 10초 후 강제 로딩 해제
        // loadingTimeout = setTimeout(() => {
        //   if (get().isLoading) {
        //     setIsLoading(false)
        //     addMessage("assistant", "서버 응답이 지연되고 있습니다. 다시 시도해 주세요.")
        //   }
        // }, 30000)
        return
      }

      // 기존 연결이 있으면 해제
      if (wsManager) {
        console.log('기존 WebSocket 연결 해제')
        wsManager.disconnect()
      }

      // 새로운 WebSocket 연결 (currentSessionId를 path로)
      const wsUrl = getWebSocketUrl("saju", currentSessionId)
      console.log('사주 채팅 WebSocket 연결 시도:', wsUrl)
      
      const newWsManager = new WebSocketManager(
        wsUrl,
        (data) => {
          // 수신 데이터와 타입을 로그로 출력
          console.log('사주 채팅 WebSocket 수신 데이터:', data, 'content 타입:', typeof data.content)
          // 데이터 타입에 따른 처리
          if (data.type === 'stream' && data.content !== undefined && data.content !== null) {
            appendToStreamingMessage(String(data.content))
          } else if (data.type === 'complete') {
            const { currentStreamingMessage } = get()
            if (currentStreamingMessage) {
              addMessage("assistant", currentStreamingMessage)
            }
            setCurrentStreamingMessage("")
            setIsLoading(false)
            clearLoadingTimeout() // <- ADD
            console.log('스트리밍 후 isLoading:', get().isLoading)
          } else if (data.type === 'error') {
            // 에러 처리
            console.error('서버에서 에러 응답:', data)
            const { currentStreamingMessage } = get()
            if (currentStreamingMessage) {
              addMessage("assistant", currentStreamingMessage)
              setCurrentStreamingMessage("")
            }
            setIsLoading(false)
            clearLoadingTimeout() // <- ADD
            addMessage("assistant", data.content || "죄송합니다. 처리 중 오류가 발생했습니다.")
            newWsManager.disconnect()
            set({ wsManager: null })
          } else if (data.type === 'json') {
            // JSON 데이터 처리
            console.log('JSON 데이터 수신:', data)
            setLastJsonData(data.content || data)
          } else {
            // 기본 텍스트 처리 (type이 없거나 stream이 아닌 경우)
            console.log('기본 텍스트 데이터:', data.content)
            if (data.content) {
              appendToStreamingMessage(data.content)
            }
          }
        },
        () => {
          // 연결 성공
          console.log('사주 채팅 WebSocket 연결 성공')
          setIsConnected(true)
          const messageData = { message: content, session_id: currentSessionId }
          console.log('메시지 전송:', messageData)
          newWsManager.send(messageData)
        },
        () => {
          // 연결 종료
          console.log('사주 채팅 WebSocket 연결 종료')
          setIsConnected(false)
          
          // 연결 종료 시 현재 스트리밍 메시지 완료
          const { currentStreamingMessage } = get()
          if (currentStreamingMessage) {
            addMessage("assistant", currentStreamingMessage)
            setCurrentStreamingMessage("")
          }
          
          // 로딩 상태 해제
          setIsLoading(false)
          clearLoadingTimeout() // <- ADD
          set({ wsManager: null })
        },
        (error) => {
          // 연결 에러
          console.error('사주 채팅 WebSocket 에러:', error)
          console.error('에러 타입:', error.type)
          console.error('에러 타겟:', error.target)
          
          setIsLoading(false)
          clearLoadingTimeout() // <- ADD
          setIsConnected(false)
          
          // 연결 종료 시 현재 스트리밍 메시지 완료
          const { currentStreamingMessage } = get()
          if (currentStreamingMessage) {
            addMessage("assistant", currentStreamingMessage)
            setCurrentStreamingMessage("")
          }
          
          // 에러 메시지 개선
          let errorMessage = "죄송합니다. 연결에 문제가 발생했습니다."
          if (error.type === 'error') {
            errorMessage = "서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요."
          }
          
          addMessage("assistant", errorMessage)
          set({ wsManager: null })
        }
      )

      set({ wsManager: newWsManager })
      newWsManager.connect()

    } catch (error) {
      console.error("메시지 전송 에러:", error)
      setIsLoading(false)
      clearLoadingTimeout() // <- ADD
      setCurrentStreamingMessage("") // ← Clear streaming message
      addMessage("assistant", "죄송합니다. 서버와의 연결에 문제가 발생했습니다.")
    }
  },
  setCurrentSessionId: (sessionId) => set({ currentSessionId: sessionId }),
  setLastJsonData: (data) => set({ lastJsonData: data })
}))
