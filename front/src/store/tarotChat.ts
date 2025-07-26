import { create } from "zustand"
import { WebSocketManager, getWebSocketUrl } from "@/lib/websocket"

export type ChatMessage = {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: number
}

interface TarotChatStoreType {
  messages: ChatMessage[]
  isLoading: boolean
  isConnected: boolean
  currentStreamingMessage: string
  wsManager: WebSocketManager | null
  currentSessionId: string | null
  lastJsonData: any | null
  finalStateData: any | null
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
  setFinalStateData: (data: any) => void
}

export const useTarotChatStore = create<TarotChatStoreType>((set, get) => ({
  messages: [],
  isLoading: false,
  isConnected: false,
  currentStreamingMessage: "",
  wsManager: null,
  currentSessionId: null,
  lastJsonData: null,
  finalStateData: null,
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
    const { addMessage, setIsLoading, setCurrentStreamingMessage, appendToStreamingMessage, setIsConnected, wsManager, currentSessionId, setLastJsonData, setFinalStateData } = get()
    
    console.log('[sendMessage] called. isLoading:', get().isLoading, 'wsManager:', wsManager, 'currentSessionId:', currentSessionId)

    if (!currentSessionId) {
      console.error('sessionId가 설정되지 않았습니다.')
      addMessage("assistant", "세션 ID가 설정되지 않았습니다. 페이지를 새로고침해주세요.")
      return
    }
    
    addMessage("user", content)
    setIsLoading(true)
    setCurrentStreamingMessage("")

    try {
      if (wsManager && wsManager.isConnected()) {
        console.log('[sendMessage] Reusing existing WebSocket connection.')
        const messageData = { message: content, session_id: currentSessionId }
        wsManager.send(messageData)
        return
      }

      if (wsManager) {
        console.log('[sendMessage] Disconnecting old WebSocket connection.')
        wsManager.disconnect()
      }

      // Tarot WebSocket 연결
      const wsUrl = getWebSocketUrl("tarot", currentSessionId)
      console.log('[sendMessage] Creating new WebSocketManager:', wsUrl)
      
      const newWsManager = new WebSocketManager(
        wsUrl,
        (data) => {
          console.log('[WebSocket onMessage] data:', data)
          // Branching logic for incoming data
          if (typeof data === "string") {
            // Plain text: show as assistant message
            addMessage("assistant", data)
            setIsLoading(false)
            setCurrentStreamingMessage("")
            return
          }
          if (typeof data === "object" && data !== null) {
            // Accept both string and object for type, to handle 'final_state' as a string
            if ((data as any).type === 'final_state') {
              set((state) => ({
                finalStateData: data
                // Do NOT clear currentStreamingMessage or set isLoading here
              }))
              return
            }
            if (data.type === 'stream' && data.content) {
              appendToStreamingMessage(data.content)
            } else if (data.type === 'complete') {
              set((state) => {
                if (state.currentStreamingMessage) {
                  const newMessages = [
                    ...state.messages,
                    {
                      id: Math.random().toString(36).slice(2),
                      role: "assistant" as const,
                      content: state.currentStreamingMessage,
                      timestamp: Date.now(),
                    },
                  ]
                  console.log('[Zustand] Assistant message added:', state.currentStreamingMessage)
                  return {
                    messages: newMessages,
                    currentStreamingMessage: "",
                    isLoading: false,
                  }
                }
                return { currentStreamingMessage: "", isLoading: false }
              })
              return
            } else if (data.type === 'error') {
              console.error('서버에서 에러 응답:', data)
              const { currentStreamingMessage } = get()
              if (currentStreamingMessage) {
                addMessage("assistant", currentStreamingMessage)
                setCurrentStreamingMessage("")
              }
              setIsLoading(false)
              addMessage("assistant", data.content || "죄송합니다. 처리 중 오류가 발생했습니다.")
              newWsManager.disconnect()
              set({ wsManager: null, currentSessionId: null })
            } else if (data.type === 'json') {
              console.log('JSON 데이터 수신:', data)
              setLastJsonData(data.content || data)
            } else if (data.content) {
              appendToStreamingMessage(data.content)
            }
            return
          }
        },
        () => {
          console.log('[WebSocket onOpen] 연결 성공')
          setIsConnected(true)
          const messageData = { message: content, session_id: currentSessionId }
          console.log('[WebSocket onOpen] 메시지 전송:', messageData)
          newWsManager.send(messageData)
        },
        () => {
          console.log('[WebSocket onClose] 연결 종료')
          setIsConnected(false)
          const { currentStreamingMessage } = get()
          if (currentStreamingMessage) {
            addMessage("assistant", currentStreamingMessage)
            setCurrentStreamingMessage("")
          }
          setIsLoading(false)
          set({ wsManager: null, currentSessionId: null })
        },
        (error) => {
          console.error('[WebSocket onError] 에러 발생:', error)
          setIsLoading(false)
          setIsConnected(false)
          const { currentStreamingMessage } = get()
          if (currentStreamingMessage) {
            addMessage("assistant", currentStreamingMessage)
            setCurrentStreamingMessage("")
          }
          let errorMessage = "죄송합니다. 연결에 문제가 발생했습니다."
          if (error.type === 'error') {
            errorMessage = "서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요."
          }
          addMessage("assistant", errorMessage)
          set({ wsManager: null, currentSessionId: null })
        }
      )

      set({ wsManager: newWsManager })
      newWsManager.connect()

      console.log('[sendMessage] New WebSocketManager set and connect called.')

    } catch (error) {
      console.error('[sendMessage] 메시지 전송 에러:', error)
      setIsLoading(false)
      addMessage("assistant", "죄송합니다. 서버와의 연결에 문제가 발생했습니다.")
    }
  },
  setCurrentSessionId: (sessionId) => set({ currentSessionId: sessionId }),
  setLastJsonData: (data) => set({ lastJsonData: data }),
  setFinalStateData: (data) => set({ finalStateData: data })
}))
