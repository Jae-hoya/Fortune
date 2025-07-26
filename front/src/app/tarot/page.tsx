"use client"

import { useRef, useEffect, useState, useCallback } from "react"
import { format } from "date-fns"
import { ko } from "date-fns/locale"
import { Card } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { useSidebarStore } from "@/store/sidebar"
import { SIDEBAR_WIDTH } from "@/components/layout/Sidebar"
import { useTarotChatStore } from "@/store/tarotChat"
import { v4 as uuidv4 } from "uuid"

export default function TarotPage() {
  const [sessionId, setSessionId] = useState<string>("")
  const [isSessionReady, setIsSessionReady] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const isSidebarOpen = useSidebarStore((state) => state.isOpen)
  const left = isSidebarOpen ? SIDEBAR_WIDTH : 0
  const width = isSidebarOpen ? `calc(100vw - ${SIDEBAR_WIDTH}px)` : "100vw"
  const transition = "all 0.3s"
  const [isUserAtBottom, setIsUserAtBottom] = useState(true)

  // Zustand store for tarot chat
  const {
    messages,
    addMessage,
    isLoading,
    setIsLoading,
    sendMessage,
    isConnected,
    currentStreamingMessage,
    disconnect,
    reset,
    setCurrentSessionId,
    lastJsonData,
    finalStateData // <--- Add this line
  } = useTarotChatStore()

  // Session ID management (similar to Saju)
  useEffect(() => {
    const stored = window.sessionStorage.getItem("tarot_session_id")
    if (stored) {
      setSessionId(stored)
      setCurrentSessionId(stored)
    } else {
      const newId = uuidv4()
      window.sessionStorage.setItem("tarot_session_id", newId)
      setSessionId(newId)
      setCurrentSessionId(newId)
    }
    setIsSessionReady(true)
  }, [])

  // Scroll event handler to track if user is at bottom
  const handleScroll = useCallback(() => {
    const scrollArea = scrollAreaRef.current
    if (!scrollArea) return
    const isAtBottom =
      scrollArea.scrollHeight - scrollArea.scrollTop - scrollArea.clientHeight < 50
    setIsUserAtBottom(isAtBottom)
  }, [])

  // Attach scroll event listener
  useEffect(() => {
    const scrollArea = scrollAreaRef.current
    if (!scrollArea) return
    scrollArea.addEventListener("scroll", handleScroll)
    // Set initial state
    handleScroll()
    return () => {
      scrollArea.removeEventListener("scroll", handleScroll)
    }
  }, [handleScroll])

  // Auto-scroll only if user is at bottom
  useEffect(() => {
    const scrollArea = scrollAreaRef.current
    if (!scrollArea) return
    if (isUserAtBottom) {
      scrollArea.scrollTop = scrollArea.scrollHeight
    }
  }, [messages, currentStreamingMessage, isUserAtBottom])

  // JSON 데이터 처리 (사주 페이지 참고)
  useEffect(() => {
    if (lastJsonData) {
      console.log('타로 JSON 데이터 처리:', lastJsonData)
      if (lastJsonData.type === 'error') {
        addMessage("assistant", lastJsonData.message || "처리 중 오류가 발생했습니다.")
      }
      // 추가적인 JSON 데이터 처리는 여기에 구현
    }
  }, [lastJsonData, addMessage])

  // Handle final_state JSON data
  useEffect(() => {
    if (finalStateData) {
      console.log('final_state 데이터 수신:', finalStateData)
      // TODO: Use finalStateData as needed in this page
    }
  }, [finalStateData])

  // Debug log for isLoading state
  useEffect(() => {
    console.log('[TarotPage] isLoading changed:', isLoading)
  }, [isLoading])

  useEffect(() => {
    if (!sessionId) return
    // Only add assistant welcome message if not present
    const hasAssistant = messages.some((msg) => msg.role === "assistant")
    if (!hasAssistant) {
      addMessage(
        "assistant",
        `안녕하세요! 타로 리딩을 도와드릴게요. (세션: ${sessionId})\n어떤 질문이 있으신가요?`
      )
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId])

  useEffect(() => {
    return () => {
      disconnect()
    }
  }, [disconnect])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const content = textareaRef.current?.value.trim()
    if (!content || isLoading) return
    textareaRef.current.value = ""
    await sendMessage(content)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  if (!isSessionReady || !sessionId) {
    return (
      <div className="flex min-h-screen bg-white dark:bg-gradient-to-br dark:from-slate-950 dark:to-slate-900 transition-colors items-center justify-center">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce [animation-delay:-0.3s]" />
          <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce [animation-delay:-0.15s]" />
          <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" />
          <span className="text-sm text-gray-600 dark:text-gray-400">세션 로딩 중...</span>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col min-h-screen bg-white dark:bg-gradient-to-br dark:from-slate-950 dark:to-slate-900 transition-colors">
      {/* 헤더 */}
      <div className="fixed top-0 left-0 right-0 z-20 bg-white dark:bg-slate-900 border-b border-gray-200 dark:border-slate-700 px-4 py-2" style={{ left, width, transition }}>
        <div className="max-w-2xl mx-auto flex items-center justify-between">
          <div className="text-lg font-semibold text-purple-800 dark:text-purple-300">
            타로 리딩
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-500">
            {format(new Date(), "yyyy.MM.dd HH:mm", { locale: ko })}
          </div>
        </div>
      </div>

      {/* 채팅 메시지 영역 */}
      <div
        ref={scrollAreaRef}
        onScroll={handleScroll}
        className="fixed overflow-y-auto px-1 pt-20 pb-32 bg-white dark:bg-transparent transition-colors"
        style={{
          left,
          width,
          top: 72,
          bottom: 96,
          height: "calc(100vh - 72px - 96px)",
          transition,
          maxWidth: "100vw",
          zIndex: 10,
        }}
      >
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex w-full mb-8 ${message.role === "user" ? "justify-end" : "justify-start"}`}
            style={{ paddingTop: "0.5rem" }}
          >
            <div
              className={`max-w-[80vw] px-3 py-2 rounded-2xl shadow-lg
                ${message.role === "user"
                  ? "bg-gradient-to-r from-purple-500 to-fuchsia-500 text-white rounded-br-none mr-[10vw] dark:from-purple-700 dark:to-fuchsia-700"
                  : "bg-white dark:bg-slate-950/90 text-purple-700 dark:text-purple-300 border border-gray-200 dark:border-slate-800 rounded-bl-none ml-[10vw]"}
              `}
            >
              <div className="text-sm font-semibold mb-1">
                {message.role === "user" ? "나" : "타로 리더"}
              </div>
              <div className="whitespace-pre-line">{message.content}</div>
              <div className="text-xs text-right mt-1 opacity-60">
                {format(new Date(message.timestamp), "a h:mm", { locale: ko })}
              </div>
            </div>
          </div>
        ))}

        {/* 스트리밍 메시지 표시 */}
        {isLoading && currentStreamingMessage && (
          <div className="flex w-full mb-8 justify-start" style={{ paddingTop: "0.5rem" }}>
            <div className="max-w-[80vw] px-3 py-2 rounded-2xl shadow-lg bg-white dark:bg-slate-950/90 text-purple-700 dark:text-purple-300 border border-gray-200 dark:border-slate-800 rounded-bl-none ml-[10vw]">
              <div className="text-sm font-semibold mb-1">타로 리더</div>
              <div className="whitespace-pre-line">
                {currentStreamingMessage}
                <span className="animate-pulse">▋</span>
              </div>
              <div className="text-xs text-right mt-1 opacity-60">
                {format(new Date(), "a h:mm", { locale: ko })}
              </div>
            </div>
          </div>
        )}

        {/* 로딩 인디케이터 */}
        {isLoading && !currentStreamingMessage && (
          <div className="flex w-full mb-8 justify-start" style={{ paddingTop: "0.5rem" }}>
            <Card className="p-4 ml-[10vw]">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce [animation-delay:-0.3s]" />
                <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce [animation-delay:-0.15s]" />
                <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" />
              </div>
            </Card>
          </div>
        )}

        {/* 연결 상태 표시 */}
        {!isConnected && messages.length > 1 && (
          <div className="flex w-full mb-4 justify-center">
            <div className="bg-yellow-100 dark:bg-yellow-900/20 text-yellow-800 dark:text-yellow-200 px-3 py-1 rounded-full text-xs flex items-center gap-2">
              <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse"></div>
              서버에 연결 중...
            </div>
          </div>
        )}

        {/* 연결 성공 표시 */}
        {isConnected && (
          <div className="flex w-full mb-4 justify-center">
            <div className="bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-200 px-3 py-1 rounded-full text-xs flex items-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              연결됨
            </div>
          </div>
        )}

        {/* 서버 연결 실패 안내 */}
        {!isConnected && messages.length > 1 && messages.some(msg => msg.content.includes("서버에 연결할 수 없습니다")) && (
          <div className="flex w-full mb-4 justify-center">
            <div className="bg-red-100 dark:bg-red-900/20 text-red-800 dark:text-red-200 px-4 py-2 rounded-lg text-sm max-w-md text-center">
              <div className="font-semibold mb-1">서버 연결 실패</div>
              <div className="text-xs">
                WebSocket 서버가 실행되지 않았습니다.<br/>
                <a href="/test-websocket" className="text-blue-600 dark:text-blue-400 underline">
                  연결 테스트 페이지
                </a>에서 서버 상태를 확인해보세요.
              </div>
            </div>
          </div>
        )}
      </div>
      
      {/* 입력창 */}
      <form
        onSubmit={handleSubmit}
        className="fixed bottom-0 left-0 right-0 w-full flex justify-center items-center bg-white dark:bg-slate-900 py-4 border-t border-gray-200 dark:border-slate-700 z-50 transition-colors"
        style={{ left, width, transition: "all 0.3s" }}
      >
        <div className="w-full max-w-2xl flex items-center bg-gray-50 dark:bg-slate-800 rounded-2xl shadow px-4 py-2 mx-auto transition-colors">
          <Textarea
            ref={textareaRef}
            placeholder={isLoading ? "응답을 기다리는 중..." : "질문을 입력하세요..."}
            className="min-h-[48px] max-h-40 flex-1 resize-none border border-gray-200 dark:border-slate-700 bg-white dark:bg-slate-900 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-purple-400 dark:focus:ring-purple-700 px-4 py-3 text-base text-slate-900 dark:text-slate-100 placeholder:text-gray-400 dark:placeholder:text-slate-500 placeholder:text-center text-left transition-colors"
            onKeyDown={handleKeyDown}
            disabled={isLoading}
          />
        </div>
      </form>
    </div>
  )
}
