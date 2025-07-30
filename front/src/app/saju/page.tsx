"use client"

import { useRef, useEffect, useState } from "react"
import { useSearchParams } from "next/navigation"
import { v4 as uuidv4 } from "uuid"
import { format } from "date-fns"
import { ko } from "date-fns/locale"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Card } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Textarea } from "@/components/ui/textarea"
import { Markdown } from "@/components/ui/markdown"
import { useSajuChatStore } from "@/store/sajuChat"
import { useSidebarStore } from "@/store/sidebar"
import { SIDEBAR_WIDTH } from "@/components/layout/Sidebar"

export default function SajuPage() {
  const searchParams = useSearchParams()
  const urlSessionId = searchParams.get("session_id")
  const [sessionId, setSessionId] = useState<string>("")
  const [isSessionReady, setIsSessionReady] = useState(false)

  // 최초 1회만 session ID 생성/설정
  useEffect(() => {
    if (urlSessionId) {
      // URL에 session_id가 있으면 사용
      setSessionId(urlSessionId)
      setCurrentSessionId(urlSessionId)
    } else {
      // URL에 session_id가 없으면 저장된 것 사용하거나 새로 생성
      const stored = window.sessionStorage.getItem("saju_session_id")
      if (stored) {
        setSessionId(stored)
        setCurrentSessionId(stored)
      } else {
        const newId = uuidv4()
        window.sessionStorage.setItem("saju_session_id", newId)
        setSessionId(newId)
        setCurrentSessionId(newId)
      }
    }
    setIsSessionReady(true)
  }, [urlSessionId]) // urlSessionId가 변경될 때만 실행

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
    lastJsonData
  } = useSajuChatStore()
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const isSidebarOpen = useSidebarStore((state) => state.isOpen)

  // 사이드바 열림/닫힘에 따라 동적 스타일 적용
  const sidebarWidth = SIDEBAR_WIDTH
  const left = isSidebarOpen ? sidebarWidth : 0
  const width = isSidebarOpen ? `calc(100vw - ${sidebarWidth}px)` : "100vw"
  const transition = "all 0.3s"

  useEffect(() => {
    const scrollArea = scrollAreaRef.current;
    if (!scrollArea) return;

    // 현재 스크롤이 하단(50px 이내)에 있는지 확인
    const isAtBottom =
      scrollArea.scrollHeight - scrollArea.scrollTop - scrollArea.clientHeight < 50;

    // 새 메시지가 도착했고, 사용자가 하단에 있을 때만 자동 스크롤
    if (isAtBottom) {
      scrollArea.scrollTop = scrollArea.scrollHeight;
    }
  }, [messages, currentStreamingMessage]);

  // 로딩 상태 디버깅
  useEffect(() => {
    console.log('로딩 상태 변경:', isLoading, '현재 시간:', new Date().toISOString())
  }, [isLoading])

  // JSON 데이터 처리
  useEffect(() => {
    if (lastJsonData) {
      console.log('사주 JSON 데이터 처리:', lastJsonData)
      
      // JSON 데이터 타입에 따른 분기 처리
      if (lastJsonData.type === 'error') {
        // 에러 처리
        addMessage("assistant", lastJsonData.message || "처리 중 오류가 발생했습니다.")
      }
      // 사주 페이지에서는 추가적인 JSON 데이터 처리를 여기에 구현
      // 예: 사주 결과, 운세 정보 등
    }
  }, [lastJsonData, addMessage])

  useEffect(() => {
    if (!sessionId) return
    
    // URL 파라미터로 새로운 session_id가 들어올 때만 reset
    if (urlSessionId && urlSessionId !== sessionId) {
      console.log('새로운 session_id로 변경:', urlSessionId)
      reset()
      // 새로운 session_id로 저장
      window.sessionStorage.setItem("saju_session_id", urlSessionId)
      setCurrentSessionId(urlSessionId)
    }
    
    // 이미 assistant 메시지가 있는지 체크 (새로운 session이거나 처음 접속한 경우만)
    const hasAssistant = messages.some((msg) => msg.role === "assistant")
    if (!hasAssistant) {
      addMessage(
        "assistant",
        `안녕하세요! 사주 리딩을 도와드릴게요. \n어떤 질문이 있으신가요?` //(세션: ${sessionId})
      )
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, urlSessionId])

  // 컴포넌트 언마운트 시 WebSocket 연결 정리
  useEffect(() => {
    return () => {
      disconnect()
    }
  }, [disconnect])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const content = textareaRef.current?.value.trim()
    if (!content || isLoading) return

    // 입력창 초기화
    textareaRef.current.value = ""

    // WebSocket을 통해 메시지 전송 (현재 sessionId 사용)
    await sendMessage(content)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  // sessionId가 로딩 중이면 로딩 표시
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
    <div className="flex min-h-screen bg-white dark:bg-gradient-to-br dark:from-slate-950 dark:to-slate-900 transition-colors">
      {/* 채팅 영역 */}
      <div className="flex flex-col flex-1">
        {/* 세션 정보 표시 */}
        <div className="fixed top-0 left-0 right-0 z-20 bg-white dark:bg-slate-900 border-b border-gray-200 dark:border-slate-700 px-4 py-2" style={{ left, width, transition }}>
          <div className="max-w-2xl mx-auto flex items-center justify-between">
            <div className="text-sm text-gray-600 dark:text-gray-400">
              세션 ID: <span className="font-mono text-purple-600 dark:text-purple-400">{sessionId}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="text-xs text-gray-500 dark:text-gray-500">
                {format(new Date(), "yyyy.MM.dd HH:mm", { locale: ko })}
              </div>
              {isLoading && (
                <button
                  onClick={() => {
                    console.log('강제 로딩 해제 - 현재 스트리밍 메시지:', currentStreamingMessage)
                    setIsLoading(false)
                    if (currentStreamingMessage) {
                      addMessage("assistant", currentStreamingMessage)
                    }
                  }}
                  className="text-xs bg-red-500 text-white px-2 py-1 rounded"
                >
                  로딩 해제
                </button>
              )}
            </div>
          </div>
        </div>

        {/* 채팅 메시지 영역 */}
        <div
          ref={scrollAreaRef}
          className="fixed overflow-y-auto px-1 pt-20 pb-56 bg-white dark:bg-transparent transition-colors"
          style={{
            left,
            width,
            top: 72, // 헤더(56px) + 추가 여백(16px)
            bottom: 96, // 입력창 높이(예시)
            height: "calc(100vh - 72px - 96px)", // 헤더+여백+입력창 높이만큼 제외
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
                  {message.role === "user" ? "나" : "사주 마스터"}
                </div>
                {message.role === "user" ? (
                  <div className="whitespace-pre-line">{message.content}</div>
                ) : (
                  <Markdown>{message.content}</Markdown>
                )}
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
                <div className="text-sm font-semibold mb-1">사주 마스터</div>
                <div className="whitespace-pre-line">
                  <Markdown>{currentStreamingMessage}</Markdown>
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
    </div>
  )
}
