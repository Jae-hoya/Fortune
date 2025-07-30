"use client"

import { useRef, useEffect, useState, useCallback } from "react"
import { format } from "date-fns"
import { ko } from "date-fns/locale"
import { Card } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Button } from "@/components/ui/button"
import { Markdown } from "@/components/ui/markdown"
import { useSidebarStore } from "@/store/sidebar"
import { SIDEBAR_WIDTH } from "@/components/layout/Sidebar"
import { useTarotChatStore } from "@/store/tarotChat"
import { v4 as uuidv4 } from "uuid"
import { getCardImagePath, getCardName } from "@/lib/tarot-utils"

export default function TarotPage() {
  const [sessionId, setSessionId] = useState<string>("")
  const [isSessionReady, setIsSessionReady] = useState(false)
  const [showSpreadSelection, setShowSpreadSelection] = useState(false)
  const [showCardSelection, setShowCardSelection] = useState(false)
  const [recommendedSpreads, setRecommendedSpreads] = useState<any[]>([])
  const [selectedCards, setSelectedCards] = useState<number[]>([])
  const [requiredCardCount, setRequiredCardCount] = useState(0)
  const [animatedCards, setAnimatedCards] = useState<number[]>([])
  const [flippedCards, setFlippedCards] = useState<number[]>([])
  const [shuffledCards, setShuffledCards] = useState<number[]>([])
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
    finalStateData
  } = useTarotChatStore()

  // Session ID management - Always create new session on page reload
  useEffect(() => {
    // Always create a new session ID on page reload
      const newId = uuidv4()
      window.sessionStorage.setItem("tarot_session_id", newId)
      setSessionId(newId)
      setCurrentSessionId(newId)
    // Reset chat store for new session
    reset()
    setIsSessionReady(true)
  }, [reset])

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
      
      // Check if consultation_data.status is "spread_selection"
      if (finalStateData.state?.consultation_data?.status === "spread_selection") {
        console.log('스프레드 선택 모드 활성화')
        setShowSpreadSelection(true)
        setShowCardSelection(false)
        setSelectedCards([])
        setRecommendedSpreads(finalStateData.state.consultation_data.recommended_spreads || [])
      } 
      // Check if consultation_data.status is "card_selection"
      else if (finalStateData.state?.consultation_data?.status === "card_selection") {
        console.log('카드 선택 모드 활성화')
        setShowSpreadSelection(false)
        setShowCardSelection(true)
        setSelectedCards([])
        setAnimatedCards([])
        setFlippedCards([])
        setRequiredCardCount(finalStateData.state.consultation_data.selected_spread?.card_count || 0)
        
        // Shuffle cards before displaying
        const shuffled = shuffleCards()
        setShuffledCards(shuffled)
        
        // Start card spreading animation
        setTimeout(() => {
          const totalCards = 78
          const animationDuration = 2000 // 2 seconds total
          const delayPerCard = animationDuration / totalCards
          
          for (let i = 0; i < totalCards; i++) {
            setTimeout(() => {
              setAnimatedCards(prev => [...prev, shuffled[i]])
            }, i * delayPerCard)
          }
        }, 100)
      } else {
        setShowSpreadSelection(false)
        setShowCardSelection(false)
      }
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
        `안녕하세요! 타로 리딩을 도와드릴게요. \n어떤 질문이 있으신가요?` //(세션: ${sessionId})
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

  const handleSpreadSelection = async (index: number) => {
    console.log('스프레드 선택 (위치):', index + 1)
    setShowSpreadSelection(false)
    await sendMessage((index + 1).toString())
  }

  const handleCardSelection = (cardIndex: number) => {
    setSelectedCards(prev => {
      const isSelected = prev.includes(cardIndex)
      if (isSelected) {
        // Remove card if already selected
        setFlippedCards(flipped => flipped.filter(card => card !== cardIndex))
        return prev.filter(card => card !== cardIndex)
      } else {
        // Add card if not selected and under limit
        if (prev.length < requiredCardCount) {
          // Trigger flip animation
          setTimeout(() => {
            setFlippedCards(flipped => [...flipped, cardIndex])
          }, 300) // Start flip after selection
          return [...prev, cardIndex].sort((a, b) => a - b)
        }
        return prev
      }
    })
  }

  const handleCardSelectionSubmit = async () => {
    if (selectedCards.length === requiredCardCount) {
      console.log('선택된 카드들:', selectedCards)
      setShowCardSelection(false)
      const cardString = selectedCards.join(', ')
      await sendMessage(cardString)
    }
  }

  // Fisher-Yates shuffle algorithm
  const shuffleCards = () => {
    const cards = Array.from({ length: 78 }, (_, i) => i + 1)
    for (let i = cards.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[cards[i], cards[j]] = [cards[j], cards[i]]
    }
    return cards
  }

  const handleCardSelectionReset = () => {
    setSelectedCards([])
    setFlippedCards([])
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
        className="fixed overflow-y-auto px-1 pt-20 bg-white dark:bg-transparent transition-colors"
        style={{
          left,
          width,
          top: 72,
          bottom: showCardSelection ? 600 : 96, // 카드 선택 UI가 활성화되면 더 많은 공간 확보
          height: showCardSelection ? "calc(100vh - 72px - 600px)" : "calc(100vh - 72px - 96px)",
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
              <div className="whitespace-pre-line">
                {message.role === "assistant" ? (
                  <Markdown>{message.content}</Markdown>
                ) : (
                  message.content
                )}
              </div>
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
      
      {/* 스프레드 선택 UI */}
      {showSpreadSelection && (
        <div className="fixed bottom-0 left-0 right-0 w-full bg-white dark:bg-slate-900 py-4 border-t border-gray-200 dark:border-slate-700 z-50 transition-colors" style={{ left, width, transition }}>
          <div className="max-w-4xl mx-auto px-4">
            <div className="text-center mb-4">
              <h3 className="text-lg font-semibold text-purple-800 dark:text-purple-300 mb-2">
                추천 스프레드를 선택해주세요
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                질문에 가장 적합한 타로 스프레드를 선택하세요
              </p>
            </div>
            <div className="flex gap-4 justify-center">
              {recommendedSpreads.map((spread, index) => (
                <Button
                  key={spread.number}
                  onClick={() => handleSpreadSelection(index)}
                  className="flex-1 max-w-xs h-auto p-4 flex flex-col items-center gap-2 bg-gradient-to-br from-purple-500 to-fuchsia-500 hover:from-purple-600 hover:to-fuchsia-600 text-white border-0 shadow-lg hover:shadow-xl transition-all duration-200"
                >
                  <div className="text-sm font-semibold text-center leading-tight">
                    {spread.spread_name}
                  </div>
                  <div className="text-xs opacity-90">
                    {spread.card_count}장의 카드
                  </div>
                </Button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* 카드 선택 UI */}
      {showCardSelection && (
        <div className="fixed bottom-0 left-0 right-0 w-full bg-white dark:bg-slate-900 py-6 border-t border-gray-200 dark:border-slate-700 z-50 transition-colors" style={{ left, width, transition }}>
          <div className="max-w-6xl mx-auto px-4">
            <div className="text-center mb-4">
              <h3 className="text-lg font-semibold text-purple-800 dark:text-purple-300 mb-2">
                {requiredCardCount}장의 카드를 선택해주세요
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                반원형으로 펼쳐진 카드 중에서 선택하세요 • 선택된 카드: {selectedCards.length}/{requiredCardCount}
              </p>
              {selectedCards.length > 0 && (
                <p className="text-xs text-purple-600 dark:text-purple-400">
                  선택된 카드: {selectedCards.map(cardIndex => getCardName(cardIndex)).join(', ')}
                </p>
              )}
            </div>
            
            {/* 카드 반원형 스프레드 */}
            <div className="relative mb-3 overflow-hidden">
              <div className="flex justify-center">
                <div className="relative" style={{ width: '800px', height: '500px' }}>
                  {/* 반원형 카드 배치 */}
                  {shuffledCards.length > 0 && shuffledCards.map((cardIndex, arrayIndex) => {
                    // 반원형 배치 계산 (180도 스프레드) - 90도 왼쪽으로 회전
                    const angle = (arrayIndex / 77) * (90 * Math.PI / 180) - (Math.PI) +0.75;
                    const radius = 500
                    const x = Math.cos(angle) * radius + 400 // 중앙에서 400px
                    const y = Math.sin(angle) * radius + 600 // 중앙에서 200px (더 아래로 이동)
                    const rotation = (angle * 180) / Math.PI + 90
                    const isAnimated = animatedCards.includes(cardIndex)
                    const isSelected = selectedCards.includes(cardIndex)
                    const isFlipped = flippedCards.includes(cardIndex)
                    
                    // If card is flipped, position it at the bottom
                    if (isFlipped) {
                      const selectedIndex = selectedCards.indexOf(cardIndex)
                      const bottomX = 400 + (selectedIndex - Math.floor(selectedCards.length / 2)) * 100 // Center the cards horizontally
                      const bottomY = 420 // Position higher up from the bottom
                      
                      return (
                        <button
                          key={cardIndex}
                          className="absolute w-20 h-28 rounded-lg border-2 transition-all duration-700 transform z-20 bg-gradient-to-br from-purple-500 to-fuchsia-500 text-white border-purple-400 shadow-lg shadow-purple-500/50"
                          style={{
                            left: `${bottomX}px`,
                            top: `${bottomY}px`,
                            transform: 'translate(-50%, -50%) rotateY(180deg)',
                          }}
                        >
                          <div className="w-full h-full flex items-center justify-center relative">
                            <img 
                              src={getCardImagePath(cardIndex)} 
                              alt={`${getCardName(cardIndex)}`}
                              className="w-full h-full object-cover rounded-lg"
                              style={{ transform: 'rotateY(180deg)' }}
                            />
                          </div>
                        </button>
                      )
                    }
                    
                    return (
                      <button
                        key={cardIndex}
                        onClick={() => handleCardSelection(cardIndex)}
                        className={`absolute w-20 h-28 rounded-lg border-2 transition-all duration-500 transform hover:scale-110 ${
                          isSelected
                            ? 'bg-gradient-to-br from-purple-500 to-fuchsia-500 text-white border-purple-400 shadow-lg shadow-purple-500/50 z-10'
                            : 'bg-gradient-to-br from-amber-50 to-amber-100 dark:from-amber-900/20 dark:to-amber-800/20 text-amber-800 dark:text-amber-200 border-amber-300 dark:border-amber-600 hover:border-purple-400 dark:hover:border-purple-400 hover:shadow-md'
                        } ${isAnimated ? 'opacity-100 scale-100' : 'opacity-0 scale-75'}`}
                        style={{
                          left: `${x}px`,
                          top: `${y}px`,
                          transform: `translate(-50%, -50%) rotate(${rotation}deg) ${isAnimated ? 'scale(1)' : 'scale(0.75)'}`,
                          zIndex: isSelected ? 10 : 1
                        }}
                      >
                        <div className="w-full h-full flex items-center justify-center relative">
                          <img 
                            src="/tarot/cardback.png" 
                            alt={`Card ${cardIndex}`}
                            className="w-full h-full object-cover rounded-lg"
                          />
                        </div>
                      </button>
                    )
                  })}
                </div>
              </div>
            </div>

            {/* 액션 버튼들 */}
            <div className="flex gap-2 justify-center">
              <Button
                onClick={handleCardSelectionReset}
                variant="outline"
                size="sm"
                className="text-xs"
              >
                초기화
              </Button>
              <Button
                onClick={handleCardSelectionSubmit}
                disabled={selectedCards.length !== requiredCardCount}
                className="bg-gradient-to-r from-purple-500 to-fuchsia-500 hover:from-purple-600 hover:to-fuchsia-600 text-white"
                size="sm"
              >
                카드 선택 완료 ({selectedCards.length}/{requiredCardCount})
              </Button>
            </div>
          </div>
        </div>
      )}
      
      {/* 입력창 - 스프레드 선택이나 카드 선택 중에는 숨김 */}
      {!showSpreadSelection && !showCardSelection && (
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
      )}
    </div>
  )
}
