"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { getWebSocketUrl } from "@/lib/websocket"

export default function TestWebSocketPage() {
  const [connectionStatus, setConnectionStatus] = useState<string>("연결되지 않음")
  const [messages, setMessages] = useState<string[]>([])
  const [inputMessage, setInputMessage] = useState("")
  const [ws, setWs] = useState<WebSocket | null>(null)

  const connectWebSocket = () => {
    try {
      const wsUrl = getWebSocketUrl("saju")
      console.log("테스트 WebSocket URL:", wsUrl)
      
      const websocket = new WebSocket(wsUrl)
      
      websocket.onopen = () => {
        console.log("WebSocket 연결 성공")
        setConnectionStatus("연결됨")
        setMessages(prev => [...prev, "✅ 연결 성공"])
      }
      
      websocket.onmessage = (event) => {
        console.log("메시지 수신:", event.data)
        setMessages(prev => [...prev, `📨 수신: ${event.data}`])
      }
      
      websocket.onclose = (event) => {
        console.log("WebSocket 연결 종료:", event.code, event.reason)
        setConnectionStatus("연결 종료")
        setMessages(prev => [...prev, `❌ 연결 종료: ${event.code}`])
      }
      
      websocket.onerror = (error) => {
        console.error("WebSocket 에러:", error)
        setConnectionStatus("에러 발생")
        setMessages(prev => [...prev, "❌ 연결 에러"])
      }
      
      setWs(websocket)
    } catch (error) {
      console.error("WebSocket 연결 실패:", error)
      setConnectionStatus("연결 실패")
      setMessages(prev => [...prev, `❌ 연결 실패: ${error}`])
    }
  }

  const disconnectWebSocket = () => {
    if (ws) {
      ws.close()
      setWs(null)
      setConnectionStatus("연결되지 않음")
    }
  }

  const sendMessage = () => {
    if (ws && ws.readyState === WebSocket.OPEN && inputMessage.trim()) {
      const messageData = { message: inputMessage, session_id: "test" }
      ws.send(JSON.stringify(messageData))
      setMessages(prev => [...prev, `📤 전송: ${JSON.stringify(messageData)}`])
      setInputMessage("")
    }
  }

  const clearMessages = () => {
    setMessages([])
  }

  return (
    <div className="container mx-auto p-4 max-w-4xl">
      <h1 className="text-2xl font-bold mb-4">WebSocket 연결 테스트</h1>
      
      <Card className="p-4 mb-4">
        <div className="flex gap-2 mb-4">
          <Button onClick={connectWebSocket} disabled={ws?.readyState === WebSocket.OPEN}>
            연결
          </Button>
          <Button onClick={disconnectWebSocket} disabled={!ws || ws.readyState !== WebSocket.OPEN}>
            연결 해제
          </Button>
          <Button onClick={clearMessages} variant="outline">
            메시지 지우기
          </Button>
        </div>
        
        <div className="mb-4">
          <strong>상태:</strong> {connectionStatus}
        </div>
        
        <div className="flex gap-2 mb-4">
          <Input
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="메시지를 입력하세요..."
            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          />
          <Button onClick={sendMessage} disabled={!ws || ws.readyState !== WebSocket.OPEN}>
            전송
          </Button>
        </div>
      </Card>
      
      <Card className="p-4">
        <h2 className="text-lg font-semibold mb-2">메시지 로그</h2>
        <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded max-h-96 overflow-y-auto">
          {messages.length === 0 ? (
            <p className="text-gray-500">메시지가 없습니다.</p>
          ) : (
            <div className="space-y-1">
              {messages.map((message, index) => (
                <div key={index} className="text-sm font-mono">
                  {message}
                </div>
              ))}
            </div>
          )}
        </div>
      </Card>
      
      <Card className="p-4 mt-4">
        <h2 className="text-lg font-semibold mb-2">환경 정보</h2>
        <div className="space-y-2 text-sm">
          <div><strong>WebSocket URL:</strong> {getWebSocketUrl("saju")}</div>
          <div><strong>환경 변수:</strong> {process.env.NEXT_PUBLIC_WEBSOCKET_URL || 'ws://localhost:8000'}</div>
          <div><strong>브라우저 WebSocket 지원:</strong> {typeof WebSocket !== 'undefined' ? '✅ 지원' : '❌ 미지원'}</div>
        </div>
      </Card>
    </div>
  )
} 