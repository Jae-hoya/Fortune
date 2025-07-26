"use client"

import * as React from "react"
import { Moon, Sun } from "lucide-react"
import { useTheme } from "next-themes"

export function ThemeToggle() {
  const { theme, setTheme } = useTheme()
  const [isMounted, setIsMounted] = React.useState(false)

  React.useEffect(() => {
    setIsMounted(true)
  }, [])

  // 드래그 방지 핸들러
  const handleDragStart = (e: React.DragEvent) => {
    e.preventDefault()
  }

  // 현재 테마에 따라 반대 테마의 아이콘만 보여줌 (전환 의도 명확)
  const nextTheme = theme === "dark" ? "light" : "dark"
  const Icon = theme === "dark" ? Moon : Sun

  return (
    <button
      onClick={() => setTheme(nextTheme)}
      className="fixed top-4 right-4 z-50 rounded-md w-10 h-10 flex items-center justify-center select-none"
      draggable="false"
      aria-label="테마 변경"
      onDragStart={handleDragStart}
    >
      {isMounted && <Icon className="h-[1.2rem] w-[1.2rem]" draggable="false" onDragStart={handleDragStart} />}
      <span className="sr-only">테마 변경</span>
    </button>
  )
} 