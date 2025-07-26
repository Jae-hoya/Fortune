"use client"

import { Sun } from 'lucide-react'
import { useSidebarStore } from '@/store/sidebar'

export function Header({ onMenuClick }: { onMenuClick?: () => void }) {
  const { isOpen } = useSidebarStore()
  return (
    <header className="sticky top-0 h-14 w-full flex items-center justify-between px-4 bg-white dark:bg-slate-900 z-50 select-none">
      {/* 햄버거 버튼 (왼쪽) */}
      {(!isOpen) && (
        <button
          onClick={onMenuClick}
          className="p-2 rounded-full hover:bg-neutral-200/50"
          aria-label="메뉴 열기"
          type="button"
        >
          <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="4" y1="7" x2="20" y2="7" />
            <line x1="4" y1="12" x2="20" y2="12" />
            <line x1="4" y1="17" x2="20" y2="17" />
          </svg>
        </button>
      )}
      {/* 중앙 공간 */}
      <div className="flex-1" />
    </header>
  )
}
