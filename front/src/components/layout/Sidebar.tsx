"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { Menu, X, Home, User, Settings, Star, LayoutGrid, Book, Crown } from "lucide-react"
import { useSidebarStore } from "@/store/sidebar"

export const SIDEBAR_WIDTH = 256;

export function Sidebar() {
  const pathname = usePathname()
  const { isOpen, toggle } = useSidebarStore()

  const sections = [
    {
      label: "홈",
      icon: Home,
      items: [
        { href: "/", label: "홈", icon: Home },
      ],
    },
    {
      label: "사주",
      icon: Star,
      items: [
        { href: "/saju", label: "사주" },
        { href: "/saju/history", label: "히스토리" },
      ],
    },
    {
      label: "타로",
      icon: LayoutGrid,
      items: [
        { href: "/tarot", label: "타로" },
        { href: "/tarot/history", label: "히스토리" },
      ],
    },
    {
      label: "기타",
      icon: null,
      items: [
        { href: "/profile", label: "프로필", icon: User },
        { href: "/settings", label: "설정", icon: Settings },
        { href: "/premium", label: "프리미엄", icon: Crown },
      ],
    },
  ]

  return (
    <>
      <aside
        className={cn(
          `fixed top-0 left-0 z-[90] h-screen transition-all duration-300 bg-white/70 dark:bg-slate-950/70 border-r border-slate-200 dark:border-slate-800 overflow-hidden`,
          isOpen ? `w-[${SIDEBAR_WIDTH}px]` : "w-0"
        )}
        style={{ width: isOpen ? SIDEBAR_WIDTH : 0 }}
      >
        <div className={cn(
          "flex flex-col h-full pt-20 pb-4 transition-opacity duration-300",
          isOpen ? "opacity-100" : "opacity-0 pointer-events-none"
        )}>
          <nav className="space-y-6 px-4 min-w-[256px]">
            {sections.map((section) => (
              <div key={section.label}>
                <div className="flex items-center gap-2 mb-2 text-purple-700 font-bold text-base">
                  {section.icon && <section.icon className="h-5 w-5" />} {section.label}
                </div>
                <ul className="space-y-1">
                  {section.items.map((item) => {
                    const isActive = pathname === item.href
                    const Icon = item.icon
                    return (
                      <li key={item.href}>
                        <Link
                          href={item.href}
                          className={cn(
                            "flex items-center gap-3 px-3 py-2 rounded-lg transition-colors duration-200",
                            isActive
                              ? "bg-purple-100 text-purple-900 dark:bg-purple-900/20 dark:text-purple-100"
                              : "text-purple-700 hover:bg-purple-100 dark:text-purple-200 dark:hover:bg-purple-900/20"
                          )}
                        >
                          {Icon && <Icon className="h-5 w-5" />}
                          <span className="font-medium">{item.label}</span>
                        </Link>
                      </li>
                    )
                  })}
                </ul>
              </div>
            ))}
          </nav>
        </div>
        {isOpen && (
          <button
            onClick={toggle}
            className="absolute top-4 right-4 z-[101] p-2 rounded-full bg-white/80 dark:bg-slate-800/80 text-slate-700 dark:text-white hover:bg-gray-200 dark:hover:bg-gray-700 transition"
            aria-label="사이드바 닫기"
            type="button"
          >
            <X className="h-5 w-5" />
          </button>
        )}
      </aside>
    </>
  )
}
