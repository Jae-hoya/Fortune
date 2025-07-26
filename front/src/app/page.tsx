"use client"

import Link from "next/link"

export default function HomePage() {
  return (
    <main className="relative flex flex-col items-center justify-center min-h-[calc(100vh-56px)] bg-white dark:bg-gradient-to-br dark:from-slate-950 dark:to-slate-900 transition-colors">
      <section className="relative z-10 text-center max-w-2xl mx-auto px-4">
        <h1 className="relative text-5xl md:text-7xl font-black mb-8">
          <span className="relative inline-block">
            <span className="absolute inset-0 bg-gradient-to-r from-purple-600 via-fuchsia-500 to-purple-600 dark:from-white dark:via-purple-100 dark:to-white blur-[2px] bg-clip-text text-transparent animate-gradient opacity-90">
              Fortune Rhythm
            </span>
            <span className="relative bg-gradient-to-r from-purple-700 via-fuchsia-600 to-purple-700 dark:from-white dark:via-purple-200 dark:to-white bg-clip-text text-transparent animate-gradient drop-shadow-[0_2px_2px_rgba(0,0,0,0.5)] dark:drop-shadow-[0_2px_2px_rgba(0,0,0,0.3)]">
              Fortune Rhythm
            </span>
          </span>
        </h1>
        <p className="text-xl md:text-2xl text-slate-700 dark:text-white mb-12 transition-colors duration-300 font-semibold drop-shadow-sm">
          신비로운 AI와 함께 사주·타로를 경험하세요
        </p>
        <div className="flex flex-col md:flex-row justify-center gap-6 md:gap-10 mb-10">
          <Link 
            href="/saju" 
            className="group relative px-12 py-4 rounded-2xl bg-slate-900 dark:bg-white/25 text-white dark:text-white dark:backdrop-blur-sm text-xl font-semibold shadow-lg transition-all duration-300 hover:scale-105 hover:bg-slate-800 dark:hover:bg-white/40 dark:border dark:border-white/40"
          >
            <span className="relative z-10">사주 보기</span>
          </Link>
          <Link 
            href="/tarot" 
            className="px-12 py-4 rounded-2xl text-white text-xl font-semibold shadow-lg transition-all duration-300 hover:scale-105 bg-gradient-to-r from-purple-600 to-fuchsia-600 hover:from-purple-700 hover:to-fuchsia-700 dark:from-purple-400 dark:to-fuchsia-400 dark:hover:from-purple-500 dark:hover:to-fuchsia-500 backdrop-blur-sm dark:border dark:border-white/40"
          >
            타로 보기
          </Link>
        </div>
      </section>
    </main>
  )
}
