"use client"

import Link from "next/link"

export default function ReadingsPage() {
  return (
    <main className="p-6 min-h-screen bg-white dark:bg-gradient-to-br dark:from-slate-950 dark:to-slate-900 transition-colors">
      <div className="flex gap-4 mb-6">
          <button className="px-4 py-2 rounded-lg bg-purple-500 text-white">전체</button>
          <button className="px-4 py-2 rounded-lg bg-white/50 text-purple-700 dark:text-purple-400">사주</button>
          <button className="px-4 py-2 rounded-lg bg-white/50 text-purple-700 dark:text-purple-400">타로</button>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {Array.from({ length: 3 }).map((_, i) => (
          <div key={i} className="bg-white dark:bg-slate-950/90 border border-gray-200 dark:border-slate-800 rounded-2xl p-6 shadow-lg transition-colors">
            <h3 className="font-bold text-lg text-black dark:text-white mb-2">2025년 7월 {21 + i}일의 운세</h3>
            <p className="text-purple-700 dark:text-purple-400 truncate">AI가 해석한 운세 내용의 일부가 여기에 표시됩니다...</p>
            <Link
              href={i % 2 === 0 ? "/saju" : "/tarot"}
              className="mt-4 inline-block px-6 py-2 rounded-xl bg-purple-500 text-white font-semibold shadow hover:bg-purple-600 transition"
            >
              {i % 2 === 0 ? "사주 보기" : "타로 보기"}
            </Link>
          </div>
        ))}
      </div>
    </main>
  )
}
