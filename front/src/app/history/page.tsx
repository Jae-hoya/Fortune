"use client"

import { useState } from "react"

const historyData = [
  { id: 1, type: "saju", title: "2025년 7월 21일 사주", summary: "사주 내용 예시" },
  { id: 2, type: "tarot", title: "2025년 7월 22일 타로", summary: "타로 내용 예시" },
  { id: 3, type: "saju", title: "2025년 7월 23일 사주", summary: "사주 내용 예시2" },
]

const tabs = [
  { key: "all", label: "전체" },
  { key: "saju", label: "사주" },
  { key: "tarot", label: "타로" },
]

export default function HistoryPage() {
  const [selected, setSelected] = useState("all")
  const filtered = selected === "all" ? historyData : historyData.filter(item => item.type === selected)

  return (
    <div className="p-10 min-h-screen bg-white dark:bg-gradient-to-br dark:from-slate-950 dark:to-slate-900 transition-colors">
      <h2 className="text-3xl md:text-4xl font-bold mb-8 text-slate-900 dark:text-white">질문 히스토리</h2>
      <div className="flex gap-4 mb-8">
        {tabs.map(tab => (
          <button
            key={tab.key}
            type="button"
            onClick={() => setSelected(tab.key)}
            className={`px-8 py-3 rounded-xl font-bold text-lg border-none transition-colors duration-200
              ${selected === tab.key
                ? 'bg-purple-500 text-white dark:bg-purple-700 dark:text-white'
                : 'bg-purple-100 text-purple-700 dark:bg-slate-800 dark:text-purple-200 hover:bg-purple-200 dark:hover:bg-slate-700'}
            `}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div>
        {filtered.length === 0 ? (
          <div className="text-slate-400 dark:text-slate-500 text-xl">히스토리가 없습니다.</div>
        ) : (
          filtered.map(item => (
            <div key={item.id} className="border border-gray-200 dark:border-slate-800 rounded-xl p-6 mb-4 bg-white dark:bg-slate-950/90 shadow-lg transition-colors">
              <div className="font-bold text-lg text-slate-900 dark:text-white">{item.title}</div>
              <div className="text-purple-700 dark:text-purple-300 mt-2">{item.summary}</div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
