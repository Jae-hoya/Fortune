"use client"

import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"

const sajuHistory = [
  {
    id: 1,
    title: "2025년 7월 21일의 운세",
    summary: "AI가 해석한 운세 내용의 일부가 여기에 표시됩니다...",
  },
  {
    id: 3,
    title: "2025년 7월 23일의 운세",
    summary: "AI가 해석한 운세 내용의 일부가 여기에 표시됩니다...",
  },
]

export default function SajuHistoryPage() {
  return (
    <main className="w-full min-h-screen bg-white dark:bg-gradient-to-br dark:from-slate-950 dark:to-slate-900 transition-colors">
      <div className="max-w-4xl mx-auto py-10 px-4">
        <h2 className="text-2xl font-bold mb-6 text-slate-900 dark:text-white">사주 질문 히스토리</h2>
        <div className="flex flex-wrap gap-8 justify-center">
          {sajuHistory.length === 0 ? (
            <div className="text-gray-400 dark:text-gray-500 text-lg">히스토리가 없습니다.</div>
          ) : (
            sajuHistory.map(item => (
              <Card key={item.id} className="w-[350px] p-6 flex flex-col gap-4 items-start bg-white dark:bg-slate-950/90 border border-gray-200 dark:border-slate-800 shadow-lg transition-colors">
                <CardHeader className="p-0 pb-2">
                  <CardTitle className="text-lg font-bold text-slate-900 dark:text-white">{item.title}</CardTitle>
                </CardHeader>
                <CardContent className="p-0 pb-2 text-violet-500 dark:text-purple-300">
                  {item.summary}
                </CardContent>
              </Card>
            ))
          )}
        </div>
      </div>
    </main>
  )
}
