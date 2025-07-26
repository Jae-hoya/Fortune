"use client"

export default function AboutPage() {
  return (
    <div>
      <main className="p-6 min-h-screen bg-white dark:bg-gradient-to-br dark:from-slate-950 dark:to-slate-900 transition-colors">
        <div className="text-center mb-10">
            <h2 className="text-3xl font-bold text-secondary-700 dark:text-white">우리에 대해</h2>
            <p className="text-neutral-600 dark:text-neutral-300 mt-2">AI 운세 플랫폼을 만드는 사람들</p>
        </div>
        <div className="bg-white dark:bg-slate-950/90 border border-gray-200 dark:border-slate-800 rounded-2xl p-6 shadow-lg transition-colors">
            <p className="text-neutral-600 dark:text-neutral-300">타임라인 스크롤 애니메이션, 팀 소개 카드가 구현될 예정입니다.</p>
        </div>
      </main>
    </div>
  )
}
