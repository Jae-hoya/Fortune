"use client"

export default function PremiumPage() {
  return (
    <main className="p-6 min-h-screen bg-white dark:bg-gradient-to-br dark:from-slate-950 dark:to-slate-900 transition-colors">
      <div className="text-center mb-10">
          <h2 className="text-3xl font-bold text-black dark:text-white">전문가와 함께하는 프리미엄 운세</h2>
          <p className="text-purple-700 dark:text-purple-400 mt-2">더 깊이 있는 해석을 경험해보세요.</p>
      </div>
      <div className="bg-white dark:bg-slate-950/90 border border-gray-200 dark:border-slate-800 rounded-2xl p-6 shadow-lg transition-colors">
          <p className="text-purple-700 dark:text-purple-400">전문가 카드 슬라이더, 상세 모달, 결제 CTA가 구현될 예정입니다.</p>
      </div>
    </main>
  )
}
