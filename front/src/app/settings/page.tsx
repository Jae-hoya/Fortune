"use client"

export default function SettingsPage() {
  return (
    <main className="p-6 min-h-screen bg-white dark:bg-gradient-to-br dark:from-slate-950 dark:to-slate-900 transition-colors">
      <div className="max-w-2xl mx-auto bg-white dark:bg-slate-950/90 border border-gray-200 dark:border-slate-800 rounded-2xl p-6 shadow-lg space-y-6 transition-colors">
          <div>
              <h3 className="font-bold text-lg text-black dark:text-white mb-2">다크 모드</h3>
              <p className="text-purple-700 dark:text-purple-400">토글 스위치가 구현될 예정입니다.</p>
          </div>
           <div>
              <h3 className="font-bold text-lg text-black dark:text-white mb-2">언어 설정</h3>
              <p className="text-purple-700 dark:text-purple-400">드롭다운 메뉴가 구현될 예정입니다.</p>
          </div>
           <div>
              <h3 className="font-bold text-lg text-black dark:text-white mb-2">TTS 미리듣기</h3>
              <p className="text-purple-700 dark:text-purple-400">미리듣기 버튼이 구현될 예정입니다.</p>
          </div>
      </div>
    </main>
  )
}
