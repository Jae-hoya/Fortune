"use client"

import { Plus } from 'lucide-react'

export default function ProfilePage() {
  return (
    <div className="relative min-h-screen bg-white dark:bg-gradient-to-br dark:from-slate-950 dark:to-slate-900 transition-colors">
      <main className="p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-slate-950/90 border border-gray-200 dark:border-slate-800 rounded-2xl p-6 shadow-lg transition-colors">
                <h3 className="font-bold text-lg text-black dark:text-white mb-4">내 프로필</h3>
                <p className="text-purple-700 dark:text-purple-400">프로필 입력 폼이 여기에 표시됩니다.</p>
            </div>
            <div className="bg-white dark:bg-slate-950/90 border border-gray-200 dark:border-slate-800 rounded-2xl p-6 shadow-lg transition-colors">
                <h3 className="font-bold text-lg text-black dark:text-white mb-4">지인 프로필</h3>
                <p className="text-purple-700 dark:text-purple-400">지인 프로필 목록이 여기에 표시됩니다.</p>
            </div>
        </div>
      </main>
      <button className="absolute bottom-10 right-10 bg-purple-500 text-white p-4 rounded-full shadow-lg hover:bg-purple-600 transition">
        <Plus className="w-6 h-6" />
      </button>
    </div>
  )
}
