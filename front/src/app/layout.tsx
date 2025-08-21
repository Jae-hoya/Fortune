"use client"
import "./globals.css"
import { Inter } from "next/font/google"
import { ThemeProvider } from "@/components/theme-provider"
import { Sidebar, SIDEBAR_WIDTH } from "@/components/layout/Sidebar"
import { useSidebarStore } from "@/store/sidebar"
import { Header } from "@/components/layout/Header"
import { ThemeToggle } from "@/components/theme-toggle"

const inter = Inter({ subsets: ["latin"] })

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const { isOpen, toggle } = useSidebarStore()
  return (
    <html lang="ko" suppressHydrationWarning>
      <body className={inter.className}>
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
          <Header onMenuClick={toggle} />
          <div className="fixed top-16 right-4 md:top-8 md:right-8 z-50">
            <ThemeToggle />
          </div>
          <div className="flex min-h-screen pt-14">
            <Sidebar />
            <main
              className="flex-1 min-h-[calc(100vh-56px)] bg-white transition-all duration-300"
              style={{
                marginLeft: isOpen ? SIDEBAR_WIDTH : 0,
                transition: "margin-left 0.3s cubic-bezier(0.4,0,0.2,1)",
              }}
            >
              {children}
            </main>
          </div>
        </ThemeProvider>
      </body>
    </html>
  )
}
