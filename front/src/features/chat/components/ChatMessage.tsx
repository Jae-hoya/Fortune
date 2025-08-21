"use client"
import { cn } from "@/lib/utils"

type ChatMessageProps = {
  role: "user" | "master"
  message: string
}

export function ChatMessage({ role, message }: ChatMessageProps) {
  const isUser = role === "user"
  return (
    <div className={cn(
      "flex w-full mb-2",
      isUser ? "justify-end" : "justify-start"
    )}>
      <div className={cn(
        "max-w-[70%] px-4 py-2 rounded-2xl text-base break-words shadow",
        isUser
          ? "bg-purple-100 text-purple-900 rounded-br-md"
          : "bg-gray-100 text-gray-800 rounded-bl-md"
      )}>
        {message}
      </div>
    </div>
  )
}
