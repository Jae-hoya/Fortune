"use client"

import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { cn } from '@/lib/utils'

interface MarkdownProps {
  children: string
  className?: string
}

export function Markdown({ children, className }: MarkdownProps) {
  return (
    <div className={cn(
      "prose prose-sm dark:prose-invert max-w-none",
      "prose-headings:text-purple-800 dark:prose-headings:text-purple-200 prose-headings:my-2",
      "prose-p:text-gray-700 dark:prose-p:text-gray-300 prose-p:my-1",
      "prose-strong:text-purple-700 dark:prose-strong:text-purple-300",
      "prose-em:text-purple-600 dark:prose-em:text-purple-400",
      "prose-blockquote:border-purple-300 dark:prose-blockquote:border-purple-600 prose-blockquote:my-2",
      "prose-blockquote:text-gray-600 dark:prose-blockquote:text-gray-400",
      "prose-code:text-purple-700 dark:prose-code:text-purple-300",
      "prose-code:bg-purple-50 dark:prose-code:bg-purple-900/20",
      "prose-pre:bg-gray-50 dark:prose-pre:bg-gray-900 prose-pre:my-2",
      "prose-ul:text-gray-700 dark:prose-ul:text-gray-300 prose-ul:my-1",
      "prose-ol:text-gray-700 dark:prose-ol:text-gray-300 prose-ol:my-1",
      "prose-li:text-gray-700 dark:prose-li:text-gray-300 prose-li:my-0.5",
      "prose-ul:space-y-0 prose-ol:space-y-0",
      "prose-p:leading-tight prose-p:mb-0",
      "prose-headings:leading-tight prose-headings:mb-1 prose-headings:mt-2",
      "prose-li:leading-tight prose-li:mb-0",
      className
    )}>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>
        {children}
      </ReactMarkdown>
    </div>
  )
} 