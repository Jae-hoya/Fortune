import { create } from 'zustand'

type Role = 'user' | 'assistant'

export interface Message {
  id: string
  role: Role
  content: string
  timestamp: string
}

interface ChatStore {
  messages: Message[]
  addMessage: (role: Role, content: string) => void
  isLoading: boolean
  setIsLoading: (loading: boolean) => void
  reset: () => void
}

export const useChatStore = create<ChatStore>((set) => ({
  messages: [],
  isLoading: false,
  setIsLoading: (loading) => set({ isLoading: loading }),
  addMessage: (role, content) =>
    set((state) => {
      const nextId = (state.messages.length + 1).toString();
      return {
        messages: [
          ...state.messages,
          {
            id: nextId,
            role,
            content,
            timestamp: new Date().toISOString(),
          },
        ],
      }
    }),
  reset: () => set({ messages: [], isLoading: false }),
})) 