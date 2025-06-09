from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from operator import itemgetter

from models import get_openai_llm
from prompts import get_fortune_rag_prompt
from reranker import get_flashrank_reranker, rerank_documents

# 세션 기록을 저장할 딕셔너리
session_store = {}

# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in session_store:
        session_store[session_ids] = ChatMessageHistory()
    return session_store[session_ids]

class RagPipeline:
    """RAG(Retrieval-Augmented Generation) 파이프라인 클래스"""
    
    def __init__(self, retriever: BaseRetriever, llm: BaseChatModel, reranker, prompt: Optional[ChatPromptTemplate] = None):
        """
        RagPipeline 초기화
        
        Args:
            retriever: 문서 검색기 (EnsembleRetriever 등)
            llm: LLM 모델 (기본값: gpt-3.5-turbo)
            reranker: 리랭커 객체 (기본값: FlashRank)
            prompt: 프롬프트 템플릿 (기본값: get_fortune_rag_prompt())
        """
        self.retriever = retriever
        self.llm = llm
        self.reranker = reranker
        self.prompt = prompt or get_fortune_rag_prompt()
        self.session_id = "default_session"
        self.chat_histories = {}  # 세션별 대화 기록 저장
    
    def _summarize_long_message(self, message: str, max_length: int = 200) -> str:
        """
        긴 메시지를 요약하는 함수
        
        Args:
            message: 원본 메시지
            max_length: 최대 길이
            
        Returns:
            요약된 메시지
        """
        if len(message) <= max_length:
            return message
        
        # 너무 긴 메시지는 앞부분만 유지하고 요약 표시
        return message[:max_length] + "... (이전 대화 내용 요약)"
    
    def _format_chat_history(self, messages):
        """
        ChatMessageHistory 객체의 메시지를 문자열로 변환
        
        Args:
            messages: 메시지 리스트
            
        Returns:
            포맷팅된 대화 기록 문자열
        """
        if not messages:
            return "이전 대화 내용이 없습니다."
        
        formatted = []
        for message in messages:
            if isinstance(message, HumanMessage):
                formatted.append(f"사용자: {message.content}")
            elif isinstance(message, AIMessage):
                # 긴 AI 메시지는 요약
                formatted.append(f"AI: {self._summarize_long_message(message.content)}")
        
        return "\n".join(formatted)
    
    def _get_context(self, query: str, top_k: int = 8) -> str:
        """
        검색 및 리랭킹을 수행하여 컨텍스트를 추출하는 함수
        
        Args:
            query: 사용자 질문/쿼리
            top_k: 검색할 문서 수
        
        Returns:
            컨텍스트 문자열
        """
        # 1. 검색 (하이브리드)
        docs = self.retriever.invoke(query)
        
        # 2. 리랭킹 - top_k개 결과만 유지
        reranked_docs = rerank_documents(self.reranker, docs, query)
        if len(reranked_docs) > top_k:
            reranked_docs = reranked_docs[:top_k]
        
        # 3. 문맥 결합
        return "\n\n".join([doc.page_content for doc in reranked_docs])
    
    def run(self, query: str, session_id: str = None) -> str:
        """
        RAG 파이프라인 실행 - 검색, 리랭킹, 생성 수행
        
        Args:
            query: 사용자 질문/쿼리
            session_id: 대화 세션 ID (기본값: None, 현재 설정된 세션 사용)
            
        Returns:
            생성된 응답 텍스트
        """
        if session_id:
            self.session_id = session_id
        
        # 세션별 대화 기록 초기화
        if self.session_id not in self.chat_histories:
            self.chat_histories[self.session_id] = []
        
        # 컨텍스트 검색
        context = self._get_context(query)
        
        # 현재 세션의 대화 기록 가져오기
        chat_history = self._format_chat_history(self.chat_histories[self.session_id])
        
        # 프롬프트 생성 및 LLM 호출
        prompt_input = {
            "context": context,
            "chat_history": chat_history,
            "question": query
        }
        
        # 체인 실행
        try:
            # 프롬프트 템플릿 적용
            prompt_value = self.prompt.format(**prompt_input)
            
            # 모든 경우에 동일한 프롬프트 사용 (대화 시작 여부와 관계없이)
            result = self.llm.invoke(prompt_value).content
            
            # 대화 기록 업데이트
            self.chat_histories[self.session_id].append(HumanMessage(content=query))
            self.chat_histories[self.session_id].append(AIMessage(content=result))
            
            # 대화 기록 길이 제한 (최근 10개 메시지만 유지)
            if len(self.chat_histories[self.session_id]) > 10:
                self.chat_histories[self.session_id] = self.chat_histories[self.session_id][-10:]
            
            return result
        except Exception as e:
            print(f"체인 실행 중 오류 발생: {e}")
            # 오류 발생 시에도 사용자 메시지는 저장
            self.chat_histories[self.session_id].append(HumanMessage(content=query))
            return f"죄송합니다. 요청을 처리하는 중 오류가 발생했습니다: {str(e)}"

def retrieve_and_generate(query: str, pipeline: RagPipeline, session_id: str = None) -> str:
    """
    RAG 파이프라인을 사용하여 질의응답을 수행하는 편의 함수
    
    Args:
        query: 사용자 질문/쿼리
        pipeline: RAG 파이프라인 객체
        session_id: 대화 세션 ID (기본값: None)
        
    Returns:
        생성된 응답 텍스트
    """
    return pipeline.run(query, session_id) 