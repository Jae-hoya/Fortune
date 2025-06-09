import numpy as np
from typing import List, Dict, Any, Union, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import EnsembleRetriever
from rank_bm25 import BM25Okapi
import re
import string

class BM25Searcher:
    """BM25 알고리즘 기반 키워드 검색을 위한 클래스"""
    
    def __init__(self, documents: List[Document]):
        """
        BM25Searcher 초기화
        
        Args:
            documents: 검색 대상 문서 리스트
        """
        self.documents = documents
        
        # 문서 텍스트 추출 및 토큰화
        self.texts = [doc.page_content for doc in documents]
        self.tokenized_corpus = [self._tokenize(text) for text in self.texts]
        
        # BM25 모델 초기화
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def _tokenize(self, text: str) -> List[str]:
        """
        텍스트를 토큰화합니다. 간단한 전처리 포함.
        
        Args:
            text: 토큰화할 텍스트
            
        Returns:
            토큰 리스트
        """
        # 소문자 변환
        text = text.lower()
        # 구두점 제거
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        # 여러 공백을 하나로 변환
        text = re.sub(r'\s+', ' ', text)
        # 단어 분할
        return text.strip().split()
    
    def search(self, query: str, top_k: int = 10) -> List[Document]:
        """
        BM25 기반 검색 수행
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 문서 수
            
        Returns:
            관련 문서 리스트
        """
        # 쿼리 토큰화
        tokenized_query = self._tokenize(query)
        
        # BM25 점수 계산
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # 상위 k개 문서 인덱스
        top_indices = np.argsort(doc_scores)[-top_k:][::-1]
        
        # 관련 문서 반환
        return [self.documents[i] for i in top_indices]

# Pydantic 호환 BM25 검색기
class BM25Retriever(BaseRetriever):
    """LangChain 호환 BM25 검색기"""
    
    def __init__(self, documents: List[Document], top_k: int = 10):
        """
        BM25Retriever 초기화
        
        Args:
            documents: 검색 대상 문서 리스트
            top_k: 검색할 문서 수
        """
        super().__init__()
        # pydantic 모델이므로 _searcher로 내부 속성 사용
        self._searcher = BM25Searcher(documents)
        self._top_k = top_k
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[Any] = None
    ) -> List[Document]:
        """
        관련 문서 검색 (LangChain BaseRetriever 구현)
        
        Args:
            query: 검색 쿼리
            run_manager: 실행 관리자 (선택 사항)
            
        Returns:
            관련 문서 리스트
        """
        return self._searcher.search(query, top_k=self._top_k)

def create_hybrid_retriever(
    vectorstore,
    documents: List[Document],
    weights: List[float] = [0.7, 0.3],
    top_k: int = 10
) -> EnsembleRetriever:
    """
    하이브리드 검색을 위한 EnsembleRetriever 생성
    
    Args:
        vectorstore: 벡터 스토어 객체 (시맨틱 검색용)
        documents: 검색 대상 문서 리스트 (BM25 검색용)
        weights: 각 검색기의 가중치 [시맨틱_가중치, BM25_가중치]
        top_k: 각 검색기가 반환할 문서 수
        
    Returns:
        EnsembleRetriever 객체
    """
    # 벡터 스토어 검색기
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    
    # BM25 검색기 (top_k 전달)
    bm25_retriever = BM25Retriever(documents, top_k=top_k)
    
    # EnsembleRetriever 생성
    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=weights
    ) 