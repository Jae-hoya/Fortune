from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank, CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from typing import List
from langchain_core.documents import Document

def get_flashrank_reranker():
    """
    FlashRank 기반 리랭커를 생성합니다.
    
    Returns:
        FlashRank 리랭커 객체
    """
    # 최신 API는 model_name 매개변수를 받지 않음
    return FlashrankRerank()

def get_crossencoder_reranker(top_n: int = 10):
    """
    CrossEncoder 기반 리랭커를 생성합니다. (사주 전용)
    
    Args:
        top_n: 리랭킹 후 반환할 문서 수 (기본값: 10)
        
    Returns:
        CrossEncoder 리랭커 객체
    """
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    return CrossEncoderReranker(model=model, top_n=top_n)

def create_compression_retriever(base_retriever, compressor):
    """
    리랭커를 사용하는 검색기를 생성합니다.
    
    Args:
        base_retriever: 기본 검색기
        compressor: 문서 압축기/리랭커
        
    Returns:
        ContextualCompressionRetriever 객체
    """
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

def create_saju_compression_retriever():
    """
    사주 전용 압축 검색기를 생성합니다.
    CrossEncoder를 사용하여 검색 결과를 리랭킹합니다.
    
    Returns:
        사주 전용 ContextualCompressionRetriever 객체
    """
    from vector_store import create_saju_retriever
    
    # 사주 전용 기본 검색기 생성
    base_retriever = create_saju_retriever(k=20)
    
    # CrossEncoder 리랭커 생성
    compressor = get_crossencoder_reranker(top_n=10)
    
    # 압축 검색기 생성
    return create_compression_retriever(base_retriever, compressor)

def rerank_documents(reranker, documents: List[Document], query: str) -> List[Document]:
    """
    문서 리스트를 쿼리 관련성에 따라 재정렬합니다.
    
    Args:
        reranker: 리랭커 객체
        documents: 재정렬할 문서 리스트
        query: 검색 쿼리
        
    Returns:
        재정렬된 문서 리스트
    """
    return reranker.compress_documents(documents, query) 