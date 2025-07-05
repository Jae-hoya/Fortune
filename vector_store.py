from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from typing import List
# from embeddings import get_bge_embeddings
from models import get_bge_embeddings

def load_vector_store(db_path: str):
    """
    지정된 경로에서 FAISS 벡터 스토어를 로드합니다.
    
    Args:
        db_path: 벡터 스토어가 저장된 디렉토리 경로
        
    Returns:
        로드된 FAISS 벡터 스토어 객체
    """
    embeddings = get_bge_embeddings()
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

def load_saju_vector_store():
    """
    사주 전용 FAISS 벡터 스토어를 로드합니다.
    
    Returns:
        사주 전용 FAISS 벡터 스토어 객체
    """
    # embeddings = OllamaEmbeddings(model="bge-m3")
    embeddings = get_bge_embeddings()
    return FAISS.load_local("faiss_saju/all_saju_data", embeddings, allow_dangerous_deserialization=True)

def get_all_documents(vectorstore, query: str = "", top_k: int = 1000) -> List[Document]:
    """
    벡터 스토어에서 최대 top_k개의 문서를 가져옵니다.
    빈 쿼리를 사용하면 대부분의 문서를 가져올 수 있습니다.
    
    Args:
        vectorstore: FAISS 벡터 스토어 객체
        query: 검색 쿼리 (기본값: 빈 문자열)
        top_k: 가져올 최대 문서 수 (기본값: 1000)
        
    Returns:
        문서 객체 리스트
    """
    return vectorstore.similarity_search(query, k=top_k)

def create_retriever(vectorstore, k: int = 10):
    """
    벡터 스토어에서 기본 검색기를 생성합니다.
    
    Args:
        vectorstore: FAISS 벡터 스토어 객체
        k: 검색할 문서 수
        
    Returns:
        벡터 스토어 검색기
    """
    return vectorstore.as_retriever(search_kwargs={"k": k})

def create_saju_retriever(k: int = 20):
    """
    사주 전용 검색기를 생성합니다.
    
    Args:
        k: 검색할 문서 수 (기본값: 20)
        
    Returns:
        사주 전용 벡터 스토어 검색기
    """
    vector_store = load_saju_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": k}) 