import os
import uuid
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# from embeddings import get_bge_embeddings
from vector_store import load_vector_store, get_all_documents
from search import create_hybrid_retriever
from models import get_openai_llm, get_gemini_llm, get_bge_embeddings
from reranker import get_flashrank_reranker
from rag import RagPipeline, retrieve_and_generate

def initialize_rag_pipeline():
    """RAG 파이프라인 초기화"""
    
    # 임베딩 모델 로드
    embeddings = get_bge_embeddings()
    
    # 벡터 스토어 로드
    vectorstore = load_vector_store("saju_vectordb")
    
    # 모든 문서 가져오기
    all_docs = get_all_documents(vectorstore)
    
    # 하이브리드 검색기 초기화 (EnsembleRetriever 사용, 시맨틱 70%, BM25 30%)
    hybrid_retriever = create_hybrid_retriever(
        vectorstore=vectorstore, 
        documents=all_docs, 
        weights=[0.8, 0.2],
        top_k=20  # 명시적으로 top_k 설정
    )
    
    # LLM 모델 초기화
    # llm = get_openai_llm()
    llm = get_gemini_llm()  # Gemini 모델 사용
    
    # 리랭커 초기화
    reranker = get_flashrank_reranker()
    
    # RAG 파이프라인 생성
    pipeline = RagPipeline(hybrid_retriever, llm, reranker)
    
    return pipeline

def print_chatbot_message(message):
    """챗봇 메시지를 출력하는 함수"""
    print(f"\n🔮 {message}")

def print_system_message(message):
    """시스템 메시지를 출력하는 함수"""
    print(f"\n💬 {message}")

def print_welcome_message():
    """환영 메시지 출력 함수"""
    print("\n" + "=" * 60)
    print("✨ FortuneAI 사주 상담사 ✨")
    print("=" * 60)
    print("• 자연스러운 대화로 사주팔자를 분석해 드립니다.")
    print("• 원하는 내용을 자유롭게 질문해 보세요.")
    print("• 새 대화를 시작하려면 'new'를 입력하세요.")
    print("• 종료하려면 'q' 또는 'exit'를 입력하세요.")
    print("=" * 60)

def start_conversation(pipeline, session_id):
    """새 대화를 시작하는 함수"""
    # 초기 메시지 생성
    initial_greeting = retrieve_and_generate("안녕하세요", pipeline, session_id)
    print_chatbot_message(initial_greeting)
    return session_id

def main():
    """메인 함수"""
    # 환영 메시지 출력
    print_welcome_message()
    
    # 고유한 세션 ID 생성
    session_id = str(uuid.uuid4())
    print_system_message(f"새로운 대화가 시작되었습니다 (세션 ID: {session_id[:8]}...)")
    
    # RAG 파이프라인 초기화
    pipeline = initialize_rag_pipeline()
    print_system_message("사주 상담사가 준비되었습니다. 무엇이든 물어보세요!")
    
    # 대화 시작
    session_id = start_conversation(pipeline, session_id)
    
    # 대화 루프
    while True:
        # 사용자 입력 프롬프트
        query = input("\n🙋 ")
        
        if query.lower() in ['q', 'exit']:
            print_system_message("대화를 종료합니다. 다음에 또 찾아주세요!")
            break
        
        if query.lower() == 'new':
            # 새 세션 시작
            session_id = str(uuid.uuid4())
            print_system_message(f"새로운 대화가 시작되었습니다 (세션 ID: {session_id[:8]}...)")
            # 새 대화 시작
            session_id = start_conversation(pipeline, session_id)
            continue
        
        if not query.strip():
            continue
        
        try:
            # RAG 파이프라인 실행 (세션 ID 전달)
            answer = retrieve_and_generate(query, pipeline, session_id)
            print_chatbot_message(answer)
        except Exception as e:
            print_system_message(f"오류가 발생했습니다: {e}")
            print_system_message("다시 질문해 주세요.")

if __name__ == "__main__":
    main() 