from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

load_dotenv()

# vector store
embeddings = OllamaEmbeddings(model="bge-m3")
vector_store = FAISS.load_local("faiss_saju/all_saju_data", embeddings, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever(search_kwargs={"k":20})

# reranker retriever
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=model, top_n=10)

def compression_retriever():
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    return compression_retriever

from langchain_core.prompts import load_prompt

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from operator import itemgetter

# prompt = hub.pull("teddynote/rag-prompt-chat-history")
prompt = load_prompt("prompt/saju-rag-promt_korea2.yaml")

# 
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.1)
llm = ChatOpenAI(model="gpt-4.1-mini")

pdf_chain = {
    "question": itemgetter("question"),
    "context": itemgetter("context"),
    "chat_history": itemgetter("chat_history"),
} | prompt | model | StrOutputParser()


from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 세션 기록을 저장할 딕셔너리
store = {}

# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    print(f"[대화 세션ID]: {session_ids}")
    if session_ids not in store:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환

# 대화를 기록하는 RAG 체인 생성
def pdf_rag_chain():
    rag_with_history = RunnableWithMessageHistory(
    pdf_chain,
    get_session_history,  # 세션 기록을 가져오는 함수
    input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
    history_messages_key="chat_history",  # 기록 메시지의 키
    )
    return rag_with_history

