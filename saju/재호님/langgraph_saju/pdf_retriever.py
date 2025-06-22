from langchain_core.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings

from abc import ABC, abstractmethod
from operator import itemgetter
from langchain import hub


class RetrievalChain(ABC):
    def create_embedding(self):
        return OllamaEmbeddings(model="bge-m3")

    def load_faiss(self):
        return FAISS.load_local(f"faiss_pdf/manse", embeddings=self.create_embedding, allow_dangerous_deserialization=True)

    def create_retriever(self, vectorstore):
        # MMR을 사용하여 검색을 수행하는 retriever를 생성합니다.
        dense_retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": self.k}
        )
        return dense_retriever

    def load_prompt(self):
        return load_prompt("prompt/saju-rag-prompt_korea.yaml")

    def create_model(self):
        return ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)


    def create_chain(self):
        self.vectorstore = self.load_faiss()
        self.retriever = self.create_retriever(self.vectorstore)
        model = self.create_model()
        prompt = self.create_prompt()
        self.chain = (
            {
                "question": itemgetter("question"),
                "context": itemgetter("context"),
                "chat_history": itemgetter("chat_history"),
            }
            | prompt
            | model
            | StrOutputParser()
        )
        return self