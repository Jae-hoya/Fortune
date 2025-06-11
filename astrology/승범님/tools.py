import os

from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

from vectorstore import get_vectorstore


def get_document_retriever_tool():
    split_docs, vectorstore = get_vectorstore()

    SYNTACTIC_SEARCH_RATIO = round(float(os.getenv("SYNTACTIC_SEARCH_RATIO", 0.5)), 1)
    SEMANTIC_SEARCH_RATIO = round(1 - SYNTACTIC_SEARCH_RATIO, 1)

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 30})
    bm25_retriever = BM25Retriever.from_documents(split_docs, k=20)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[SYNTACTIC_SEARCH_RATIO, SEMANTIC_SEARCH_RATIO],
        top_k=int(os.getenv("TOP_K_RETRIEVER", 15)),
    )

    compressor = FlashrankRerank(top_n=10)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
        top_k=int(os.getenv("TOP_K_RERANKER", 5)),
    )

    retriever_tool = create_retriever_tool(
        compression_retriever,
        name="astrology_pdf_search",
        description="use this tool to search information about astrological interpretation of planets or aspects of user's natal chart.",
    )

    return retriever_tool


def get_websearch_tool():
    return TavilySearchResults(
        k=5,
        description="use this tool to search for meaning of user's planets, aspects or houses if necessary")
