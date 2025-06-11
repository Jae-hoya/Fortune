import os

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

load_dotenv()

embedding_model = OllamaEmbeddings(model="bge-m3")
splitter = os.getenv("TEXT_SPLITTER", "RecursiveCharacterTextSplitter")
headers_to_split_on = [
    ("#", "Header 1"),
]


splitter_dict = {
    "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP")),
    ),
    "MarkdownHeaderTextSplitter": MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    ),
    "SemanticChunker": SemanticChunker(embeddings=embedding_model),
}

splitter = os.getenv("TEXT_SPLITTER", "RecursiveCharacterTextSplitter")
selected_splitter = splitter_dict[splitter]


def get_vectorstore(load_saved=False):
    if not load_saved:
        documents = list()
        path_list = [
            "../../markdown_file/Astrology_master_combined",
            "../../markdown_file/The Essential Guide to Practical Astrology",
        ]

        for path in path_list:
            loader = DirectoryLoader(
                path=path,
                glob="*.md",
                show_progress=True,
                use_multithreading=True,
                loader_cls=TextLoader,
            )

            documents.extend(loader.load())

        if type(selected_splitter) in [
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        ]:
            split_docs = list()
            for doc in documents:
                split = selected_splitter.split_text(doc.page_content)
                split = [Document(page_content=st) for st in split]
                split_docs.extend(split)
        else:
            split_docs = splitter.split_documents(documents)
        print(f"Split into {len(split_docs)} chunks.")

        vectorstore = FAISS.from_documents(split_docs, embedding=embedding_model)
        print(f"Created FAISS index with {vectorstore.index.ntotal} vectors.")

        return split_docs, vectorstore
