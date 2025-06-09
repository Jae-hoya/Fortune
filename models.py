from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import torch

# 환경 변수 로드
load_dotenv()

def get_openai_llm(model_name: str = "gpt-4o-mini"):
    """
    OpenAI 기반 LLM 모델을 초기화합니다.
    
    Args:
        model_name: 사용할 OpenAI 모델 이름
        
    Returns:
        ChatOpenAI 모델 객체
    """
    return ChatOpenAI(model=model_name)


def get_bge_embeddings():
    """BGE-M3 임베딩 모델을 초기화하고 반환합니다."""
    # 환경 변수에서 USE_CUDA 값을 확인하여 device 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    ) 

def get_gemini_llm(model_name: str = "gemini-2.0-flash"):
    """
    Google Gemini 기반 LLM 모델을 초기화합니다.
    
    Args:
        model_name: 사용할 Gemini 모델 이름
        
    Returns:
        ChatGoogleGenerativeAI 모델 객체
    """
    return ChatGoogleGenerativeAI(model=model_name) 