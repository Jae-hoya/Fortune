# 필요한 라이브러리 설치
# !pip install flashrank faiss-cpu sentence-transformers langchain langchain-community rank-bm25

import os
import numpy as np
from typing import List, Tuple, Dict, Any
from flashrank import Ranker, RerankRequest
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from rank_bm25 import BM25Okapi

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    else:
        return obj

class BM25Retriever:
    """BM25 키워드 검색기"""
    
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.doc_contents = [doc.page_content for doc in documents]
        
        # 문서 토큰화 (간단한 공백 기반 토큰화)
        self.tokenized_docs = [doc.split() for doc in self.doc_contents]
        
        # BM25 인덱스 생성
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print(f"🔤 BM25 index created with {len(documents)} documents")
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """키워드 기반 검색을 수행하고 관련 문서와 점수를 반환합니다."""
        # 쿼리 토큰화
        tokenized_query = query.split()
        
        # BM25 점수 계산
        scores = self.bm25.get_scores(tokenized_query)
        
        # 상위 k개 결과 가져오기
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # 결과 문서와 점수 결합
        results = [(self.documents[idx], scores[idx]) for idx in top_indices]
        
        return results

class FlashRankReranker:
    """FlashRank를 사용한 Reranker 클래스"""
    
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        try:
            self.ranker = Ranker(model_name=model_name, cache_dir="/tmp")
            print(f"⚡ FlashRank Reranker initialized with model: {model_name}")
        except Exception as e:
            print(f"Error initializing FlashRank: {e}")
            # 폴백 모델 시도
            self.ranker = Ranker(model_name="ms-marco-MiniLM-L-6-v2", cache_dir="/tmp")
            print("FlashRank initialized with fallback model")
    
    def rerank(self, query: str, documents: List[Document], top_k: int = None) -> List[Tuple[Document, float]]:
        """문서들을 재순위화하고 점수와 함께 반환"""
        if not documents:
            return []
        
        if top_k is None or top_k > len(documents):
            top_k = len(documents)
        
        print(f"⚡ Reranking {len(documents)} documents...")
        
        try:
            # FlashRank용 데이터 준비
            passages = []
            for i, doc in enumerate(documents):
                passages.append({
                    "id": i,
                    "text": doc.page_content,
                    "meta": doc.metadata
                })
            
            # RerankRequest 객체 생성
            rerank_request = RerankRequest(query=query, passages=passages)
            
            # 재순위화 실행
            results = self.ranker.rerank(rerank_request)
            
            # 결과 정렬 (점수 내림차순)
            reranked_docs = []
            for result in results[:top_k]:
                original_doc = documents[result['id']]
                score = result['score']
                reranked_docs.append((original_doc, score))
            
            print(f"✅ Reranking completed. Top score: {reranked_docs[0][1]:.4f}")
            return reranked_docs
            
        except Exception as e:
            print(f"❌ Error during reranking: {e}")
            # 폴백: 원본 순서 유지하며 더미 점수 할당
            return [(doc, 0.0) for doc in documents[:top_k]]

class HybridRetriever:
    """FAISS(Semantic) + BM25(Keyword) 하이브리드 검색기"""
    
    def __init__(
        self, 
        semantic_retriever: FAISS, 
        keyword_retriever: BM25Retriever,
        semantic_weight: float = 0.8,
        keyword_weight: float = 0.2
    ):
        self.semantic_retriever = semantic_retriever
        self.keyword_retriever = keyword_retriever
        
        # 가중치 검증 및 설정
        assert abs(semantic_weight + keyword_weight - 1.0) < 1e-6, "가중치 합은 1이어야 합니다"
        
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        
        print(f"🔀 HybridRetriever initialized: Semantic={semantic_weight}, Keyword={keyword_weight}")
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        """하이브리드 검색을 수행하고 결과를 반환합니다."""
        print(f"🔍 Hybrid search for: '{query}' (top_k={top_k})")
        
        # 1. 시멘틱 검색 수행
        semantic_results = self.semantic_retriever.similarity_search_with_score(query, k=top_k*2)
        print(f"📊 Semantic search: {len(semantic_results)} results")
        
        # 2. 키워드 검색 수행
        keyword_results = self.keyword_retriever.retrieve(query, top_k=top_k*2)
        print(f"🔤 Keyword search: {len(keyword_results)} results")
        
        # 3. 결과 병합 및 가중치 적용
        combined_scores = {}
        
        # 시멘틱 결과 처리
        for doc, score in semantic_results:
            doc_id = self._get_doc_id(doc)
            # FAISS는 거리를 반환하므로 유사도로 변환 (1 / (1 + distance))
            similarity = 1.0 / (1.0 + score)
            combined_scores[doc_id] = {
                "doc": doc,
                "semantic_score": similarity,
                "keyword_score": 0.0,
                "final_score": self.semantic_weight * similarity
            }
        
        # 키워드 결과 처리
        for doc, score in keyword_results:
            doc_id = self._get_doc_id(doc)
            # BM25 점수 정규화 (0-1 범위로)
            normalized_score = score / (score + 1.0) if score > 0 else 0.0
            
            if doc_id in combined_scores:
                combined_scores[doc_id]["keyword_score"] = normalized_score
                combined_scores[doc_id]["final_score"] += self.keyword_weight * normalized_score
            else:
                combined_scores[doc_id] = {
                    "doc": doc,
                    "semantic_score": 0.0,
                    "keyword_score": normalized_score,
                    "final_score": self.keyword_weight * normalized_score
                }
        
        # 최종 점수로 정렬
        sorted_results = sorted(
            combined_scores.values(), 
            key=lambda x: x["final_score"], 
            reverse=True
        )
        
        print(f"✅ Hybrid search completed: {len(sorted_results)} combined results")
        
        # 상위 k개 결과 반환
        return [item["doc"] for item in sorted_results[:top_k]]
    
    def _get_doc_id(self, doc: Document) -> str:
        """문서의 고유 ID를 반환합니다."""
        # 타로 CSV 메타데이터에 ID가 있으면 사용
        if doc.metadata and "id" in doc.metadata:
            return str(doc.metadata["id"])
        
        # 폴백: 내용 해시로 ID 생성
        content_hash = hash(doc.page_content[:100])
        return f"doc_{content_hash}"

class TarotRAGSystem:
    """타로 Hybrid RAG 시스템: 분리된 카드/스프레드 FAISS + BM25 + FlashRank"""
    
    def __init__(self, 
                 card_faiss_path: str = "tarot_card_faiss_index",
                 spread_faiss_path: str = "tarot_spread_faiss_index",
                 embedding_model_name: str = "BAAI/bge-m3",
                 reranker_model_name: str = "ms-marco-MiniLM-L-12-v2",
                 semantic_weight: float = 0.8,
                 keyword_weight: float = 0.2):
        """
        타로 RAG 시스템 초기화
        
        Args:
            card_faiss_path: 카드 FAISS 인덱스 저장 경로
            spread_faiss_path: 스프레드 FAISS 인덱스 저장 경로
            embedding_model_name: 임베딩 모델명
            reranker_model_name: 리랭커 모델명
            semantic_weight: 시멘틱 검색 가중치
            keyword_weight: 키워드 검색 가중치
        """
        print("=" * 80)
        print("🔮 TAROT HYBRID RAG SYSTEM INITIALIZATION")
        print("=" * 80)
        
        # 1. 임베딩 모델 로드
        print("📚 Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},  # GPU 사용 시 'cuda'로 변경
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"✅ Embedding model loaded: {embedding_model_name}")
        
        # 2. 카드 FAISS 벡터스토어 로드
        print("🃏 Loading Card FAISS index...")
        try:
            self.card_vectorstore = FAISS.load_local(
                card_faiss_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"✅ Card FAISS loaded: {self.card_vectorstore.index.ntotal} documents")
        except Exception as e:
            print(f"❌ Error loading Card FAISS index: {e}")
            self.card_vectorstore = None
        
        # 3. 스프레드 FAISS 벡터스토어 로드
        print("🔮 Loading Spread FAISS index...")
        try:
            self.spread_vectorstore = FAISS.load_local(
                spread_faiss_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"✅ Spread FAISS loaded: {self.spread_vectorstore.index.ntotal} documents")
        except Exception as e:
            print(f"❌ Error loading Spread FAISS index: {e}")
            self.spread_vectorstore = None
        
        # 4. 카드 하이브리드 검색기 초기화
        if self.card_vectorstore:
            print("🃏 Initializing Card hybrid retriever...")
            card_docs = self._extract_documents(self.card_vectorstore)
            self.card_bm25 = BM25Retriever(card_docs)
            self.card_hybrid = HybridRetriever(
                semantic_retriever=self.card_vectorstore,
                keyword_retriever=self.card_bm25,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight
            )
        
        # 5. 스프레드 하이브리드 검색기 초기화
        if self.spread_vectorstore:
            print("🔮 Initializing Spread hybrid retriever...")
            spread_docs = self._extract_documents(self.spread_vectorstore)
            self.spread_bm25 = BM25Retriever(spread_docs)
            self.spread_hybrid = HybridRetriever(
                semantic_retriever=self.spread_vectorstore,
                keyword_retriever=self.spread_bm25,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight
            )
        
        # 6. FlashRank Reranker 초기화
        print("⚡ Initializing FlashRank reranker...")
        self.reranker = FlashRankReranker(model_name=reranker_model_name)
        
        print("🎉 TAROT HYBRID RAG SYSTEM READY!")
        print("=" * 80)
    
    def _extract_documents(self, vectorstore: FAISS) -> List[Document]:
        """FAISS에서 모든 문서 추출"""
        try:
            all_docs = []
            docstore = vectorstore.docstore._dict
            for doc_id, doc in docstore.items():
                all_docs.append(doc)
            return all_docs
        except Exception as e:
            print(f"❌ Error extracting documents: {e}")
            return []
    
    def search_cards(self, 
                    query: str, 
                    hybrid_k: int = 20, 
                    final_k: int = 5,
                    show_details: bool = True) -> List[Tuple[Document, float]]:
        """
        카드 의미 검색
        
        Args:
            query: 검색 쿼리
            hybrid_k: 하이브리드 검색에서 가져올 문서 수
            final_k: 최종 반환할 문서 수
            show_details: 검색 과정 상세 출력 여부
            
        Returns:
            (문서, 점수) 튜플의 리스트
        """
        if not self.card_vectorstore:
            print("❌ Card FAISS index not loaded")
            return []
        
        if show_details:
            print(f"\n🃏 CARD SEARCH: {query}")
            print("-" * 50)
        
        # 하이브리드 검색
        hybrid_docs = self.card_hybrid.retrieve(query, top_k=hybrid_k)
        
        if not hybrid_docs:
            return []
        
        # FlashRank 재순위화
        if show_details:
            print("⚡ Reranking card results...")
            
        reranked_results = self.reranker.rerank(query, hybrid_docs, top_k=final_k)
            
        return reranked_results
    
    def search_spreads(self, 
                      query: str, 
                      hybrid_k: int = 20, 
                      final_k: int = 5,
                      show_details: bool = True) -> List[Tuple[Document, float]]:
        """
        스프레드 설명 검색
        
        Args:
            query: 검색 쿼리
            hybrid_k: 하이브리드 검색에서 가져올 문서 수
            final_k: 최종 반환할 문서 수
            show_details: 검색 과정 상세 출력 여부
            
        Returns:
            (문서, 점수) 튜플의 리스트
        """
        if not self.spread_vectorstore:
            print("❌ Spread FAISS index not loaded")
            return []
        
        if show_details:
            print(f"\n🔮 SPREAD SEARCH: {query}")
            print("-" * 50)
        
        # 하이브리드 검색
        hybrid_docs = self.spread_hybrid.retrieve(query, top_k=hybrid_k)
        
        if not hybrid_docs:
            return []
        
        # FlashRank 재순위화
        if show_details:
            print("⚡ Reranking spread results...")
            
        reranked_results = self.reranker.rerank(query, hybrid_docs, top_k=final_k)
            
        return reranked_results
    
    def search_auto(self, 
                   query: str, 
                   final_k: int = 5,
                   show_details: bool = True) -> Dict[str, List[Tuple[Document, float]]]:
        """
        자동 검색 - 쿼리에 따라 카드/스프레드 자동 판단
        
        Args:
            query: 검색 쿼리
            final_k: 각 타입별 최종 반환할 문서 수
            show_details: 검색 과정 상세 출력 여부
            
        Returns:
            {"cards": [...], "spreads": [...]} 형태의 결과
        """
        if show_details:
            print(f"\n🔮 AUTO SEARCH: {query}")
            print("=" * 60)
        
        results = {"cards": [], "spreads": []}
        
        # 쿼리 키워드 분석
        query_lower = query.lower()
        
        # 카드 관련 키워드
        card_keywords = ["card", "meaning", "interpretation", "arcana", "reversed", "upright"]
        
        # 스프레드 관련 키워드 (실제 스프레드명 기반으로 확장)
        spread_keywords = ["spread", "layout", "reading", "position", "cross", "celtic", "three card", 
                          "past present future", "chakra", "horseshoe", "tree of life", "decision", 
                          "love", "relationship", "career", "money", "health", "daily", "yes no", 
                          "magical lottery", "astrological", "one card", "pull", "oracle", "guidance",
                          "triangle", "heart", "soul mate", "karmic", "wish", "bottom line", "choice",
                          "options", "advice", "spiritual", "weekly", "monthly", "yearly", "star",
                          "marketplace", "business", "lawsuit", "family", "pregnancy", "divorce",
                          "vacation", "trip", "relocation", "destiny", "fears", "reincarnation"]
        
        # 메이저 아르카나 카드명
        major_arcana = ["fool", "magician", "priestess", "empress", "emperor", "hierophant", "lovers", 
                       "chariot", "strength", "hermit", "wheel", "justice", "hanged", "death", 
                       "temperance", "devil", "tower", "star", "moon", "sun", "judgement", "world"]
        
        # 마이너 아르카나 카드명 (56장 전체)
        minor_arcana = [
            # Cups (14장)
            "ace of cups", "two of cups", "three of cups", "four of cups", "five of cups",
            "six of cups", "seven of cups", "eight of cups", "nine of cups", "ten of cups",
            "page of cups", "knight of cups", "queen of cups", "king of cups",
            
            # Pentacles (14장)
            "ace of pentacles", "two of pentacles", "three of pentacles", "four of pentacles", "five of pentacles",
            "six of pentacles", "seven of pentacles", "eight of pentacles", "nine of pentacles", "ten of pentacles",
            "page of pentacles", "knight of pentacles", "queen of pentacles", "king of pentacles",
            
            # Swords (14장)
            "ace of swords", "two of swords", "three of swords", "four of swords", "five of swords",
            "six of swords", "seven of swords", "eight of swords", "nine of swords", "ten of swords",
            "page of swords", "knight of swords", "queen of swords", "king of swords",
            
            # Wands (14장)
            "ace of wands", "two of wands", "three of wands", "four of wands", "five of wands",
            "six of wands", "seven of wands", "eight of wands", "nine of wands", "ten of wands",
            "page of wands", "knight of wands", "queen of wands", "king of wands",
            
            # 수트명들
            "cups", "pentacles", "swords", "wands", "ace", "king", "queen", "knight", "page"
        ]
        
        # 검색 우선순위 결정
        has_card_keywords = any(keyword in query_lower for keyword in card_keywords)
        has_spread_keywords = any(keyword in query_lower for keyword in spread_keywords)
        has_specific_card = any(card in query_lower for card in major_arcana + minor_arcana)
        
        # 카드 검색
        if has_card_keywords or has_specific_card or not has_spread_keywords:
            if show_details:
                print("🃏 Searching cards...")
            results["cards"] = self.search_cards(query, final_k=final_k, show_details=False)
        
        # 스프레드 검색
        if has_spread_keywords or not (has_card_keywords or has_specific_card):
            if show_details:
                print("🔮 Searching spreads...")
            results["spreads"] = self.search_spreads(query, final_k=final_k, show_details=False)
        
        return results
    
    def pretty_print_results(self, results: List[Tuple[Document, float]], result_type: str = ""):
        """타로 검색 결과를 예쁘게 출력"""
        if not results:
            print(f"❌ No {result_type} results found")
            return
        
        type_emoji = "🃏" if "card" in result_type.lower() else "🔮" if "spread" in result_type.lower() else "🔍"
        print(f"\n{type_emoji} {result_type.upper()} RESULTS ({len(results)} documents)")
        print("=" * 60)
        
        for i, (doc, score) in enumerate(results, 1):
            metadata = doc.metadata
            
            print(f"\n{type_emoji} Result {i} (Score: {score:.4f})")
            print("-" * 30)
            
            # 카드 정보 출력
            if metadata.get("card_name"):
                card_name = metadata['card_name']
                card_type = metadata.get('card_type', '')
                orientation = metadata.get('orientation', '')
                print(f"🃏 Card: {card_name} ({card_type})")
                if orientation and orientation != "both":
                    print(f"🔄 Orientation: {orientation}")
            
            # 스프레드 정보 출력
            if metadata.get("spread_name"):
                spread_name = metadata['spread_name']
                card_count = metadata.get('card_count', 0)
                print(f"🔮 Spread: {spread_name}")
                if card_count > 0:
                    print(f"🎯 Cards: {card_count}")
            
            # 메타데이터 출력
            print(f"📚 Source: {metadata.get('source', 'Unknown')}")
            
            # 내용 출력 (처음 200자)
            content = doc.page_content
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"📝 Content: {content}")

    def doc_to_card_info(self, doc: Document, orientation: str = "upright") -> dict:
        """카드 Document에서 메타데이터와 의미를 추출해 dict로 반환 (모든 메타데이터 포함)"""
        meta = doc.metadata
        info = dict(meta)  # 모든 메타데이터 복사
        info["orientation"] = orientation
        # orientation별 의미 추출
        if orientation == "upright":
            if meta.get("upright_keywords"):
                info["meaning"] = meta.get("upright_keywords")
            elif meta.get("tarot_keywords"):
                info["meaning"] = ", ".join(meta.get("tarot_keywords"))
            else:
                info["meaning"] = doc.page_content[:200]
        elif orientation == "reversed":
            if meta.get("reversed_keywords"):
                info["meaning"] = meta.get("reversed_keywords")
            elif meta.get("tarot_keywords"):
                info["meaning"] = ", ".join(meta.get("tarot_keywords"))
            else:
                info["meaning"] = doc.page_content[:200]
        else:
            info["meaning"] = doc.page_content[:200]
        info["content"] = doc.page_content
        return convert_numpy_types(info)

    def search_card_meaning(self, card_name: str, orientation: str = "upright", show_details: bool = False) -> dict:
        """카드 이름과 방향으로 의미/키워드 등 메타데이터 dict 반환"""
        if not self.card_vectorstore:
            return {"success": False, "message": "Card FAISS index not loaded"}
        query = f"{card_name} {orientation} meaning"
        results = self.search_cards(query, final_k=1, show_details=show_details)
        if not results:
            return {"success": False, "message": "No result"}
        doc, score = results[0]
        info = self.doc_to_card_info(doc, orientation)
        info["success"] = True
        info["score"] = score
        return info

    def doc_to_spread_info(self, doc: Document) -> dict:
        """스프레드 Document에서 메타데이터와 positions 등 추출해 dict로 반환 (모든 메타데이터 포함)"""
        meta = doc.metadata
        info = dict(meta)  # 모든 메타데이터 복사
        info["content"] = doc.page_content
        return convert_numpy_types(info)

    def search_spread_info(self, spread_name: str, show_details: bool = False) -> dict:
        """스프레드 이름으로 positions 등 메타데이터 dict 반환"""
        if not self.spread_vectorstore:
            return {"success": False, "message": "Spread FAISS index not loaded"}
        query = f"{spread_name} positions"
        results = self.search_spreads(query, final_k=1, show_details=show_details)
        if not results:
            return {"success": False, "message": "No result"}
        doc, score = results[0]
        info = self.doc_to_spread_info(doc)
        info["success"] = True
        info["score"] = score
        return info

def main():
    """메인 실행 함수 - 타로 RAG 시스템 테스트"""
    
    # RAG 시스템 초기화
    rag_system = TarotRAGSystem(
        card_faiss_path="tarot_card_faiss_index",
        spread_faiss_path="tarot_spread_faiss_index",
        embedding_model_name="BAAI/bge-m3",
        reranker_model_name="ms-marco-MiniLM-L-12-v2",
        semantic_weight=0.8,
        keyword_weight=0.2
    )
    
    # 타로 관련 테스트 쿼리들
    test_queries = [
        "What does The Fool card mean?",  # 카드 쿼리
        "How to do a Celtic Cross spread?",  # 스프레드 쿼리
        "Ace of Cups meaning in love",  # 카드 쿼리
        "Three card spread for relationships",  # 스프레드 쿼리
        "Death card interpretation",  # 카드 쿼리
        "Past present future reading"  # 스프레드 쿼리
    ]
    
    print("\n🧪 TESTING TAROT RAG SYSTEM")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        
        # 자동 검색
        results = rag_system.search_auto(query, final_k=2, show_details=False)
        
        # 결과 출력
        if results["cards"]:
            rag_system.pretty_print_results(results["cards"], "CARD")
        
        if results["spreads"]:
            rag_system.pretty_print_results(results["spreads"], "SPREAD")
        
        print("\n" + "-" * 80)
    
    # 카드 전용 검색 테스트
    print("\n🃏 TESTING CARD-ONLY SEARCH")
    print("=" * 80)
    card_results = rag_system.search_cards("What does Strength card represent?", final_k=3, show_details=False)
    rag_system.pretty_print_results(card_results, "CARD")
    
    # 스프레드 전용 검색 테스트
    print("\n🔮 TESTING SPREAD-ONLY SEARCH")
    print("=" * 80)
    spread_results = rag_system.search_spreads("chakra energy spread", final_k=3, show_details=False)
    rag_system.pretty_print_results(spread_results, "SPREAD")

if __name__ == "__main__":
    main()