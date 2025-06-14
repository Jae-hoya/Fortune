import os
import pandas as pd
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import ast

# CSV 파일 경로 설정
CARD_CSV_PATH = "parsed_chunks/tarot_card_chunk.csv"  # 카드 데이터
SPREAD_CSV_PATH = "parsed_chunks/tarot_spread_chunk.csv"  # 스프레드 데이터

def parse_positions(positions_raw):
    """
    positions 컬럼에서 포지션 정보를 리스트로 파싱
    - 파이썬 리스트(dict의 리스트) 문자열을 안전하게 파싱
    """
    if not positions_raw or not isinstance(positions_raw, str):
        return []
    try:
        positions = ast.literal_eval(positions_raw)
        if isinstance(positions, list):
            return positions
    except Exception:
        pass
    return []

def load_csv_to_documents(csv_path: str) -> List[Document]:
    """CSV 파일을 Document 객체 리스트로 변환 - 메타데이터 포함 임베딩"""
    print(f"📄 CSV 파일 로드 중: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        print(f"✅ CSV 로드 완료: {len(df)}개 행")
    except Exception as e:
        print(f"❌ CSV 로드 오류: {e}")
        return []
    
    documents = []
    
    for idx, row in df.iterrows():
        # 메타데이터 구성
        metadata = {
            "id": row.get("id", f"doc_{idx}"),
            "source": row.get("source", "unknown"),
            "chapter": row.get("chapter", ""),
            "content_type": row.get("content_type", ""),
            "word_count": row.get("word_count", 0)
        }
        
        # 페이지 콘텐츠 구성 (메타데이터 포함)
        page_content_parts = []
        
        # 카드 정보가 있으면 추가
        if pd.notna(row.get("card_name")) and row.get("card_name"):
            card_name = row["card_name"]
            card_type = row.get("card_type", "")
            orientation = row.get("orientation", "")
            
            # 기본 메타데이터
            metadata["card_name"] = card_name
            metadata["card_type"] = card_type
            metadata["orientation"] = orientation
            
            # 추가 카드 메타데이터
            if card_type == "major_arcana":
                metadata["is_major_arcana"] = True
                
                # Tarot-for-beginners 소스에서 추가 메타데이터 파싱
                if metadata["source"] == "Tarot-for-beginners":
                    content = row.get("content", "")
                    if content:
                        # 대체 이름 추출
                        also_known_as = extract_also_known_as(content)
                        if also_known_as:
                            metadata["also_known_as"] = also_known_as
                            page_content_parts.append(f"ALSO KNOWN AS: {', '.join(also_known_as)}")
                            
                        # 요소 추출
                        element = extract_element(content)
                        if element:
                            metadata["element"] = element
                            page_content_parts.append(f"ELEMENT: {element}")
                            
                        # 별자리 추출
                        astrology = extract_astrology(content)
                        if astrology:
                            metadata["astrology"] = astrology
                            page_content_parts.append(f"ASTROLOGY: {astrology}")
                            
                        # 숫자학 추출
                        numerology = extract_numerology(content)
                        if numerology:
                            metadata["numerology"] = numerology
                            page_content_parts.append(f"NUMEROLOGY: {numerology}")
                            
                        # 키워드 추출 (KEYWORDS 섹션에서)
                        keywords = extract_tarot_keywords(content)
                        if keywords:
                            metadata["tarot_keywords"] = keywords
                            page_content_parts.append(f"TAROT KEYWORDS: {', '.join(keywords)}")
                            
                        # 신화적 연관 추출
                        mythological_association = extract_mythological_association(content)
                        if mythological_association:
                            metadata["mythological_association"] = mythological_association
                            page_content_parts.append(f"MYTHOLOGICAL ASSOCIATION: {mythological_association}")
                            
                        # 상징 추출
                        symbols = extract_symbols(content)
                        if symbols:
                            metadata["symbols"] = symbols
                            page_content_parts.append(f"SYMBOLS: {', '.join(symbols)}")
                            
                        # 관련 카드 추출
                        related_cards = extract_related_cards(content)
                        if related_cards:
                            if related_cards.get("supporting_cards"):
                                metadata["supporting_cards"] = related_cards["supporting_cards"]
                                page_content_parts.append(f"SUPPORTING CARDS: {', '.join(related_cards['supporting_cards'])}")
                            if related_cards.get("opposing_cards"):
                                metadata["opposing_cards"] = related_cards["opposing_cards"]
                                page_content_parts.append(f"OPPOSING CARDS: {', '.join(related_cards['opposing_cards'])}")
                
            elif card_type == "minor_arcana":
                metadata["is_minor_arcana"] = True
                
                # 카드 슈트(문양) 추출 (Cups, Wands, Pentacles, Swords)
                for suit in ["Cups", "Wands", "Pentacles", "Swords"]:
                    if suit.lower() in card_name.lower():
                        metadata["suit"] = suit
                        break
                        
                # 카드 숫자/인물 추출 (Ace, Two, Three, ..., Page, Knight, Queen, King)
                card_ranks = ["Ace", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", 
                              "Page", "Knight", "Queen", "King"]
                for rank in card_ranks:
                    if rank.lower() in card_name.lower():
                        metadata["rank"] = rank
                        if rank in ["Page", "Knight", "Queen", "King"]:
                            metadata["is_court_card"] = True
                        break
                
                # Tarot-for-beginners 소스에서 추가 메타데이터 파싱
                if metadata["source"] == "Tarot-for-beginners":
                    content = row.get("content", "")
                    if content:
                        # 별자리 추출
                        astrology = extract_astrology(content)
                        if astrology:
                            metadata["astrology"] = astrology
                            page_content_parts.append(f"ASTROLOGY: {astrology}")
                            
                        # 키워드 추출 (KEYWORDS 섹션에서)
                        keywords = extract_tarot_keywords(content)
                        if keywords:
                            metadata["tarot_keywords"] = keywords
                            page_content_parts.append(f"TAROT KEYWORDS: {', '.join(keywords)}")
                            
                        # 특성 및 역할 추출
                        traits = extract_traits(content)
                        if traits:
                            metadata["personality_traits"] = traits
                            page_content_parts.append(f"PERSONALITY TRAITS: {', '.join(traits)}")
                            
                        roles = extract_roles(content)
                        if roles:
                            metadata["roles"] = roles
                            page_content_parts.append(f"ROLES: {', '.join(roles)}")
                            
                        # 상징 추출
                        symbols = extract_symbols(content)
                        if symbols:
                            metadata["symbols"] = symbols
                            page_content_parts.append(f"SYMBOLS: {', '.join(symbols)}")
            
            # 카드 메타데이터를 임베딩에 포함
            page_content_parts.append(f"CARD: {card_name}")
            page_content_parts.append(f"TYPE: {card_type}")
            
            # 슈트와 랭크 정보 추가
            if metadata.get("suit"):
                page_content_parts.append(f"SUIT: {metadata['suit']}")
            if metadata.get("rank"):
                page_content_parts.append(f"RANK: {metadata['rank']}")
            
            # 방향 정보 추가
            if orientation:
                page_content_parts.append(f"ORIENTATION: {orientation}")
                
                # 방향별 키워드 추출 및 추가
                if orientation == "upright" or orientation == "both":
                    upright_keywords = extract_keywords_from_content(row.get("content", ""), "upright")
                    if upright_keywords:
                        metadata["upright_keywords"] = upright_keywords
                        page_content_parts.append(f"UPRIGHT KEYWORDS: {upright_keywords}")
                        
                if orientation == "reversed" or orientation == "both":
                    reversed_keywords = extract_keywords_from_content(row.get("reversed", ""), "reversed")
                    if reversed_keywords:
                        metadata["reversed_keywords"] = reversed_keywords
                        page_content_parts.append(f"REVERSED KEYWORDS: {reversed_keywords}")
            
            # Reversed 내용이 있으면 추가
            if pd.notna(row.get("reversed")) and row.get("reversed").strip():
                reversed_content = row["reversed"].strip()
                metadata["has_reversed"] = True
                
                page_content_parts.append(f"REVERSED MEANING: {reversed_content}")
        
        # 스프레드 정보가 있으면 추가  
        if pd.notna(row.get("spread_name")) and row.get("spread_name"):
            spread_name = row["spread_name"]
            card_count = row.get("card_count", 0)
            description = row.get("description", "")
            
            # 개선 1: normalized_name 메타데이터 추가
            normalized_name = row.get("normalized_name", "")
            if normalized_name:
                metadata["normalized_name"] = normalized_name
            
            # 개선 2: keywords 메타데이터 추가
            keywords = row.get("keywords", "")
            if keywords:
                metadata["keywords"] = keywords
            
            metadata["spread_name"] = spread_name
            metadata["card_count"] = card_count
            
            # 스프레드 메타데이터를 임베딩에 포함
            page_content_parts.append(f"SPREAD: {spread_name}")
            
            # 개선 3: normalized_name 임베딩에 포함
            if normalized_name:
                page_content_parts.append(f"NORMALIZED NAME: {normalized_name}")
            
            # 개선 4: keywords 임베딩에 포함
            if keywords:
                page_content_parts.append(f"KEYWORDS: {keywords}")
            
            if card_count > 0:
                page_content_parts.append(f"CARDS: {card_count} cards")
            if description:
                page_content_parts.append(f"DESCRIPTION: {description}")
                
            # positions 정보 임베딩 (스프레드 데이터에만 해당)
            positions_raw = row.get("positions", "")
            positions_list = parse_positions(positions_raw)
            if positions_list:
                page_content_parts.append("POSITIONS:")
                for pos in positions_list:
                    if isinstance(pos, dict):
                        pos_str = f"{pos.get('position_num', '')}: {pos.get('position_name', '')} - {pos.get('position_meaning', '')}"
                    else:
                        pos_str = str(pos)
                    page_content_parts.append(pos_str)
        
        # 메인 콘텐츠 추가
        content = row.get("content", "")
        if content:
            # 전체 내용을 임베딩에 포함
            page_content_parts.append(f"CONTENT: {content}")
        
        # 최종 페이지 콘텐츠 생성
        final_page_content = "\n\n".join(page_content_parts)
        
        # Document 객체 생성
        doc = Document(
            page_content=final_page_content,
            metadata=metadata
        )
        
        documents.append(doc)
    
    print(f"✅ {len(documents)}개 Document 객체 생성 완료 (메타데이터 포함)")
    return documents

# Tarot-for-beginners 소스에서 메타데이터 추출 함수들
def extract_also_known_as(content):
    """ALSO KNOWN AS 정보 추출"""
    import re
    pattern = r"ALSO KNOWN AS\s+(.*?)(?:\n|$)"
    match = re.search(pattern, content, re.IGNORECASE)
    if match:
        items = [item.strip() for item in match.group(1).split(",")]
        return items
    return []

def extract_element(content):
    """ELEMENT 정보 추출"""
    import re
    pattern = r"ELEMENT\s+(.*?)(?:\n|$)"
    match = re.search(pattern, content, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

def extract_astrology(content):
    """ASTROLOGY 정보 추출"""
    import re
    pattern = r"ASTROLOGY\s+(.*?)(?:\n|$)"
    match = re.search(pattern, content, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

def extract_numerology(content):
    """NUMEROLOGY 정보 추출"""
    import re
    pattern = r"NUMEROLOGY\s+(.*?)(?:\n|$)"
    match = re.search(pattern, content, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

def extract_tarot_keywords(content):
    """KEYWORDS 정보 추출"""
    import re
    pattern = r"KEYWORDS\s+(.*?)(?:\n|$)"
    match = re.search(pattern, content, re.IGNORECASE)
    if match:
        keywords = [kw.strip() for kw in match.group(1).split(",")]
        return keywords
    return []

def extract_mythological_association(content):
    """신화적 연관성 추출"""
    import re
    # Mystic Meanings 섹션에서 추출
    pattern = r"Mystic Meanings.*?Chiron(.*?)(?:\n|$)"
    match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
    if match:
        return "Centaur Chiron, the wounded healer"
    return ""

def extract_symbols(content):
    """상징 추출"""
    import re
    # 내용에서 키 단어로 상징 추출
    symbols = []
    key_symbols = ["Keys", "Chalice", "Cup", "Throne", "Cross", "Wand", "Pentacle", "Sword"]
    
    for symbol in key_symbols:
        if re.search(rf"\b{symbol}\b", content, re.IGNORECASE):
            symbols.append(symbol)
    
    return symbols

def extract_related_cards(content):
    """관련 카드 추출"""
    import re
    result = {}
    
    # Supporting and Opposing Cards 섹션 찾기
    pattern = r"Supporting and Opposing Cards(.*?)(?:(?:^#)|$)"
    match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
    
    if match:
        section = match.group(1)
        
        # Supporting 카드 찾기
        supporting_pattern = r"combined with\s+(.*?),\s+The\s+(\w+)\s+indicates"
        supporting_matches = re.findall(supporting_pattern, section, re.IGNORECASE)
        
        supporting_cards = []
        for match in supporting_matches:
            if len(match) >= 2:
                supporting_cards.append(f"The {match[1]}")
                supporting_cards.append(match[0])
        
        # Opposing 카드 찾기
        opposing_pattern = r"free spirits,\s+(.*?),\s+often indicate"
        opposing_match = re.search(opposing_pattern, section, re.IGNORECASE)
        
        opposing_cards = []
        if opposing_match:
            cards_text = opposing_match.group(1)
            cards = re.findall(r"The\s+(\w+)", cards_text)
            opposing_cards = [f"The {card}" for card in cards]
        
        if supporting_cards:
            result["supporting_cards"] = supporting_cards
        if opposing_cards:
            result["opposing_cards"] = opposing_cards
    
    return result

def extract_traits(content):
    """카드의 특성 추출"""
    import re
    traits = []
    
    # 특성 관련 키워드 찾기
    trait_keywords = ["compassionate", "understanding", "nurturing", "healing", "empathetic", "intuitive"]
    
    for trait in trait_keywords:
        if re.search(rf"\b{trait}\b", content, re.IGNORECASE):
            traits.append(trait.capitalize())
    
    return traits

def extract_roles(content):
    """카드의 역할 추출"""
    import re
    roles = []
    
    # 역할 관련 키워드 찾기
    role_keywords = ["healer", "mother", "nurturer", "teacher", "guide", "mentor"]
    
    for role in role_keywords:
        if re.search(rf"\b{role}\b", content, re.IGNORECASE):
            roles.append(role.capitalize())
    
    return roles

def extract_keywords_from_content(text, orientation_type):
    """카드 의미 텍스트에서 키워드 추출"""
    if not text or not isinstance(text, str):
        return ""
    
    # 키워드 추출 로직 (간단한 버전)
    # 실제로는 NLP 기법을 사용하여 더 정교하게 추출 가능
    max_words = 10
    words = text.lower().split()
    filtered_words = [word for word in words if len(word) > 3 and word.isalpha()]
    
    # 중복 제거 및 최대 길이 제한
    unique_words = list(set(filtered_words))[:max_words]
    
    return ", ".join(unique_words)

def summarize_text(text, max_length=100):
    """텍스트 요약 (간단한 버전)"""
    if not text or not isinstance(text, str):
        return ""
    
    # 첫 문장 또는 단락 추출
    if len(text) <= max_length:
        return text
    
    # 첫 문장 추출 시도
    import re
    sentences = re.split(r'[.!?]\s+', text)
    if sentences and len(sentences[0]) < max_length:
        return sentences[0]
    
    # 첫 문장이 너무 길면 단순 절단
    return text[:max_length] + "..."

def main():
    # 임베딩 모델 초기화 (한 번만)
    print("🤖 임베딩 모델 초기화 중...")
    model_name = "BAAI/bge-m3"
    model_kwargs = {"device": "cuda"}  # GPU 사용 시 "cuda"로 변경
    encode_kwargs = {"normalize_embeddings": True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print(f"✅ 임베딩 모델 로드 완료: {model_name}")
    
    # 1. 카드 데이터 처리
    print("\n" + "="*50)
    print("📚 카드 데이터 처리 중...")
    card_documents = load_csv_to_documents(CARD_CSV_PATH)
    
    if card_documents:
        print(f"✅ {len(card_documents)}개 카드 문서 로드 완료")
        
        # 카드 FAISS 인덱스 생성
        print("\n🔍 카드 FAISS 인덱스 생성 중...")
        try:
            card_db = FAISS.from_documents(card_documents, embeddings)
            print("✅ 카드 FAISS 인덱스 생성 완료")
            
            # 카드 인덱스 저장
            card_save_path = "tarot_card_faiss_index"
            card_db.save_local(card_save_path)
            print(f"✅ 카드 FAISS 인덱스 저장: {card_save_path}")
            
            # 카드 데이터 통계
            print_card_statistics(card_documents)
            
        except Exception as e:
            print(f"❌ 카드 FAISS 인덱스 생성 오류: {e}")
    
    # 2. 스프레드 데이터 처리
    print("\n" + "="*50)
    print("📚 스프레드 데이터 처리 중...")
    spread_documents = load_csv_to_documents(SPREAD_CSV_PATH)
    
    if spread_documents:
        print(f"✅ {len(spread_documents)}개 스프레드 문서 로드 완료")
        
        # 스프레드 FAISS 인덱스 생성
        print("\n🔍 스프레드 FAISS 인덱스 생성 중...")
        try:
            spread_db = FAISS.from_documents(spread_documents, embeddings)
            print("✅ 스프레드 FAISS 인덱스 생성 완료")
            
            # 스프레드 인덱스 저장
            spread_save_path = "tarot_spread_faiss_index"
            spread_db.save_local(spread_save_path)
            print(f"✅ 스프레드 FAISS 인덱스 저장: {spread_save_path}")
            
            # 스프레드 데이터 통계
            print_spread_statistics(spread_documents)
            
        except Exception as e:
            print(f"❌ 스프레드 FAISS 인덱스 생성 오류: {e}")
    
    # 3. 테스트 쿼리
    if card_documents and spread_documents:
        print("\n" + "="*50)
        print("🧪 쿼리 테스트 중...")
        
        # 카드 쿼리 테스트
        print("\n🃏 카드 쿼리 테스트:")
        card_queries = ["What does The Fool card mean?", "Ace of Cups meaning", "Death card reversed"]
        test_queries(card_db, card_queries, "카드")
        
        # 스프레드 쿼리 테스트
        print("\n🔮 스프레드 쿼리 테스트:")
        spread_queries = ["How to do Celtic Cross spread?", "3 card spread", "love relationship spread"]
        test_queries(spread_db, spread_queries, "스프레드")
    
    print(f"\n🎉 모든 FAISS 인덱스 생성 및 저장 완료!")

def print_card_statistics(documents):
    """카드 데이터 통계 출력"""
    print(f"\n📊 카드 데이터 통계:")
    print(f"총 카드 수: {len(documents)}")
    
    # 카드 타입별
    card_types = {}
    orientations = {}
    suits = {}
    ranks = {}
    
    for doc in documents:
        meta = doc.metadata
        
        # 카드 타입 통계
        card_type = meta.get("card_type", "unknown")
        card_types[card_type] = card_types.get(card_type, 0) + 1
        
        # 방향 통계
        orientation = meta.get("orientation", "unknown")
        orientations[orientation] = orientations.get(orientation, 0) + 1
        
        # 슈트 통계 (Minor Arcana 카드만)
        if meta.get("suit"):
            suit = meta.get("suit")
            suits[suit] = suits.get(suit, 0) + 1
            
        # 랭크 통계 (Minor Arcana 카드만)
        if meta.get("rank"):
            rank = meta.get("rank")
            ranks[rank] = ranks.get(rank, 0) + 1
    
    print("🃏 카드 타입별:")
    for card_type, count in card_types.items():
        print(f"  {card_type}: {count}개")
    
    print("🔄 방향별:")
    for orientation, count in orientations.items():
        print(f"  {orientation}: {count}개")
    
    if suits:
        print("♠️ 슈트별:")
        for suit, count in suits.items():
            print(f"  {suit}: {count}개")
    
    if ranks:
        print("👑 랭크별:")
        for rank, count in sorted(ranks.items(), key=lambda x: (
            0 if x[0] == "Ace" else
            1 if x[0] == "Two" else
            2 if x[0] == "Three" else
            3 if x[0] == "Four" else
            4 if x[0] == "Five" else
            5 if x[0] == "Six" else
            6 if x[0] == "Seven" else
            7 if x[0] == "Eight" else
            8 if x[0] == "Nine" else
            9 if x[0] == "Ten" else
            10 if x[0] == "Page" else
            11 if x[0] == "Knight" else
            12 if x[0] == "Queen" else
            13 if x[0] == "King" else 14
        )):
            print(f"  {rank}: {count}개")

def print_spread_statistics(documents):
    """스프레드 데이터 통계 출력"""
    print(f"\n📊 스프레드 데이터 통계:")
    print(f"총 스프레드 수: {len(documents)}")
    
    # 카드 수별 분포
    card_counts = {}
    # 개선: 포지션 수 분포 추가
    position_counts = {}
    
    for doc in documents:
        card_count = doc.metadata.get("card_count", 0)
        card_counts[card_count] = card_counts.get(card_count, 0) + 1
        
        # 포지션 수 통계 추가
        positions_count = doc.metadata.get("positions_count", 0)
        position_counts[positions_count] = position_counts.get(positions_count, 0) + 1
    
    print("🎯 카드 수별 분포:")
    for card_count in sorted(card_counts.keys()):
        count = card_counts[card_count]
        print(f"  {card_count}장: {count}개")
    
    # 포지션 수 통계 출력
    print("📍 포지션 수별 분포:")
    for positions_count in sorted(position_counts.keys()):
        count = position_counts[positions_count]
        print(f"  포지션 {positions_count}개: {count}개 스프레드")

def test_queries(db, queries, db_type):
    """쿼리 테스트"""
    for query in queries:
        print(f"\n🔍 쿼리: {query}")
        try:
            results = db.similarity_search(query, k=2)
            print(f"상위 2개 {db_type} 결과:")
            
            for i, doc in enumerate(results, 1):
                metadata = doc.metadata
                print(f"\n  결과 {i}:")
                print(f"    ID: {metadata.get('id', 'Unknown')}")
                
                # 카드 또는 스프레드 정보
                if metadata.get('card_name'):
                    print(f"    카드: {metadata['card_name']} ({metadata.get('card_type', '')})")
                    # 개선: 카드 메타데이터 추가 표시
                    if metadata.get('suit'):
                        print(f"    슈트: {metadata['suit']}")
                    if metadata.get('rank'):
                        print(f"    랭크: {metadata['rank']}")
                    if metadata.get('upright_keywords'):
                        print(f"    정방향 키워드: {metadata['upright_keywords'][:100]}...")
                    if metadata.get('reversed_keywords'):
                        print(f"    역방향 키워드: {metadata['reversed_keywords'][:100]}...")
                
                if metadata.get('spread_name'):
                    print(f"    스프레드: {metadata['spread_name']} ({metadata.get('card_count', 0)}장)")
                    # 개선: 정규화된 이름과 키워드 정보 표시
                    if metadata.get('normalized_name'):
                        print(f"    정규화 이름: {metadata['normalized_name']}")
                    if metadata.get('keywords'):
                        print(f"    키워드: {metadata['keywords'][:100]}...")
                    if metadata.get('positions_count'):
                        print(f"    포지션 수: {metadata['positions_count']}")
                
                print(f"    소스: {metadata.get('source', 'Unknown')}")
                print(f"    내용: {doc.page_content[:100]}...")
                
        except Exception as e:
            print(f"    ❌ 쿼리 테스트 오류: {e}")

if __name__ == "__main__":
    main()