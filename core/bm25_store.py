import re
import jieba
from rank_bm25 import BM25Okapi
from config.config import get_config
from core.text_cleaner import clean_pdf_text_336_style, restore_protected_periods

class BM25Store:
    def __init__(self, chunk_size=None):
        self.chunk_size = chunk_size if chunk_size is not None else get_config("rag.chunk_size", 80)
        self.default_top_k = get_config("retriever.bm25_top_k", 6)
        self.corpus = []
        self.raw_docs = []
        self.bm25 = None

    def split_text(self, text, chunk_size=80):
        text = clean_pdf_text_336_style(text)
        sentences = re.findall(r'[^.!?]+[.!?]?', text)
        sentences = [restore_protected_periods(s) for s in sentences]
        chunk = []
        term = ""
        term_length = 0
        for i in sentences:
            i_length = len(i.split())
            if i_length >= chunk_size:
                if term:
                    chunk.append(term)
                chunk.append(i)
                term = ""
                term_length = 0
            elif term_length + i_length >= chunk_size:
                term = term + " " + i
                chunk.append(term)
                term = i
                term_length = i_length
            else:
                if term is None:
                    term = i
                else:
                    term = term + " " + i
                term_length += i_length
        if term:
            chunk.append(term)
        return chunk

    def _tokenize(self, text: str) -> list[str]:
        """分词：英文直接按空格切，中文用 jieba 切"""
        text = text.lower()
        tokens = jieba.lcut(text)
        tokens = [t.strip() for t in tokens if t.strip()]
        return tokens

    def add(self, text: str):
        if not text or not text.strip():
            return
        chunks = self.split_text(text, self.chunk_size)
        for chunk in chunks:
            self.raw_docs.append(chunk)
            self.corpus.append(self._tokenize(chunk))

    def build(self):
        """所有文档 add 完成后调用一次"""
        if not self.corpus:
            print("[BM25] Warning: No documents to build index.")
            return
        self.bm25 = BM25Okapi(self.corpus)
        print(f"[BM25] Index built with {len(self.corpus)} chunks.")

    def search(self, query: str, top_k: int = None) -> list[str]:
        if top_k is None:
            top_k = self.default_top_k
        if self.bm25 is None:
            print("[BM25] Error: Index not built. Call build() first.")
            return []

        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = [self.raw_docs[i] for i in top_indices]
        return results

    def __len__(self):
        return len(self.raw_docs)
