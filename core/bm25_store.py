import re
import jieba
from rank_bm25 import BM25Okapi

class BM25Store:
    def __init__(self, chunk_size=80):
        self.chunk_size = chunk_size
        self.corpus = []      # 分词后的 chunk 列表
        self.raw_docs = []    # 原始 chunk 列表（返回用）
        self.bm25 = None      # BM25 实例

    # 和之前vector里面一样
    def split_text(self, text, chunk_size=80):
        sentences = re.findall(r'[^.!?]+[.!?]?', text)
        chunk = []
        term = ""
        term_length = 0
        for i in sentences:
            i_length = len(i.split())
            if i_length >= chunk_size:    # 单个句子就超长！
                if term:
                    chunk.append(term)
                chunk.append(i)
                term = ""
                term_length = 0
            elif term_length + i_length >= chunk_size:  # 加起来溢出
                term = term + " " + i
                chunk.append(term)
                term = i
                term_length = i_length
                # overlap 逻辑
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

    def search(self, query: str, top_k: int = 5) -> list[str]:
        """搜索与 query 最相关的文档"""
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
