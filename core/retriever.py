from config.config import get_config

class Retriever:
    def __init__(self, vector_store, bm25_store=None, reranker=None,
                 rrf_k=None, rrf_alpha=None):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.reranker = reranker
        self.rrf_k = rrf_k if rrf_k is not None else get_config("retriever.rrf_k", 60)
        self.rrf_alpha = rrf_alpha if rrf_alpha is not None else get_config("retriever.rrf_alpha", 0.4)

    def _merge_and_dedup(self, list_a, list_b, use_rrf=True):
        """
        合并两个列表并去重
        use_rrf: 是否使用 Reciprocal Rank Fusion
        """
        k = self.rrf_k
        alpha = self.rrf_alpha
        if not use_rrf:
            seen = set()
            merged = []
            for doc in list_a + list_b:
                if doc not in seen:
                    seen.add(doc)
                    merged.append(doc)
            return merged
        
        scores = {}
        for rank, doc in enumerate(list_a, 1):
            if doc not in scores:
                scores[doc] = 0
            scores[doc] += alpha * 1 / (k + rank)
        
        for rank, doc in enumerate(list_b, 1):
            if doc not in scores:
                scores[doc] = 0
            scores[doc] += (1 - alpha) * 1 / (k + rank)
        
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in sorted_docs]


    def retrieve(self, query, k=3, threshold=0.3, debug=False):
        """
        混合检索流程:
        1. 向量检索召回
        2. BM25 检索召回
        3. 合并去重
        4. Rerank 精排
        :param query: 用户查询
        :param k: 最终返回数量
        :param threshold: 向量检索阈值
        :param debug: 是否打印调试信息
        :return: 精排后的文档列表
        """
        # ========== 第 1 步：向量检索 ==========
        vector_docs = self.vector_store.search(query, k * 2, threshold, debug)

        # ========== 第 2 步：BM25 检索 ==========
        bm25_docs = []
        if self.bm25_store is not None:
            bm25_docs = self.bm25_store.search(query, top_k=k * 2)
            if debug:
                print(f"[Retriever] BM25 search returned {len(bm25_docs)} docs")

        # ========== 第 3 步：合并去重 ==========
        merged_docs = self._merge_and_dedup(vector_docs, bm25_docs)
        if debug:
            print(f"[Retriever] After merge & dedup: {len(merged_docs)} docs")

        # 如果合并后没有结果
        if not merged_docs:
            return []

        # ========== 第 4 步：Rerank 精排 ==========
        # 如果没有 reranker，或者候选数量已经 <= k，直接返回
        if self.reranker is None or len(merged_docs) <= k:
            return merged_docs[:k]

        return self.reranker.rerank(query, merged_docs, k)
    
"""
    def retrieve(self, query, k=3,threshold=0.3,debug=False):
        docs = self.vector_store.search(query, 2 * k, threshold, debug)
        if not docs:
            return []

        if self.reranker is None or len(docs) <= k:
            return docs[:k]

        return self.reranker.rerank(query, docs, k)"""
