from sentence_transformers import CrossEncoder
import os
import numpy as np
class Rerank:
    def __init__(self, model_path="models/cross-encoder-ms-marco"):
        # 1. 强制离线（防止误触发下载）
        os.environ["HF_HUB_OFFLINE"] = "1"
        # 2. 转绝对路径（避免WSL路径问题）
        self.model_path = os.path.abspath(model_path)

        print(f"[Embedding] Loading model from: {self.model_path}")
        # 3. 加载模型
        self.model = CrossEncoder(self.model_path)

    def rerank(self,query,top_k,top_n=3):
        if len(top_k) < top_n:
            raise ValueError("top_n cannot be larger than number of documents")
        pairs = [(query, doc) for doc in top_k]
        scores = self.model.predict(pairs)
        result=[(score,text) for score,text in zip(scores,top_k)]
        result.sort(reverse=True)
        return  [doc for _,doc in result[:top_n]]
        #result=[(text,score) for text,score in zip(top_k,scores)]
        #result=sorted(result,key=lambda x:x[1] ,reverse=True)
    
    

"""[(query, doc1), (query, doc2), ...]
pairs = [
    (query, doc1),
    (query, doc2),
    (query, doc3)
]
"""