import numpy as np
import yaml
import re

class VectorStore:
    def __init__(self, embedding):
        #self.embed_class = Embedding()
        #这样会每次都加载embed模型！
        self.embedding =embedding
        self.texts = []
        self.vectors = []
        self.add=self.add_text

    def add_text(self, text, chunk_size=80):
        if not text or not text.strip():
            return
        chunks = self.split_text(text, chunk_size)
        # batch embedding
        vectors = self.embedding.embed(chunks)
        if len(vectors)==0: return 
        vectors = np.array(vectors).reshape(len(chunks), -1)
        # 归一化
        #为之后求相似度做准备
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-6
        vectors = vectors / norms
        self.texts.extend(chunks)
        self.vectors.extend(vectors)
        return len(chunks)
    
    def add_chunk(self, text: str):
        emb = self.embedding.embed([text])[0]
        emb = np.array(emb)

        # 归一化
        norm = np.linalg.norm(emb) + 1e-6
        emb = emb / norm

        self.texts.append(text)
        self.vectors.append(emb)

    def split_text(self,text,chunk_size=80):
        sentences = re.findall(r'[^.!?]+[.!?]?', text)
        chunk=[]
        term=""
        term_length=0
        for i in sentences:
            i_length=len(i.split())
            #print("#####chunk####",i_length,term_length)
            #print(term_length+i_length,chunk_size)
            if i_length>=chunk_size:    #单个句子就超长！
                if term:
                    chunk.append(term)
                chunk.append(i)
                term=""
                term_length=0
            elif term_length+i_length>=chunk_size:  #加起来溢出
                term = term + " " + i
                chunk.append(term)
                term=i
                term_length=i_length
                #overlap
            else:
                if term is None:
                    term=i
                else:
                    term = term + " " + i
                term_length+=i_length
        if term:
            chunk.append(term)
        return chunk

    def search(self, query, k=3, threshold=0.3,debug=False):
        if len(self.vectors) == 0:
            return []
        # query embedding
        query_vec = np.array(self.embedding.embed(query)).squeeze()
        # query归一化
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-6)
        
        # 转为矩阵
        matrix = np.array(self.vectors)   # shape: (N, D)
        # 直接点积 = cosine similarity（因为已归一化）
        sims = np.dot(matrix, query_vec)  # shape: (N,)
        
        topk_idx = np.argsort(sims)[::-1][:k]
        
        if debug:
            print("\n[Retriever Debug]")
            for i in range(len(sims)):
                print(f"{i}: sim={sims[i]:.4f} | {self.texts[i][:50]}")
            
            print("\n[Top-K]")
            for idx in topk_idx:
                print(f"sim={sims[idx]:.4f} | {self.texts[idx][:100]}")
        

        result = []
        for idx in topk_idx:
            # threshold过滤
            if sims[idx] >= threshold:
                result.append(self.texts[idx])
        return result
    
    def show_text_chunk(self):
        for count,sen in enumerate(self.texts):
            print(f"len(sen)={len(sen.split())}")
            print(f"{count}:{sen}")
            print()