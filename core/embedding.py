import yaml
from sentence_transformers import SentenceTransformer
import os

class Embedding:
    def __init__(self, model_path="models/all-MiniLM-L6-v2"):
        # 1. 强制离线（防止误触发下载）
        os.environ["HF_HUB_OFFLINE"] = "1"
        # 2. 转绝对路径（避免WSL路径问题）
        self.model_path = os.path.abspath(model_path)

        print(f"[Embedding] Loading model from: {self.model_path}")
        # 3. 加载模型
        self.model = SentenceTransformer(self.model_path)

    def embed(self, texts):
        # 统一输入格式
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, normalize_embeddings=False)

    def embed_batch(self, texts):
        """
        批量 embedding
        """
        return self.model.encode(texts)