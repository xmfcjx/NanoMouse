import os
import numpy as np

from core.llm import LLM
from core.vector_store import VectorStore
from core.embedding import Embedding
from core.memory import Memory
from core.retriever import Retriever

print("start to run!")
# ====== Step 1: Init ======
embedding = Embedding()
vector_store = VectorStore(embedding)
retriever = Retriever(vector_store)

# ====== Step 2: English Knowledge Base ======
text = """
Cybersecurity involves protecting systems from attacks such as SQL injection, XSS, and malware.
Encryption ensures that data remains confidential during transmission.
Access control mechanisms restrict unauthorized users.
Intrusion detection systems monitor abnormal activities in networks.

Machine learning enables systems to learn patterns from data.
Neural networks are composed of multiple layers for complex tasks.
Transformers use attention mechanisms for NLP tasks.

Databases use indexing techniques like B-trees to improve query performance.
Hash indexes allow fast lookup for exact matches.
"""

print("step3!")
"""
# ====== Step 3: Simple split ======
def split_text(text, chunk_size=100):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

chunks = split_text(text)

print(f"\n[DEBUG] Total chunks: {len(chunks)}")
for i, c in enumerate(chunks):
    print(f"{i}: {c[:60]}")


print("step4!")
# ====== Step 4: Add to Vector Store ======
for chunk in chunks:
    vector_store.add_chunk(chunk)
"""
vector_store.add("data/knowledge.txt")

print("\n[DEBUG] Vector store size:", len(vector_store.texts))

print("step5!")
# ====== Step 5: Test Queries ======
queries = [
    "How to prevent cyber attacks?",
    "What is machine learning?",
    "How do databases improve query speed?"
]

for q in queries:
    print("\n" + "="*60)
    print("Query:", q)

    results = retriever.retrieve(q,debug=True)

    """print("\n[Top Results]")
    for r in results:
        print("->", r.strip())"""