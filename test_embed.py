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