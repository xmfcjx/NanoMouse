"""
RAG 消融实验
测试不同检索配置的效果
用法: python test_rag_ablation.py
"""
import os
import time
from pypdf import PdfReader

from core.llm import LLM
from core.embedding import Embedding
from core.vector_store import VectorStore
from core.bm25_store import BM25Store
from core.rerank import Rerank
from core.retriever import Retriever
from eval.test_cases import RAG_TEST_CASES
from eval.metrics import check_recall, check_answer_quality
from eval.report import save_results, print_rag_summary


def load_knowledge(vector_store, bm25_store, data_dir="data"):
    """从磁盘读取文件，同时喂给向量库和 BM25 库"""
    if not os.path.isdir(data_dir):
        print(f"数据目录不存在: {data_dir}")
        return 0
    total_chunks = 0
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            path = os.path.join(root, file)
            text_content = ""
            if file.endswith(".txt"):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        text_content = f.read()
                except Exception as e:
                    print(f"Error reading {file}: {e}")
                    continue
            elif file.endswith(".pdf"):
                try:
                    reader = PdfReader(path)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text_content += extracted + "\n"
                except Exception as e:
                    print(f"Error reading {file}: {e}")
                    continue

            if text_content.strip():
                vec_count = vector_store.add_text(text_content)
                bm25_store.add(text_content)
                total_chunks += vec_count
                print(f"  Loaded {file}: {vec_count} chunks")

    bm25_store.build()
    return total_chunks


def build_prompt(question, context):
    return f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""


def postprocess_response(text):
    if "Answer:" in text:
        return text.split("Answer:")[-1].strip()
    return text.strip()


def run_rag_ablation():
    print("\n" + "=" * 60)
    print("RAG 消融实验")
    print("=" * 60)

    print("\n初始化组件...")
    llm = LLM()
    embedding = Embedding()
    vector_store = VectorStore(embedding)
    bm25_store = BM25Store()
    reranker = Rerank()

    print("\n加载知识库...")
    total_chunks = load_knowledge(vector_store, bm25_store)
    if total_chunks == 0:
        print("❌ 知识库为空，请检查 data 目录")
        return
    print(f"✅ 知识库加载完成: {len(vector_store.texts)} chunks (Vector) + {len(bm25_store)} chunks (BM25)")

    configs = {
        "仅Vector检索": Retriever(vector_store, bm25_store=None, reranker=None),
        "Vector+BM25": Retriever(vector_store, bm25_store=bm25_store, reranker=None),
        "Vector+Rerank": Retriever(vector_store, bm25_store=None, reranker=reranker),
        "完整混合检索": Retriever(vector_store, bm25_store=bm25_store, reranker=reranker),
    }

    test_cases = RAG_TEST_CASES

    results = {}

    for config_name, retriever in configs.items():
        print(f"\n--- 配置: {config_name} ---")
        recall_scores = []
        quality_scores = []
        latencies = []

        for case in test_cases:
            start = time.time()
            docs = retriever.retrieve(case["question"], k=3, threshold=0.3)
            retrieve_time = time.time() - start

            recall = check_recall(docs, case["expected_keywords"])

            context = "\n".join(docs) if docs else "(empty)"
            prompt = build_prompt(case["question"], context)
            gen_start = time.time()
            output = llm.generate(prompt, max_new_tokens=128)
            gen_time = time.time() - gen_start
            answer = postprocess_response(output)

            quality = check_answer_quality(answer, case["expected_keywords"])

            recall_scores.append(recall)
            quality_scores.append(quality)
            latencies.append(retrieve_time + gen_time)

            status = "✓" if quality > 0.5 else "✗"
            print(f"  [{case['id']}] {status} recall={recall:.2f} quality={quality:.2f} | docs={len(docs)} | answer={answer[:40]}...")

        avg_recall = sum(recall_scores) / len(recall_scores)
        avg_quality = sum(quality_scores) / len(quality_scores)
        avg_latency = sum(latencies) / len(latencies)

        results[config_name] = {
            "avg_recall": round(avg_recall, 4),
            "avg_quality": round(avg_quality, 4),
            "avg_latency": round(avg_latency, 2),
        }

        print(f"  → 平均召回率: {avg_recall:.2%} | 平均准确率: {avg_quality:.2%} | 平均延迟: {avg_latency:.2f}s")

    save_results("rag_ablation", results)
    print_rag_summary(results)

    return results


if __name__ == "__main__":
    run_rag_ablation()
