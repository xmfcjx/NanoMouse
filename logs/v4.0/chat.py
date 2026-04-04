"""
NanoChat-Lab 主程序 (v4.0 - 混合检索版)
新增 BM25 关键词检索，与向量检索互补
"""

import os
import re
from pypdf import PdfReader

from core.llm import LLM
from core.memory import Memory
from core.vector_store import VectorStore
from core.bm25_store import BM25Store      # 🎯 新增
from core.embedding import Embedding
from core.retriever import Retriever
from core.rerank import Rerank
from core.input_classifier import InputClassifier

# =========================
# 1. 全局初始化
# =========================
print("NanoChat-Lab Initializing... Please wait.")

embedding = Embedding()
vector_store = VectorStore(embedding)
bm25_store = BM25Store()                   # 🎯 新增
reranker = Rerank()
# 🎯 Retriever 同时接收 vector_store 和 bm25_store
retriever = Retriever(vector_store, bm25_store=bm25_store, reranker=reranker)
llm = LLM()
classifier = InputClassifier()


# =========================
# 2. 知识库加载（同时喂给 VectorStore 和 BM25Store）
# =========================
def load_knowledge(vector_store, bm25_store, data_dir="data"):
    """
    从磁盘读取文件，同时喂给向量库和 BM25 库
    """
    if not os.path.isdir(data_dir):
        return

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
                # 🎯 同时写入两个存储
                vec_count = vector_store.add_text(text_content)
                bm25_store.add(text_content)  # BM25 也加一份
                total_chunks += vec_count
                print(f"  Loaded {file}: {vec_count} chunks")

    # 🎯 BM25 需要手动 build（一次性建索引）
    bm25_store.build()
    return total_chunks


load_knowledge(vector_store, bm25_store)
print(f"✅ System Ready! Knowledge Base: {len(vector_store.texts)} chunks (Vector) + {len(bm25_store)} chunks (BM25).\n")


# =========================
# 3. 辅助函数
# =========================
def format_history(messages, max_turns=3):
    useful = [m for m in messages if m.get("role") in ("user", "assistant")]
    useful = useful[-2 * max_turns:]
    lines = []
    for msg in useful:
        role = msg["role"]
        content = msg["content"].strip()
        if role == "user":
            lines.append(f"User: {content}")
        else:
            lines.append(f"Assistant: {content}")
    return "\n".join(lines)


def build_prompt(query, context, history_text=""):
    """构建 RAG 专用的 Prompt"""
    context_block = context if context.strip() else "(empty)"
    history_block = history_text if history_text.strip() else "(none)"

    return f"""You are a strict factual assistant. Answer the question STRICTLY based on the provided Context.
Instructions:
1. IGNORE ALL YOUR PRIOR KNOWLEDGE. Use ONLY the provided Context to formulate your answer.
2. Provide a detailed explanation in at least 3 to 4 complete sentences. Elaborate on the key points from the Context.
3. If the Context is empty or does not contain the answer, you MUST say "I don't know." Do not guess.
4. Do not continue the conversation.

History: {history_block}
Context: {context_block}
Question: {query}
Answer: """


def postprocess_response(text: str) -> str:
    text = text.strip()
    prefixes = ["Assistant:", "assistant:", "Answer:", "answer:"]
    for p in prefixes:
        if text.startswith(p):
            text = text[len(p):].strip()
    markers = ["\nUser:", "\nAssistant:", "\nQuestion:"]
    cut = len(text)
    for mk in markers:
        idx = text.find(mk)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut].strip() if text else "I don't know."


# =========================
# 4. 核心接口
# =========================
def get_response(query, history=None):
    """
    :param query: 用户当前输入
    :param history: 由外部传入的历史对话列表（交互模式下来自 Memory，测试模式下来自脚本组装）
    :return: (response, predict_type, subtype)
    """
    if history is None:
        history = []
        
    # 🎯 将当前 query 临时加入历史，解决改名、连续提问等时序穿透问题
    temp_history = history + [{"role": "user", "content": query}]
    
    # 分类器判断（必须用 temp_history，才能看到最新输入）
    result = classifier.classify(query, temp_history)
    predict_type = result["type"]
    subtype = result.get("subtype", "")
    response = ""
    
    if result["handled"]:
        # Tool / Identity 等直接处理的类型
        response = result["value"]
    else:
        # RAG 处理流
        rag_data = result["value"]
        actual_question = rag_data["question"]
        inline_context = rag_data["context"]
        
        # 混合检索
        docs = retriever.retrieve(actual_question, k=3, threshold=0.3)
        retrieved_context = "\n".join(docs)
        
        if inline_context:
            # 🎯 核心防御：如果用户自带了 Context（如测试题），强行拼接到最前面
            context = inline_context + "\n" + retrieved_context
            # 🎯 绝杀防御：自带 Context 时，清空历史传给 LLM，彻底杜绝 Q16 污染 Q17 的乱码问题
            history_text = ""
        else:
            # 正常检索：只看检索到的内容
            context = retrieved_context
            # 正常检索：传递历史，让模型理解上下文
            history_text = format_history(temp_history, max_turns=3)
            
        context = context[:1000]
        
        # 构建 Prompt 并生成
        prompt = build_prompt(actual_question, context, history_text)
        output = llm.generate(prompt)
        response = postprocess_response(output)
        
    return response, predict_type, subtype



# =========================
# 5. 交互模式
# =========================
def main():
    print("NanoChat (v4.0 - Hybrid Retrieval)")
    print("Commands: /clear /exit\n")

    memory = Memory(max_history=8)

    while True:
        query = input("\nYou: ").strip()
        if not query:
            continue
        if query.lower() in ["/exit", "exit", "quit"]:
            break
        if query.lower() in ["/clear", "clear"]:
            memory.clear()
            print("\nAssistant: Memory cleared.")
            continue

        history = memory.get_messages()
        response, pred_type, subtype = get_response(query, history)

        # 显示分类信息
        if subtype:
            print(f"\n[Classify: {pred_type} > {subtype}]")
        else:
            print(f"\n[Classify: {pred_type}]")

        print(f"Assistant: {response}")

        memory.add_user_message(query)
        memory.add_assistant_message(response)


if __name__ == "__main__":
    main()
