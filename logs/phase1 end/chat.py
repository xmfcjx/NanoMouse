import os
import re
from pypdf import PdfReader
from core.llm import LLM
from core.memory import Memory
from core.vector_store import VectorStore
from core.embedding import Embedding
from core.retriever import Retriever
from core.rerank import Rerank
from core.input_classifier import InputClassifier  # ✅ 导入新的分类器

# =========================
# 加载知识库
# =========================
def load_knowledge(vector_store, data_dir="data"):
    """
    数据加载器：负责从磁盘读取各种格式的文件，并喂给 VectorStore
    """
    if not os.path.isdir(data_dir):
        print(f"Knowledge dir not found: {data_dir}")
        return

    # 遍历文件夹
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            path = os.path.join(root, file)
            text_content = ""
            
            # 1. 处理 TXT
            if file.endswith(".txt"):
                print(f"Loading TXT: {file}")
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        text_content = f.read()
                except Exception as e:
                    print(f"Error reading {file}: {e}")
                    continue
            
            # 2. 处理 PDF
            elif file.endswith(".pdf"):
                print(f"Loading PDF: {file}")
                try:
                    reader = PdfReader(path)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text_content += extracted + "\n"
                except Exception as e:
                    print(f"Error reading {file}: {e}")
                    continue
            
            # 3. 喂给 VectorStore
            if text_content.strip():
                count = vector_store.add(text_content)
                print(f"  └─ Added {count} chunks from {file}")

    print(f"\n✅ Total chunks in DB: {len(vector_store.texts)}\n")

# =========================
# History 格式化
# =========================
def format_history(messages, max_turns=3):
    """将 Memory 格式化为文本历史，用于构建 Prompt"""
    useful = [m for m in messages if m.get("role") in ("user", "assistant")]
    useful = useful[-2 * max_turns:]  # 取最近 N 轮
    lines = []
    for msg in useful:
        role = msg["role"]
        content = msg["content"].strip()
        if role == "user":
            lines.append(f"User: {content}")
        else:
            lines.append(f"Assistant: {content}")
    return "\n".join(lines)

# =========================
# Prompt 构建
# =========================
def build_prompt(query, context, history_text=""):
    context_block = context if context.strip() else "(empty)"
    history_block = history_text if history_text.strip() else "(none)"
    
    return f"""Answer the question based on the provided context.

Instructions:
1. Use the context to formulate your answer.
2. If the context is empty, say "I don't know".
3. Answer in 4-5 complete sentences. Do not just output keywords.
4. Do not continue the conversation.

History:
{history_block}

Context:
{context_block}

Question: {query}
Answer:
"""

# =========================
# 后处理
# =========================
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
    text = text[:cut].strip()
    
    return text if text else "I don't know."

# =========================
# 🎯 主函数
# =========================
def main():
    print("NanoChat (v3.1) starting...\n")
    print("Commands: /clear /exit\n")
    
    # 初始化组件
    embedding = Embedding()
    vector_store = VectorStore(embedding)
    reranker = Rerank()
    retriever = Retriever(vector_store, reranker=reranker)
    llm = LLM()
    memory = Memory(max_history=8)
    classifier = InputClassifier()  # ✅ 初始化分类器
    
    # 加载知识库
    load_knowledge(vector_store)
    
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
        
        #  1. 获取历史并使用分类器
        history = memory.get_messages()
        result = classifier.classify(query, history)
        
        response = ""
        
        # =========================
        # 2. 根据分类结果处理
        # =========================
        if result["handled"]:
            # --- 情况 A：规则已处理 (Identity/Arithmetic/Equation) ---
            response = result["value"]
            print(f"\nAssistant: {response}")
            
            # 调试信息：显示命中类型
            if result["type"] == "identity":
                print(f"[Debug] Identity matched from {'current query' if 'fallback_reason' not in result else 'history'}")
            elif result["type"] in ["arithmetic", "equation"]:
                print(f"[Debug] {result['type'].capitalize()} tool executed")

        else:
            # --- 情况 B：需要 LLM 处理 (RAG / Identity降级) ---
            rag_data = result["value"]
            actual_question = rag_data["question"]
            inline_context = rag_data["context"]
            
            # 如果是 Identity 降级过来的，给出提示
            if result["type"] == "rag" and result.get("fallback_reason") == "identity_not_found":
                print("[Debug] Identity not found in history, falling back to LLM...")
            
            # 1. 检索
            docs = retriever.retrieve(actual_question, k=3, threshold=0.3)
            retrieved_context = "\n".join(docs)
            
            # 2. 合并上下文
            if inline_context:
                context = inline_context + "\n" + retrieved_context
            else:
                context = retrieved_context
            
            context = context[:1000]  # 截断防止过长
            
            print(f"\n[Retrieved {len(docs)} chunks for RAG]")
            
            # 3. 格式化历史 (关键：让 LLM 记住上下文)
            history_text = format_history(history, max_turns=3)
            
            # 4. 构建 Prompt
            prompt = build_prompt(actual_question, context, history_text)
            
            # 5. LLM 生成
            output = llm.generate(prompt)
            response = postprocess_response(output)
            
            print(f"\nAssistant: {response}")
        
        # =========================
        # 3. 更新 Memory
        # =========================
        # 无论是规则处理还是LLM处理，都要存入 Memory，保证下一轮对话能获取到
        memory.add_user_message(query)
        memory.add_assistant_message(response)

# chat.py

# ... 原有的导入和初始化代码 ...

# 假设你的主循环类似这样，我们需要把核心逻辑提取出来
def get_response(query):
    """
    对外暴露的接口：输入问题，返回回答
    """
    # 1. 意图识别
    classify_result = classifier.classify(query)
    
    # 2. 根据意图分发
    if classify_result == "IDENTITY":
        # ... 你的 identity 处理逻辑 ...
        response = "..." 
    elif classify_result == "TOOL":
        # ... 你的 tool 调用逻辑 ...
        response = "..."
    elif classify_result == "RAG":
        # ... 你的 rag 逻辑 ...
        response = "..."
    else:
        response = "I don't know."
        
    return response

# ... main 函数里调用 get_response ...



if __name__ == "__main__":
    main()
