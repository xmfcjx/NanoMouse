import os
from core.llm import LLM
from core.memory import Memory
from core.vector_store import VectorStore
from core.embedding import Embedding
from core.retriever import Retriever
from core.rerank import Rerank


def load_knowledge(vector_store, data_dir="data"):
    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            path = os.path.join(data_dir, file)
            print(f"Loading: {path}")
            vector_store.add(path)

    print(f"\nTotal chunks: {len(vector_store.texts)}\n")

def format_history(messages):
    lines = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")

    return "\n".join(lines)

def build_prompt(query, context, history):
    history_text = format_history(history)

    return f"""You are an AI assistant.

You are talking to a human user.

IMPORTANT:
- "User" is the human.
- "Assistant" is you.
- Always distinguish between User and Assistant identities.
- If the user says "my name is Tom", it means the USER's name is Tom.
- Do NOT confuse user identity with yourself.

Rules:
- If the question is about the user, use Conversation History.
- If about knowledge, use Context.
- If unknown, say "I don't know".

Conversation History:
{history_text}

Context:
{context}

User: {query}
Assistant:"""

def main():
    print("NanoChat (Memory + RAG) starting...\n")

    # 初始化
    embedding = Embedding()
    vector_store = VectorStore(embedding)

    reranker = Rerank()
    retriever = Retriever(vector_store, reranker=reranker)

    llm = LLM()

    # 🔥 Memory 初始化（控制长度）
    memory = Memory(max_history=5)

    # 加载知识库
    load_knowledge(vector_store)

    # 对话循环
    while True:
        query = input("\nYou: ")

        if query.lower() in ["exit", "quit"]:
            break

        # 1️⃣ 检索
        docs = retriever.retrieve(query, k=3, threshold=0.3)
        context = "\n".join(docs)

        print(f"\nRetrieved {len(docs)} chunks")

        # 控制 context 长度
        context = context[:800]

        # 2️⃣ 获取历史
        history = memory.get_messages()

        # 3️⃣ 构造 prompt
        prompt = build_prompt(query, context, history)

        # 4️⃣ 调用 LLM
        output = llm.generate(prompt)

        # 5️⃣ 去掉 prompt（避免回显）
        if prompt in output:
            response = output.split(prompt)[-1].strip()
        else:
            response = output.strip()

        print("\nAssistant:", response)

        # 6️⃣ 写入 memory（关键）
        memory.add_user_message(query)
        memory.add_assistant_message(response)


if __name__ == "__main__":
    main()