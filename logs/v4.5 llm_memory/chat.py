""" 
NanoChat-Lab 主程序 (v4.5 - LLM 驱动记忆系统)
使用 LLM 理解语义，而非硬编码规则
""" 
import os 
import re 
from pypdf import PdfReader 

from core.llm import LLM 
from core.memory import Memory 
from core.memory_store import MemoryStore
from core.vector_store import VectorStore 
from core.bm25_store import BM25Store 
from core.embedding import Embedding 
from core.retriever import Retriever 
from core.rerank import Rerank 
from core.input_classifier import InputClassifier 
from core.ReActAgent import ReActAgent 

# ========================= 
# 1. 全局初始化 
# ========================= 
print("NanoChat-Lab Initializing... Please wait.") 
embedding = Embedding() 
vector_store = VectorStore(embedding) 
bm25_store = BM25Store() 
reranker = Rerank() 

retriever = Retriever(vector_store, bm25_store=bm25_store, reranker=reranker) 

llm = LLM() 
agent = ReActAgent(llm, max_steps=3) 
classifier = InputClassifier() 
memory_store = MemoryStore(llm=llm) 

# ========================= 
# 2. 知识库加载（同时喂给 VectorStore 和 BM25Store） 
# ========================= 
def load_knowledge(vector_store, bm25_store, data_dir="data"): 
    """ 从磁盘读取文件，同时喂给向量库和 BM25 库 """ 
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
                vec_count = vector_store.add_text(text_content) 
                bm25_store.add(text_content) 
                total_chunks += vec_count 
                print(f"  Loaded {file}: {vec_count} chunks") 

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

        if role == "user" and "Context:" in content: 
            q_match = re.search(r'(?:Q|Question):\s*(.*)', content, re.IGNORECASE | re.DOTALL) 
            if q_match: 
                content = q_match.group(1).strip() 
            else: 
                continue 

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
def get_response(query, history=None, memory=None): 
    """ 
    :param query: 用户当前输入 
    :param history: 由外部传入的历史对话列表 
    :param memory: Memory 对象，用于对话历史
    :return: (response, predict_type, subtype) 
    """ 
    if history is None: history = [] 
    if memory is None:
        from core.memory import Memory
        memory = Memory()
    
    temp_history = history + [{"role": "user", "content": query}] 

    # ================= 1. LLM 驱动的记忆处理 =================
    response, pred_type, subtype = memory_store.process(query)
    if response:
        return response, pred_type, subtype

    # ================= 2. 分类器判断 =================
    result = classifier.classify(query, temp_history) 
    predict_type = result["type"] 
    subtype = result.get("subtype", "") 
    response = "" 

    if result["handled"]: 
        response = result["value"] 
        
    elif predict_type == "agent": 
        # ================= Agent 直通车：不经过 LLM，直接调工具 =================
        
        if subtype == "time" or subtype == "date_today":
            response = agent.direct_call("time")
            
        elif subtype == "weekday":
            response = agent.direct_call("weekday", query)
            
        elif subtype == "days_between":
            response = agent.direct_call("days_between", query)
            
        elif subtype == "equation":
            response = agent.direct_call("solve", query)
            
        elif subtype == "arithmetic":
            # 提取纯算术表达式，包装成 print() 交给 calc
            expr = classifier.extract_arithmetic_expr(query)
            if expr:
                response = agent.direct_call("calc", f"print({expr})")
            else:
                response = "无法解析算术表达式"
                
        elif subtype == "base_convert":
            # 进制转换：直接调用 base_convert 工具
            response = agent.direct_call("base_convert", query)
                
        elif subtype == "multi_tool":
            # 多工具问题：走 ReAct 循环，让模型自己决定调用什么工具
            agent_response = agent.run(query)
            final_match = re.search(r"Final Answer:\s*(.*)", agent_response, re.IGNORECASE | re.DOTALL)
            response = final_match.group(1).strip() if final_match else agent_response.strip()
                
        else:
            # 未知 agent 子类型，走 ReAct 循环兜底
            agent_response = agent.run(query)
            final_match = re.search(r"Final Answer:\s*(.*)", agent_response, re.IGNORECASE | re.DOTALL)
            response = final_match.group(1).strip() if final_match else agent_response.strip()
            
    else: 
        # RAG 处理流 
        rag_data = result["value"] 
        actual_question = rag_data["question"] 
        inline_context = rag_data["context"] 

        docs = retriever.retrieve(actual_question, k=3, threshold=0.3) 
        retrieved_context = "\n".join(docs) 

        if inline_context: 
            context = inline_context + "\n" + retrieved_context 
            history_text = "" 
        else: 
            context = retrieved_context 
            history_text = format_history(temp_history, max_turns=3) 

        context = context[:1000] 
        prompt = build_prompt(actual_question, context, history_text) 
        output = llm.generate(prompt) 
        response = postprocess_response(output) 

    return response, predict_type, subtype 

# ========================= 
# 5. 交互模式 
# ========================= 
def main(): 
    print("NanoChat (v4.5 - LLM-Driven Memory)") 
    print("Commands: /clear /exit\n") 
    memory = Memory(max_history=8) 

    while True: 
        query = input("\nYou: ").strip() 
        if not query: continue 
        if query.lower() in ["/exit", "exit", "quit"]: break 
        if query.lower() in ["/clear", "clear"]: 
            memory.clear() 
            memory_store.clear()
            print("\nAssistant: Memory cleared.") 
            continue 

        history = memory.get_messages() 
        response, pred_type, subtype = get_response(query, history, memory) 

        if subtype: 
            print(f"\n[Classify: {pred_type} > {subtype}]") 
        else: 
            print(f"\n[Classify: {pred_type}]") 

        print(f"Assistant: {response}") 
        memory.add_user_message(query) 
        memory.add_assistant_message(response) 

if __name__ == "__main__": 
    main()
