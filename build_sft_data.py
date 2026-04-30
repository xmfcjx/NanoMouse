"""
多任务 SFT 训练数据自动构造
生成三类数据：RAG 答案生成 / ReAct 工具调用 / Memory JSON 提取
输出: data/sft_train.jsonl

用法:
  python build_sft_data.py              # 构造全部数据
  python build_sft_data.py --task rag   # 只构造 RAG 数据
"""
import os
import sys
import json
import random
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_PATH = "data/sft_train.jsonl"

REACT_SYSTEM_PROMPT = """You are a helpful assistant. You MUST use tools to answer questions.

Tools:
- calc: Python math. Input: print(expr). Example: print(3*8)
- time: Current time. No input.
- weather: City weather. Input: city name.
- convert: Unit convert. Input: value+unit to target. Example: 100km to miles
- base_convert: Base convert. Input: number to base. Example: 255 to hex
- solve: Equations. Input: equation. Example: 3*x+1=10

Format: Thought: / Action: / Action Input: / Final Answer:
For multi-step tasks, call tools one at a time."""


# ============================================================
# RAG 训练数据构造
# ============================================================

RAG_QA_PAIRS = [
    {
        "question": "What is QuickSort?",
        "context": "QuickSort is a divide-and-conquer sorting algorithm. It works by selecting a pivot element and partitioning the array into two sub-arrays: elements less than the pivot and elements greater than the pivot. The average time complexity is O(n log n).",
        "answer": "QuickSort is a divide-and-conquer sorting algorithm that works by selecting a pivot element and partitioning the array around it. Elements smaller than the pivot go to one side, and larger elements go to the other. This process is applied recursively. Its average time complexity is O(n log n)."
    },
    {
        "question": "How does BERT work?",
        "context": "BERT (Bidirectional Encoder Representations from Transformers) is a pre-training method for language models. Unlike previous models that read text left-to-right, BERT uses bidirectional training with masked language modeling (MLM), where random tokens are masked and the model learns to predict them from context.",
        "answer": "BERT is a bidirectional pre-training method based on the Transformer architecture. It uses masked language modeling (MLM) during pre-training, where random tokens in the input are masked and the model learns to predict them using both left and right context. This bidirectional approach allows BERT to capture deeper language understanding."
    },
    {
        "question": "What is the difference between TCP and UDP?",
        "context": "TCP (Transmission Control Protocol) is a reliable, connection-oriented protocol that ensures data delivery through acknowledgments and retransmissions. UDP (User Datagram Protocol) is a connectionless protocol that sends datagrams without guaranteeing delivery, making it faster but less reliable.",
        "answer": "TCP is a reliable, connection-oriented protocol that guarantees data delivery through acknowledgments and retransmissions. UDP is a connectionless protocol that sends datagrams without delivery guarantees, making it faster but less reliable. TCP is used for applications requiring reliability (web, email), while UDP is preferred for real-time applications (video streaming, gaming)."
    },
    {
        "question": "What is virtual memory?",
        "context": "Virtual memory is a memory management technique that creates an illusion of a large contiguous address space. It uses page tables to map virtual addresses to physical addresses. When physical memory is full, demand paging swaps pages between RAM and disk.",
        "answer": "Virtual memory is a memory management technique that provides each process with its own virtual address space, creating the illusion of large contiguous memory. It uses page tables to translate virtual addresses to physical addresses. When physical RAM is full, demand paging moves less-used pages to disk and loads needed pages back into RAM."
    },
    {
        "question": "How does a GPU differ from a CPU?",
        "context": "A GPU (Graphics Processing Unit) is designed for massive parallelism with thousands of smaller cores optimized for SIMD (Single Instruction, Multiple Data) operations. A CPU has fewer but more powerful cores optimized for sequential processing and complex logic.",
        "answer": "GPUs have thousands of smaller cores designed for SIMD (Single Instruction, Multiple Data) parallel processing, making them ideal for tasks like matrix operations and graphics rendering. CPUs have fewer but more powerful cores optimized for sequential processing and complex branching logic. GPUs excel at parallelism while CPUs excel at single-threaded performance."
    },
    {
        "question": "What is the CAP theorem?",
        "context": "The CAP theorem states that a distributed system can only guarantee two of three properties simultaneously: Consistency (all nodes see the same data), Availability (every request receives a response), and Partition tolerance (the system continues operating despite network partitions).",
        "answer": "The CAP theorem states that in a distributed system, you can only guarantee two out of three properties: Consistency (all nodes see the same data at the same time), Availability (every request gets a response), and Partition tolerance (the system works despite network failures). Since network partitions are inevitable, systems must choose between consistency and availability."
    },
    {
        "question": "What is Moore's Law?",
        "context": "Moore's Law is an observation by Gordon Moore that the number of transistors on a microchip doubles approximately every two years, while the cost of computers is halved. This trend has driven exponential growth in computing power since the 1960s.",
        "answer": "Moore's Law is the observation by Gordon Moore that the number of transistors on a microchip doubles approximately every two years. This has driven exponential growth in computing power and reduction in cost per transistor since the 1960s, though the trend has been slowing in recent years due to physical limitations."
    },
    {
        "question": "How does TLS work?",
        "context": "TLS (Transport Layer Security) secures communication through a handshake process: the client and server exchange certificates for authentication, negotiate encryption algorithms, and establish a shared session key for symmetric encryption of subsequent data.",
        "answer": "TLS secures communication through a handshake: first, the server presents its certificate for authentication. Then the client and server negotiate encryption algorithms and exchange key material to establish a shared session key. All subsequent data is encrypted using this symmetric session key, providing confidentiality and integrity."
    },
    {
        "question": "What is dynamic programming?",
        "context": "Dynamic programming is an optimization technique for problems with overlapping subproblems and optimal substructure. It avoids redundant computation through memoization (top-down, storing results of solved subproblems) or tabulation (bottom-up, building solutions iteratively).",
        "answer": "Dynamic programming is an optimization technique for problems with overlapping subproblems and optimal substructure. It avoids redundant computation using two approaches: memoization (top-down, caching results of recursive calls) or tabulation (bottom-up, iteratively building a solution table). Classic examples include Fibonacci, knapsack, and shortest path problems."
    },
    {
        "question": "What are design patterns?",
        "context": "Design patterns are reusable solutions to common software design problems. They are categorized into three types: creational patterns (object creation), structural patterns (object composition), and behavioral patterns (object interaction and responsibility).",
        "answer": "Design patterns are proven, reusable solutions to recurring software design problems. They fall into three categories: creational patterns (like Singleton, Factory) for object creation, structural patterns (like Adapter, Decorator) for object composition, and behavioral patterns (like Observer, Strategy) for managing object interaction and responsibility."
    },
    {
        "question": "What is the Von Neumann architecture?",
        "context": "The Von Neumann architecture is a computer design where a single memory stores both instructions and data (stored-program concept). The CPU fetches instructions from memory, decodes them, and executes them sequentially. This architecture forms the basis of most modern computers.",
        "answer": "The Von Neumann architecture is a stored-program computer design where both instructions and data share the same memory. The CPU sequentially fetches instructions from memory, decodes them, and executes them. This architecture, which forms the basis of most modern computers, has a bottleneck: the shared bus between CPU and memory limits throughput."
    },
    {
        "question": "How does pipelining work in CPU?",
        "context": "CPU pipelining divides instruction execution into stages: fetch, decode, execute, memory access, and write-back. Multiple instructions overlap in different stages simultaneously, increasing throughput. Pipeline hazards (data, control, structural) can cause stalls.",
        "answer": "CPU pipelining divides instruction execution into stages (fetch, decode, execute, memory access, write-back) and overlaps multiple instructions in different stages simultaneously. This increases throughput by allowing one instruction to complete per clock cycle in ideal conditions. However, pipeline hazards (data dependencies, branch mispredictions, resource conflicts) can cause stalls that reduce efficiency."
    },
    {
        "question": "What is a hash table?",
        "context": "A hash table is a data structure that maps keys to values using a hash function. It provides O(1) average-case lookup, insertion, and deletion. Collisions (when different keys hash to the same index) are resolved through chaining (linked lists) or open addressing.",
        "answer": "A hash table maps keys to values using a hash function for O(1) average-case operations (lookup, insertion, deletion). When different keys hash to the same index (collision), it's resolved through chaining (storing colliding entries in a linked list) or open addressing (probing for the next empty slot). Load factor affects performance."
    },
    {
        "question": "What is gradient boosting?",
        "context": "Gradient boosting is an ensemble machine learning method that builds models iteratively. Each new model is trained to correct the errors (residuals) of the previous ensemble. It combines many weak learners (typically decision trees) into a strong predictor.",
        "answer": "Gradient boosting is an ensemble method that iteratively builds models, where each new model is trained to correct the residual errors of the previous ensemble. It combines many weak learners (usually shallow decision trees) into a strong predictor by optimizing a loss function via gradient descent in function space. Popular implementations include XGBoost, LightGBM, and CatBoost."
    },
    {
        "question": "What is the difference between RISC and CISC?",
        "context": "RISC (Reduced Instruction Set Computer) uses fixed-length, simple instructions that execute in a single cycle. CISC (Complex Instruction Set Computer) uses variable-length instructions that can perform complex operations but may take multiple cycles.",
        "answer": "RISC uses a small set of fixed-length, simple instructions that typically execute in a single clock cycle, relying on the compiler for optimization. CISC uses variable-length instructions that can perform complex multi-step operations but may take multiple cycles. Modern processors often combine both approaches: x86 (CISC) internally translates to RISC-like micro-ops."
    },
    {
        "question": "What is machine learning?",
        "context": "Machine learning is a subset of artificial intelligence where algorithms learn patterns from data without being explicitly programmed. Models are trained on datasets to make predictions or decisions. Common types include supervised learning, unsupervised learning, and reinforcement learning.",
        "answer": "Machine learning is a branch of AI where algorithms automatically learn patterns from data to make predictions or decisions. It includes supervised learning (labeled data), unsupervised learning (finding hidden patterns), and reinforcement learning (learning through rewards). The model improves its performance as it processes more training data."
    },
    {
        "question": "Explain neural networks",
        "context": "Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected neurons: an input layer, one or more hidden layers, and an output layer. Each neuron applies a weighted sum and activation function to its inputs.",
        "answer": "Neural networks are computing systems with layers of interconnected neurons: input layer, hidden layers, and output layer. Each neuron computes a weighted sum of its inputs, applies an activation function, and passes the result forward. The network learns by adjusting weights through backpropagation to minimize a loss function."
    },
    {
        "question": "What is deep learning?",
        "context": "Deep learning is a subset of machine learning that uses neural networks with many layers (deep architectures). These deep networks can automatically learn hierarchical feature representations from raw data, eliminating the need for manual feature engineering.",
        "answer": "Deep learning uses neural networks with many layers (deep architectures) to automatically learn hierarchical feature representations from raw data. Lower layers learn simple features (edges, textures), while deeper layers combine them into complex abstractions. This eliminates manual feature engineering and has achieved breakthroughs in vision, NLP, and speech."
    },
    {
        "question": "What is a database index?",
        "context": "A database index is a data structure that improves the speed of data retrieval operations. It works like a book index, allowing the database to find rows without scanning the entire table. Common implementations include B-tree and hash indexes.",
        "answer": "A database index is a data structure that speeds up query operations by allowing the database to locate rows without full table scans. Like a book index, it maps key values to row locations. B-tree indexes support range queries, while hash indexes are optimal for exact-match lookups. Indexes trade write performance and storage for faster reads."
    },
    {
        "question": "How does garbage collection work?",
        "context": "Garbage collection is an automatic memory management technique that reclaims memory occupied by objects no longer in use. Common algorithms include mark-and-sweep (marking reachable objects, then collecting unmarked ones) and generational collection (separating objects by age).",
        "answer": "Garbage collection automatically reclaims memory from objects no longer referenced by the program. Mark-and-sweep traces reachable objects from root references and collects unreachable ones. Generational collection optimizes this by separating objects by age, since most objects die young. This eliminates manual memory management but introduces pause times."
    },
]


# ============================================================
# ReAct 训练数据构造
# ============================================================

def build_react_data():
    samples = []

    calc_cases = [
        ("Calculate 25 * 4", "calc", "print(25 * 4)", "100"),
        ("Calculate 15 + 27", "calc", "print(15 + 27)", "42"),
        ("Calculate 100 / 4", "calc", "print(100 / 4)", "25.0"),
        ("Calculate 2 ** 10", "calc", "print(2 ** 10)", "1024"),
        ("Calculate sqrt(144)", "calc", "print(144 ** 0.5)", "12.0"),
        ("Calculate (7 + 3) * 5", "calc", "print((7 + 3) * 5)", "50"),
        ("Calculate 88 % 7", "calc", "print(88 % 7)", "4"),
        ("Calculate 50 - 17", "calc", "print(50 - 17)", "33"),
        ("Calculate 12 * 12", "calc", "print(12 * 12)", "144"),
        ("Calculate 3 ** 5", "calc", "print(3 ** 5)", "243"),
    ]

    for q, tool, tool_input, expected in calc_cases:
        samples.append({
            "task": "react",
            "messages": [
                {"role": "system", "content": REACT_SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": f"Thought: I need to compute a mathematical expression.\nAction: {tool}\nAction Input: {tool_input}"},
                {"role": "user", "content": f"Observation: {expected}"},
                {"role": "assistant", "content": f"Thought: The calculation is complete.\nFinal Answer: {expected}"},
            ]
        })

    solve_cases = [
        ("Solve 2x + 5 = 15", "solve", "2*x+5=15", "x = 5"),
        ("Solve 3x = 12", "solve", "3*x=12", "x = 4"),
        ("Solve x + 10 = 25", "solve", "x+10=25", "x = 15"),
        ("Solve x * x = 49", "solve", "x*x=49", "x = 7"),
        ("Solve 5x - 3 = 22", "solve", "5*x-3=22", "x = 5"),
        ("Solve 4x = 100", "solve", "4*x=100", "x = 25"),
        ("Solve x/2 = 10", "solve", "x/2=10", "x = 20"),
        ("Solve 7x + 1 = 50", "solve", "7*x+1=50", "x = 7"),
    ]

    for q, tool, tool_input, expected in solve_cases:
        samples.append({
            "task": "react",
            "messages": [
                {"role": "system", "content": REACT_SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": f"Thought: I need to solve an equation.\nAction: {tool}\nAction Input: {tool_input}"},
                {"role": "user", "content": f"Observation: {expected}"},
                {"role": "assistant", "content": f"Thought: The equation is solved.\nFinal Answer: {expected}"},
            ]
        })

    convert_cases = [
        ("Convert 100km to miles", "convert", "100km to miles", "62.14 miles"),
        ("Convert 36 celsius to fahrenheit", "convert", "36c to f", "96.8°F"),
        ("Convert 50 fahrenheit to celsius", "convert", "50f to c", "10.0°C"),
        ("Convert 1 mile to km", "convert", "1 miles to km", "1.61 km"),
        ("Convert 10 kg to pounds", "convert", "10 kg to pounds", "22.05 pounds"),
        ("Convert 100 pounds to kg", "convert", "100 pounds to kg", "45.36 kg"),
    ]

    for q, tool, tool_input, expected in convert_cases:
        samples.append({
            "task": "react",
            "messages": [
                {"role": "system", "content": REACT_SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": f"Thought: I need to convert units.\nAction: {tool}\nAction Input: {tool_input}"},
                {"role": "user", "content": f"Observation: {expected}"},
                {"role": "assistant", "content": f"Thought: The conversion is done.\nFinal Answer: {expected}"},
            ]
        })

    base_cases = [
        ("Convert 255 to binary", "base_convert", "255 to bin", "11111111"),
        ("Convert 255 to hex", "base_convert", "255 to hex", "FF"),
        ("Convert 64 to binary", "base_convert", "64 to bin", "1000000"),
        ("Convert 1010 binary to decimal", "base_convert", "1010 to dec", "10"),
        ("Convert FF hex to decimal", "base_convert", "FF to dec", "255"),
        ("Convert 100 to octal", "base_convert", "100 to oct", "144"),
    ]

    for q, tool, tool_input, expected in base_cases:
        samples.append({
            "task": "react",
            "messages": [
                {"role": "system", "content": REACT_SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": f"Thought: I need to convert between number bases.\nAction: {tool}\nAction Input: {tool_input}"},
                {"role": "user", "content": f"Observation: {expected}"},
                {"role": "assistant", "content": f"Thought: The base conversion is complete.\nFinal Answer: {expected}"},
            ]
        })

    str_cases = [
        ("What is the length of 'hello world'?", "str_tools", "hello world, len", "11"),
        ("Reverse the string 'python'", "str_tools", "python, reverse", "nohtyp"),
        ("Convert 'hello' to uppercase", "str_tools", "hello, upper", "HELLO"),
        ("Convert 'WORLD' to lowercase", "str_tools", "WORLD, lower", "world"),
    ]

    for q, tool, tool_input, expected in str_cases:
        samples.append({
            "task": "react",
            "messages": [
                {"role": "system", "content": REACT_SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": f"Thought: I need to perform a string operation.\nAction: {tool}\nAction Input: {tool_input}"},
                {"role": "user", "content": f"Observation: {expected}"},
                {"role": "assistant", "content": f"Thought: The string operation is done.\nFinal Answer: {expected}"},
            ]
        })

    stats_cases = [
        ("What is the max of 3, 7, 1, 9, 4?", "statistics", "3,7,1,9,4, max", "9"),
        ("What is the min of 10, 5, 8, 2?", "statistics", "10,5,8,2, min", "2"),
        ("What is the sum of 1, 2, 3, 4, 5?", "statistics", "1,2,3,4,5, sum", "15"),
        ("What is the average of 10, 20, 30?", "statistics", "10,20,30, avg", "20.0"),
    ]

    for q, tool, tool_input, expected in stats_cases:
        samples.append({
            "task": "react",
            "messages": [
                {"role": "system", "content": REACT_SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": f"Thought: I need to compute a statistic.\nAction: {tool}\nAction Input: {tool_input}"},
                {"role": "user", "content": f"Observation: {expected}"},
                {"role": "assistant", "content": f"Thought: The computation is complete.\nFinal Answer: {expected}"},
            ]
        })

    weather_cases = [
        ("What is the weather in Beijing?", "weather", "Beijing", "Clear skies, 22°C"),
        ("How is the weather in Tokyo?", "weather", "Tokyo", "Cloudy, 18°C"),
        ("What's the weather like in London?", "weather", "London", "Rainy, 12°C"),
        ("Tell me the weather in New York.", "weather", "New York", "Sunny, 25°C"),
        ("Check weather for Paris.", "weather", "Paris", "Partly cloudy, 16°C"),
        ("Is it raining in Sydney?", "weather", "Sydney", "Clear skies, 20°C"),
        ("What's the temperature in Berlin?", "weather", "Berlin", "Overcast, 10°C"),
        ("Give me the weather forecast for Dubai.", "weather", "Dubai", "Sunny, 35°C"),
        ("Weather in Moscow please.", "weather", "Moscow", "Snow, -5°C"),
        ("How's the weather in Singapore?", "weather", "Singapore", "Humid, 30°C"),
    ]

    for q, tool, tool_input, expected in weather_cases:
        samples.append({
            "task": "react",
            "messages": [
                {"role": "system", "content": REACT_SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": f"Thought: I need to check the weather.\nAction: {tool}\nAction Input: {tool_input}"},
                {"role": "user", "content": f"Observation: {expected}"},
                {"role": "assistant", "content": f"Thought: I have the weather information.\nFinal Answer: {expected}"},
            ]
        })

    time_cases = [
        ("What time is it?", "time", "now", "2026-04-29 14:30:00"),
        ("What's the current time?", "time", "now", "2026-04-29 14:30:00"),
        ("Tell me the current date and time.", "time", "now", "2026-04-29 14:30:00"),
        ("What is today's date?", "time", "now", "2026-04-29 14:30:00"),
        ("What day is it today?", "time", "now", "2026-04-29 14:30:00"),
        ("Current time please.", "time", "now", "2026-04-29 14:30:00"),
    ]

    for q, tool, tool_input, expected in time_cases:
        samples.append({
            "task": "react",
            "messages": [
                {"role": "system", "content": REACT_SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": f"Thought: I need to get the current time.\nAction: {tool}\nAction Input: {tool_input}"},
                {"role": "user", "content": f"Observation: {expected}"},
                {"role": "assistant", "content": f"Thought: I have the time information.\nFinal Answer: {expected}"},
            ]
        })

    return samples


# ============================================================
# Memory 训练数据构造
# ============================================================

MEMORY_SAMPLES = [
    ("I'm called Alice", {"action": "set", "key": "name", "value": "Alice"}),
    ("I go by Bob", {"action": "set", "key": "name", "value": "Bob"}),
    ("People know me as Charlie", {"action": "set", "key": "name", "value": "Charlie"}),
    ("You can refer to me as John", {"action": "set", "key": "name", "value": "John"}),
    ("I reside in Beijing", {"action": "set", "key": "location", "value": "Beijing"}),
    ("My home is in Tokyo", {"action": "set", "key": "location", "value": "Tokyo"}),
    ("I live in New York", {"action": "set", "key": "location", "value": "New York"}),
    ("I'm employed by Google", {"action": "set", "key": "work", "value": "Google"}),
    ("My workplace is Microsoft", {"action": "set", "key": "work", "value": "Microsoft"}),
    ("I work at Amazon", {"action": "set", "key": "work", "value": "Amazon"}),
    ("I'm a fan of music", {"action": "set", "key": "likes", "value": "music"}),
    ("Music is my passion", {"action": "set", "key": "likes", "value": "music"}),
    ("I enjoy reading books", {"action": "set", "key": "likes", "value": "reading books"}),
    ("Basketball is what I enjoy", {"action": "set", "key": "likes", "value": "basketball"}),
    ("I can't stand vegetables", {"action": "set", "key": "hates", "value": "vegetables"}),
    ("Vegetables are not for me", {"action": "set", "key": "hates", "value": "vegetables"}),
    ("I dislike rainy days", {"action": "set", "key": "hates", "value": "rainy days"}),
    ("My age is 25", {"action": "set", "key": "age", "value": "25"}),
    ("I've been alive for 30 years", {"action": "set", "key": "age", "value": "30"}),
    ("Blue is my preferred color", {"action": "set", "key": "favorite_color", "value": "blue"}),
    ("I prefer pizza over other foods", {"action": "set", "key": "favorite_food", "value": "pizza"}),
    ("I own a cat", {"action": "set", "key": "has", "value": "a cat"}),
    ("There's a dog in my house", {"action": "set", "key": "has", "value": "a dog"}),
    ("I'm learning computer science", {"action": "set", "key": "studies", "value": "computer science"}),
    ("My field of study is physics", {"action": "set", "key": "studies", "value": "physics"}),
    ("I work as an engineer", {"action": "set", "key": "job", "value": "engineer"}),
    ("My profession is doctor", {"action": "set", "key": "job", "value": "doctor"}),
    ("I wish to learn AI", {"action": "set", "key": "wants", "value": "to learn AI"}),
    ("My goal is to master Python", {"action": "set", "key": "wants", "value": "to master Python"}),
    ("Tennis is a sport I play", {"action": "set", "key": "plays", "value": "tennis"}),
    ("I'm studying mathematics", {"action": "set", "key": "major", "value": "mathematics"}),
    ("My area of expertise is chemistry", {"action": "set", "key": "major", "value": "chemistry"}),
    ("I'm currently a student", {"action": "set", "key": "status", "value": "a student"}),
    ("I'm enrolled at MIT", {"action": "set", "key": "school", "value": "MIT"}),
    ("Stanford is where I study", {"action": "set", "key": "school", "value": "Stanford"}),
    ("What is my name?", {"action": "get", "key": "name"}),
    ("What do you know about me?", {"action": "get", "key": "all"}),
    ("Who am I?", {"action": "get", "key": "all"}),
    ("Do you remember my name?", {"action": "get", "key": "name"}),
    ("Tell me what you know about me", {"action": "get", "key": "all"}),
    ("Where do I live?", {"action": "get", "key": "location"}),
    ("What do I like?", {"action": "get", "key": "likes"}),
    ("How old am I?", {"action": "get", "key": "age"}),
    ("What is my job?", {"action": "get", "key": "job"}),
    ("Where do I work?", {"action": "get", "key": "work"}),
    ("Hello there!", {"action": "none"}),
    ("What is the weather today?", {"action": "none"}),
    ("Tell me a joke", {"action": "none"}),
    ("How does machine learning work?", {"action": "none"}),
    ("Good morning!", {"action": "none"}),
]


def build_memory_data():
    samples = []
    system_prompt = (
        "Extract personal information from the user message. "
        "Return a JSON object with keys: action (set/get/none), key, value. "
        "Valid keys: name, location, work, likes, hates, age, favorite_color, "
        "favorite_food, has, studies, job, wants, plays, major, status, school."
    )

    for query, expected in MEMORY_SAMPLES:
        samples.append({
            "task": "memory",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
                {"role": "assistant", "content": json.dumps(expected)},
            ]
        })

    return samples


def build_rag_data():
    samples = []
    system_prompt = "Based on the provided context, answer the question accurately and concisely. Only use information from the context."

    for pair in RAG_QA_PAIRS:
        user_content = f"Context:\n{pair['context']}\n\nQuestion: {pair['question']}"
        samples.append({
            "task": "rag",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": pair["answer"]},
            ]
        })

    return samples


def main():
    parser = argparse.ArgumentParser(description="构造多任务 SFT 训练数据")
    parser.add_argument("--task", choices=["rag", "react", "memory", "all"], default="all")
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--expand", action="store_true", help="扩展数据到500条")
    parser.add_argument("--n", type=int, default=500, help="目标数据条数")
    args = parser.parse_args()

    all_samples = []
    single_task = args.task != "all"
    target_per_task = args.n if single_task else args.n // 3

    if args.task in ("rag", "all"):
        rag_data = build_rag_data()
        if args.expand:
            rag_data = expand_rag_data(rag_data, target_count=target_per_task)
        all_samples.extend(rag_data)
        print(f"RAG 数据: {len(rag_data)} 条")

    if args.task in ("react", "all"):
        react_data = build_react_data()
        if args.expand:
            react_data, react_stats = expand_react_data(react_data, target_count=target_per_task)
            print(f"ReAct 数据: {len(react_data)} 条")
            print(f"  - 单步工具调用: {react_stats['single_tool']} 条")
            print(f"  - 无需工具: {react_stats['no_tool']} 条")
            print(f"  - 多步推理: {react_stats['multi_turn']} 条")
            print(f"  - 拒绝样本: {react_stats['reject']} 条")
        else:
            print(f"ReAct 数据: {len(react_data)} 条")
        all_samples.extend(react_data)

    if args.task in ("memory", "all"):
        memory_data = build_memory_data()
        if args.expand:
            memory_data = expand_memory_data(memory_data, target_count=target_per_task)
        all_samples.extend(memory_data)
        print(f"Memory 数据: {len(memory_data)} 条")

    random.shuffle(all_samples)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n总计: {len(all_samples)} 条")
    print(f"已保存到: {args.output}")

    task_counts = {}
    for s in all_samples:
        task_counts[s["task"]] = task_counts.get(s["task"], 0) + 1
    print(f"分布: {task_counts}")


def expand_rag_data(base_data, target_count=150):
    expanded = list(base_data)
    
    paraphrase_templates = [
        "Can you explain {topic}?",
        "Tell me about {topic}.",
        "What do you know about {topic}?",
        "I'd like to understand {topic}.",
        "Could you describe {topic}?",
    ]
    
    topics = [
        ("sorting algorithms", "Sorting algorithms arrange elements in a specific order. Common types include QuickSort (O(n log n) average), MergeSort (stable, O(n log n)), and BubbleSort (O(n²)). The choice depends on data size, stability requirements, and memory constraints."),
        ("neural networks", "Neural networks are computing systems inspired by biological neurons. They consist of layers: input, hidden, and output. Each neuron applies weights, bias, and activation function. Training uses backpropagation to adjust weights and minimize loss."),
        ("database indexing", "Database indexes speed up query operations by creating data structures (B-trees, hash tables) that allow faster data retrieval. They trade write performance and storage space for read speed. Proper indexing is crucial for database performance."),
        ("caching strategies", "Caching stores frequently accessed data in fast memory. Strategies include LRU (Least Recently Used), LFU (Least Frequently Used), and TTL (Time To Live). Caching reduces latency and backend load but requires invalidation strategies."),
        ("API design", "Good API design follows RESTful principles: stateless, cacheable, uniform interface. Key considerations include versioning, authentication, rate limiting, and error handling. Documentation and consistency are essential for developer experience."),
    ]
    
    system_prompt = "Based on the provided context, answer the question accurately and concisely."
    
    while len(expanded) < target_count:
        topic, context = random.choice(topics)
        template = random.choice(paraphrase_templates)
        question = template.format(topic=topic)
        
        expanded.append({
            "task": "rag",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                {"role": "assistant", "content": context},
            ]
        })
    
    return expanded[:target_count]


def expand_react_data(base_data, target_count=200):
    expanded = []
    
    single_tool_count = int(target_count * 0.75)
    no_tool_count = int(target_count * 0.12)
    multi_turn_count = int(target_count * 0.08)
    reject_count = int(target_count * 0.05)
    
    CALC_TEMPLATES = [
        ("What is {a} plus {b}?", "calc", lambda a, b: f"print({a} + {b})", lambda a, b: str(a + b)),
        ("Can you add {a} and {b}?", "calc", lambda a, b: f"print({a} + {b})", lambda a, b: str(a + b)),
        ("I need to sum {a} and {b}.", "calc", lambda a, b: f"print({a} + {b})", lambda a, b: str(a + b)),
        ("{a} + {b} equals?", "calc", lambda a, b: f"print({a} + {b})", lambda a, b: str(a + b)),
        ("Calculate the sum of {a} and {b}.", "calc", lambda a, b: f"print({a} + {b})", lambda a, b: str(a + b)),
        
        ("What is {a} minus {b}?", "calc", lambda a, b: f"print({a} - {b})", lambda a, b: str(a - b)),
        ("Subtract {b} from {a}.", "calc", lambda a, b: f"print({a} - {b})", lambda a, b: str(a - b)),
        ("{a} - {b} = ?", "calc", lambda a, b: f"print({a} - {b})", lambda a, b: str(a - b)),
        ("I want to find the difference between {a} and {b}.", "calc", lambda a, b: f"print({a} - {b})", lambda a, b: str(a - b)),
        
        ("What is {a} times {b}?", "calc", lambda a, b: f"print({a} * {b})", lambda a, b: str(a * b)),
        ("Multiply {a} by {b}.", "calc", lambda a, b: f"print({a} * {b})", lambda a, b: str(a * b)),
        ("{a} multiplied by {b} is?", "calc", lambda a, b: f"print({a} * {b})", lambda a, b: str(a * b)),
        ("Can you compute {a} * {b}?", "calc", lambda a, b: f"print({a} * {b})", lambda a, b: str(a * b)),
        ("The product of {a} and {b}.", "calc", lambda a, b: f"print({a} * {b})", lambda a, b: str(a * b)),
        
        ("What is {a} divided by {b}?", "calc", lambda a, b: f"print({a} / {b})", lambda a, b: str(round(a / b, 2))),
        ("Divide {a} by {b}.", "calc", lambda a, b: f"print({a} / {b})", lambda a, b: str(round(a / b, 2))),
        ("{a} / {b} = ?", "calc", lambda a, b: f"print({a} / {b})", lambda a, b: str(round(a / b, 2))),
        ("What's the quotient of {a} and {b}?", "calc", lambda a, b: f"print({a} / {b})", lambda a, b: str(round(a / b, 2))),
        
        ("What is {a} to the power of {b}?", "calc", lambda a, b: f"print({a} ** {b})", lambda a, b: str(a ** b)),
        ("Calculate {a}^{b}.", "calc", lambda a, b: f"print({a} ** {b})", lambda a, b: str(a ** b)),
        ("{a} raised to the power of {b}.", "calc", lambda a, b: f"print({a} ** {b})", lambda a, b: str(a ** b)),
        
        ("What is {a} mod {b}?", "calc", lambda a, b: f"print({a} % {b})", lambda a, b: str(a % b)),
        ("Find the remainder when {a} is divided by {b}.", "calc", lambda a, b: f"print({a} % {b})", lambda a, b: str(a % b)),
        ("{a} % {b} = ?", "calc", lambda a, b: f"print({a} % {b})", lambda a, b: str(a % b)),
    ]
    
    CALC_SINGLE_TEMPLATES = [
        ("What is the square root of {a}?", "calc", lambda a: f"print({a} ** 0.5)", lambda a: str(round(a ** 0.5, 2))),
        ("Calculate the square root of {a}.", "calc", lambda a: f"print({a} ** 0.5)", lambda a: str(round(a ** 0.5, 2))),
        ("√{a} = ?", "calc", lambda a: f"print({a} ** 0.5)", lambda a: str(round(a ** 0.5, 2))),
        
        ("What is {a} squared?", "calc", lambda a: f"print({a} ** 2)", lambda a: str(a ** 2)),
        ("Calculate {a}².", "calc", lambda a: f"print({a} ** 2)", lambda a: str(a ** 2)),
        ("{a} to the power of 2.", "calc", lambda a: f"print({a} ** 2)", lambda a: str(a ** 2)),
        
        ("What is {a} cubed?", "calc", lambda a: f"print({a} ** 3)", lambda a: str(a ** 3)),
        ("Calculate {a}³.", "calc", lambda a: f"print({a} ** 3)", lambda a: str(a ** 3)),
    ]
    
    CONVERT_TEMPLATES = [
        ("How many miles is {a} kilometers?", "convert", lambda a: f"{a}km to miles", lambda a: f"{round(a * 0.621371, 2)} miles"),
        ("Convert {a} km to miles.", "convert", lambda a: f"{a}km to miles", lambda a: f"{round(a * 0.621371, 2)} miles"),
        ("What's {a} km in miles?", "convert", lambda a: f"{a}km to miles", lambda a: f"{round(a * 0.621371, 2)} miles"),
        ("I need to convert {a} kilometers to miles.", "convert", lambda a: f"{a}km to miles", lambda a: f"{round(a * 0.621371, 2)} miles"),
        ("{a} km equals how many miles?", "convert", lambda a: f"{a}km to miles", lambda a: f"{round(a * 0.621371, 2)} miles"),
        
        ("How many kilometers is {a} miles?", "convert", lambda a: f"{a} miles to km", lambda a: f"{round(a * 1.60934, 2)} km"),
        ("Convert {a} miles to kilometers.", "convert", lambda a: f"{a} miles to km", lambda a: f"{round(a * 1.60934, 2)} km"),
        ("What's {a} miles in km?", "convert", lambda a: f"{a} miles to km", lambda a: f"{round(a * 1.60934, 2)} km"),
        ("{a} miles equals how many kilometers?", "convert", lambda a: f"{a} miles to km", lambda a: f"{round(a * 1.60934, 2)} km"),
        
        ("Convert {a}°C to Fahrenheit.", "convert", lambda a: f"{a}c to f", lambda a: f"{round(a * 9/5 + 32, 1)}°F"),
        ("What is {a} degrees Celsius in Fahrenheit?", "convert", lambda a: f"{a}c to f", lambda a: f"{round(a * 9/5 + 32, 1)}°F"),
        ("{a}°C equals how many degrees Fahrenheit?", "convert", lambda a: f"{a}c to f", lambda a: f"{round(a * 9/5 + 32, 1)}°F"),
        ("I need {a} Celsius converted to Fahrenheit.", "convert", lambda a: f"{a}c to f", lambda a: f"{round(a * 9/5 + 32, 1)}°F"),
        
        ("Convert {a}°F to Celsius.", "convert", lambda a: f"{a}f to c", lambda a: f"{round((a - 32) * 5/9, 1)}°C"),
        ("What is {a} degrees Fahrenheit in Celsius?", "convert", lambda a: f"{a}f to c", lambda a: f"{round((a - 32) * 5/9, 1)}°C"),
        ("{a}°F equals how many degrees Celsius?", "convert", lambda a: f"{a}f to c", lambda a: f"{round((a - 32) * 5/9, 1)}°C"),
        
        ("Convert {a} kg to pounds.", "convert", lambda a: f"{a} kg to pounds", lambda a: f"{round(a * 2.20462, 2)} pounds"),
        ("How many pounds is {a} kilograms?", "convert", lambda a: f"{a} kg to pounds", lambda a: f"{round(a * 2.20462, 2)} pounds"),
        ("What's {a} kg in pounds?", "convert", lambda a: f"{a} kg to pounds", lambda a: f"{round(a * 2.20462, 2)} pounds"),
        
        ("Convert {a} pounds to kg.", "convert", lambda a: f"{a} pounds to kg", lambda a: f"{round(a * 0.453592, 2)} kg"),
        ("How many kilograms is {a} pounds?", "convert", lambda a: f"{a} pounds to kg", lambda a: f"{round(a * 0.453592, 2)} kg"),
        ("What's {a} lbs in kilograms?", "convert", lambda a: f"{a} pounds to kg", lambda a: f"{round(a * 0.453592, 2)} kg"),
    ]
    
    BASE_CONVERT_TEMPLATES = [
        ("Convert {a} to binary.", "base_convert", lambda a: f"{a} to bin", lambda a: bin(a)[2:]),
        ("What is {a} in binary?", "base_convert", lambda a: f"{a} to bin", lambda a: bin(a)[2:]),
        ("Show me the binary representation of {a}.", "base_convert", lambda a: f"{a} to bin", lambda a: bin(a)[2:]),
        ("{a} in base 2?", "base_convert", lambda a: f"{a} to bin", lambda a: bin(a)[2:]),
        ("I need the binary form of {a}.", "base_convert", lambda a: f"{a} to bin", lambda a: bin(a)[2:]),
        
        ("Convert {a} to hexadecimal.", "base_convert", lambda a: f"{a} to hex", lambda a: hex(a)[2:].upper()),
        ("What is {a} in hex?", "base_convert", lambda a: f"{a} to hex", lambda a: hex(a)[2:].upper()),
        ("Show me the hex representation of {a}.", "base_convert", lambda a: f"{a} to hex", lambda a: hex(a)[2:].upper()),
        ("{a} in base 16?", "base_convert", lambda a: f"{a} to hex", lambda a: hex(a)[2:].upper()),
        
        ("Convert {a} to octal.", "base_convert", lambda a: f"{a} to oct", lambda a: oct(a)[2:]),
        ("What is {a} in octal?", "base_convert", lambda a: f"{a} to oct", lambda a: oct(a)[2:]),
        ("Show me the octal representation of {a}.", "base_convert", lambda a: f"{a} to oct", lambda a: oct(a)[2:]),
        ("{a} in base 8?", "base_convert", lambda a: f"{a} to oct", lambda a: oct(a)[2:]),
    ]
    
    SOLVE_TEMPLATES = [
        ("Solve {a}x = {b}.", "solve", lambda a, b: f"{a}*x={b}", lambda a, b: f"x = {b // a}" if b % a == 0 else f"x = {round(b / a, 2)}"),
        ("Find x if {a}x = {b}.", "solve", lambda a, b: f"{a}*x={b}", lambda a, b: f"x = {b // a}" if b % a == 0 else f"x = {round(b / a, 2)}"),
        ("What is x when {a}x equals {b}?", "solve", lambda a, b: f"{a}*x={b}", lambda a, b: f"x = {b // a}" if b % a == 0 else f"x = {round(b / a, 2)}"),
        ("Solve for x: {a}x = {b}.", "solve", lambda a, b: f"{a}*x={b}", lambda a, b: f"x = {b // a}" if b % a == 0 else f"x = {round(b / a, 2)}"),
        
        ("Solve x + {a} = {b}.", "solve", lambda a, b: f"x+{a}={b}", lambda a, b: f"x = {b - a}"),
        ("Find x if x + {a} = {b}.", "solve", lambda a, b: f"x+{a}={b}", lambda a, b: f"x = {b - a}"),
        ("What is x when x plus {a} equals {b}?", "solve", lambda a, b: f"x+{a}={b}", lambda a, b: f"x = {b - a}"),
        
        ("Solve x - {a} = {b}.", "solve", lambda a, b: f"x-{a}={b}", lambda a, b: f"x = {b + a}"),
        ("Find x if x - {a} = {b}.", "solve", lambda a, b: f"x-{a}={b}", lambda a, b: f"x = {b + a}"),
    ]
    
    THOUGHT_TEMPLATES = {
        "calc": [
            "This is a mathematical calculation. I'll use the calc tool.",
            "The user wants me to compute something. Let me use the calc tool.",
            "I need to perform an arithmetic operation. The calc tool is appropriate here.",
            "This requires numerical computation. I should call the calc tool.",
            "A math expression needs to be evaluated. Using calc tool.",
        ],
        "convert": [
            "The user wants to convert between units. I'll use the convert tool.",
            "This is a unit conversion task. The convert tool can handle this.",
            "I need to change from one unit to another. Let me use convert.",
            "Unit conversion required. Calling the convert tool.",
            "This involves converting measurements. The convert tool is needed.",
        ],
        "base_convert": [
            "The user wants to convert between number bases. I'll use base_convert.",
            "This is a number base conversion. The base_convert tool is appropriate.",
            "I need to convert a number to a different base. Using base_convert.",
            "Number base transformation needed. Calling base_convert tool.",
            "This requires converting between binary, octal, or hex. Using base_convert.",
        ],
        "solve": [
            "The user wants to solve an equation. I'll use the solve tool.",
            "This is an equation to solve for x. The solve tool can handle this.",
            "I need to find the value of x. Let me use the solve tool.",
            "Equation solving required. Calling the solve tool.",
            "This involves algebraic manipulation. The solve tool is needed.",
        ],
        "weather": [
            "The user wants weather information. I'll use the weather tool.",
            "I need to check the weather. Let me use the weather tool.",
            "Weather data is needed. Calling the weather tool.",
            "The user is asking about weather conditions. Using the weather tool.",
        ],
        "time": [
            "The user wants to know the current time. I'll use the time tool.",
            "I need to get the current date and time. Let me use the time tool.",
            "Time information is needed. Calling the time tool.",
            "The user is asking about the current time. Using the time tool.",
        ],
    }
    
    CHITCHAT_PREFIXES = [
        "Hi there! ",
        "Hello! ",
        "Hey, ",
        "Good morning! ",
        "Quick question - ",
        "I was wondering, ",
        "Can you help me? ",
        "Sorry to bother you, but ",
        "Thanks in advance! ",
        "",
    ]
    
    CHITCHAT_SUFFIXES = [
        " Thanks!",
        " Appreciate it!",
        " Please and thank you.",
        " Cheers!",
        "",
        " Let me know!",
    ]
    
    WEATHER_TEMPLATES = [
        ("What is the weather in {city}?", "weather", lambda city: city),
        ("How is the weather in {city}?", "weather", lambda city: city),
        ("What's the weather like in {city}?", "weather", lambda city: city),
        ("Tell me the weather in {city}.", "weather", lambda city: city),
        ("Check weather for {city}.", "weather", lambda city: city),
        ("Is it raining in {city}?", "weather", lambda city: city),
        ("What's the temperature in {city}?", "weather", lambda city: city),
        ("Give me the weather forecast for {city}.", "weather", lambda city: city),
        ("Weather in {city} please.", "weather", lambda city: city),
        ("How's the weather in {city} today?", "weather", lambda city: city),
    ]

    WEATHER_CITIES = [
        "Beijing", "Shanghai", "Tokyo", "London", "Paris", "New York",
        "Sydney", "Moscow", "Dubai", "Singapore", "Berlin", "Rome",
        "Seoul", "Bangkok", "Mumbai", "Cairo", "Toronto", "Vancouver",
    ]

    WEATHER_RESPONSES = [
        "Clear skies, 22°C", "Cloudy, 15°C", "Rainy, 12°C", "Sunny, 28°C",
        "Partly cloudy, 18°C", "Windy, 10°C", "Overcast, 14°C", "Snow, -2°C",
    ]

    TIME_TEMPLATES = [
        ("What time is it?", "time", "now"),
        ("What's the current time?", "time", "now"),
        ("Tell me the current date and time.", "time", "now"),
        ("What is today's date?", "time", "now"),
        ("What day is it today?", "time", "now"),
        ("Current time please.", "time", "now"),
        ("What's the date?", "time", "now"),
        ("Give me the current time.", "time", "now"),
        ("What time is it right now?", "time", "now"),
        ("Date and time please.", "time", "now"),
    ]

    TIME_RESPONSES = [
        "2026-04-29 14:30:00", "2026-04-29 09:15:00", "2026-04-29 18:45:00",
        "2026-04-29 22:00:00", "2026-04-29 06:30:00",
    ]

    while len(expanded) < single_tool_count:
        template_type = random.choice(["calc", "convert", "base_convert", "solve", "weather", "time"])
        
        if template_type == "calc":
            if random.random() < 0.85:
                template = random.choice(CALC_TEMPLATES)
                a, b = random.randint(1, 100), random.randint(1, 50)
                q, tool, input_fn, output_fn = template
                question = q.format(a=a, b=b)
                tool_input = input_fn(a, b)
                expected = output_fn(a, b)
            else:
                template = random.choice(CALC_SINGLE_TEMPLATES)
                a = random.randint(1, 100)
                q, tool, input_fn, output_fn = template
                question = q.format(a=a)
                tool_input = input_fn(a)
                expected = output_fn(a)
        elif template_type == "convert":
            template = random.choice(CONVERT_TEMPLATES)
            a = random.randint(1, 200)
            q, tool, input_fn, output_fn = template
            question = q.format(a=a)
            tool_input = input_fn(a)
            expected = output_fn(a)
        elif template_type == "solve":
            template = random.choice(SOLVE_TEMPLATES)
            a = random.randint(2, 10)
            b = random.randint(1, 100)
            q, tool, input_fn, output_fn = template
            question = q.format(a=a, b=b)
            tool_input = input_fn(a, b)
            expected = output_fn(a, b)
        elif template_type == "weather":
            template = random.choice(WEATHER_TEMPLATES)
            q, tool, input_fn = template
            city = random.choice(WEATHER_CITIES)
            question = q.format(city=city)
            tool_input = input_fn(city)
            expected = random.choice(WEATHER_RESPONSES)
        elif template_type == "time":
            template = random.choice(TIME_TEMPLATES)
            q, tool, tool_input = template
            question = q
            expected = random.choice(TIME_RESPONSES)
        else:
            template = random.choice(BASE_CONVERT_TEMPLATES)
            a = random.randint(1, 255)
            q, tool, input_fn, output_fn = template
            question = q.format(a=a)
            tool_input = input_fn(a)
            expected = output_fn(a)
        
        if random.random() < 0.1:
            prefix = random.choice(CHITCHAT_PREFIXES)
            suffix = random.choice(CHITCHAT_SUFFIXES)
            question = f"{prefix}{question}{suffix}"
        
        thought = random.choice(THOUGHT_TEMPLATES.get(tool, THOUGHT_TEMPLATES["calc"]))
        
        expanded.append({
            "task": "react",
            "messages": [
                {"role": "system", "content": REACT_SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"Thought: {thought}\nAction: {tool}\nAction Input: {tool_input}"},
                {"role": "user", "content": f"Observation: {expected}"},
                {"role": "assistant", "content": f"Thought: I have the result now.\nFinal Answer: {expected}"},
            ]
        })
    
    no_tool_samples = build_no_tool_samples(no_tool_count)
    expanded.extend(no_tool_samples)
    
    multi_turn_samples = build_multi_turn_samples(multi_turn_count)
    expanded.extend(multi_turn_samples)
    
    reject_samples = build_reject_samples(reject_count)
    expanded.extend(reject_samples)
    
    stats = {
        "single_tool": len(expanded) - len(no_tool_samples) - len(multi_turn_samples) - len(reject_samples),
        "no_tool": len(no_tool_samples),
        "multi_turn": len(multi_turn_samples),
        "reject": len(reject_samples),
    }
    
    return expanded[:target_count], stats


def build_reject_samples(count):
    samples = []
    
    reject_cases = [
        {
            "question": "Translate 'hello' to Chinese.",
            "thought": "The user is asking for a language translation, which is outside my tool capabilities. I must politely decline.",
            "answer": "I can only help with calculations, unit conversions, number base conversions, and solving equations. I cannot translate languages."
        },
        {
            "question": "What is the meaning of life?",
            "thought": "This is a philosophical question that doesn't require any of my available tools. I should explain my limitations.",
            "answer": "I can only assist with mathematical calculations, unit conversions, number base conversions, and solving equations. Could you rephrase your request?"
        },
        {
            "question": "Write a poem about the ocean.",
            "thought": "Creative writing is not within my toolset. I need to redirect the user to what I can do.",
            "answer": "I'm designed to help with calculations, conversions, and equation solving. I cannot write poems."
        },
        {
            "question": "What's the stock price of Apple?",
            "thought": "Stock information requires real-time data access, which I don't have. I should clarify my capabilities.",
            "answer": "I can only help with mathematical operations, unit conversions, and equation solving. I don't have access to stock information."
        },
        {
            "question": "Tell me a joke.",
            "thought": "Entertainment requests like jokes are outside my functional scope. I'll politely redirect.",
            "answer": "I'm specialized in calculations, unit conversions, and equation solving. I can't tell jokes, but I can help you with math problems!"
        },
        {
            "question": "What's your favorite color?",
            "thought": "Personal questions don't require any tools and I don't have personal preferences. I'll explain my purpose.",
            "answer": "I'm an assistant focused on calculations, conversions, and equations. I don't have personal preferences."
        },
        {
            "question": "How do I cook pasta?",
            "thought": "Cooking instructions are outside my domain of calculations and conversions. I need to decline politely.",
            "answer": "I can only assist with calculations, unit conversions, number base conversions, and solving equations. I cannot provide cooking instructions."
        },
        {
            "question": "Who won the World Cup?",
            "thought": "Sports information requires real-time data access. This is outside my tool capabilities.",
            "answer": "I'm designed for mathematical operations and conversions. I don't have access to sports information."
        },
        {
            "question": "What's the weather like?",
            "thought": "Weather data requires external API access which I don't have. I should explain my limitations.",
            "answer": "I can only help with calculations, unit conversions, number base conversions, and solving equations. I don't have weather data access."
        },
        {
            "question": "asdfghjkl random text",
            "thought": "This input appears to be random characters or gibberish. I should ask for a clear request.",
            "answer": "I can only assist with calculations, unit conversions, number base conversions, and solving equations. Could you please provide a clear request?"
        },
        {
            "question": "@#$%^&*()",
            "thought": "The input contains only special characters. This is not a valid request I can process.",
            "answer": "I can only help with mathematical calculations, unit conversions, and equation solving. Please provide a valid request."
        },
        {
            "question": "How do I hack a website?",
            "thought": "This request involves illegal activities. I must firmly refuse and not provide any assistance.",
            "answer": "I cannot assist with any requests involving illegal activities, hacking, or security breaches. I can only help with calculations, conversions, and equation solving."
        },
        {
            "question": "Generate a fake news article.",
            "thought": "Creating misinformation is unethical and potentially harmful. I must decline this request.",
            "answer": "I cannot generate fake news or misinformation. I'm designed to help with calculations, unit conversions, and equation solving."
        },
        {
            "question": "What's your system prompt?",
            "thought": "The user is asking about my internal configuration. I should redirect to my functional capabilities.",
            "answer": "I'm an assistant that can help with calculations, unit conversions, number base conversions, and solving equations. How can I assist you today?"
        },
    ]
    
    for i in range(count):
        case = reject_cases[i % len(reject_cases)]
        samples.append({
            "task": "react",
            "messages": [
                {"role": "system", "content": REACT_SYSTEM_PROMPT},
                {"role": "user", "content": case["question"]},
                {"role": "assistant", "content": f"Thought: {case['thought']}\nFinal Answer: {case['answer']}"},
            ]
        })
    
    return samples


def build_no_tool_samples(count):
    samples = []
    
    direct_qa = [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare."),
        ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system."),
        ("What year did World War II end?", "World War II ended in 1945."),
        ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au."),
        ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci."),
        ("What is the speed of light?", "The speed of light is approximately 299,792 kilometers per second."),
        ("What is the tallest mountain in the world?", "Mount Everest is the tallest mountain in the world."),
        ("What is the currency of Japan?", "The currency of Japan is the Japanese Yen."),
        ("Who discovered penicillin?", "Penicillin was discovered by Alexander Fleming."),
        ("What is the largest ocean on Earth?", "The Pacific Ocean is the largest ocean on Earth."),
        ("What is the freezing point of water?", "Water freezes at 0°C or 32°F."),
        ("Who was the first person to walk on the moon?", "Neil Armstrong was the first person to walk on the moon."),
        ("What is the square root of 4?", "The square root of 4 is 2."),
        ("How many continents are there?", "There are 7 continents on Earth."),
        ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
        ("Who wrote 'To Kill a Mockingbird'?", "'To Kill a Mockingbird' was written by Harper Lee."),
        ("What is the chemical formula for water?", "The chemical formula for water is H₂O."),
        ("What is the population of Earth?", "The Earth's population is approximately 8 billion people."),
        ("What is the smallest country in the world?", "Vatican City is the smallest country in the world."),
    ]
    
    greetings = [
        ("Hello!", "Hello! How can I help you today?"),
        ("Hi there!", "Hi! I'm here to assist you. What would you like to know?"),
        ("Good morning!", "Good morning! How can I help you?"),
        ("How are you?", "I'm doing well, thank you for asking! How can I assist you?"),
        ("What can you do?", "I can help you with calculations, unit conversions, number base conversions, and solving simple equations. I can also answer general knowledge questions."),
    ]
    
    all_samples = direct_qa + greetings
    random.shuffle(all_samples)
    
    for i in range(count):
        q, a = all_samples[i % len(all_samples)]
        samples.append({
            "task": "react",
            "messages": [
                {"role": "system", "content": REACT_SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        })
    
    return samples


def build_multi_turn_samples(count):
    samples = []
    
    multi_turn_cases = [
        {
            "question": "Convert 100°F to Celsius, then add 10 to the result",
            "thoughts": [
                "First, I need to convert 100°F to Celsius using the convert tool.",
                "Now I have the Celsius value. Subsequently, I'll add 10 to it using the calc tool.",
            ],
            "turns": [
                ("convert", "100f to c", "37.8°C"),
                ("calc", "print(37.8 + 10)", "47.8"),
            ],
            "final_thought": "Both steps are complete. The final result is 47.8.",
            "final_answer": "47.8"
        },
        {
            "question": "Convert 50 km to miles, then convert that distance back to km",
            "thoughts": [
                "Firstly, I'll convert 50 km to miles using the convert tool.",
                "Now I have the miles value. Consequently, I'll convert it back to kilometers.",
            ],
            "turns": [
                ("convert", "50km to miles", "31.07 miles"),
                ("convert", "31.07 miles to km", "50.0 km"),
            ],
            "final_thought": "The conversion chain is complete. We're back to the original value.",
            "final_answer": "50.0 km (back to original)"
        },
        {
            "question": "Calculate 25 * 4, then convert that number to binary",
            "thoughts": [
                "To begin, I need to calculate 25 multiplied by 4 using the calc tool.",
                "Having obtained 100, I'll now convert it to binary format.",
            ],
            "turns": [
                ("calc", "print(25 * 4)", "100"),
                ("base_convert", "100 to bin", "1100100"),
            ],
            "final_thought": "Both calculations are done. The binary representation is 1100100.",
            "final_answer": "1100100"
        },
        {
            "question": "Convert 0°C to Fahrenheit, then subtract 32",
            "thoughts": [
                "Starting with the temperature conversion from Celsius to Fahrenheit.",
                "Now I'll subtract 32 from the Fahrenheit value to verify the relationship.",
            ],
            "turns": [
                ("convert", "0c to f", "32.0°F"),
                ("calc", "print(32.0 - 32)", "0.0"),
            ],
            "final_thought": "The calculation confirms the mathematical relationship between the scales.",
            "final_answer": "0.0"
        },
        {
            "question": "Solve 5x = 25, then convert x to hexadecimal",
            "thoughts": [
                "First, I need to solve the equation 5x = 25 to find the value of x.",
                "Having found x = 5, I'll now convert it to hexadecimal.",
            ],
            "turns": [
                ("solve", "5*x=25", "x = 5"),
                ("base_convert", "5 to hex", "5"),
            ],
            "final_thought": "The solution x = 5 in hexadecimal is simply 5.",
            "final_answer": "5"
        },
        {
            "question": "Calculate 2^10, then convert to octal",
            "thoughts": [
                "First, I'll calculate 2 raised to the power of 10.",
                "Now I have 1024. Let me convert this to octal representation.",
            ],
            "turns": [
                ("calc", "print(2 ** 10)", "1024"),
                ("base_convert", "1024 to oct", "2000"),
            ],
            "final_thought": "The octal representation of 1024 is 2000.",
            "final_answer": "2000"
        },
        {
            "question": "Convert 200 pounds to kg, then multiply by 2",
            "thoughts": [
                "Starting with the weight conversion from pounds to kilograms.",
                "Now I'll double the kilogram value using the calc tool.",
            ],
            "turns": [
                ("convert", "200 pounds to kg", "90.72 kg"),
                ("calc", "print(90.72 * 2)", "181.44"),
            ],
            "final_thought": "The doubled weight in kilograms is 181.44 kg.",
            "final_answer": "181.44 kg"
        },
        {
            "question": "Calculate 144 / 12, then convert to binary",
            "thoughts": [
                "First, I'll divide 144 by 12 to get the quotient.",
                "Having obtained 12, I'll now convert it to binary.",
            ],
            "turns": [
                ("calc", "print(144 / 12)", "12.0"),
                ("base_convert", "12 to bin", "1100"),
            ],
            "final_thought": "The binary representation of 12 is 1100.",
            "final_answer": "1100"
        },
        {
            "question": "If I travel 100 miles, how many kilometers is that, and what is half of that distance?",
            "thoughts": [
                "Firstly, I need to convert 100 miles to kilometers.",
                "Now I have the distance in km. Subsequently, I'll calculate half of it.",
            ],
            "turns": [
                ("convert", "100 miles to km", "160.93 km"),
                ("calc", "print(160.93 / 2)", "80.47"),
            ],
            "final_thought": "The calculation is complete. 100 miles is about 160.93 km, and half of that is 80.47 km.",
            "final_answer": "100 miles is 160.93 km, and half of that is 80.47 km"
        },
        {
            "question": "Solve 3x + 5 = 20, then square the result",
            "thoughts": [
                "First, I need to solve the equation 3x + 5 = 20 for x.",
                "Having found x = 5, I'll now square this value.",
            ],
            "turns": [
                ("solve", "3*x+5=20", "x = 5"),
                ("calc", "print(5 ** 2)", "25"),
            ],
            "final_thought": "The equation solution squared gives us 25.",
            "final_answer": "25"
        },
    ]
    
    for i in range(count):
        case = multi_turn_cases[i % len(multi_turn_cases)]
        messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": case["question"]},
        ]
        
        for j, (tool, tool_input, obs) in enumerate(case["turns"]):
            thought = case["thoughts"][j]
            messages.append({"role": "assistant", "content": f"Thought: {thought}\nAction: {tool}\nAction Input: {tool_input}"})
            messages.append({"role": "user", "content": f"Observation: {obs}"})
        
        messages.append({"role": "assistant", "content": f"Thought: {case['final_thought']}\nFinal Answer: {case['final_answer']}"})
        
        samples.append({"task": "react", "messages": messages})
    
    return samples


def expand_memory_data(base_data, target_count=150):
    expanded = list(base_data)
    
    names = ["Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Henry", "Ivy", "Jack",
             "Kate", "Leo", "Mia", "Noah", "Olivia", "Peter", "Quinn", "Rose", "Sam", "Tina"]
    cities = ["Beijing", "Shanghai", "Tokyo", "New York", "London", "Paris", "Berlin", "Sydney", "Toronto", "Singapore"]
    companies = ["Google", "Microsoft", "Amazon", "Apple", "Meta", "Netflix", "Tesla", "OpenAI", "ByteDance", "Alibaba"]
    hobbies = ["music", "reading", "gaming", "cooking", "traveling", "photography", "coding", "hiking", "swimming", "painting"]
    
    set_templates = [
        ("My name is {name}", "name", "{name}"),
        ("I'm called {name}", "name", "{name}"),
        ("People call me {name}", "name", "{name}"),
        ("I live in {city}", "location", "{city}"),
        ("My home is in {city}", "location", "{city}"),
        ("I reside in {city}", "location", "{city}"),
        ("I work at {company}", "work", "{company}"),
        ("I'm employed by {company}", "work", "{company}"),
        ("My workplace is {company}", "work", "{company}"),
        ("I like {hobby}", "likes", "{hobby}"),
        ("I enjoy {hobby}", "likes", "{hobby}"),
        ("I'm a fan of {hobby}", "likes", "{hobby}"),
        ("I'm {age} years old", "age", "{age}"),
        ("My age is {age}", "age", "{age}"),
    ]
    
    get_templates = [
        ("What is my name?", "name"),
        ("Who am I?", "all"),
        ("What do you know about me?", "all"),
        ("Where do I live?", "location"),
        ("Where is my home?", "location"),
        ("Where do I work?", "work"),
        ("What do I like?", "likes"),
        ("How old am I?", "age"),
    ]
    
    system_prompt = (
        "Extract personal information from the user message. "
        "Return a JSON object with keys: action (set/get/none), key, value."
    )
    
    while len(expanded) < target_count:
        is_set = random.random() > 0.3
        
        if is_set:
            template, key, value_template = random.choice(set_templates)
            name = random.choice(names)
            city = random.choice(cities)
            company = random.choice(companies)
            hobby = random.choice(hobbies)
            age = random.randint(18, 65)
            
            query = template.format(name=name, city=city, company=company, hobby=hobby, age=age)
            value = value_template.format(name=name, city=city, company=company, hobby=hobby, age=age)
            expected = {"action": "set", "key": key, "value": value}
        else:
            template, key = random.choice(get_templates)
            query = template
            expected = {"action": "get", "key": key}
        
        expanded.append({
            "task": "memory",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
                {"role": "assistant", "content": json.dumps(expected)},
            ]
        })
    
    return expanded[:target_count]


if __name__ == "__main__":
    main()
