# NanoChat-Lab 项目文档

## 一、项目概述

NanoChat-Lab 是一个在 4GB 显卡（GTX 1650）上运行的轻量级 LLM 应用，实现了完整的 RAG + Agent + Memory 系统。核心代码约 1000 行，不依赖 LangChain 等重型框架，所有模块自定义实现。

**核心模型**：Qwen2.5-1.5B（本地推理）

**设计理念**：轻量、可控、可学习

---

## 二、项目进度

| 阶段 | 内容 | 状态 |
|------|------|------|
| 阶段1 | 基础 RAG 架构（文档加载、向量检索、对话流程） | ✅ 完成 |
| 阶段2 | 工业级 RAG 深化（BM25+向量混合检索、Rerank、评估体系） | ✅ 完成 |
| 阶段3 | Agent 智能体进化（ReAct 循环、12+ 工具、LLM 驱动记忆系统） | ✅ 完成 |
| 阶段4 | 模型量化（bitsandbytes ✅ / GGUF ⏳ / GPTQ ⬜） | ⏳ 进行中 |
| 阶段4 | 评测体系（benchmark.py 已创建） | ⏳ 进行中 |
| 阶段5 | LoRA 微调 / 部署优化 | ⬜ 未开始 |

---

## 三、系统架构

```
用户输入
    ↓
┌─────────────────────────────────────────────────┐
│              MemoryStore（记忆处理）              │
│         正则优先 + LLM 兜底提取用户信息           │
└─────────────────────────────────────────────────┘
    ↓ 未命中
┌─────────────────────────────────────────────────┐
│           InputClassifier（分类路由）             │
│     identity / agent / rag / preference          │
└─────────────────────────────────────────────────┘
    ↓
┌──────────┬──────────────┬──────────────┐
│ identity │    agent     │     rag      │
│ 直接回复  │  工具调用     │  知识检索     │
│          │ direct_call  │  混合检索     │
│          │ / ReAct循环   │  + Rerank    │
└──────────┴──────────────┴──────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│                 LLM（文本生成）                   │
│            Qwen2.5-1.5B + stop_strings           │
└─────────────────────────────────────────────────┘
    ↓
返回响应
```

---

## 四、核心模块详解

### 4.1 LLM（`core/llm.py`）

LLM 封装类，负责模型加载与文本生成。

**初始化参数**：
- `model_path`：模型路径，默认 `models/Qwen2.5-1.5B`

**核心属性**：
- `self.model`：AutoModelForCausalLM 实例
- `self.tokenizer`：AutoTokenizer 实例
- `self.device`：cuda / cpu 自动选择
- `self.default_stop_strings`：默认停止字符串 `["\nObservation:", "\nFinal Answer:"]`

**核心方法**：

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `generate()` | `prompt`, `max_new_tokens=128`, `stop_strings=None` | `str` | 生成文本，支持自定义停止条件 |

**生成流程**：
1. 构建 messages（system + user）
2. apply_chat_template 格式化
3. model.generate 生成（含 stop_strings 停止条件）
4. 解码新 token，去除特殊 token

**停止条件设计**：
- 默认在 `\nObservation:` 和 `\nFinal Answer:` 处停止
- 防止 LLM 编造工具结果或生成多余内容
- 可通过参数覆盖默认值

---

### 4.2 ReActAgent（`core/ReActAgent.py`）

ReAct 架构的 Agent 实现，支持工具注册、直通车调用和多步推理循环。

**初始化参数**：
- `llm_instance`：LLM 实例
- `max_steps`：最大推理步数，默认 5

**注册工具（12+）**：

| 工具名 | 函数 | 说明 |
|--------|------|------|
| `calc` | `_python_repl` | 执行 Python 数学表达式 |
| `time` | `_time_tool` | 获取当前日期时间 |
| `weather` | `_weather` | 查询城市天气（wttr.in API） |
| `date` | `_date_calc` | 日期加减计算 |
| `weekday` | `_weekday` | 查询星期几 |
| `days_between` | `_days_between` | 计算两日期间隔天数 |
| `convert` | `_convert` | 单位换算（长度/重量/温度） |
| `base_convert` | `_base_convert` | 进制转换 |
| `str_tools` | `_str_tools` | 字符串处理（len/reverse/upper/lower） |
| `random` | `_random_tool` | 随机数生成（骰子/硬币） |
| `statistics` | `_statistics` | 数据统计（max/min/avg/sum） |
| `solve` | `_solve` | 解一元方程（sympy） |

**核心方法**：

| 方法 | 说明 |
|------|------|
| `register_tool(name, func, desc)` | 注册自定义工具 |
| `direct_call(tool_name, tool_input)` | 直通车：不经过 LLM，直接调用工具 |
| `run(question, debug=False)` | ReAct 循环：LLM 推理 + 工具调用 |

**ReAct 循环流程**：
1. 构建 Prompt（system_prompt + 工具描述 + 问题）
2. LLM 生成（在 Observation/Final Answer 处停止）
3. 解析输出：Final Answer / Action / Error
4. 若 Action：执行工具 → 追加 Observation → 继续循环
5. 若 Final Answer：返回结果
6. 超过 max_steps：返回 "Max steps reached."

**Prompt 设计要点**：
- 强制使用工具，禁止凭记忆回答
- 提供正确/错误示例
- 格式严格：Thought → Action → Action Input

**容错机制**：
- 工具名模糊匹配（小写化 + 子串匹配）
- 工具执行异常捕获

---

### 4.3 MemoryStore（`core/memory_store.py`）

混合记忆系统，正则优先 + LLM 兜底。

**核心属性**：
- `self.facts`：存储的事实列表 `[(key, value), ...]`
- `self.name`：用户名字（单独存储）
- `self.llm`：LLM 实例（用于兜底提取）

**支持的信息类型**：

| key | 示例输入 | 提取正则 |
|-----|---------|---------|
| `name` | "My name is Tom" | `my name is ([a-z]+)` |
| `location` | "I live in Beijing" | `i live in (.+)` |
| `age` | "I'm 25 years old" | `i'm (\d+) years? old` |
| `likes` | "I like music" | `i (?:like\|love\|enjoy) (.+)` |
| `hates` | "I hate bugs" | `i (?:hate\|dislike) (.+)` |
| `activity` | "I am playing piano" | `i am (?:playing\|doing\|...) (.+)` |
| `has` | "I have a cat" | `i have (?:a \|an )?(.+)` |
| `work` | "I work at Google" | `i work (?:at\|for\|in) (.+)` |
| `job` | "My job is engineer" | `my (?:job\|occupation) is (.+)` |
| `major` | "My major is CS" | `my major is (.+)` |
| `studies` | "I study math" | `i study (.+)` |
| `school` | "I go to MIT" | `i (?:go to\|attend) (.+)` |
| `favorite_*` | "My favorite color is blue" | `my favorite (.+?) is (.+)` |

**核心方法**：

| 方法 | 说明 |
|------|------|
| `extract_fact(query)` | 提取用户信息（正则优先，LLM 兜底） |
| `_extract_with_llm(query)` | LLM 兜底提取（返回 JSON） |
| `process(query)` | 处理查询，返回 (response, type, subtype) |
| `_add_fact(key, value)` | 存储事实（同 key 覆盖更新） |
| `get_all_facts()` | 获取所有存储的事实 |
| `clear()` | 清空记忆 |

**处理流程**：
1. 判断是否为查询类问题（问名字/问偏好）
2. 正则匹配 20 种模式
3. 正则未命中 → LLM 兜底提取
4. set 操作：存储事实 + 返回确认
5. get 操作：返回所有已知信息

**名称黑名单**：happy, fine, tired, ok, good, sad, sure, sorry 等（防止 "I am happy" 被误识别为名字）

---

### 4.4 InputClassifier（`core/input_classifier.py`）

输入分类路由器，将用户输入分发到对应处理模块。

**分类类型**：

| type | subtype | 说明 | 处理方式 |
|------|---------|------|---------|
| `identity` | `set_name` | 名字设置 | 直接回复 |
| `identity` | `question` | 身份查询 | 直接回复 |
| `identity` | `bot` | AI 自身问题 | 直接回复 |
| `preference` | `set` / `get` | 偏好设置/查询 | 交给 MemoryStore |
| `agent` | `time` / `date_today` | 时间查询 | direct_call |
| `agent` | `weekday` | 星期查询 | direct_call |
| `agent` | `weather` | 天气查询 | direct_call |
| `agent` | `arithmetic` | 算术计算 | direct_call(calc) |
| `agent` | `equation` | 方程求解 | direct_call(solve) |
| `agent` | `base_convert` | 进制转换 | direct_call |
| `agent` | `convert` | 单位转换 | direct_call |
| `agent` | `days_between` | 日期间隔 | direct_call |
| `agent` | `multi_tool` | 多工具问题 | ReAct 循环 |
| `rag` | - | 知识问答 | RAG 检索 |

**核心判断方法**：

| 方法 | 说明 |
|------|------|
| `is_time_question(query)` | 判断时间类问题，返回子类型 |
| `is_weather_question(query)` | 判断天气问题，返回城市名 |
| `is_identity_question(query)` | 判断身份查询 |
| `is_bot_question(query)` | 判断 AI 自身问题 |
| `is_arithmetic_question(query)` | 判断纯算术表达式 |
| `is_equation_question(query)` | 判断方程问题 |
| `is_multi_tool_question(query)` | 判断多工具问题（含 "then"） |
| `is_base_convert_question(query)` | 判断进制转换 |
| `is_convert_question(query)` | 判断单位转换 |
| `extract_arithmetic_expr(query)` | 提取算术表达式 |
| `extract_inline_context(query)` | 提取内联 Context 和 Question |

**路由优先级**：identity → preference → time → weather → equation → arithmetic → base_convert → convert → multi_tool → rag

---

### 4.5 VectorStore（`core/vector_store.py`）

基于 NumPy 的向量存储与检索，自定义实现。

**核心属性**：
- `self.embedding`：Embedding 实例（共享，不重复加载）
- `self.texts`：文本块列表
- `self.vectors`：归一化向量列表

**核心方法**：

| 方法 | 说明 |
|------|------|
| `add_text(text, chunk_size=80)` | 添加文本：切分 → 批量 embedding → 归一化存储 |
| `add_chunk(text)` | 添加单个文本块 |
| `split_text(text, chunk_size=80)` | 文本切分（按句子，累积到 chunk_size） |
| `search(query, k=3, threshold=0.3)` | 向量检索：query embedding → 点积相似度 → Top-K |

**检索原理**：
1. Query → Embedding → 归一化
2. 矩阵点积 = Cosine Similarity（因已归一化）
3. argsort 降序取 Top-K
4. threshold 过滤低相似度结果

**文本切分策略**：
1. 清洗 PDF 文本（text_cleaner）
2. 按句子切分（`.!?` 分隔）
3. 累积句子到 chunk_size（按词数计算）
4. 超长句子单独成块

---

### 4.6 BM25Store（`core/bm25_store.py`）

基于 rank_bm25 的稀疏检索，支持中英文分词。

**核心属性**：
- `self.corpus`：分词后的 chunk 列表
- `self.raw_docs`：原始 chunk 列表
- `self.bm25`：BM25Okapi 实例

**核心方法**：

| 方法 | 说明 |
|------|------|
| `add(text)` | 添加文本：切分 → 分词 → 存入语料 |
| `build()` | 构建 BM25 索引（所有文档 add 后调用） |
| `search(query, top_k=5)` | BM25 检索 |
| `_tokenize(text)` | 分词：英文按空格，中文用 jieba |

**分词策略**：
- 英文：按空格切分
- 中文：jieba 分词
- 统一小写

---

### 4.7 Embedding（`core/embedding.py`）

文本向量化封装，基于 SentenceTransformer。

**初始化参数**：
- `model_path`：模型路径，默认 `models/all-MiniLM-L6-v2`

**核心方法**：

| 方法 | 说明 |
|------|------|
| `embed(texts)` | 单条/批量向量化（自动统一输入格式） |
| `embed_batch(texts)` | 批量向量化 |

**模型信息**：
- all-MiniLM-L6-v2：384 维向量，80MB
- 支持中英文（英文为主）

---

### 4.8 Retriever（`core/retriever.py`）

混合检索协调器，整合向量检索 + BM25 + Rerank。

**初始化参数**：
- `vector_store`：VectorStore 实例
- `bm25_store`：BM25Store 实例（可选）
- `reranker`：Rerank 实例（可选）

**检索流程**：
1. 向量检索：召回 k×2 个候选
2. BM25 检索：召回 k×2 个候选
3. 合并去重
4. Rerank 精排：取 Top-K

**核心方法**：

| 方法 | 说明 |
|------|------|
| `retrieve(query, k=3, threshold=0.3)` | 混合检索 |
| `_merge_and_dedup(list_a, list_b)` | 合并去重 |

---

### 4.9 Rerank（`core/rerank.py`）

基于 Cross-Encoder 的重排序。

**初始化参数**：
- `model_path`：模型路径，默认 `models/cross-encoder-ms-marco`

**核心方法**：

| 方法 | 说明 |
|------|------|
| `rerank(query, top_k, top_n=3)` | Cross-Encoder 打分 → 排序 → 取 Top-N |

**模型信息**：
- cross-encoder-ms-marco：MS MARCO 数据集训练
- 输入：(query, doc) 对
- 输出：相关性分数

---

### 4.10 Memory（`core/memory.py`）

对话历史管理，基于 deque 的滑动窗口。

**初始化参数**：
- `max_history`：最大历史轮数，默认 20

**核心方法**：

| 方法 | 说明 |
|------|------|
| `add_user_message(message)` | 添加用户消息 |
| `add_assistant_message(message)` | 添加助手消息 |
| `get_messages()` | 获取完整消息列表（system + history） |
| `clear()` | 清空历史 |

**数据结构**：
- `self.head`：system prompt（固定）
- `self.history`：deque(maxlen=max_history)，自动淘汰旧消息

---

### 4.11 TextCleaner（`core/text_cleaner.py`）

CS336 风格的 PDF 文本清洗模块。

**核心函数**：

| 函数 | 说明 |
|------|------|
| `clean_pdf_text_336_style(text)` | 主清洗函数 |
| `_is_valid_line(line)` | 判断有效行（长度/符号比例/重复词/噪声模式） |
| `_protect_fake_periods(text)` | 保护假句号（小数点/缩写/代码/URL） |
| `restore_protected_periods(text)` | 恢复被保护的点号 |

**清洗规则**：
1. 长度过滤：< 15 字符的行丢弃
2. 符号比例：字母占比 < 40% 丢弃
3. 重复词：唯一词占比 < 30% 丢弃
4. 噪声模式：Figure/Table/Page/公式编号等丢弃

**假句号保护**：
- 小数点：`1.5` → `1__DECIMAL__5`
- 缩写：`e.g.` → `e__ABBR__g__ABBR__`
- 代码：`nn.Linear()` → `nn__CODE__Linear()`
- URL：`http://` → `http__URL__//`

---

## 五、主程序（`chat.py`）

主程序负责初始化所有模块、加载知识库、协调处理流程。

**初始化流程**：
1. 加载 Embedding → VectorStore → BM25Store → Rerank → Retriever
2. 加载 LLM → ReActAgent → InputClassifier → MemoryStore
3. 加载知识库（data/ 目录下的 txt/pdf 文件）

**核心函数 `get_response(query, history, memory)`**：

```
query → MemoryStore.process() → [命中] → 返回记忆响应
                              → [未命中] → Classifier.classify()
                                         → identity → 直接回复
                                         → agent → direct_call / ReAct
                                         → rag → 混合检索 → LLM 生成
```

**辅助函数**：
- `format_history(messages, max_turns=3)`：格式化对话历史
- `build_prompt(query, context, history_text)`：构建 RAG Prompt
- `postprocess_response(text)`：后处理（去除前缀/截断）

---

## 六、技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| LLM | Qwen2.5-1.5B | 文本生成 |
| Embedding | all-MiniLM-L6-v2 (SentenceTransformer) | 文本向量化 |
| Rerank | cross-encoder-ms-marco (CrossEncoder) | 重排序 |
| 向量检索 | NumPy (自定义) | 语义相似度 |
| 稀疏检索 | BM25Okapi + jieba | 关键词匹配 |
| 模型框架 | PyTorch + Transformers | 模型加载与推理 |
| 量化 | bitsandbytes / GGUF / GPTQ | 模型压缩 |

---

## 七、项目结构

```
NanoChat-Lab/
├── core/
│   ├── llm.py              # LLM 封装（Qwen2.5-1.5B + stop_strings）
│   ├── ReActAgent.py       # ReAct Agent（12+ 工具）
│   ├── memory_store.py     # 记忆系统（正则 + LLM）
│   ├── memory.py           # 对话历史（deque 滑动窗口）
│   ├── input_classifier.py # 输入分类路由器
│   ├── vector_store.py     # 向量存储（NumPy）
│   ├── bm25_store.py       # BM25 稀疏检索
│   ├── embedding.py        # 文本向量化（SentenceTransformer）
│   ├── retriever.py        # 混合检索协调器
│   ├── rerank.py           # Cross-Encoder 重排序
│   └── text_cleaner.py     # PDF 文本清洗（CS336 风格）
├── data/                   # 知识库文档（txt/pdf）
├── models/                 # 本地模型文件
│   ├── Qwen2.5-1.5B/
│   ├── all-MiniLM-L6-v2/
│   ├── cross-encoder-ms-marco/
│   └── qwen2.5-1.5b-instruct-q4_k_m.gguf
├── eval/                   # 评估测试用例
├── chat.py                 # 主程序入口
├── run_eval.py             # 评估脚本
├── test_quantization.py    # 量化对比测试
├── quantize_model.py       # 量化工具脚本
└── benchmark.py            # 系统 vs 裸LLM 评测
```

---

## 八、评测体系

### 8.1 功能测试（`run_eval.py`）

- 99 项测试用例
- 100% 通过率
- 覆盖：工具函数、分类器、集成测试、边界容错

### 8.2 量化评测（`test_quantization.py`）

| 方案 | 显存 | 加载时间 | 推理速度 | 精度 |
|------|------|---------|---------|------|
| FP16 基准 | ~2.88GB | - | - | 基准 |
| bitsandbytes INT4 | ~1.11GB | - | - | 待测 |
| GGUF INT4 | 待测 | 待测 | 待测 | 待测 |
| GPTQ INT4 | 待测 | 待测 | 待测 | 待测 |

### 8.3 系统评测（`benchmark.py`）

- 项目系统 vs 裸 LLM 对比
- 指标：准确率、延迟
- 10+ 测试用例（RAG / Agent / Memory）
