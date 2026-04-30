# NanoChat-Lab

在 4GB 显存（GTX 1650）约束下，基于 Qwen2.5-1.5B 构建的端到端 RAG + Agent + Memory 系统。通过 LoRA 微调 + Prompt 压缩实现"更准、更快、更省"的工具调用能力。全部模块自研实现，不依赖 LangChain。

## 核心成果

| 指标 | 原生模型+长Prompt | 微调模型+短Prompt | 提升 |
|------|-----------------|-----------------|------|
| 工具准确率 | 80.5% | 85.5% | +5.0% |
| 答案准确率 | 55.0% | 63.5% | +8.5% |
| 格式正确率 | 90.5% | 98.5% | +8.0% |
| 平均延迟 | 10.26s | 9.01s | -1.25s |
| Prompt Token | 282 tokens | 149 tokens | -47.2% |

## 系统架构

```
用户输入
    ↓
┌─────────────────────────────────────┐
│       MemoryStore（记忆处理）         │
│     正则优先 + LLM 兜底提取信息       │
└─────────────────────────────────────┘
    ↓ 未命中
┌─────────────────────────────────────┐
│      InputClassifier（分类路由）      │
│   identity / agent / rag / reject    │
└─────────────────────────────────────┘
    ↓
┌──────────┬───────────────┬──────────┐
│ identity │    agent      │   rag    │
│ 直接回复  │ direct_call   │ 混合检索  │
│          │ / ReAct 循环   │ + Rerank │
└──────────┴───────────────┴──────────┘
    ↓
┌─────────────────────────────────────┐
│     LLM（Qwen2.5-1.5B + LoRA）      │
│   int4 量化 | LoRA adapter 热加载    │
└─────────────────────────────────────┘
```

## 快速开始

### 环境安装

```bash
pip install -r requirement.txt
```

### 运行（默认加载 LoRA 微调模型）

```bash
python chat.py
```

### 运行选项

```bash
# 指定量化模式
python chat.py -q int4

# 禁用 LoRA，使用原生模型
python chat.py --lora none

# 使用 GGUF 格式模型
python chat.py -q gguf -m models/Qwen2.5-1.5B-GGUF
```

## 项目结构

```
NanoChat-Lab/
├── chat.py                  # 主程序入口
├── core/
│   ├── llm.py              # LLM 推理（支持 LoRA 热加载 + 5种量化）
│   ├── ReActAgent.py       # ReAct 智能体（12+ 工具）
│   ├── input_classifier.py # 输入分类路由
│   ├── memory_store.py     # LLM 驱动记忆系统
│   ├── retriever.py        # 混合检索（Vector + BM25）
│   ├── vector_store.py     # 自研 NumPy 向量库
│   ├── bm25_store.py       # BM25 检索
│   └── rerank.py           # Cross-Encoder 重排
├── eval/
│   ├── test_cases.jsonl    # 200 条测试用例
│   └── results/            # 评估结果
├── eval_sbs_compare.py     # SBS 对比评估框架
├── build_sft_data.py       # SFT 训练数据构建
├── train_lora.py           # LoRA 微调脚本
├── inference_lora.py       # LoRA 推理测试
└── quantize_model.py       # 量化工具
```

## 技术亮点

### 1. LoRA 微调 + Prompt 压缩

- 3000 条 SFT 数据，云端 fp32 训练（rank=8, 300 steps）
- 精简 System Prompt 从 282 tokens 压缩到 149 tokens
- 通过 verify 诊断模式定位性能瓶颈：确认问题在 Prompt 设计而非微调质量

### 2. InputClassifier 路由 + Agent 直通

- 正则匹配 + 语义识别实现任务预分类
- Agent 直通模式跳过 ReAct 循环：准确率 +40%，延迟降低 800 倍
- 解决小模型 ReAct 循环不稳定性问题

### 3. 混合检索架构

- Vector + BM25 双路召回
- Cross-Encoder 重排序
- 消融实验 Top-3 命中率 92%

### 4. 多维评测体系

- SBS 对比评估：工具准确率 / 答案准确率 / 格式正确率 / 延迟
- DPO 偏好数据自动生成
- verify 诊断模式：分离 Prompt 影响与模型能力
- 训练 → 评估 → 诊断 → 迭代闭环

### 5. 模型量化

| 模式 | 显存占用 | 适用场景 |
|------|---------|---------|
| fp16 | ~3.5GB | 精度优先 |
| int8 | ~2.2GB | 平衡 |
| int4 | ~1.8GB | 显存受限（默认） |
| fp4  | ~1.8GB | 实验对比 |
| GGUF | ~1.5GB | CPU/极低显存 |

## 工具列表

| 工具 | 功能 | 示例输入 |
|------|------|---------|
| calc | Python 数学计算 | `print(3*8)` |
| time | 当前时间 | 无需输入 |
| weather | 城市天气查询 | `Beijing` |
| convert | 单位换算 | `100km to miles` |
| base_convert | 进制转换 | `255 to hex` |
| solve | 方程求解 | `3*x+1=10` |
| date | 日期计算 | `2025-07-05 + 30 days` |
| weekday | 星期查询 | `2025-07-05` |
| days_between | 日期间隔 | `2025-01-01, 2025-07-05` |
| str_tools | 字符串操作 | `hello, len` |
| random | 随机生成 | `dice 3` |
| statistics | 统计计算 | `1,5,3,9,2, max` |

## 评估复现

```bash
# 完整 200 条 SBS 对比评估
python eval_sbs_compare.py

# 快速测试 20 条
python eval_sbs_compare.py --quick --n 20

# 诊断模式（同时测试微调模型+长/短Prompt）
python eval_sbs_compare.py --verify
```

## 硬件要求

- GPU: 4GB+ 显存（GTX 1650 验证通过）
- RAM: 8GB+
- 磁盘: ~5GB（模型 + 数据）
