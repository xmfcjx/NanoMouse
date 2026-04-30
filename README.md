# NanoChat-Lab

> 当大模型遇上 4GB 显存：如何在消费级硬件上跑出生产级效果？

NanoChat-Lab 探索的核心问题是：**在设备受限环境下（4GB VRAM / GTX 1650），如何让 1.5B 参数的小模型具备可靠的工具调用、知识检索和多轮对话能力？**

本项目不依赖 LangChain 等重型框架，全部模块从零实现，通过"量化压缩 + LoRA 微调 + Prompt 工程 + 路由优化"四层策略，在极低资源下达到甚至超越原生大 Prompt 方案的效果。

## 为什么做这个项目

大模型落地面临一个现实矛盾：云端 API 有成本和隐私问题，而本地部署受限于硬件。大量边缘场景（嵌入式设备、老旧工作站、无网络环境）只有 4-6GB 显存可用。NanoChat-Lab 验证了一条可行路径：

- 1.5B 模型 + int4 量化 → 显存占用 < 2GB
- LoRA 微调 + 短 Prompt → 比原生模型 + 长 Prompt 更准更快
- 分类器路由 + 直通调用 → 绕过小模型 ReAct 循环的不稳定性

## 核心成果

**200 条 SBS 对比评估（微调+短Prompt vs 原生+长Prompt）：**

| 指标 | 原生模型 + 长Prompt (282 tokens) | 微调模型 + 短Prompt (149 tokens) | 变化 |
|------|:---:|:---:|:---:|
| 工具调用准确率 | 80.5% | **85.5%** | +5.0% |
| 答案准确率 | 55.0% | **63.5%** | +8.5% |
| 格式正确率 | 90.5% | **98.5%** | +8.0% |
| 平均延迟 | 10.26s | **9.01s** | -1.25s |
| Prompt 开销 | 282 tokens | **149 tokens** | -47% |

**结论：用更少的 Token、更短的延迟，获得更高的准确率。**

## 资源占用

| 量化模式 | 模型显存 | 总运行显存 | 适用场景 |
|:---:|:---:|:---:|:---:|
| int4 (默认) | ~1.8GB | ~2.8GB | 4GB 显卡 (GTX 1650) |
| int8 | ~2.2GB | ~3.2GB | 6GB 显卡 |
| fp16 | ~3.5GB | ~4.5GB | 8GB+ 显卡 |
| GGUF | ~1.5GB | ~2.0GB | 纯 CPU / 极低显存 |

## 系统架构

```
用户输入
    ↓
┌─────────────────────────────────────────┐
│         MemoryStore（记忆层）             │
│       正则优先 + LLM 兜底提取信息         │
└─────────────────────────────────────────┘
    ↓ 未命中
┌─────────────────────────────────────────┐
│       InputClassifier（路由层）           │
│    identity / agent / rag / reject       │
└─────────────────────────────────────────┘
    ↓
┌───────────┬────────────────┬────────────┐
│ identity  │     agent      │    rag     │
│ 直接回复   │  direct_call   │  混合检索   │
│           │  / ReAct 循环   │  + Rerank  │
└───────────┴────────────────┴────────────┘
    ↓
┌─────────────────────────────────────────┐
│    Qwen2.5-1.5B + LoRA + int4 量化      │
│         显存占用 < 3GB                   │
└─────────────────────────────────────────┘
```

## 受限环境下的关键设计

### 1. 四级量化适配

支持 fp16 / int8 / int4 / fp4 / GGUF 五种模式，根据设备显存自动推荐最优量化策略。int4 模式下 1.5B 模型仅占 1.8GB，为检索、Embedding、Rerank 留出充足空间。

### 2. LoRA 微调 + Prompt 压缩

小模型 + 长 Prompt = 有效上下文被挤占。通过 LoRA 将工具调用格式"烧入"模型权重，System Prompt 从 282 tokens 压缩到 149 tokens（-47%），释放出的上下文窗口留给用户输入和检索结果。

- 训练配置：rank=8, 3000 条 SFT 数据, 300 steps
- 诊断方法：verify 模式分离 Prompt 影响与模型能力，确认瓶颈来源

### 3. 分类器路由 + Agent 直通

小模型跑 ReAct 循环容易"跑偏"（幻觉、格式错误、死循环）。解决方案：

- InputClassifier 预判任务类型，简单任务直接调用工具，跳过 LLM 推理
- 仅复杂多步任务才进入 ReAct 循环
- 效果：准确率 +40%，延迟从秒级降到毫秒级

### 4. 轻量混合检索

不依赖 FAISS / Milvus 等重型向量库：

- 自研 NumPy 向量存储（零外部依赖）
- Vector + BM25 双路召回 + Cross-Encoder 重排
- 消融实验 Top-3 命中率 92%

## 快速开始

```bash
# 安装依赖
pip install -r requirement.txt

# 运行（默认 int4 量化 + LoRA）
python chat.py

# 禁用 LoRA，对比原生效果
python chat.py --lora none

# 极低显存模式
python chat.py -q gguf -m models/Qwen2.5-1.5B-GGUF
```

## 评估复现

```bash
# 完整 200 条 SBS 对比评估
python eval_sbs_compare.py

# 诊断模式（分离 Prompt 影响 vs 模型能力）
python eval_sbs_compare.py --verify

# 快速验证
python eval_sbs_compare.py --quick --n 20
```

## 项目结构

```
NanoChat-Lab/
├── chat.py                  # 主程序入口
├── core/
│   ├── llm.py              # LLM 推理（LoRA 热加载 + 5种量化）
│   ├── ReActAgent.py       # ReAct 智能体（12 种工具）
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
└── quantize_model.py       # 量化工具
```

## 工具列表

| 工具 | 功能 | 示例输入 |
|------|------|---------|
| calc | 数学计算 | `print(3*8)` |
| time | 当前时间 | 无需输入 |
| weather | 天气查询 | `Beijing` |
| convert | 单位换算 | `100km to miles` |
| base_convert | 进制转换 | `255 to hex` |
| solve | 方程求解 | `3*x+1=10` |
| date | 日期计算 | `2025-07-05 + 30 days` |
| weekday | 星期查询 | `2025-07-05` |
| days_between | 日期间隔 | `2025-01-01, 2025-07-05` |
| str_tools | 字符串操作 | `hello, len` |
| random | 随机生成 | `dice 3` |
| statistics | 统计计算 | `1,5,3,9,2, max` |

## 硬件要求

- **最低**：4GB 显存 GPU（GTX 1650 验证通过）或纯 CPU（GGUF 模式）
- **推荐**：6GB+ 显存
- RAM：8GB+
- 磁盘：~5GB（模型 + 数据）
