# NanoChat 架构升级方案

## 1. 项目主线

将 NanoChat 从“多个模块的功能组合”升级为：

> **Resource-Aware Adaptive NanoChat：面向低显存设备的置信度路由、多适配器专家与动态上下文系统。**

核心研究问题：

> 在 4GB 显存、1.5B 基座模型和有限上下文预算下，如何让系统根据任务难度、路由置信度、检索证据和计算预算，自适应选择最低成本且可靠的执行路径？

项目含金量不依赖堆叠技术名词，而来自三个部分：

1. 发现并形式化当前架构的瓶颈；
2. 设计具有明确目标函数和决策机制的新架构；
3. 用端到端实验说明质量、成本和可靠性的变化。

## 2. 当前架构的主要问题

### 2.1 路由依赖规则，缺少置信度

`InputClassifier` 通过正则和关键词决定 identity、agent、rag 等路径。优点是快，但存在：

- 规则覆盖不足，表达变体容易误路由；
- 没有概率或置信度，无法判断何时回退模型；
- 规则逻辑与工具参数抽取耦合；
- 新增工具需要持续增加硬编码。

### 2.2 单一 LoRA 存在多任务干扰

当前一份 LoRA 同时学习 ReAct、RAG 和 Memory：

- 不同任务的输出格式差异明显；
- 任务数据量不平衡时会发生负迁移；
- 无法定位某项能力提升或退化来自哪个任务；
- 为所有请求加载同一适配器，不符合按需计算目标。

### 2.3 RAG 使用固定预算

当前检索固定 Top-k，最终按字符截断到 1000：

- 简单问题可能召回过多，浪费上下文；
- 复杂问题可能证据不足；
- 字符截断可能破坏文档语义；
- 缺少 evidence confidence 和无答案判定；
- 检索、重排和生成没有统一 token budget。

### 2.4 工具调用依赖自由文本协议

当前 ReAct 使用 `Thought / Action / Action Input / Final Answer` 文本解析：

- 小模型容易重复、漏字段或输出非法工具名；
- 多步任务可能陷入循环；
- 训练目标包含不可验证的自由文本 Thought；
- 工具执行结果和模型最终回答缺少结构化状态。

### 2.5 产品入口重复编排

`chat.py` 与 `api.py` 分别维护相似流程，容易产生行为偏差。模型、Embedding、Reranker 均在启动时加载，缺少统一生命周期、追踪和资源调度。

## 3. 目标架构

```text
                         ┌──────────────────────┐
User Request ---------->│ Unified Orchestrator │
                         └──────────┬───────────┘
                                    |
                         Feature / Cost / Budget
                                    |
                         ┌──────────v───────────┐
                         │ Confidence Router     │
                         │ rule + learned model  │
                         └───┬─────┬─────┬──────┘
                             |     |     |
                  direct tool|     |     |memory
                             |     |     |
                     ┌───────v┐ ┌──v──────────┐
                     │Tool FSM│ │Adaptive RAG │
                     └───────┬┘ └──┬──────────┘
                             |     |
                             └──┬──┘
                                |
                    ┌───────────v────────────┐
                    │ Adapter Expert Manager │
                    │ tool / rag / memory    │
                    └───────────┬────────────┘
                                |
                       Qwen2.5-1.5B INT4
                                |
                    Structured Output Validator
                                |
                         Trace + Evaluation
```

## 4. 核心创新一：置信度与成本感知路由

### 4.1 两级路由

第一层保留高精度规则，用于明显的时间、算术等请求。第二层使用轻量分类器输出：

```text
P(route | query, history) =
    [direct_tool, rag, memory, model, reject]
```

当最大概率低于阈值，或第一、第二候选差距过小时，进入安全回退路径，而不是强行路由。

### 4.2 决策函数

路由不只最大化分类概率，还考虑执行成本：

```text
score(route) =
    confidence(route)
    - lambda_latency * estimated_latency(route)
    - lambda_memory * estimated_vram(route)
    - lambda_risk * failure_risk(route)
```

在 4GB 环境下，这个目标比单纯追求路由准确率更符合产品约束。

### 4.3 可展示点

- 路由置信度校准；
- coverage-risk 曲线；
- 不同阈值下准确率、回退率和延迟的 Pareto 前沿；
- 与纯规则、纯 LLM 路由的对比。

## 5. 核心创新二：任务专用 LoRA 专家

不修改 1.5B 基座为传统 MoE，而是在低显存条件下使用 **Mixture of LoRA Experts**：

```text
Base INT4 model
  ├── tool_adapter
  ├── rag_adapter
  └── memory_adapter
```

路由器根据任务选择一个 adapter。第一阶段只激活单专家，避免同时加载多个专家带来的显存和实现复杂度。

### 5.1 训练方式

- 共享同一冻结基座；
- 按任务拆分当前 3000 条 SFT；
- 每个专家独立训练；
- 保留一份 unified multi-task LoRA 作为对照；
- 统一使用 assistant-only loss 和相同训练 token 预算。

### 5.2 研究问题

- 专家 LoRA 是否减少多任务负迁移？
- adapter 切换延迟是否可接受？
- 相同可训练参数预算下，专家方案是否优于单一 LoRA？
- 路由错误对最终质量的敏感度有多高？

### 5.3 可选升级

在单专家路由稳定后，再研究：

- weighted adapter fusion；
- rsLoRA / DoRA；
- 按层选择不同 rank；
- 高频专家常驻、低频专家按需加载。

## 6. 核心创新三：动态上下文预算

将固定 Top-k 和字符截断改为 token-aware context packing。

### 6.1 动态检索

根据查询难度和检索分数决定：

- 是否需要 RAG；
- 候选数量；
- 是否启用 CrossEncoder；
- 最终可使用的 context token 数。

### 6.2 证据选择目标

在上下文预算 `B` 内选择文档集合：

```text
maximize:
    relevance + diversity - redundancy

subject to:
    sum(document_tokens) <= B
```

可使用贪心 MMR 作为第一版，不必一开始引入复杂优化器。

### 6.3 无答案控制

使用检索置信度、top-1/top-2 margin 和证据覆盖率决定回答或拒答，重点评估：

- answerable accuracy；
- unanswerable rejection accuracy；
- hallucination rate；
- context tokens 与延迟。

## 7. 核心创新四：结构化工具执行

用结构化动作协议替换自由文本 ReAct：

```json
{
  "action": "base_convert",
  "arguments": {"value": "128", "target": "hex"}
}
```

执行器维护显式状态：

```text
PLAN -> TOOL_CALL -> OBSERVATION -> TOOL_CALL / FINAL
```

每一步执行：

- JSON schema 校验；
- 工具白名单检查；
- 参数类型检查；
- 最大步数与重复动作检测；
- 工具结果验证。

训练数据不要求模型生成冗长 Thought，只监督可执行 action 和 final answer。这样更适合小模型，也减少不可验证推理链污染。

## 8. 工程结构调整

建议逐步演进为：

```text
nanochat/
├── orchestration/
│   ├── orchestrator.py
│   ├── router.py
│   ├── budget.py
│   └── trace.py
├── models/
│   ├── backend.py
│   ├── transformers_backend.py
│   ├── gguf_backend.py
│   └── adapter_manager.py
├── retrieval/
│   ├── pipeline.py
│   ├── context_packer.py
│   └── confidence.py
├── tools/
│   ├── registry.py
│   ├── executor.py
│   └── schema.py
├── memory/
├── training/
└── evaluation/
```

CLI 与 FastAPI 只调用同一个 `Orchestrator.handle()`，不再复制业务流程。

## 9. 实施顺序

### A. 统一编排与 Trace

先抽取统一 Orchestrator 和请求 Trace，保持行为不变。Trace 至少记录：

- route、subtype、confidence；
- adapter；
- retrieval candidates 和最终 context；
- tool steps；
- prompt/output tokens；
- latency 和错误。

### B. 结构化工具协议

先解决当前最明显的小模型 ReAct 不稳定问题，建立可校验执行器。

### C. 置信度路由

构造路由数据集，训练轻量分类器，并与现有规则组成两级路由。

### D. 多 LoRA 专家

拆分训练数据，训练 tool/rag/memory adapter，并加入 AdapterManager。

### E. 动态 RAG

实现 token-aware packing、动态 Top-k 和拒答判定。

### F. 偏好优化

在结构化输出和可验证指标稳定后，再进行 DPO/SimPO/ORPO。偏好数据优先由执行正确性自动产生，而不是仅靠启发式文本评分。

## 10. 面试叙事

项目不应表述为：

> 我给 Qwen 做了 LoRA、量化、RAG 和 DPO。

应表述为：

> 我发现 1.5B 模型在低显存设备上主要受多任务干扰、路由误差、固定上下文预算和自由文本工具协议影响。因此设计了一个资源感知的自适应架构：两级置信度路由负责选择计算路径，任务专用 LoRA 专家减少负迁移，动态检索在 token 预算内选择证据，结构化状态机保证工具调用可执行。随后在相同模型、数据和计算预算下，对质量、延迟、显存、拒答和失败类型进行端到端评估。

这套叙事能够同时回答模型训练、检索、Agent、系统优化和实验设计问题。
