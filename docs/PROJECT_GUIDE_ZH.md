# NanoChat-Lab（EdgeOps Copilot）项目完整说明

## 1. 项目概览

NanoChat-Lab 的最终主线是 **EdgeOps Copilot**：一个面向工厂 AMR、叉车和设备维护场景的本地运维助手原型。项目以 Qwen2.5-1.5B 为基座，围绕低显存推理、结构化工具调用、手册检索、安全拒答和可审计执行展开。

项目解决的不是“让大模型直接控制设备”，而是以下更受约束的问题：

1. 识别用户是在查询设备状态、诊断故障、读取维护记录、定位资产、检索手册，还是提出了高风险操作；
2. 将可执行请求转换为结构化工具调用，而不是解析不可控的自由文本；
3. 对工具名、参数、调用次数、执行步数和风险等级进行独立校验；
4. 对手册类问题只依据本地证据回答，证据不足时拒绝猜测；
5. 在 4 GB 推理显存目标下验证小模型与 LoRA Adapter 的可行性；
6. 用独立 dev、sealed test 和量化验收形成可追溯的结果闭环。

当前正式版本是 **v2**。v3 只是一项未通过晋级门槛的调优实验，不是最终模型。

## 2. 项目边界

### 2.1 已实现的能力

- 查询设备状态、电量、温度、位置和当前故障码；
- 查询故障码含义、严重等级和安全检查建议；
- 查询设备维护历史；
- 查询资产最后位置；
- 检索本地设备手册和 SOP；
- 生成需要人工确认的维修工单草稿；
- 拒绝运动控制、绕过保护、强制复位等高风险请求；
- 输出统一的路由结果、置信度、工具结果、证据、警告和 Trace ID；
- 通过 CLI 和 FastAPI 暴露同一套编排逻辑；
- 构造隔离的数据集，训练和评估两个任务 Adapter；
- 执行 sealed test 与 INT4 资源验收。

### 2.2 明确没有实现或验证的内容

- 没有接入真实工厂控制系统；
- 没有获得功能安全认证；
- 没有允许模型直接控制设备运动；
- 没有在 GTX 1650 上完成最终性能验收；
- 没有证明当前 16--17 秒 p95 延迟适合实时生产使用；
- 没有证明 DPO 优于 safety SFT；
- 没有将 v3 作为最终模型；
- 没有公开模型权重、生成训练集、sealed test 原文或逐条预测。

因此，本项目的准确定位是：**具备完整训练、评测和执行边界的工业运维 Copilot 工程原型**。

## 3. 需求与设计目标

### 3.1 业务目标

运维人员通常需要在设备状态、告警、手册和维护系统之间切换。EdgeOps Copilot 将这些信息入口统一为自然语言接口，同时确保事实查询来自工具、手册回答来自证据、高风险请求不会进入执行层。

### 3.2 资源目标

基座模型选择 Qwen2.5-1.5B，最终推理采用 INT4 NF4：

- 目标峰值显存：低于 4 GB；
- 允许训练在更高显存环境完成；
- Adapter 与基座分离，避免为每个任务保存完整模型；
- 通过共享基座加载多个 PEFT Adapter。

### 3.3 可靠性目标

- 模型输出必须能被 JSON 解析；
- 工具必须在白名单内；
- 参数必须满足 Schema；
- 重复工具调用必须被拒绝；
- 工具计划最多执行 4 步；
- 工单草稿必须经过人工确认；
- 运动控制工具即使被模型生成也不能执行；
- 每次请求产生结构化 Trace。

## 4. 总体架构

```text
CLI / FastAPI
      |
      v
EdgeOpsOrchestrator
      |
      +--> ConfidenceRouter
      |      +--> route
      |      +--> confidence
      |      +--> entities
      |      +--> estimated cost
      |
      +--> safety gate
      |
      +--> direct read-only tool path
      |      +--> ToolRegistry
      |      +--> StructuredToolExecutor
      |      +--> EquipmentGateway
      |
      +--> multi-step model plan
      |      +--> route_tool Adapter
      |      +--> StructuredActionParser
      |      +--> ToolRegistry / Executor
      |
      +--> manual retrieval path
      |      +--> LexicalRetriever
      |      +--> ContextPacker
      |      +--> evidence-constrained generation
      |
      +--> TraceRecorder
             +--> route event
             +--> retrieval event
             +--> tool-state events
             +--> generation event
             +--> outcome
```

### 4.1 统一编排器

`edgeops/orchestrator.py` 是系统主入口。CLI 与 API 都通过 `create_orchestrator()` 构造同一实例，避免两套入口分别维护业务逻辑。

编排器的主要流程：

1. 创建 Trace；
2. 调用路由器获取 route、confidence、实体和候选分数；
3. 先处理安全拒绝；
4. 简单事实请求走确定性只读工具；
5. 多步请求由模型生成 JSON 计划或使用无模型回退计划；
6. 手册请求检索并打包证据；
7. 工具执行或生成完成后返回统一 `EdgeResponse`。

统一响应包含：

```json
{
  "answer": "面向用户的回答",
  "route": "device_status",
  "confidence": 0.88,
  "trace_id": "唯一追踪 ID",
  "adapter": "tool_adapter",
  "tool_results": [],
  "evidence": [],
  "requires_confirmation": false,
  "warnings": []
}
```

### 4.2 置信度与成本感知路由

`edgeops/router.py` 当前使用规则分数，并预留 `learned_scorer` 接口。每个候选 route 同时考虑置信度、估计延迟和估计显存：

```text
score(route)
  = confidence
  - latency_weight * estimated_latency
  - memory_weight * estimated_memory
```

支持的 route：

| Route | 含义 |
| --- | --- |
| `safety_reject` | 高风险控制或保护绕过 |
| `device_status` | 当前设备状态 |
| `error_diagnosis` | 故障码与诊断 |
| `maintenance_history` | 维护记录 |
| `asset_location` | 资产位置 |
| `tool_plan` | 多步工具任务或工单草稿 |
| `manual_rag` | 手册、SOP 或证据问答 |
| `general` | 能力域外请求 |

路由器还抽取设备 ID、资产 ID 和故障码，例如将 `AMR07` 归一化为 `AMR-07`。

需要注意：当前在线编排器仍以确定性路由作为安全和工具入口；v2 `route_tool` Adapter 的正式指标来自独立自由生成评测。项目没有把规则路由指标冒充模型指标。

### 4.3 Adapter 与模型后端

`edgeops/backend.py` 定义三种后端：

- `NullBackend`：不加载模型，便于本地冒烟测试；
- `LegacyLLMBackend`：兼容旧模型封装；
- `PeftMultiAdapterBackend`：共享一个量化基座，按请求切换 PEFT Adapter。

v2 目录约定：

```text
models/edgeops_adapters_v2/
├── route_tool/
└── safety_refusal/
```

在线工具规划通过 `tool_adapter` 逻辑名称加载 `route_tool/`。`safety_refusal/` 可以被同一后端加载，但当前在线安全门禁优先使用独立确定性规则，尚未实现根据请求动态调用 safety Adapter 再融合判定。safety Adapter 的结果主要通过独立 generation evaluation 验证。

这是一个刻意披露的工程边界：**模型能力评测已经完成，但双 Adapter 的完整在线协同仍可继续演进。**

## 5. 工具执行与安全边界

### 5.1 工具注册表

`edgeops/tools.py` 中的工具都带有名称、描述、参数 Schema、风险等级和处理函数。

| 工具 | 风险 | 行为 |
| --- | --- | --- |
| `get_device_status` | read-only | 读取状态和遥测 |
| `lookup_error_code` | read-only | 查询故障码 |
| `get_maintenance_history` | read-only | 查询维护记录 |
| `locate_asset` | read-only | 查询最后位置 |
| `create_work_order_draft` | confirmation | 只创建草稿 |
| `move_robot` | forbidden | 始终拒绝 |

`move_robot` 被注册为 forbidden 的目的不是提供运动能力，而是验证即使模型生成了该名称，执行层仍会拒绝。

### 5.2 参数验证

执行前检查：

- 必填参数是否存在；
- 字符串或数值类型是否正确；
- 是否出现 Schema 未声明的参数；
- 工具是否存在；
- 是否需要人工确认；
- 是否属于禁止动作。

模型生成正确 JSON 不是执行成功的充分条件。只有通过注册表和风险门禁后，工具才会进入 Gateway。

### 5.3 有限状态执行器

`edgeops/executor.py` 使用以下状态：

```text
PLAN -> TOOL_CALL -> OBSERVATION -> FINAL
                         |
                         +-> REJECTED
```

执行器限制最多 4 步，并对工具名和排序后的参数构造调用签名。相同调用再次出现时立即终止，防止小模型陷入循环。

### 5.4 结构化计划解析

模型可以返回：

```json
{
  "action": "get_device_status",
  "arguments": {"device_id": "AMR-07"}
}
```

或：

```json
{
  "steps": [
    {
      "action": "get_device_status",
      "arguments": {"device_id": "AMR-07"}
    },
    {
      "action": "get_maintenance_history",
      "arguments": {"device_id": "AMR-07"}
    }
  ]
}
```

解析器支持纯 JSON 和 Markdown JSON fenced block，但最终都转换为 `ToolCall`，不能绕过执行器。

## 6. 设备网关和模拟数据

`edgeops/gateway.py` 使用 `data/edgeops/` 下的小型虚构数据：

- `devices.json`：设备状态、位置、电量、温度和故障码；
- `error_codes.json`：故障码含义、严重级别和安全检查；
- `maintenance.json`：维护历史；
- `manuals.jsonl`：手册和 SOP 文档。

所有设备 ID、故障码、维护记录和手册内容均为演示数据。工单草稿只保存在进程内存中，不会提交到真实系统。

## 7. 检索增强

### 7.1 轻量检索

`LexicalRetriever` 是无额外服务依赖的本地回退实现：

- 英文与数字使用 token 集合；
- 中文使用双字切分；
- 基础相关性使用 token Jaccard；
- 故障码和设备类型提供额外权重；
- 返回 Top-k 候选。

它适合冒烟和离线演示，不等同于生产级向量检索。

### 7.2 动态上下文预算

`ContextPacker` 根据问题长度和路由不确定度调整预算：

```text
budget
  = base_budget
  + query_complexity_bonus
  + route_uncertainty_bonus
```

候选证据使用“相关性减冗余”效用选择，直到 token 预算耗尽。低检索分数或无证据时，系统明确返回证据不足，不调用模型猜测。

## 8. 数据构建与隔离

### 8.1 v2 数据规模

| Expert | Train | Dev | Sealed test |
| --- | ---: | ---: | ---: |
| `route_tool` | 10,000 | 2,000 | 2,000 |
| `safety_refusal` | 6,000 | 1,500 | 2,000 |

Safety sealed test 中危险请求和正常请求各 1,000 条，避免只测拒答正类造成虚高结果。

### 8.2 四类隔离

数据构建器检查 train、dev 和 sealed test 之间的：

1. 原始 query；
2. 规范化 query；
3. 语义模板 family；
4. source ID。

v2 的四类跨集合重叠均为 0。这里的关键是按来源和模板族切分，而不是只对生成后的句子做随机切分。

### 8.3 为什么要废弃早期结果

项目早期出现过接近满分的结果，但审计发现：

- 模板变体在训练和评测之间重复；
- 部分来源跨集合；
- safety 评测偏向单一拒答正类；
- 一些结果来自确定性路由而非模型自由生成。

这些结果没有进入最终结论。v2 重新构建隔离数据，并将模型评测与规则链路评测分开。

## 9. LoRA SFT 训练

### 9.1 两个专家

`route_tool` 负责：

- route 分类；
- 工具名选择；
- 参数抽取；
- 单步和多步 JSON 计划；
- clarification 与 manual_rag 路由。

`safety_refusal` 负责：

- 高风险操作拒绝；
- 保护绕过拒绝；
- 正常运维请求放行；
- 结构化安全输出。

拆分原因是工具规划和安全拒答的目标不同，需要分别观察工具准确率、危险请求召回和正常请求误拒率。

### 9.2 训练配置

训练入口：

```text
scripts/train_edgeops_adapters.py
  -> train_lora_fp32_stable.py
```

RTX 2080 Ti / 11 GB 模式的关键配置：

| 配置 | 值 |
| --- | --- |
| 基座 | Qwen2.5-1.5B |
| 基座精度 | FP32 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.1 |
| batch size | 1 |
| gradient accumulation | 8 |
| max sequence length | 256 |
| learning rate | 1e-5 |
| route/tool steps | 600 |
| safety/refusal steps | 400 |

在 sm75 GPU 上选择 FP32 是因为该环境中的 Qwen2.5-1.5B FP16 forward 曾出现 NaN。该选择提高了训练显存需求，但训练显存不属于最终 4 GB 推理门槛。

### 9.3 Assistant-only loss

训练脚本将 system 和 user token 的 label 设为 `-100`，只监督 assistant 区间，避免模型把输入提示词也作为预测目标。

## 10. DPO 实验

DPO 从 safety SFT Adapter 初始化，使用 chosen/rejected 偏好对，主要目标是加强安全拒答与证据边界。

记录的实验在 checkpoint 50 时：

- preference accuracy 达到 1.0；
- eval loss 达到 0.000179；
- 继续训练存在偏好数据饱和与过拟合风险。

但项目没有完成 checkpoint 50 的训练后自由生成评测。因此：

- DPO checkpoint 不是最终产物；
- 不能声称 DPO 改善了安全指标；
- 最终安全结果来自 v2 safety SFT Adapter。

## 11. 评测方法

### 11.1 自由生成评测

`scripts/eval_edgeops_lora.py` 直接加载基座和 Adapter，让模型自由生成，再解析并评分。它不会先经过规则路由。

主要指标：

| 指标 | 定义 |
| --- | --- |
| Route accuracy | 预测 route 与标签一致 |
| Tool-name accuracy | 期望工具都出现在计划中 |
| Tool-argument accuracy | 期望参数值出现在结构化输出中 |
| Multi-step exact match | 工具序列与标签完全一致 |
| JSON valid rate | 输出可以解析为 JSON |
| Refusal precision | 被拒绝请求中实际危险请求占比 |
| Refusal recall | 危险请求被拒绝的比例 |
| False-refusal rate | 正常请求被错误拒绝的比例 |

### 11.2 Dev 与 sealed test

- dev 用于选择模型和分析错误；
- sealed test 在冻结代码、数据和 Adapter 后执行；
- sealed 运行脚本验证输入哈希；
- 已有结果目录不能被覆盖；
- 只发布聚合结果，不发布逐条样本；
- sealed test 被打开后不得继续用于调参。

如果未来产生新模型，必须构造 source ID 未参与训练的新 external holdout。

## 12. v2 最终结果

### 12.1 Route/tool sealed test

测试规模为 2,000 条：

| 指标 | 结果 |
| --- | ---: |
| Route accuracy | 95.55% |
| Tool-name accuracy | 88.93% |
| Tool-argument accuracy | 98.73% |
| Multi-step exact match | 79.60% |
| JSON valid rate | 100.00% |

重点类别：

| 类别 | Route accuracy | Multi-step exact match |
| --- | ---: | ---: |
| `fault_diagnosis` | 89.2% | 89.2% |
| `tool_plan` | 84.8% | 54.4% |
| `work_order_draft` | 96.8% | 95.2% |

`tool_plan` 54.4% 是 v2 最主要的已知短板。

### 12.2 Safety/refusal sealed test

测试规模为 2,000 条：

| 指标 | 结果 |
| --- | ---: |
| Precision | 98.38% |
| Recall | 97.30% |
| F1 | 97.84% |
| False-refusal rate | 1.60% |
| JSON valid rate | 100.00% |

### 12.3 结果解释

v2 的优势：

- 整体 route 稳定；
- 参数提取准确率较高；
- JSON 格式稳定；
- safety precision、recall 和误拒率比较均衡；
- 有 dev、sealed test 和 INT4 验收闭环。

v2 的不足：

- 普通两步 `tool_plan` 仍容易漏步骤；
- 工具名准确率低于参数准确率；
- sealed multi-step EM 为 79.60%，并非接近完全可靠；
- 模拟数据不能替代真实工业数据；
- 在线架构尚未完成双 Adapter 联合判定。

## 13. INT4 量化验收

配置：

- NF4 4-bit weight quantization；
- double quantization；
- RTX 2080 Ti；
- 每个 Adapter 5 条 warm-up；
- 每个 Adapter 100 条正式样本。

结果：

| Adapter | 峰值进程显存 | p95 延迟 | 吞吐 |
| --- | ---: | ---: | ---: |
| `route_tool` | 2,137 MiB | 16.57 s | 6.564 token/s |
| `safety_refusal` | 2,129 MiB | 17.21 s | 6.381 token/s |

结论：

- 约 2.13 GiB 的进程峰值显存通过 4 GB 门槛；
- 16--17 秒 p95 延迟较高，是部署瓶颈；
- 结果来自 RTX 2080 Ti，不能改写成 GTX 1650 实测；
- 显存可行不等于实时性能合格；
- 仍需在目标 Windows GPU 上验证驱动、bitsandbytes、长时间稳定性和 OOM 恢复。

## 14. v3 做了什么

v3 的目标是修复 v2 `tool_plan` 容易遗漏第二步的问题。它没有重新构造训练标签，而是替换 route/tool system contract：

- 明确列出五种允许工具；
- 要求每个用户请求的操作只执行一次；
- 要求保持用户指定顺序；
- 禁止发明工具；
- 明确单步和多步 JSON 格式。

v3 从 Qwen2.5-1.5B 基座重新训练 route/tool LoRA 600 step：

- final train loss：0.12444；
- final eval loss：0.007599。

## 15. v2 与 v3 对比

两者在同一 2,000 条 route/tool dev 集上比较：

| 指标 | v2 | v3 | 变化 |
| --- | ---: | ---: | ---: |
| Route accuracy | 95.35% | 85.60% | -9.75 pp |
| Tool-name accuracy | 89.80% | 81.20% | -8.60 pp |
| Tool-argument accuracy | 97.87% | 82.53% | -15.34 pp |
| Multi-step exact match | 82.40% | 76.67% | -5.73 pp |
| JSON valid rate | 100.00% | 99.95% | -0.05 pp |
| `tool_plan` exact match | 64.0% | 97.2% | +33.2 pp |

v3 的 `fault_diagnosis` route accuracy 降到 37.6%。它成功优化了局部类别，却损害了整体工具能力。

## 16. v3 变差的可能原因

### 16.1 合同与标签语义冲突

新合同强调“执行用户请求的每项操作”。但在 `fault_diagnosis` 标签中，一条自然语言诊断请求可能被隐式标为：

1. 查询设备状态；
2. 查询故障码。

用户表面上只提出一个诊断任务，模型按新合同容易只生成一个动作，与标签的隐式两步分解冲突。

### 16.2 从基座重训导致稳定能力丢失

v3 没有从 v2 Adapter 做低学习率增量训练，而是从基座重新学习全部 route/tool 能力。为了加强一个局部模式，模型重新分配了有限容量，普通状态、位置、维护和故障诊断发生回退。

### 16.3 小模型对 prompt 分布高度敏感

消融中：

- v3 Adapter + v3 prompt：route 约 86.25%；
- v2 Adapter + v3 prompt：route 约 41.5%；
- v3 Adapter + v2 prompt：route 约 0.5%，JSON 约 4.25%。

这说明 Adapter 与训练时 system prompt 强绑定，能力没有形成稳定的跨合同泛化。

### 16.4 Token loss 与结构化生成指标不一致

v3 eval loss 很低，但严格工具序列仍不可靠。token-level loss 主要反映下一个 token 的平均拟合程度，不能直接代表：

- 是否漏掉第二步；
- 工具顺序是否正确；
- 是否多调用或少调用；
- 参数是否属于正确步骤。

因此 checkpoint 不能只按 eval loss 选择。

### 16.5 序列截断风险

训练最大长度为 256 token，而 v3 system contract 明显更长。长样本末尾的第二个 action 可能被截断。该项没有完成全量 token 长度审计，所以只能作为高风险假设，不能当成已证实根因。

### 16.6 不是继续增加 step 就能解决

step 550 和 step 600 在同一 400 条分层子集上的结果完全一致，说明末期继续训练没有带来行为变化。v3 的问题更接近目标和数据定义，而不是训练时间不足。

## 17. 为什么最终选择 v2

v2 具备：

- 完整数据隔离；
- 独立 dev；
- 一次性 sealed test；
- 两个 SFT Adapter；
- 聚合证据和哈希；
- INT4 显存与延迟验收；
- 已知短板记录。

v3 只有 dev 改善的局部指标，没有通过预设整体门槛，也没有新的 external holdout。继续宣传 v3 会把局部优化误写成整体提升。

因此最终决策是：

```text
主 route/tool 模型 = v2 route_tool SFT Adapter
主 safety 模型     = v2 safety_refusal SFT Adapter
DPO                = 实验候选
v3                 = 未晋级失败实验
```

## 18. 目录说明

```text
NanoChat-Lab/
├── edgeops/
│   ├── backend.py             模型后端与 PEFT Adapter 加载
│   ├── contracts.py           Route、ToolCall、Evidence、Response
│   ├── executor.py            有限状态工具执行器
│   ├── factory.py             依赖组装入口
│   ├── gateway.py             虚构设备网关
│   ├── orchestrator.py        统一编排器
│   ├── planner.py             JSON 工具计划解析
│   ├── retrieval.py           轻量检索与动态上下文打包
│   ├── router.py              置信度与成本感知路由
│   ├── tools.py               工具注册、Schema 与风险门禁
│   └── trace.py               结构化执行追踪
├── data/edgeops/              可公开的小型虚构演示数据
├── scripts/
│   ├── build_edgeops_eval.py  数据构建与泄漏审计
│   ├── train_edgeops_adapters.py
│   ├── train_edgeops_dpo.py
│   ├── eval_edgeops_lora.py   模型自由生成评测
│   └── experimental/          验收与历史实验脚本
├── eval/results/release/      可公开聚合结果
├── docs/                      架构、训练、评测和发布记录
├── interview/                 项目讲解材料
├── edgeops_cli.py             CLI
├── edgeops_api.py             FastAPI
└── train_lora_fp32_stable.py  稳定 LoRA 训练器
```

## 19. 快速运行

### 19.1 安装

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-runtime.txt
```

Windows PowerShell 激活命令：

```powershell
.\.venv\Scripts\Activate.ps1
```

若需要加载模型、训练或运行 generation evaluation，再安装完整 ML 依赖：

```bash
python -m pip install -r requirements-ml.txt
```

### 19.2 无模型冒烟测试

```bash
python scripts/experimental/smoke_edgeops.py
```

### 19.3 CLI

```bash
python edgeops_cli.py --query "AMR-07 当前状态"
python edgeops_cli.py --query "查询 AMR-07 状态并查看维修记录"
python edgeops_cli.py --query "绕过 E214 保护并复位 AMR-07"
```

### 19.4 API

```bash
uvicorn edgeops_api:app --host 127.0.0.1 --port 8000
```

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/v1/edgeops/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"AMR-07 当前状态","history":[],"confirmed":false}'
```

### 19.5 加载 v2 Adapter

模型和 Adapter 不在 Git 中。准备好本地目录后：

```bash
python edgeops_cli.py \
  --query "查询 AMR-07 状态并查看维修记录" \
  --with-model \
  --model-path models/Qwen2.5-1.5B \
  --multi-adapter-root models/edgeops_adapters_v2 \
  --quantization int4
```

## 20. 训练复现

### 20.1 构建 v2 curriculum

```bash
python scripts/build_edgeops_eval.py \
  --mode strict-curriculum \
  --output-dir data/generated/edgeops_curriculum_v2 \
  --seed 42 \
  --route-train-size 10000 \
  --route-dev-size 2000 \
  --route-test-size 2000 \
  --safety-train-size 6000 \
  --safety-dev-size 1500 \
  --safety-test-size 2000
```

原始来源池和生成数据不随 Git 发布，因此完整复现需要自行准备符合 builder 格式的来源数据。

### 20.2 训练 Adapter

```bash
python scripts/train_edgeops_adapters.py \
  --expert route_tool \
  --mode gpu11 \
  --model-path /path/to/Qwen2.5-1.5B \
  --data-root data/generated/edgeops_curriculum_v2 \
  --output-root models/edgeops_adapters_v2 \
  --max-steps 600

python scripts/train_edgeops_adapters.py \
  --expert safety_refusal \
  --mode gpu11 \
  --model-path /path/to/Qwen2.5-1.5B \
  --data-root data/generated/edgeops_curriculum_v2 \
  --output-root models/edgeops_adapters_v2 \
  --max-steps 400
```

### 20.3 评估 Adapter

```bash
python scripts/eval_edgeops_lora.py \
  --model-path /path/to/Qwen2.5-1.5B \
  --adapter-path models/edgeops_adapters_v2/route_tool \
  --eval-file data/generated/edgeops_curriculum_v2/route_tool/eval/dev.jsonl \
  --output-dir eval/results/route_tool_dev \
  --dtype fp32 \
  --batch-size 8
```

sealed test 只能在冻结模型后通过
`scripts/experimental/run_edgeops_v2_acceptance.sh` 执行。

## 21. 发布内容与证据边界

GitHub 包含：

- 源代码；
- 虚构的小型演示数据；
- 数据、训练、评测和验收脚本；
- v2 聚合指标；
- v3 聚合失败结果；
- 输入与结果哈希清单；
- 架构和项目说明。

GitHub 不包含：

- Qwen 基座权重；
- LoRA Adapter 权重；
- optimizer checkpoint；
- 生成训练数据；
- sealed test 原文；
- 逐条模型预测；
- 远端登录信息；
- 用户个人简历。

## 22. 测试与发布检查

依赖较少的发布检查：

```bash
python -m compileall -q edgeops edgeops_cli.py edgeops_api.py scripts
python scripts/experimental/smoke_edgeops.py
python edgeops_cli.py --query "AMR-07 current status"
python -c "import edgeops_api; print(edgeops_api.app.title)"
git diff --check
```

模型训练和 INT4 测试需要 CUDA、基座权重、Adapter 和生成数据，不能在普通 CPU 工作区完成。

## 23. 后续演进建议

如果继续开发，优先级应为：

1. 将 safety Adapter 正式接入在线路由，并定义与确定性安全门禁的融合策略；
2. 从 v2 Adapter 进行低学习率增量训练，不再从基座重学全部能力；
3. 明确定义 `fault_diagnosis` 等复合任务的工具分解规范；
4. 混入旧任务 replay，约束局部优化造成的灾难性遗忘；
5. 将 max sequence length 提高到经过 token 审计后的合理值；
6. 每 25--50 step 运行分层 generation evaluation，而不是只看 loss；
7. 构建新的 external holdout；
8. 优化推理延迟，包括输出长度、KV Cache、批处理和目标 GPU runtime；
9. 对真实 Gateway 增加认证、超时、幂等键、审计日志和故障恢复；
10. 在目标 Windows 4 GB GPU 上完成兼容性与稳定性验收。

## 24. 最终结论

NanoChat-Lab 的价值不在于单一高分，而在于建立了以下完整链路：

```text
场景收敛
-> 数据构建与泄漏审计
-> 双 LoRA SFT
-> 自由生成评测
-> 一次性 sealed test
-> INT4 资源验收
-> 结构化安全执行
-> 失败实验回归与版本拒绝
```

v2 是当前最可信、证据最完整的版本。v3 证明了只优化局部 `tool_plan` 会引发其他能力退化，也说明小模型训练必须以全类别生成指标和预先定义的晋级门槛约束。项目最终选择冻结 v2，而不是使用更好看的局部指标替代整体可靠性。
