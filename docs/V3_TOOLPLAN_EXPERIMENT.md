# V3 Toolplan 优化实验报告

更新日期：2026-08-29

## 1. 结论摘要

V3 是一次针对 v2 普通两步 `tool_plan` 短板的定向实验，不是最终发布
模型。实验通过更明确的 system contract 强化工具集合、调用次数、执行顺序
和多步 JSON 结构。

在同一组 2,000 条 route/tool dev 样本上，v3 将 `tool_plan` exact match
从 64.0% 提升到 97.2%，但整体 route accuracy 从 95.35% 降到 85.60%，
`fault_diagnosis` route accuracy 降到 37.6%。局部收益无法覆盖核心能力
退化，因此 v3 未通过晋级门槛，没有替代 v2，也没有使用旧 sealed test
重复验收。

最终决策：

```text
最终 route/tool 模型：v2 SFT Adapter
v3 route/tool 模型：未晋级实验
```

## 2. 实验背景

V2 已完成 dev、一次性 sealed test 和 INT4 资源验收。它在 sealed
route/tool 集上的主要指标为：

| 指标 | V2 sealed test |
| --- | ---: |
| Route accuracy | 95.55% |
| Tool-name accuracy | 88.93% |
| Tool-argument accuracy | 98.73% |
| Multi-step exact match | 79.60% |
| Valid JSON | 100.00% |

V2 的主要已知短板是普通两步 `tool_plan`。该类别在 sealed test 上的
exact match 为 54.4%，在用于调优比较的 dev 集上为 64.0%。典型错误是
遗漏第二个操作，而不是 JSON 完全不可解析。

V3 的目标仅是验证：更明确的工具规划合同能否在不损害整体 route/tool
能力的前提下，修复这一局部问题。

## 3. V3 修改内容

V3 没有改变 v2 的 route/tool 原始 train/dev 行，也没有引入 sealed test
样本。它重新生成 SFT 对话表示，只替换 route/tool system contract。

新合同包含以下约束：

- 只允许五种 action：
  `get_device_status`、`lookup_error_code`、
  `get_maintenance_history`、`locate_asset` 和
  `create_work_order_draft`；
- 单操作返回 `{"action": "...", "arguments": {...}}`；
- 多操作返回 `{"steps": [...]}`；
- 每个用户请求的操作只出现一次，并保持请求顺序；
- 禁止虚构 action；
- 手册证据问题和缺少必要标识符时，分别路由到 `manual_rag` 或
  `clarification`。

对应实现位于
[`scripts/build_edgeops_eval.py`](../scripts/build_edgeops_eval.py)，构建报告
将该合同标记为：

```text
explicit_allowed_actions_and_multistep_contract_v1
```

V3 从 Qwen2.5-1.5B 基座重新训练 route/tool LoRA，而不是在 v2 Adapter
上继续训练。训练完成 600 steps：

| 训练记录 | 数值 |
| --- | ---: |
| Final train loss | 0.12444 |
| Final eval loss | 0.007599 |

## 4. 数据隔离

V3 构建范围为 `train/dev derived files only`。旧 sealed test 未复制到 v3
输出，也未参与 prompt 修改和 checkpoint 选择。

V3 route/tool SFT 文件 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| Train SFT | `31acac5f77b166cb60b699026643c4da22f3f643e19d7abfc919be9a545adcdc` |
| Dev SFT | `b9d0e6da96fc0c40de4f70167231b6e3e8aba7a000466f7a0b217c38f95609d3` |

构建前后保持不变的源 sealed route/tool SHA-256：

```text
2643e1743841a8f6d9a8125c2695e62ba098bbac296e2cee29366513fdee11f3
```

旧 sealed test 已用于 v2 最终验收。V3 如果通过 dev 晋级门槛，也必须创建
新的 external holdout，不能复用旧 sealed test 得出新的最终结论。

## 5. 同口径 Dev 对比

V2 和 v3 均在同一组 2,000 条 route/tool dev 样本上进行自由生成评测。

| 指标 | V2 | V3 | 变化 |
| --- | ---: | ---: | ---: |
| Route accuracy | 95.35% | 85.60% | -9.75 pp |
| Tool-name accuracy | 89.80% | 81.20% | -8.60 pp |
| Tool-argument accuracy | 97.87% | 82.53% | -15.34 pp |
| Multi-step exact match | 82.40% | 76.67% | -5.73 pp |
| Valid JSON | 100.00% | 99.95% | -0.05 pp |
| `tool_plan` exact match | 64.0% | 97.2% | +33.2 pp |

V3 的收益高度集中在目标类别，整体 route、工具名、参数和多步 exact match
均发生退化。99.95% 的 JSON valid rate 也说明主要问题不是格式解析，而是
动作语义和序列预测。

## 6. V3 分类结果

每个类别包含 250 条 dev 样本：

| 类别 | Route accuracy | Tool-name accuracy | Tool-argument accuracy | Multi-step exact match |
| --- | ---: | ---: | ---: | ---: |
| `asset_location` | 82.0% | 82.0% | 82.4% | n/a |
| `clarification` | 100.0% | n/a | n/a | n/a |
| `device_status` | 86.4% | 86.8% | 86.8% | n/a |
| `fault_diagnosis` | 37.6% | 37.6% | 40.0% | 37.6% |
| `maintenance_history` | 84.8% | 88.4% | 88.8% | n/a |
| `manual_rag` | 99.6% | n/a | n/a | n/a |
| `tool_plan` | 97.2% | 97.2% | 97.2% | 97.2% |
| `work_order_draft` | 97.2% | 95.2% | 100.0% | 95.2% |

最明显的回归是 `fault_diagnosis`。它抵消了 `tool_plan` 的提升，并使
v3 无法作为整体 route/tool Adapter 晋级。

## 7. Prompt 消融

在同一 400 条分层诊断子集上，已有记录显示：

| Adapter 与 prompt 组合 | Route accuracy | 其他观察 |
| --- | ---: | --- |
| V3 Adapter + v3 prompt | 约 86.25% | 与 v3 主评测趋势一致 |
| V2 Adapter + v3 prompt | 约 41.5% | 更换合同后明显退化 |
| V3 Adapter + v2 prompt | 约 0.5% | JSON valid rate 约 4.25% |

这些结果用于定位 prompt 分布依赖，不作为正式 2,000 条 dev 指标。它们
表明两个 Adapter 都与训练时的 system prompt 强绑定，v3 的局部能力没有
形成稳定的跨合同泛化。

## 8. Step 550 与 Step 600

Step 550 和 step 600 在同一 400 条分层子集上的指标相同。继续增加训练
step 没有带来可观察的行为改善，因此没有继续消耗训练资源。

这一结果只说明训练末期已经停滞，不能单独证明具体根因。

## 9. 退化原因分析

### 9.1 已有证据支持的解释

**合同与标签语义存在冲突。** 新合同要求执行用户明确请求的每个操作，
但 `fault_diagnosis` 标签会把一个自然语言诊断请求隐式拆成“查询设备
状态”和“查询故障码”两个步骤。模型按字面合同生成一个动作时，会与
隐式两步标签冲突。

**从基座重新训练造成整体能力重学。** V3 不是从已验收的 v2 Adapter
进行受控增量训练。为了强化一个局部模式，小模型重新学习全部 route/tool
分布，多个普通工具类别随之回退。

**Adapter 与 prompt 分布强耦合。** Prompt 消融中的交叉组合大幅退化，
说明能力依赖训练合同，不能把 prompt 替换视为独立、低风险的配置修改。

**Token loss 不能代表严格计划正确性。** V3 eval loss 很低，但工具序列
指标明显下降。平均 token 拟合不能直接反映漏步骤、顺序错误、重复调用或
参数归属错误。

### 9.2 尚未证实的风险

训练最大序列长度为 256 token，而 v3 system contract 更长。长样本末尾的
第二个 action 可能被截断，但本实验没有完成全量 token 长度审计。因此该项
只能记录为待验证假设，不能作为确定根因。

## 10. 晋级判断

模型版本不能只按目标类别或 eval loss 晋级。V3 的 `tool_plan` 提升没有
满足“整体指标不显著回退”的基本要求：

- route accuracy 下降 9.75 pp；
- tool-name accuracy 下降 8.60 pp；
- tool-argument accuracy 下降 15.34 pp；
- multi-step exact match 下降 5.73 pp；
- 核心 `fault_diagnosis` 类别降到 37.6%。

因此实验在 dev 阶段终止。V3 没有获得新的 external holdout 验收资格，
也没有进行 INT4 发布验收。

## 11. 后续实验建议

如果继续开展 v4，应先修正实验设计，而不是直接延长 v3 训练：

1. 明确 `fault_diagnosis` 是一个用户意图还是两个工具操作，统一合同和
   标签语义。
2. 对所有 SFT 样本做 token 长度与截断位置审计。
3. 从 v2 Adapter 做低学习率、短周期增量实验，并保留基座重训作为对照。
4. 加入旧类别回放或分层采样，避免局部样本改变整体分布。
5. 预先固定 per-category 和 aggregate 晋级阈值。
6. 只有 dev 全部门槛通过后，才构造新的 external holdout。

## 12. 证据与发布边界

Git 仓库只保留 v3 的聚合评测结果：

[`eval/results/release/v3_rejected/edgeops_lora_eval_summary.json`](../eval/results/release/v3_rejected/edgeops_lora_eval_summary.json)

仓库不包含：

- V3 Adapter 或 checkpoint；
- 基座模型权重；
- 生成训练数据；
- 逐条 dev prediction；
- 失败样本明细；
- sealed test 样本。

对外表述必须限定为：

> V3 是一次未晋级的 toolplan 定向优化实验。它显著改善了 dev
> `tool_plan`，但导致整体 route/tool 能力和 `fault_diagnosis` 明显退化，
> 最终版本仍采用通过 sealed test 与 INT4 验收的 v2。

不得将 v3 描述为最终模型，也不得把 v3 的 dev 局部指标与 v2 sealed test
拼接为同一版本的最终结果。
