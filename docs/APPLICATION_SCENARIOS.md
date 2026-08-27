# NanoChat 应用场景：EdgeOps Copilot

## 1. 产品定位

NanoChat 定位为：

> **EdgeOps Copilot：运行在受限边缘计算设备上的本地运维与操作副驾驶。**

它不是直接控制车辆或机器人的安全控制器，也不是通用聊天机器人。它位于操作人员与设备软件之间，负责：

- 理解自然语言请求；
- 查询本地手册、SOP、故障码和维护记录；
- 读取经过授权的设备状态；
- 调用白名单诊断或配置工具；
- 将结果转换为可解释的操作建议；
- 在低置信度或高风险时拒绝执行并转交人工。

## 2. 为什么适合边缘部署

工厂、仓库、车辆和现场服务环境具有共同约束：

| 约束 | 对系统的要求 |
| --- | --- |
| 网络不稳定 | 核心问答和诊断必须本地可用 |
| 数据敏感 | 设备日志、生产数据和车内信息不应默认上传 |
| 计算资源有限 | 使用 1.5B SLM、INT4 和按需 adapter |
| 响应要求高 | 确定性请求走 direct tool，不等待长推理 |
| 错误代价高 | 白名单工具、Schema 校验、置信度门控和人工确认 |
| 文档复杂 | 本地 RAG 查询手册、SOP、维修记录 |
| 任务持续变化 | LoRA 专家按设备或任务低成本更新 |

这与 NanoChat 已有的量化、RAG、工具路由、记忆和 LoRA 能力天然匹配。

## 3. 主场景：工厂 AMR / 设备维护助手

### 3.1 使用环境

- 部署位置：工控机、机器人随车计算单元或现场边缘盒子；
- 使用者：产线操作员、维修工程师、仓库调度员；
- 数据：设备手册、SOP、故障码、维修历史、只读遥测；
- 网络：厂内局域网或离线；
- 模型：Qwen2.5-1.5B INT4 + 任务 LoRA。

### 3.2 典型请求

```text
“AMR-07 为什么停止运行？”
“错误码 E214 的含义是什么？”
“根据当前电池温度和任务队列，是否建议继续执行？”
“上次更换左轮编码器是什么时候？”
“生成一份交接班故障摘要。”
“叉车 F12 当前位于哪个区域？”
```

### 3.3 执行路径

```text
用户问题
  -> 设备/意图识别
  -> 读取只读遥测
  -> 检索手册和历史记录
  -> 生成诊断解释
  -> 输出建议 / 创建待确认工单
```

### 3.4 可调用工具

| 工具 | 类型 | 风险 |
| --- | --- | --- |
| `get_device_status` | 读取状态 | 低 |
| `lookup_error_code` | 查询故障码 | 低 |
| `search_manual` | 检索手册 | 低 |
| `get_maintenance_history` | 查询记录 | 低 |
| `locate_asset` | 查询位置 | 低 |
| `create_work_order_draft` | 创建草稿 | 中，需要确认 |
| `change_device_setting` | 修改配置 | 高，默认禁止 |
| `move_robot` | 运动控制 | 安全关键，禁止交给 LLM 直接执行 |

### 3.5 项目演示方式

不需要真实机器人。可以建立一个模拟设备网关：

```text
AMR-07:
  battery: 18%
  motor_temperature: 76 C
  error_code: E214
  location: warehouse-B3
  task: deliver-bin-42
```

工具读取模拟遥测，RAG 查询本地手册，NanoChat 输出诊断和证据。这样能够展示完整算法链路，同时避免虚构已经接入工业硬件。

## 4. 第二场景：车载本地助手

车载场景复用相同核心架构，但更换数据源和工具。

### 4.1 典型请求

```text
“这个电池形状的故障灯是什么意思？”
“Auto Hold 怎么使用？”
“最近一次保养是什么时候？”
“把后排温度调低两度。”
“前方没网络，继续使用离线车辆手册。”
```

### 4.2 模块映射

| EdgeOps 核心 | 车载实现 |
| --- | --- |
| 设备手册 RAG | 车辆用户手册、维修文档 |
| 设备遥测 | 电量、胎压、故障码、保养里程 |
| 工具调用 | 空调、座椅、车窗、导航接口 |
| 用户记忆 | 驾驶员偏好、温度、常用路线 |
| 风险门控 | 行驶状态限制、驾驶员确认 |

公开车载方案已经采用本地小模型完成车辆配置、手册问答和弱网备份能力，因此该扩展具有现实依据，但 NanoChat 当前不会声称达到车规级认证。

## 5. 其他相近场景

### 5.1 仓储资产查询助手

操作员询问叉车、托盘、容器和工具位置。边缘视觉或定位系统提供结构化状态，NanoChat 负责自然语言查询和解释。

### 5.2 现场维修助手

部署在工程师笔记本或便携边缘盒子中，在矿区、风电场、船舶、通信基站等弱网区域查询手册、故障码和维护流程。

### 5.3 设备交接班助手

读取一段时间内的告警、维修记录和操作日志，生成班次摘要、风险项和待办事项。该任务适合本地批处理，不要求实时控制。

### 5.4 工业培训助手

根据岗位和设备型号提供 SOP 问答、故障演练和操作测验。它与生产控制隔离，适合作为低风险首个落地场景。

## 6. NanoChat 现有模块如何迁移

| 当前模块 | 需要的场景化改造 |
| --- | --- |
| `InputClassifier` | 从通用意图改为设备诊断、手册、状态、工单、拒绝 |
| `ReActAgent` | 将天气、字符串工具替换为设备白名单工具 |
| `Retriever` | 索引设备型号、手册章节、故障码和维修记录 |
| `MemoryStore` | 区分用户偏好、设备状态和维修事实，增加 TTL |
| LoRA SFT | 训练结构化诊断、工具选择、证据回答 |
| DPO 数据 | 依据工具执行正确性、证据一致性和安全策略生成 |
| INT4/GGUF | 保留，作为边缘部署基础 |
| FastAPI | 作为设备网关与 UI 之间的接口 |

## 7. 安全边界

面试中必须主动说明：

- LLM 不进入电机、刹车、转向、急停等实时安全控制回路；
- 所有工具均为显式白名单；
- 默认只读，写操作需要规则校验和人工确认；
- 高风险请求直接拒绝或升级到传统控制系统；
- 输出必须附带证据或状态来源；
- 模型不可用时，设备原有控制功能不受影响；
- 本项目是算法与系统原型，不声称满足 ISO 26262、IEC 61508 或其他功能安全认证。

这不是削弱项目，而是体现对 Physical AI 系统边界的理解。

## 8. 最终建议

项目主叙事选择：

> 在 4GB 级边缘计算约束下，为工厂 AMR 和设备维护场景设计本地运维副驾驶。系统使用置信度路由选择直接工具、RAG 或小模型推理，使用任务 LoRA 专家减少多任务干扰，并通过结构化工具状态机确保可执行性。车载手册与诊断作为同架构的迁移案例。

这样既保留工厂机器人和车载环境的真实动机，又避免一个项目同时实现两个完全不同产品。

## 9. 参考案例

- NVIDIA，车载 Agent 的实时边缘架构、隐私与遥测融合：https://developer.nvidia.com/blog/how-to-build-in-vehicle-ai-agents-with-nvidia-from-cloud-to-car
- SoundHound + NVIDIA，本地车辆手册、故障灯与维护问答：https://www.soundhound.com/newsroom/press-releases/soundhound-to-offer-on-chip-voice-ai-with-nvidia-that-delivers-in-vehicle-generative-ai-responses-with-no-connectivity-required
- FEV + Microsoft + NVIDIA，车端小模型与语音配置：https://www.fev.com/en/fev-collaborates-with-microsoft-on-efficient-ai-model-approach-for-in-car-applications-built-on-nvidia/
- AWS Physical AI，仓储资产定位的边缘到云架构：https://aws.amazon.com/blogs/physical-ai/edge-impulse-and-aws-combining-edge-inference-with-cloud-intelligence-for-physical-ai/
- NVIDIA Jetson，机器人和工业边缘的内存效率：https://developer.nvidia.com/blog/maximizing-memory-efficiency-to-run-bigger-models-on-nvidia-jetson
