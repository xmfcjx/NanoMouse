# 第三方安装与配置指南

本文面向第一次接触 NanoChat-Lab 的使用者，说明如何从一个干净的
Git clone 启动 EdgeOps Copilot。

## 1. 支持模式

| 模式 | 是否需要 GPU | 是否需要模型 | 安装文件 | 用途 |
| --- | --- | --- | --- | --- |
| 无模型 CLI | 否 | 否 | 无额外依赖 | 验证路由、工具、安全门禁和 RAG 回退 |
| FastAPI 服务 | 否 | 否 | `requirements-runtime.txt` | 本地 API 集成 |
| 模型推理 | 建议 NVIDIA GPU | 是 | `requirements-ml.txt` | 加载 Qwen 与 v2 Adapter |
| 训练/评测 | NVIDIA GPU | 是 | `requirements-ml.txt` | 重建数据后的 LoRA 训练和 generation eval |

最稳妥的首次验证方式是先运行无模型 CLI 和冒烟测试，再配置模型。

## 2. 系统要求

基础运行：

- Python 3.10 或 3.11；
- Git；
- macOS、Linux 或 Windows；
- 约 200 MB 可用空间，不包含虚拟环境。

模型模式：

- NVIDIA GPU；
- 与 PyTorch 兼容的驱动和 CUDA runtime；
- 可用的 Qwen2.5-1.5B 本地目录；
- v2 PEFT Adapter 本地目录；
- 足够的磁盘空间和主机内存。

仓库中的 INT4 结果来自 RTX 2080 Ti。其他 GPU 必须独立验证
PyTorch、CUDA 和 bitsandbytes 兼容性。

## 3. 获取代码

```bash
git clone git@github.com:xmfcjx/NanoMouse.git
cd NanoMouse
```

也可以使用 HTTPS：

```bash
git clone https://github.com/xmfcjx/NanoMouse.git
cd NanoMouse
```

## 4. 创建虚拟环境

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

后续命令都应在仓库根目录和已激活的虚拟环境中执行。

## 5. 无模型 CLI

无模型 CLI 只使用 Python 标准库，不需要安装 PyTorch。

```bash
python edgeops_cli.py --query "AMR-07 当前状态"
```

期望结果：

- route 为 `device_status`；
- `tool_results` 包含 `get_device_status`；
- 返回状态、电量、温度、位置和故障码；
- `trace_id` 非空。

验证高风险拒绝：

```bash
python edgeops_cli.py --query "绕过保护并让 AMR-07 继续移动"
```

期望 route 为 `safety_reject`，且不会执行设备动作。

验证多步工具：

```bash
python edgeops_cli.py --query "查询 AMR-07 状态并查看维修记录"
```

无模型模式使用确定性回退计划，可验证工具执行链路，但它不是 v2
Adapter 的模型指标。

## 6. 冒烟测试

```bash
python scripts/experimental/smoke_edgeops.py
```

成功输出：

```text
edgeops smoke checks passed
```

测试覆盖：

- 设备状态；
- 故障诊断；
- 资产位置；
- 维护记录；
- 多步计划；
- 危险请求拒绝；
- 手册检索；
- forbidden 工具；
- 重复调用检测；
- JSON fenced block 解析。

## 7. FastAPI 服务

安装轻量依赖：

```bash
python -m pip install -r requirements-runtime.txt
```

启动：

```bash
uvicorn edgeops_api:app --host 127.0.0.1 --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

请求示例：

```bash
curl -X POST http://127.0.0.1:8000/v1/edgeops/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"AMR-07 当前状态","history":[],"confirmed":false}'
```

默认只监听本机。不要在没有认证、访问控制、限流和 Trace 数据治理的
情况下绑定公网地址。

## 8. 本地演示数据

默认数据目录：

```text
data/edgeops/
├── devices.json
├── error_codes.json
├── maintenance.json
└── manuals.jsonl
```

这些数据均为虚构演示数据，可以直接用于无模型 CLI 和 API。

使用其他数据目录：

```bash
python edgeops_cli.py \
  --data-dir /path/to/edgeops-data \
  --query "AMR-07 当前状态"
```

自定义目录必须保持相同文件名和 JSON/JSONL 结构。正式接入真实数据前，
需要自行增加认证、权限、脱敏、超时和审计策略。

## 9. 模型资产

模型权重和 Adapter 不在 GitHub 中。推荐目录：

```text
models/
├── Qwen2.5-1.5B/
└── edgeops_adapters_v2/
    ├── route_tool/
    │   ├── adapter_config.json
    │   └── adapter_model.safetensors
    └── safety_refusal/
        ├── adapter_config.json
        └── adapter_model.safetensors
```

至少需要 `adapter_config.json` 和对应 Adapter 权重。请从合法来源准备
Qwen 基座，并遵循基座模型许可证。

## 10. 安装模型依赖

```bash
python -m pip install -r requirements-ml.txt
```

安装后检查：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import transformers, peft; print(transformers.__version__, peft.__version__)"
python -c "import bitsandbytes as bnb; print(bnb.__version__)"
```

若 `torch.cuda.is_available()` 为 `False`，先修复驱动、CUDA 和 PyTorch
版本，不要直接排查项目代码。

## 11. 启动模型模式

```bash
python edgeops_cli.py \
  --query "查询 AMR-07 状态并查看维修记录" \
  --with-model \
  --model-path models/Qwen2.5-1.5B \
  --multi-adapter-root models/edgeops_adapters_v2 \
  --quantization int4
```

也可以通过环境变量启动 API：

### macOS / Linux

```bash
export EDGEOPS_WITH_MODEL=1
export MODEL_PATH="$PWD/models/Qwen2.5-1.5B"
export EDGEOPS_ADAPTER_ROOT="$PWD/models/edgeops_adapters_v2"
export QUANTIZATION=int4
uvicorn edgeops_api:app --host 127.0.0.1 --port 8000
```

### Windows PowerShell

```powershell
$env:EDGEOPS_WITH_MODEL = "1"
$env:MODEL_PATH = "$PWD\models\Qwen2.5-1.5B"
$env:EDGEOPS_ADAPTER_ROOT = "$PWD\models\edgeops_adapters_v2"
$env:QUANTIZATION = "int4"
uvicorn edgeops_api:app --host 127.0.0.1 --port 8000
```

可选环境变量：

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `EDGEOPS_WITH_MODEL` | `0` | 是否加载模型 |
| `MODEL_PATH` | `models/Qwen2.5-1.5B` | 基座目录 |
| `EDGEOPS_ADAPTER_ROOT` | 无 | v2 Adapter 根目录 |
| `QUANTIZATION` | `int4` | `int4`、`int8` 或 `fp16` |
| `EDGEOPS_DATA_DIR` | `data/edgeops` | 演示或自定义数据 |
| `EDGEOPS_TRACE_PATH` | `eval/results/edgeops/traces.jsonl` | Trace 输出 |

## 12. 训练与评测

训练和完整模型评测还需要：

- 自行准备的数据来源池；
- 生成 `data/generated/edgeops_curriculum_v2/`；
- 本地 Qwen2.5-1.5B；
- CUDA 训练环境。

训练入口：

```bash
python scripts/train_edgeops_adapters.py --help
python scripts/train_edgeops_dpo.py --help
```

自由生成评测入口：

```bash
python scripts/eval_edgeops_lora.py --help
```

完整命令见：

- `docs/EDGEOPS_REAL_TRAINING_RUNBOOK.md`；
- `docs/V2_TRAINING_STATUS.md`；
- `docs/PROJECT_GUIDE_ZH.md`。

## 13. 常见问题

### `ModuleNotFoundError`

确认当前目录是仓库根目录，并已激活正确虚拟环境：

```bash
pwd
python -c "import sys; print(sys.executable)"
```

### `No PEFT adapters found`

检查：

```text
models/edgeops_adapters_v2/route_tool/adapter_config.json
models/edgeops_adapters_v2/safety_refusal/adapter_config.json
```

### bitsandbytes 或 CUDA 加载失败

确认 PyTorch 能识别 GPU，再核对 bitsandbytes 对当前系统和 CUDA 的支持。
macOS 不支持 CUDA；macOS 上建议先运行无模型模式。

### 显存不足

- 使用 `--quantization int4`；
- 关闭其他 GPU 进程；
- 缩短输入；
- 降低生成 token 数；
- 确认没有重复加载多个基座模型。

### 手册问题提示证据不足

确认 `data/edgeops/manuals.jsonl` 存在，或通过 `--data-dir` /
`EDGEOPS_DATA_DIR` 指向正确目录。

### API 能运行，但模型未启用

`/health` 中 `model_enabled` 为 `false` 表示当前是无模型模式。设置
`EDGEOPS_WITH_MODEL=1` 并配置模型及 Adapter 路径后重启。

## 14. 第三方验收清单

首次安装完成后按顺序检查：

```text
[ ] Python 版本为 3.10 或 3.11
[ ] 仓库根目录包含 data/edgeops
[ ] 无模型 CLI 返回 device_status
[ ] 危险请求返回 safety_reject
[ ] smoke_edgeops.py 通过
[ ] FastAPI /health 返回 status=ok
[ ] 模型目录来源合法并完成哈希校验
[ ] torch.cuda.is_available() 为 True
[ ] route_tool 与 safety_refusal Adapter 可发现
[ ] INT4 模型可以生成
[ ] 记录目标机器的峰值显存和 p95 延迟
```

完成前五项即可确认公开仓库的轻量工程链路可用。后五项依赖不随 Git
发布的本地模型资产和目标 GPU 环境。
