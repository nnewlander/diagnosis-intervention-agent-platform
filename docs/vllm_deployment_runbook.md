# vLLM 部署与验证速查（项目一）

本文档与主 Agent / RAG / KG 流程解耦，仅说明如何本地拉起 **OpenAI 兼容** 的 vLLM 服务，并用仓库内脚本 `scripts/smoke_test_vllm.py` 做联调验证。

## 前置条件

- 已安装 vLLM（版本以你环境为准，命令行入口一般为 `vllm` 或 `python -m vllm.entrypoints.openai.api_server`）。
- GPU 显存需与所选权重匹配；下表为常见参考，请按机器调整 `--max-model-len`、`--gpu-memory-utilization` 等。

## 7B 基座启动示例

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --served-model-name qwen2.5-7b-instruct \
  --host 0.0.0.0 \
  --port 8008
```

服务根路径为 `http://127.0.0.1:8008/v1`，与 smoke 脚本默认 `--base-url` 一致。

## 14B 基座启动示例

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct \
  --served-model-name qwen2.5-14b-instruct \
  --host 0.0.0.0 \
  --port 8008
```

smoke 脚本默认 `--base-model qwen2.5-14b-instruct` 需与 `--served-model-name`（或 vLLM 注册的模型 id）一致。

## 14B + LoRA 启动示例

将 LoRA 权重目录替换为本地实际路径；`teaching_lora` 为对外 API 中的模型名（与 smoke 脚本 `--lora-model` 默认值一致）。

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct \
  --served-model-name qwen2.5-14b-instruct \
  --enable-lora \
  --lora-modules teaching_lora=/path/to/your/lora_adapter \
  --host 0.0.0.0 \
  --port 8008
```

不同 vLLM 版本参数名可能略有差异，请以 `vllm serve --help` 为准。

## curl 快速探测

列出模型：

```bash
curl -sS "http://127.0.0.1:8008/v1/models" | head
```

最小 chat 请求（基座名按你 `served-model-name` 修改）：

```bash
curl -sS "http://127.0.0.1:8008/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"qwen2.5-14b-instruct\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":32}"
```

LoRA 作为独立模型 id 时：

```bash
curl -sS "http://127.0.0.1:8008/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"teaching_lora\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":32}"
```

## Python smoke 脚本

```bash
python scripts/smoke_test_vllm.py
python scripts/smoke_test_vllm.py --base-url http://127.0.0.1:8008/v1 --base-model qwen2.5-14b-instruct --lora-model teaching_lora --timeout 120
```

标准输出为一整份 JSON 报告（含 `/models` 与两次 `chat/completions` 的轻量字段检查）。服务不可连时会在标准错误输出中文提示，并避免打印长堆栈。

## 当前观察（业务侧）

- **LoRA** 在相同 prompt 下往往更贴近本项目的教学诊断 / 干预表述习惯（与训练数据分布相关）。
- **严格 JSON schema**（例如固定键名、嵌套层级）仍依赖：**系统提示 / 用户约束**（如「仅输出 JSON」+ 示例结构）以及 **后处理**（截断 markdown 围栏、解析失败时的降级与修复）。smoke 脚本中的 `json_parse_ok`、`has_diagnosis`、`has_intervention_plan` 仅为启发式检查，不代表线上 schema 校验已通过。
