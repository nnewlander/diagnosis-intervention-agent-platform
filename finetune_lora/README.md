# LoRA 微调复现模块（教学诊断与干预结构化生成）

## 模块定位（重要）

- 当前 **0.5B LoRA** 主要用于**验证微调闭环**：SFT 数据构造 → GPU 训练 → adapter 保存 → 推理 → 评估（`eval_lora.py`）。
- **不建议直接替代主 Agent** 的诊断与干预生成逻辑；本目录与主流程解耦，便于独立实验与指标对比（含基座对比 `--disable-adapter`、后处理评估 `--repair-json`）。
- **当前常见问题**（小模型 SFT 典型现象）：
  - JSON **停止边界**不稳定，输出可能被截断；
  - 合法 JSON 后仍可能继续生成**额外说明性文本**（见 `after_json_extra_text_rate`）；
  - 偶发**禁用领域词**（教学场景不宜的词汇，见 `forbidden_domain_term_rate`）。
- 推荐将 **`finetune_lora/outputs/eval_lora_failed_cases.jsonl`** 中失败样本**回流**到 SFT 数据或提示词约束，再迭代训练。

---

本目录用于在项目一中**独立复现**简历所述「LoRA / PEFT」微调闭环：**不替代** LangGraph Agent，也不接入现有 `workflow`、`rag_adapter` / `kg_adapter`。训练产物可作为后续将「诊断 / 干预」节点改为模型生成时的能力储备。

## 为什么在项目一中引入 LoRA

- **场景适配**：面向「教师自然语言 + 多源证据」的教学诊断与干预表述，与通用聊天分布不同；LoRA 可在较小数据下调适配分布。
- **结构化输出稳定性**：目标输出为严格 JSON（`diagnosis` + `intervention_plan`），监督微调（SFT）比纯提示更利于格式与字段稳定。
- **不替代 RAG/KG/编排**：证据仍由现有系统检索与聚合；LoRA 只增强**生成层**（本复现模块为离线闭环，与主流程解耦）。

## LoRA 与 QLoRA 的区别

|  | LoRA | QLoRA |
|--|------|--------|
| 基座权重 | 常规定精度（如 fp16 / fp32）加载 | 基座以 4bit 量化加载 |
| 训练对象 | 仅低秩 adapter | 通常仍为 LoRA adapter |
| 显存 | 相对较高 | 更低，适合大模型单卡训练 |

## 为什么当前复现选择 LoRA（而非 QLoRA）

- **环境兼容性**：LoRA 流程更简单；Windows 下避免引入 `bitsandbytes` 等依赖的安装不确定性。
- **验证闭环**：小参数量级指令模型即可验证「数据 → 训练 → 推理 → 轻量评估」全链路。
- **与简历表述一致**：项目一侧重 LoRA + Transformers + PEFT 技术栈复现。

## 目录结构

```text
finetune_lora/
  README.md
  configs/lora_config.yaml
  data/train.jsonl          # 由 build_sft_data.py 生成
  data/dev.jsonl
  scripts/build_sft_data.py
  scripts/check_gpu_env.py
  scripts/train_lora.py
  scripts/infer_lora.py
  scripts/eval_lora.py
  scripts/schema_utils.py   # JSON 抽取与 SFT schema 校验
  outputs/lora_adapter      # 训练生成的 adapter
  outputs/eval_lora_report.json
```

## 依赖

在项目根目录安装（已写入根目录 `requirements.txt`）：

- `transformers`、`peft`、`datasets`、`accelerate`
- 训练/推理还需 **`torch`**（按官方指引选择与 CUDA 匹配的版本）

未默认引入 **`bitsandbytes`**（QLoRA 常用），以免 Windows 安装失败。

## 如何运行

在项目仓库根目录执行（示例）：

```bash
# 1）构造 SFT 数据（默认约 350 条，可用 --count 300～500，写入 finetune_lora/data/）
python finetune_lora/scripts/build_sft_data.py

# 2）检查 GPU / CUDA（可选，无显卡时会以非零退出码结束）
python finetune_lora/scripts/check_gpu_env.py

# 3）仅校验数据与 tokenizer（推荐先跑；无 CUDA 时默认自动 dry-run）
python finetune_lora/scripts/train_lora.py --dry-run

# 4）GPU 上训练 LoRA（必须检测到 CUDA；详见 train_lora.py -h）
set BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct
python finetune_lora/scripts/train_lora.py --require-cuda

# 不传 --require-cuda 时：有 CUDA 则训练，无 CUDA 且未加 --force-train-cpu 则只做 dry-run。

# 5）推理（需先有 finetune_lora/outputs/lora_adapter）
python finetune_lora/scripts/infer_lora.py --request-text "学生频繁 NameError，请给诊断与干预 JSON"

# 6）轻量评估 dev 集（需已训练 adapter）
python finetune_lora/scripts/eval_lora.py
```

### 面向约 6～8GB 显存的默认训练参数

可通过命令行覆盖：默认 `per_device_train_batch_size=1`、`gradient_accumulation_steps=8`、`max_length=512`、`num_train_epochs=2`、`learning_rate=2e-4`；CUDA 下默认 `fp16=True`、`gradient_checkpointing=True`。详见 `python finetune_lora/scripts/train_lora.py -h`。

### 显存不足（OOM）

训练脚本会在检测到 CUDA OOM 时打印可行建议（减小 `--max-length`、保持 batch=1、增大 `--gradient-accumulation-steps`、使用 `--max-samples`、更换更小 `BASE_MODEL` 等）。

训练产出目录默认为：`finetune_lora/outputs/lora_adapter`（保存 tokenizer、`adapter_config`、adapter 权重）。

### 快速推理与评估（调试）

**max_new_tokens 选取**：**256** 适合快速调试，但小模型下 JSON **易被截断**（`json_parse_failed` 上升）；**384** 为推荐折中；**512** 适合做 **稳定性评估**，代价是耗时更长（例如在 **GTX 1650 Ti** 等入门级显卡上，512 平均每条生成可能达到数十秒量级）。

```bash
# 推理默认 max_new_tokens=256（偏快）；截断时可升到 384/512
python finetune_lora/scripts/infer_lora.py --request-text "李同学最近在 for循环和条件判断上一直出错，帮我先诊断一下，再给一个3天干预建议。"

python finetune_lora/scripts/infer_lora.py --request-text "..." --max-new-tokens 384

# 评估 dev 前几条（默认 eval max_new_tokens=384）；稳定性对比可用 512
python finetune_lora/scripts/eval_lora.py --max-eval-samples 3

python finetune_lora/scripts/eval_lora.py --max-eval-samples 5 --max-new-tokens 512

# 快速试跑 eval（256，适合冒烟；注意 parse 率可能下降）
python finetune_lora/scripts/eval_lora.py --max-eval-samples 3 --max-new-tokens 256

# 只检查 dev 标注 output 是否符合 schema（不加载模型）
python finetune_lora/scripts/eval_lora.py --skip-generation
```

评估结束后会写出 **`finetune_lora/outputs/eval_lora_case_details.jsonl`**（逐条详情）、**`eval_lora_failed_cases.jsonl`**（失败样本）、**`eval_lora_summary.md`**（Markdown 汇总），并在终端打印失败样本摘要 **`[failed] ...`** 与 **top error reasons** 计数；报告中包含 **`after_json_extra_text_rate`**（JSON 闭合后仍继续生成废话）。可使用 **`--disable-adapter`** 仅跑基座对比、**`--repair-json`** 单独统计修复抽取后的 JSON 可解析率（不覆盖原始 `json_parse_success_rate`）。

SFT 标签已使用 **紧凑单行 JSON**（无缩进换行），并在训练时对 target **追加 eos**，以降低输出长度与截断概率。若生成仍过长或字段漂移：检查 **`build_sft_data.py` 写盘校验** → 增补高质量样本后再训。

## 面试讲解口径（建议）

- 本模块是 **LoRA 监督微调复现**：验证「教学诊断与干预」场景下的**结构化 JSON 生成**能力。
- **不与线上 Agent 争抢职责**：证据检索与图谱仍由 RAG/KG 与子模块完成；LoRA 只管在固定 schema 下生成更稳、更贴合教务语境的文本。
- **落地扩展**：生产可换成更大基座、更长上下文，或在资源受限时用 **QLoRA（4bit + LoRA）** 训练同类 adapter。

## 评估分两层（schema + 语义）

`eval_lora.py` 报告除 JSON 可解析率、字段完整率外，建议同时关注：

**Schema 层（格式与字段）**

- `json_parse_success_rate`：模型输出能否抽取为合法 JSON。
- `required_fields_complete_rate`：顶层是否**仅有** `diagnosis` 与 `intervention_plan`，且 `intervention_plan.intervention_goal` 等非空。
- `confidence_level_valid_rate`：置信度是否在允许枚举内。
- `extra_top_level_field_rate`：是否出现顶层 `interaction_goal` 等多余项。
- `extra_day_fields_rate` / `schema_normalized_count`：是否出现 day4+ 别名或需归一化。

**语义层（教学场景合理性）**

- `forbidden_day_text_rate`：全文是否出现「第4天」「第5天」「第6天」「第4-5天」或 `day4`/`day5`/`day6` 等超出三日干预边界的表述。
- `domain_cause_error_rate`：在 **for 循环 / 条件判断** 知识点下，`probable_cause` 是否误用数据清洗话术或偏离控制流的表述（见 `schema_utils` 常量）。
- `forbidden_domain_term_rate`：是否出现医疗化/促销/职场化禁用词（如「治疗」「返现」「下班前」等）。
- `overconfident_rate`：证据对齐为 `mismatched` / `insufficient_data` 时是否仍给出 `confidence_level=high`。
- `avg_generation_seconds`：平均生成耗时（受 `max_new_tokens`、硬件影响）。

训练数据构建写盘后，`schema_utils` 会对每条 **output 字符串** 做静态校验（顶层键、禁用词、`day_3_action` 不提「第二天」、`optional_followup` 不提「三个月后」等）。

## 数据格式（单条 JSONL）

每条样本包含：

- `case_id`：样本编号（如 `SFT-00042`），校验失败时用于定位。
- `instruction`：任务说明（统一前缀）。
- `input`：JSON 字符串，字段包括 `request_text`、`parsed_slots`、`student_evidence`、`rag_evidence`、`kg_evidence`、`evidence_alignment_status`。
- `output`：JSON 字符串，严格符合 `diagnosis`（含 `observed_problem`/`probable_cause`/…）与 `intervention_plan`（仅 `day_1_action`～`day_3_action` 等），详见 `schema_utils.py`。

## 说明

- `infer_lora.py` 在未检测到 adapter 时会提示：**请先运行 train_lora.py**。
- `eval_lora.py` 在未检测到 adapter 时仍会写出报告文件，并在 JSON 中注明需先训练。
