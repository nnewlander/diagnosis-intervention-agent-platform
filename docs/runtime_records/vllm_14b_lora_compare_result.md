# vLLM 14B Base vs LoRA 对比验证记录

## 1. 验证目标

本次验证目标是对比 Qwen2.5-14B-Instruct base model 与挂载 LoRA adapter 后的 teaching_lora 在项目一教学诊断与干预场景下的输出差异。

## 2. 服务信息

- Base model：qwen2.5-14b-instruct
- LoRA model：teaching_lora
- API 地址：http://127.0.0.1:8008/v1
- 部署方式：vLLM OpenAI-compatible API Server
- LoRA adapter 路径：finetune_lora/outputs/runs/14b_lora_e2_512/lora_adapter

## 3. /v1/models 验证

/v1/models 返回了两个模型：

- qwen2.5-14b-instruct
- teaching_lora

其中 teaching_lora 的 parent 为 qwen2.5-14b-instruct，说明 LoRA adapter 已成功挂载到 14B base model 上。

## 4. 宽松 JSON prompt 对比

### 请求

李同学最近在 for循环和条件判断上一直出错，帮我先诊断一下，再给一个3天干预建议。请只输出 JSON。

### teaching_lora 观察

teaching_lora 返回合法 JSON，包含：

- 诊断
- 第1天
- 第2天
- 第3天

优点是能够直接围绕教学诊断和三天干预计划输出，且无 JSON 后额外文本。

不足是使用中文键名，没有严格遵循训练期设计的 diagnosis / intervention_plan schema。

### base model 观察

qwen2.5-14b-instruct 也能返回 JSON，包含 for循环问题、条件判断问题和三天干预建议。

base model 的内容更像通用编程学习建议，包含在线教程、编程书籍、计算偶数和、判断质数等泛化练习；业务风格不如 teaching_lora 贴近项目一。

## 5. 严格 schema prompt 测试

当 prompt 明确要求：

- 顶层字段必须为 diagnosis 和 intervention_plan
- diagnosis 必须包含 observed_problem、probable_cause、evidence_basis、confidence_level
- intervention_plan 必须包含 intervention_goal、day_1_action、day_2_action、day_3_action、optional_followup

teaching_lora 能输出接近目标 schema 的 JSON，并包含所需字段。

主要不足：

- confidence_level 输出为中文“中等”，如果校验器只接受 low / medium / high，需要增加标准化或在 prompt 中进一步限制枚举值。

## 6. NameError 测试观察

在 NameError 场景下，teaching_lora 能输出 JSON，并给出课堂解释和三天巩固建议。

但如果 prompt 只要求包含 diagnosis 和 intervention_plan，而没有完整字段约束，模型会将 diagnosis 输出为字符串，并将 day_1_action 简化为 day_1/day_2/day_3。

说明当前 adapter 具备业务方向能力，但严格 schema 仍依赖 prompt 约束和后处理。

## 7. 总结结论

本轮验证表明：

1. 14B + LoRA adapter 已成功通过 vLLM 服务化调用；
2. teaching_lora 能生成教学诊断与三天干预计划风格的 JSON；
3. 相比 base model，teaching_lora 更贴近项目一业务任务；
4. base model 本身也具备较强 JSON 输出能力，因此 LoRA 不是从 0 到 1，而是增强业务风格倾向；
5. 当前 adapter 仍不建议直接替代主 Agent 默认流程；
6. 如需接入主流程，需要增加严格 prompt、schema validation、字段标准化和 JSON 后处理。

## 8. 后续建议

1. 新增 smoke_test_vllm.py，自动测试 base 与 LoRA 的 /v1/models 和 /v1/chat/completions；
2. 增加 schema validation，检查 diagnosis / intervention_plan 字段完整性；
3. 对 confidence_level 做枚举标准化，如“中等” -> medium；
4. LoRA-vLLM 作为可选实验 provider，不默认替换当前稳定主流程；
5. 继续保留 RAG/KG/规则生成作为项目一默认演示链路。
