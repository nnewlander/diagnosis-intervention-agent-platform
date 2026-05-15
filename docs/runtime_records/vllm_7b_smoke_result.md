# vLLM 7B 服务化验证记录

## 1. 验证目标

本次验证目标是将 Qwen2.5-7B-Instruct 通过 vLLM 启动为 OpenAI-compatible API 服务，并确认项目一后续可以像调用 OpenAI API 一样调用自部署模型。

## 2. 服务信息

- 模型：Qwen2.5-7B-Instruct
- vLLM served model name：qwen2.5-7b-instruct
- API 地址：http://127.0.0.1:8008/v1
- 端口：8008
- max_model_len：2048
- 部署方式：vLLM OpenAI-compatible API Server
- 模型本地路径：/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct

## 3. 启动命令

```bash
export MODEL_7B=/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct

vllm serve $MODEL_7B \
  --host 0.0.0.0 \
  --port 8008 \
  --served-model-name qwen2.5-7b-instruct \
  --dtype auto \
  --gpu-memory-utilization 0.85 \
  --max-model-len 2048
4. /v1/models 测试
请求命令
curl http://127.0.0.1:8008/v1/models
返回结果摘要

服务返回模型列表，模型已成功注册：

model id：qwen2.5-7b-instruct
object：model
owned_by：vllm
root：/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct
max_model_len：2048
说明

第一次请求曾出现 Connection refused，原因是 vLLM 服务当时尚未完全启动。服务启动完成后，再次请求 /v1/models 成功返回模型列表。

5. /v1/chat/completions 基础测试
请求命令
curl http://127.0.0.1:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b-instruct",
    "messages": [
      {"role": "user", "content": "请用一句话解释什么是 NameError。"}
    ],
    "temperature": 0,
    "max_tokens": 256
  }'
返回结果摘要

模型成功返回：

NameError 是 Python 中的一种异常，表示尝试访问一个未声明或未赋值的变量。

返回中包含：

object：chat.completion
model：qwen2.5-7b-instruct
finish_reason：stop
prompt_tokens：37
completion_tokens：22
total_tokens：59

说明 /v1/chat/completions 接口可以正常生成回答。

6. 项目一相关 prompt 测试
请求内容
你是一个少儿编程教学助手。学生问 NameError 是什么意思，请给老师一个适合课堂解释的回答，包含原因、排查步骤和一个简单类比。
返回结果摘要

模型成功生成了适合课堂解释的回答，包含：

NameError 的名词解释
变量未定义、拼写错误、作用域问题等原因
检查变量名、定义位置、拼写和调试工具等排查步骤
用数学题/角色名字等方式进行类比解释

说明该 vLLM 7B 服务不仅能处理基础问答，也能生成项目一技术答疑场景下的教学解释内容。

7. 本次验证结论

Qwen2.5-7B-Instruct 已成功通过 vLLM 部署为 OpenAI-compatible API 服务。

本次验证已经完成：

vLLM 服务启动；
/v1/models 模型列表查询成功；
/v1/chat/completions 聊天接口调用成功；
模型能够正常返回 NameError 解释；
项目一相关教学 prompt 能够正常生成结构化教学解释；
项目一后续可以通过 HTTP adapter / OpenAI-compatible client 调用该 vLLM 服务。
8. 后续计划

本次验证的是 7B base model，不包含 LoRA adapter。

后续可以继续验证：

Qwen2.5-14B-Instruct base model 的 vLLM 部署；
Qwen2.5-14B-Instruct + LoRA adapter 的 vLLM 部署；
项目一新增 scripts/smoke_test_vllm.py；
项目一通过配置项接入 VLLM_API_BASE 与 VLLM_MODEL；
在前端或 debug_trace 中展示 LLM provider 为 vLLM。
