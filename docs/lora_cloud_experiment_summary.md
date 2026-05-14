\# LoRA 云端微调实验总结



\## 1. 实验背景



本实验用于验证项目一「核桃智能教学诊断与干预 Agent 平台」中的 LoRA 微调能力。  

在本地完成 Qwen2.5-0.5B-Instruct LoRA 闭环后，进一步租用 AutoDL A800-80GB 云服务器，尝试更大规模的 Qwen2.5-7B-Instruct 与 Qwen2.5-14B-Instruct LoRA 微调。



本实验不直接替代主 Agent workflow，仅作为诊断/干预 JSON 生成节点的后续优化验证。



\## 2. 实验环境



\- 平台：AutoDL

\- GPU：NVIDIA A800 80GB PCIe

\- 数据盘：200GB

\- 训练框架：PyTorch + Transformers + PEFT

\- 微调方式：LoRA

\- 训练数据：finetune\_lora/data/train.jsonl

\- 训练样本数：298



\## 3. 7B LoRA 实验



\### 基座模型



\- Qwen2.5-7B-Instruct



\### 训练配置



\- max\_length：512

\- per\_device\_train\_batch\_size：1

\- gradient\_accumulation\_steps：8

\- num\_train\_epochs：2

\- learning\_rate：2e-4

\- fp16：True

\- gradient\_checkpointing：True



\### 训练结果



\- LoRA 可训练参数：5,046,272

\- 总参数：7,620,662,784

\- 可训练比例：0.0662%

\- 2 epoch 训练耗时：约 130 秒

\- train\_loss：约 0.8972



\### 观察结论



7B adapter 能生成 diagnosis + intervention\_plan 结构化内容，语义比 0.5B 更稳定；但仍存在 JSON 后额外文本、重复 JSON 等停止边界问题，因此不建议直接接入主流程。



\## 4. 14B LoRA 实验



\### 基座模型



\- Qwen2.5-14B-Instruct



\### 训练配置



\- max\_length：512

\- per\_device\_train\_batch\_size：1

\- gradient\_accumulation\_steps：16

\- num\_train\_epochs：2

\- learning\_rate：2e-4

\- fp16：True

\- gradient\_checkpointing：True



\### 训练结果



\- LoRA 可训练参数：12,582,912

\- 总参数：14,782,616,576

\- 可训练比例：0.0851%

\- 训练步数：38

\- 2 epoch 训练耗时：约 227 秒

\- train\_loss：约 1.454

\- adapter\_model.safetensors：约 49MB



\## 5. 14B 推理观察



测试输入：



> 李同学最近在 for循环和条件判断上一直出错，帮我先诊断一下，再给一个3天干预建议。



14B adapter 能生成较干净的 JSON，包含：



\- diagnosis.observed\_problem

\- diagnosis.probable\_cause

\- diagnosis.evidence\_basis

\- diagnosis.confidence\_level

\- intervention\_plan.intervention\_goal

\- day\_1\_action

\- day\_2\_action

\- day\_3\_action

\- optional\_followup



其中，模型能够识别 for循环与条件判断属于控制流理解问题，并在证据不足时给出 low confidence。



\## 6. 评估结果摘要



14B 在 5 条 dev 样本上的评估观察：



\- json\_parse\_success\_rate：0.6

\- confidence\_level\_valid\_rate：0.6

\- user\_points\_covered\_rate：1.0

\- evidence\_terms\_mentioned\_rate：0.6

\- forbidden\_day\_text\_rate：0.0

\- domain\_cause\_error\_rate：0.0

\- overconfident\_rate：0.0

\- forbidden\_domain\_term\_rate：0.0

\- after\_json\_extra\_text\_rate：0.4



主要失败原因：



\- optional\_followup 为空

\- 部分样本仍存在 JSON 后额外文本

\- 少量样本 json\_parse\_failed



\## 7. 当前结论



本次云端实验成功完成了从本地 0.5B 到云端 7B / 14B 的 LoRA 微调扩展。



结论：



1\. 14B 比 7B 在单例推理中的输出更干净，业务语义更稳定。

2\. LoRA 能够让模型学习教学诊断与三天干预计划的 JSON 输出格式。

3\. 当前主要问题不再是模型不会写，而是 JSON 停止边界、optional\_followup 空值和评估规则。

4\. 当前 adapter 不建议直接接入主 Agent 主流程，适合作为 LoRA 复现实验模块和后续优化方向。



\## 8. 后续优化方向



1\. 推理时使用 constrained decoding 或 JSON schema decoding。

2\. 在 infer/eval 中默认提取第一个合法 JSON。

3\. 修改训练数据，避免 optional\_followup 为空。

4\. 降低 learning\_rate 到 1e-4 做对比实验。

5\. 对比 max\_new\_tokens=256/384/512 对 JSON 后额外文本的影响。

6\. 保持主 Agent workflow 使用稳定规则/RAG/KG 逻辑，LoRA 作为实验增强模块。

