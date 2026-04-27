# 输入数据格式设计（项目一 Agent）

## 1. 设计目标
项目一不是单一知识库问答，而是一个多节点、多工具、多状态的教学闭环 Agent。
因此输入层不能只设计成一张问答表，而应拆成五类输入：

1. 教师自然语言请求
2. 学生结构化学情数据
3. 历史干预案例
4. 任务执行与练习下发数据
5. 外部工具依赖数据（项目二 RAG、项目三 KG）

## 2. 输入层拆分

### 2.1 教师历史请求（raw_teacher_support_dialogs_10pct.jsonl）
保留多轮原文、歧义表达、缺字段、混合任务请求。
适合做：
- 意图识别
- 槽位抽取
- 追问补全
- 路由验证

### 2.2 学情数据（mysql/*.csv/jsonl）
建议复现为多张表，而不是一张大宽表：
- student_profiles
- class_student_map
- practice_submissions
- student_mastery_snapshots
- intervention_feedback

这样更符合 MySQL / NL2SQL 场景，也更贴合企业真实落地。

### 2.3 历史干预案例（raw_intervention_cases_10pct.jsonl）
保留案例原文、弱知识点、证据来源、推荐动作、教师接受结果。
适合做：
- 干预规划参考
- 方案模板复用
- 教师采纳率分析

### 2.4 练习包与执行日志
- raw_assignment_package_catalog_10pct.jsonl
- raw_execution_logs_10pct.jsonl

适合复现：
- 练习下发节点
- 任务执行节点
- 回放测试

### 2.5 复用数据源
项目一明确整合项目二和项目三已有能力：
- `reused_project2_rag_subset_10pct.jsonl`
- `reused_project3_kg_subset_10pct.jsonl`

这两部分保留原始摘要与来源ID，方便做 source linking。

## 3. 推荐统一 state 字段
建议在 Agent 编排时统一输出如下字段：

```json
{
  "current_task_type": "学情诊断",
  "key_knowledge_points": ["条件判断", "报错排查"],
  "evidence_sources": ["MySQL学情", "知识图谱", "RAG"],
  "student_id": "STU-0012",
  "class_id": "CLS-PYB-001",
  "diagnosis_conclusion": "更像排查顺序不清晰",
  "intervention_goal": "先建立最小复现与排查顺序",
  "recommended_action": "生成3-5天干预方案",
  "need_clarify_fields": ["recent_submission"]
}
```

## 4. 为什么这样设计
因为项目一的核心不是“回答一句话”，而是：
- 能否识别任务
- 能否取到正确证据
- 能否把多个子节点串成闭环
- 能否稳定输出结构化结果
