# 本地完整演示运行手册（Demo Runbook）

## 一、演示目标

本手册用于本地完整演示项目一「核桃智能教学诊断与干预 Agent 平台」的端到端能力：  
项目一作为 **Agent 编排层（LangGraph）**，通过 HTTP 接口调用项目二 RAG 与项目三 KG（不直接 import 外部项目代码），联合学情数据完成：

- 教学技术问答
- 学情诊断
- 干预建议生成
- 练习推荐

## 二、启动前准备

演示前请确认以下服务或组件可用：

- 项目二 RAG 服务（端口 `8001`）
- 项目三 KG 服务（端口 `8002`）
- Neo4j（Bolt 端口 `7687`）
- 项目一 API（端口 `8000`）
- 项目一 Streamlit 前端

建议先确认项目一 `.env` 使用 remote 配置：

```env
RAG_PROVIDER=remote
RAG_API_BASE=http://127.0.0.1:8001
RAG_ENDPOINT=/search

KG_PROVIDER=remote
KG_API_BASE=http://127.0.0.1:8002
KG_ENDPOINT=/graph_query
```

## 三、启动顺序

### 1) 启动 Neo4j

先检查本机端口连通性（PowerShell）：

```powershell
Test-NetConnection 127.0.0.1 -Port 7687
```

若端口不通，请先启动 Neo4j 服务后再继续。

### 2) 启动项目二 RAG

```powershell
cd D:\projectdata\KnowledgeBaseQ_ASystem
conda activate project2_rag_strict
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --log-level debug
```

另开窗口执行 warmup：

```powershell
python scripts/warmup_rag.py
```

### 3) 启动项目三 KG

```powershell
cd D:\projectdata\KnowledgeGraph_Graph_basedQ_A_Platform
conda activate kgqa_p3_py311
python -m uvicorn app.main:app --host 127.0.0.1 --port 8002 --log-level debug
```

验证接口：

```powershell
python scripts/test_graph_query_api.py
```

### 4) 启动项目一 API

```powershell
cd D:\projectdata\Diagnosis_InterventionAgentPlatform
conda activate diagnosis_agent
python scripts/run_api.py
```

### 5) 启动项目一前端

```powershell
streamlit run frontend/app.py
```

## 四、联调验证命令

在项目一目录执行：

```powershell
python scripts/smoke_test_real_rag_kg.py `
  --rag-api-base http://127.0.0.1:8001 `
  --kg-api-base http://127.0.0.1:8002 `
  --auto-warmup-rag `
  --fail-on-rag-fallback
```

## 五、期望结果

联合 smoke 通过时，典型结果应包含：

- `RAG evidence_count=5`
- `RAG top1=seed-faq-nameerror`
- `KG evidence_count=5`
- `KG top1 source=neo4j_core_seed`
- `final_response_non_empty=True`
- `rag_validation_ok=True`
- `kg_validation_ok=True`

## 六、常见问题

### 1) RAG `fallback_only`

现象：RAG 就绪状态偏冷启动，top1 可能落到 fallback。  
解决：

- 先运行项目二 warmup：`python scripts/warmup_rag.py`
- 或在联合 smoke 使用 `--auto-warmup-rag`

### 2) KG `evidence_count=0`

重点检查 `KG_API_BASE` 是否为真实项目三端口 `8002`，而不是 mock 常用端口 `9003`。

### 3) `/ready` 超时

优先检查 Neo4j 是否启动，及 `7687` 端口是否可达：

```powershell
Test-NetConnection 127.0.0.1 -Port 7687
```

### 4) GitHub 不要提交 `.env`

`.env` 属于本地环境配置，不应提交到远程仓库。  
应提交并维护的是 `.env.example`，用于团队共享配置模板。
