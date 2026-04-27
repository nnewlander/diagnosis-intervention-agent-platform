# 核桃智能教学诊断与干预 Agent 平台（开发调试版）

本项目是面向教师场景的 Agent 平台骨架，目标是先实现**本地可调试、可验证、可回归测试**，再平滑替换为真实 RAG/KG/MySQL 服务。

## 项目目标

- 支持教学请求闭环：技术答疑、学情诊断、干预规划、练习下发、混合请求。
- 使用 LangGraph 做多节点编排，统一 `AgentState`。
- 本地 mock 检索可运行，不依赖外部大模型 API。
- 返回结构化 JSON，并附带 `debug_trace` 便于排查。

## 技术栈

- Python 3.11
- FastAPI
- Streamlit
- LangGraph
- Pydantic / pydantic-settings

## 目录结构

```text
.
├── app/
│   ├── api/
│   ├── core/
│   ├── graph/
│   ├── services/
│   ├── tools/
│   ├── models/
│   └── data_loader/
├── frontend/
├── scripts/
├── tests/
├── data/
│   └── sample_requests.json
├── outputs/
├── .env.example
└── requirements.txt
```

## .env 配置说明

复制 `.env.example` 为 `.env` 后可按需修改：

- `PROJECT_NAME`: 项目名
- `API_HOST`: FastAPI 监听地址
- `API_PORT`: FastAPI 端口
- `DATA_ROOT`: 本地数据根目录（默认 `project1_agent_raw_data_10pct`）
- `DEBUG`: 是否开启调试模式
- `TOP_K_RAG`: RAG 检索条数
- `TOP_K_KG`: KG 检索条数
- `TOP_K_PACKAGES`: 练习包推荐条数
- `MAX_SUBMISSIONS`: 每次最多读取提交记录数
- `OUTPUT_DIR`: 离线评测输出目录
- `RAG_PROVIDER`: `local|remote`
- `RAG_API_BASE`: 项目二 RAG 服务地址
- `RAG_API_KEY`: 可选鉴权令牌
- `RAG_TIMEOUT`: RAG 请求超时秒数
- `KG_PROVIDER`: `local|remote`
- `KG_API_BASE`: 项目三 KG 服务地址
- `KG_API_KEY`: 可选鉴权令牌
- `KG_TIMEOUT`: KG 请求超时秒数
- `STUDENT_DATA_PROVIDER`: `local_csv_jsonl|sqlite|mysql`
- `SQLITE_DB_PATH`: 本地 SQLite 文件路径
- `MYSQL_HOST/MYSQL_PORT/MYSQL_USER/MYSQL_PASSWORD/MYSQL_DB`: MySQL 占位连接配置

`.env` 完整示例：

```env
PROJECT_NAME=核桃智能教学诊断与干预 Agent 平台
API_HOST=0.0.0.0
API_PORT=8000
DATA_ROOT=project1_agent_raw_data_10pct
DEBUG=true
TOP_K_RAG=5
TOP_K_KG=5
TOP_K_PACKAGES=3
MAX_SUBMISSIONS=10
OUTPUT_DIR=outputs
RAG_PROVIDER=local
RAG_API_BASE=http://127.0.0.1:9002
RAG_API_KEY=
RAG_TIMEOUT=15
KG_PROVIDER=local
KG_API_BASE=http://127.0.0.1:9003
KG_API_KEY=
KG_TIMEOUT=15
STUDENT_DATA_PROVIDER=local_csv_jsonl
SQLITE_DB_PATH=outputs/local_student_data.db
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_DB=teaching_agent
```

## 本地启动顺序

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 启动 FastAPI

```bash
python scripts/run_api.py
```

3. 启动 Streamlit Demo

```bash
streamlit run frontend/app.py
```

4. 访问接口与页面

- 健康检查：`GET http://127.0.0.1:8000/health`
- Agent 接口：`POST http://127.0.0.1:8000/agent/run`

## 调试能力（debug_trace）

LangGraph 每个节点都会写入 `debug_trace`，每条记录包含：

- `node_name`
- `input_summary`
- `output_summary`
- `selected_task_type`
- `selected_tools`
- `timestamp`

查看方式：

- API 返回字段 `debug_trace`
- Streamlit 中“调试详情（debug_trace）”折叠面板

## 本地 Smoke Test

用于快速校验接口可用性和返回结构。

```bash
python scripts/smoke_test.py
```

行为：

- 读取 `data/sample_requests.json`（若不存在则使用内置请求）
- 调用本地 `/agent/run`
- 打印 `task_type / diagnosis / final_response`
- 做基本结构校验

## 离线评测

```bash
python scripts/eval_offline.py
```

评测数据来源：

- `project1_agent_raw_data_10pct/data/agent_eval_requests_10pct.jsonl`

统计指标：

- `task_type` 命中率（若样本含 `expected_primary_node` 或 `task_hint`）
- `primary_task_type` 命中率
- 槽位抽取覆盖率（`student_id`、`knowledge_points`）
- `need_clarify` 触发率
- JSON 结构完整率
- `diagnosis` 非空率
- `final_response` 非空率
- `recommended_packages` 非空率

输出文件：

- `outputs/eval_report.json`
- `outputs/eval_report.csv`
- `outputs/eval_case_details.csv`

## 混合请求路由说明

`route_task` 支持：

- 诊断 + 干预
- 技术答疑 + 学情诊断
- 其他多意图组合

状态字段：

- `task_type`（单任务或 `mixed`）
- `primary_task_type`（主任务）
- `secondary_task_types`（次任务列表）

后续节点按主任务优先执行，同时补充次任务相关证据和结果。

## 业务能力增强点

- 意图识别新增槽位抽取：`student_id`、`class_id`、`knowledge_points`、`desired_days`、`error_type`、`task_priority`
- 支持“某某同学/这个学生”实体解析与模糊定位（`app/services/entity_resolver.py`）
- 学情证据聚合升级为结构化摘要（`app/services/sql_service.py`）
- 诊断与干预改为结构化输出，且支持证据不足时保守模式
- 练习包推荐结合知识点 + 年级 + 难度，并输出推荐理由

## local/mock 与 remote/service 架构

- `app/tools/base.py` 定义统一 adapter 接口。
- `app/tools/rag_adapter.py`:
  - `LocalRAGAdapter`（本地 jsonl 检索）
  - `RemoteRAGAdapter`（HTTP 占位，后续对接项目二）
- `app/tools/kg_adapter.py`:
  - `LocalKGAdapter`（本地 jsonl 检索）
  - `RemoteKGAdapter`（HTTP 占位 + Neo4j/Cypher TODO）
- `app/tools/student_data_adapter.py`:
  - `LocalCSVJSONLStudentDataAdapter`
  - `SQLiteStudentDataAdapter`
  - `MySQLStudentDataAdapter`（占位）
- `app/tools/package_adapter.py`:
  - `LocalPackageAdapter`
  - `RemotePackageAdapter`（占位）

LangGraph 节点只通过 adapter 调用，不直接读取底层文件。

## 内部 Evidence Schema

内部统一证据对象定义在 `app/models/evidence.py`，用于隔离 graph 与外部接口字段差异：

- `RAGEvidenceItem`: `source_id/title/snippet/score/source_type/metadata`
- `KGEvidenceItem`: `entity/entity_type/relation/target/evidence/score/metadata`
- `StudentEvidenceBundle`: 学生画像、提交摘要、薄弱点、错误分布、干预反馈
- `PackageRecommendationItem`: `package_id/package_name/reason/difficulty_level`

这保证 local 与 remote 最终都回到同一内部格式。

## 为什么需要 Response Mapper

真实服务返回字段经常不一致，例如：

- RAG 可能返回 `hits/results/items`
- KG 可能返回 `records/results/paths`

Mapper 层（`app/tools/response_mappers/`）负责：

- 字段兼容映射
- 缺字段兜底
- 默认值填充

因此 graph 不需要感知 remote 原始字段变化。

## Remote Contract Validation

`app/tools/contracts.py` 定义 remote 输入/输出 contract（Pydantic）：

- 先校验请求 payload
- 再校验 remote 响应结构
- 校验失败不会让 workflow 崩溃，adapter 返回空结果并在 debug trace 中记录 validation 错误摘要

## 项目二 / 项目三接入说明

- 项目二（RAG）：
  1. 设置 `RAG_PROVIDER=remote`
  2. 配置 `RAG_API_BASE` 与可选 `RAG_API_KEY`
  3. 在 `RemoteRAGAdapter.search` 中按真实接口协议补齐 payload/response mapping
  4. 若字段变化，只改 `RAGResponseMapper`，无需改 graph

- 项目三（KG）：
  1. 设置 `KG_PROVIDER=remote`
  2. 配置 `KG_API_BASE` 与可选 `KG_API_KEY`
  3. 在 `RemoteKGAdapter.search` 对接真实 HTTP 或补齐 Neo4j/Cypher 查询实现
  4. 若字段变化，只改 `KGResponseMapper`，无需改 graph

## Mock Remote Services 启动

1) 启动 mock RAG：

```bash
python scripts/run_mock_rag.py
```

2) 启动 mock KG：

```bash
python scripts/run_mock_kg.py
```

3) 运行 remote smoke：

```bash
python scripts/smoke_test_remote.py
```

该脚本会切换 `RAG_PROVIDER=remote` 和 `KG_PROVIDER=remote`，并打印 evidence 示例，便于联调前验证 mapper 和 contract 行为。

## SQLite 初始化步骤

```bash
python scripts/build_local_sqlite.py
```

脚本会从 `project1_agent_raw_data_10pct/mysql/` 构建 SQLite，并创建：

- `student_profiles`
- `class_student_map`
- `practice_submissions`
- `student_mastery_snapshots`
- `intervention_feedback`

切换到 SQLite 模式：

- `.env` 中设置 `STUDENT_DATA_PROVIDER=sqlite`
- 配置 `SQLITE_DB_PATH` 指向刚生成的数据库

## 后续替换到 MySQL / Neo4j

- MySQL：
  - 在 `MySQLStudentDataAdapter` 中实现真实连接池与查询语句
  - 保持返回 schema 不变，确保 graph 无感切换
- Neo4j：
  - 在 `RemoteKGAdapter` 的 TODO 位置加入 driver 初始化、Cypher 模板和结果归一化
  - 保持 `search(query, keywords, top_k)` 接口不变

## 后续替换成真实项目二/三服务

- 优先保持 adapter 对外接口不变（`search()/resolve_student()/load_student_evidence()`）。
- 真实接口字段变化优先改 mapper，不动 graph。
- 协议升级优先改 contract，校验逻辑集中在 adapter 入口。
- 联调阶段建议先跑 `scripts/smoke_test_remote.py` + `tests/test_remote_mappers.py`。
