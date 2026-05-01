# 里程碑：项目一接入真实项目二 RAG 与真实项目三 KG（远程联调）

## 一、背景

项目一是基于 **LangGraph** 的 Agent 编排层，通过 **HTTP API** 调用项目二（RAG）与项目三（KG）的 evidence-only 接口，**不直接 import 外部项目代码**。  
数据面与协议面由 **Adapter → Contract validation → ResponseMapper → 内部 Evidence Schema** 串联，保证编排逻辑与外部服务字段解耦，便于独立演进与回归测试。

## 二、服务配置

### RAG（项目二）

```env
RAG_PROVIDER=remote
RAG_API_BASE=http://127.0.0.1:8001
RAG_ENDPOINT=/search
```

### KG（项目三）

```env
KG_PROVIDER=remote
KG_API_BASE=http://127.0.0.1:8002
KG_ENDPOINT=/graph_query
```

说明：本地 `.env` 若仍指向 mock 端口（例如 KG `9003`），会覆盖代码默认值；联调真实服务时建议使用上述地址，或在 smoke 命令中通过 `--rag-api-base` / `--kg-api-base` 显式覆盖。

## 三、验证命令

在项目二、项目三服务已启动的前提下，于项目一仓库根目录执行（PowerShell 换行示例）：

```powershell
python scripts/smoke_test_real_rag_kg.py `
  --rag-api-base http://127.0.0.1:8001 `
  --kg-api-base http://127.0.0.1:8002 `
  --auto-warmup-rag `
  --fail-on-rag-fallback
```

该命令会：

- 对 RAG 探测 `/health`、`/ready`，必要时自动 `GET /warmup` 后再探测；
- 对 KG 探测 `/health`、`/ready`；
- 跑完整 LangGraph workflow，并输出 RAG/KG 证据摘要与 `debug_trace` 中的 provider、校验与 mapper 状态；
- 若 RAG top1 仍为 fallback 证据且指定了 `--fail-on-rag-fallback`，则以非零退出码失败，便于 CI 或本地一键验收。

## 四、本次验证结果

以下为一次成功联调的代表性输出摘要（与脚本及项目二、三当前版本一致）。

- **pytest**：50 passed  
- **RAG `/ready` 初始**：`fallback_only`；通过 **`--auto-warmup-rag`** 自动 warmup 后进入 **`lightweight_search`**（就绪形态改善）。  
- **RAG evidence_count**：5  
- **RAG top1**：`seed-faq-nameerror`  
- **RAG source_type**：`faq`  
- **RAG metadata.route**：`bm25_faq`  
- **RAG metadata.fallback**：`False`  
- **RAG validation_ok**：`True`  
- **RAG mapper**：`RAGResponseMapper`  

- **KG ready**：`true`  
- **neo4j_connected**：`true`  
- **graph_node_count**：1154  
- **graph_relation_count**：1913  
- **KG evidence_count**：5  
- **KG top1 metadata.source**：`neo4j_core_seed`  
- **KG entity**：`NameError`  
- **KG relation**：`RELATED_ERROR` / `HAS_SOLUTION`（以实际返回为准）  
- **KG validation_ok**：`True`  
- **KG mapper**：`KGResponseMapper`  

- **final_response_non_empty**：`True`  
- **rag_provider**：`remote`  
- **kg_provider**：`remote`  

## 五、工程价值

1. **Adapter 层隔离外部服务**：项目一图节点只依赖统一 `search()` 等接口，不感知项目二、三的实现细节。  
2. **项目二 RAG 提供文本证据**：支撑课堂话术、FAQ、BM25 等检索式证据，进入统一 `RAGEvidenceItem`。  
3. **项目三 KG 提供结构化知识关系**：实体—关系—目标与证据文本进入统一 `KGEvidenceItem`，便于诊断与解释链融合。  
4. **ResponseMapper 统一外部 schema**：字段别名、嵌套结构、缺省兜底集中在 mapper，避免 graph 随远程字段变更大面积改动。  
5. **Contract validation 保证鲁棒性**：响应结构异常时 adapter 返回空证据并记录状态，**workflow 不崩溃**，`debug_trace` 可观测。  
6. **`--auto-warmup-rag` 提升联调稳定性**：冷启动或索引未就绪时自动触发 warmup，减少「第一次必 fallback」的人工重复操作。  
7. **形成 RAG + KG + 学情数据的编排架构**：同一套状态与证据摘要可继续叠加学生画像、提交与干预反馈，支撑完整教学闭环。

## 六、当前限制和后续优化

- 项目二 **`/search`** 当前默认以 **lightweight FAQ / BM25** 等路径为主，具体路由与索引形态依赖项目二部署与 warmup 策略。  
- 项目三 KG 对高频教学 case 仍较依赖 **core_seed** 等高质量种子图谱；图谱覆盖不足时 evidence 可能偏少或泛化。  
- 后续可扩展更多错误类型与知识点，例如 **TypeError、SyntaxError、IndexError、KeyError** 等，与项目二 FAQ、项目三图谱 jointly 对齐。  
- 后续可将本次联调中的 **RAG/KG 就绪状态、warmup 前后对比、fallback 标记** 等结构化展示到 **Streamlit** 或现有 API 的 debug 面板，降低教师与研发排查成本。

---

*文档版本：与仓库内联合 smoke 脚本及 pytest 通过版本对应；若外部服务接口变更，请以 `app/tools/contracts.py`、各 ResponseMapper 及 README 为准同步更新本文。*
