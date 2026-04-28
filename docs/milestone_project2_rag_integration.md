# 项目一接入真实项目二 RAG 服务里程碑

## 一、背景

项目一（核桃智能教学诊断与干预 Agent 平台）是教学诊断与干预的多节点编排系统；项目二是教师智能知识库问答系统。  
两者采用服务化集成方式：项目一不直接 `import` 项目二代码，而是通过 HTTP API 调用项目二的 evidence-only `POST /search` 接口来获取检索证据。

## 二、项目二提供的接口

- `GET /health`
- `GET /ready`
- `GET /warmup`
- `POST /search`

## 三、项目一侧配置

示例配置如下：

```env
RAG_PROVIDER=remote
RAG_API_BASE=http://127.0.0.1:8001
RAG_ENDPOINT=/search
KG_PROVIDER=local
```

## 四、验证命令

项目二：

```bash
python scripts/warmup_rag.py
python scripts/profile_search_api.py
```

项目一：

```bash
python scripts/smoke_test_real_rag.py --force-remote --auto-warmup --fail-on-fallback
```

## 五、当前验证结果

当前阶段验证记录：

- 项目二 `pytest`：`24 passed`
- 项目一 `pytest`：`38 passed`
- 项目二 `/ready`：`serving_mode=lightweight_search`
- 项目二 `/search` top1：`seed-faq-nameerror`
- 项目一 smoke：`source_id=seed-faq-nameerror`
- `source_type=faq`
- `route=bm25_faq`
- `fallback=false`
- `validation_ok=true`
- `mapper_used=RAGResponseMapper`

## 六、工程价值

- 项目一与项目二通过 HTTP API 解耦，便于独立演进与部署。
- `RemoteRAGAdapter` 屏蔽外部服务调用细节，graph 节点保持稳定。
- `RAGResponseMapper` + contract validation 屏蔽项目二返回字段变化，降低联调风险。
- `auto-warmup` 能自动完成 readiness 检查与预热，提高真实联调稳定性。
- lightweight search 模式避免 smoke/eval 场景不必要加载向量模型或 LLM，提升调试效率。

## 七、当前限制

- 项目二 `/search` 当前默认 lightweight FAQ/BM25 路径。
- vector/model 还不是默认 search 路径。
- technical_qa 离线评测仍有继续提升空间。
- dispatch 练习下发相关指标仍需后续优化。

## 八、下一步计划

- 接入真实项目三 KG 服务。
- 项目一支持 RAG + KG 双远程 evidence 协同。
- 持续优化 technical_qa 与 dispatch 指标表现。
