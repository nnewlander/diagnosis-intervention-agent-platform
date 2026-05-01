# 项目一面试演示问题清单

## 一、演示前置条件

演示前请先确认以下服务已启动并可访问：

- 项目二 RAG 服务：`http://127.0.0.1:8001`
- 项目三 KG 服务：`http://127.0.0.1:8002`
- 项目一 API：`http://127.0.0.1:8000`
- 项目一 Streamlit 前端

---

## 二、演示 case 1：技术答疑 NameError

**输入：**

课堂演示遇到 NameError，应该怎么给学生解释？

**预期结果：**

- `task_type=technical_qa`
- RAG 命中 `seed-faq-nameerror`
- KG 命中 NameError core_seed
- `final_response` 解释 NameError：程序找不到这个名字
- 学情/诊断/干预/推荐区显示“当前任务不适用”

**面试讲解重点：**

- RAG 提供 FAQ/BM25 文本证据
- KG 提供 NameError 的结构化关系
- technical_qa 不触发学情诊断链路

---

## 三、演示 case 2：学情诊断

**输入：**

李同学最近几次作业在变量定义和 for循环上反复出错，帮我诊断一下。

**预期结果：**

- `task_type=diagnosis`
- StudentData 支持变量定义
- RAG 支持变量定义
- KG 支持 for循环
- `evidence_alignment_status=partially_aligned`
- `final_response` 能区分：
  - 用户关注点
  - 学情数据命中点
  - 学情未直接支持点
  - 系统额外发现点

**面试讲解重点：**

- 系统不是盲目相信用户描述，也不是只看学情数据
- 能处理“用户关注点”和“学情数据”部分一致的场景
- RAG/KG 可补充学情缺失证据

---

## 四、演示 case 3：诊断 + 3 天干预

**输入：**

李同学最近在 for循环和条件判断上一直出错，帮我先诊断一下，再给一个 3 天干预建议。

**预期结果：**

- `task_type=mixed`
- `primary_task_type=diagnosis`
- `secondary_task_types=intervention`
- StudentData 未直接支持 for循环、条件判断
- RAG 支持条件判断
- KG 支持 for循环
- `evidence_alignment_status=mismatched`
- `confidence_level=cautious_medium`
- 生成 3 天干预计划
- 推荐练习包覆盖 for循环、条件判断

**面试讲解重点：**

- mixed 请求会触发多节点 workflow（主任务 + 次任务）
- 证据不一致时系统自动降低置信度
- 干预建议会附带证据边界提醒

---

## 五、演示 case 4：RAG + KG 混合解释

**输入：**

学生问 NameError 是什么意思，能结合知识图谱和资料给我一个课堂解释吗？

**预期结果：**

- `task_type=technical_qa`
- RAG 高相关 evidence 默认只展示 `seed-faq-nameerror`
- 低相关 RAG 候选折叠展示
- KG 推荐解决关系：`NameError -> HAS_SOLUTION -> 检查变量或函数是否先定义后使用`
- `final_response` 包含“程序找不到这个名字”

**面试讲解重点：**

- 用户明确要求“结合资料 + 知识图谱”
- RAG 负责资料证据
- KG 负责结构化关系
- 前端分层展示高相关证据与低相关候选

---

## 六、统一讲解话术

项目一是 Agent 编排层，使用 LangGraph 管理任务路由和多节点状态流转；通过 Adapter 调用项目二 RAG 和项目三 KG；通过 ResponseMapper 统一外部服务 schema；通过 Contract Validation 保证外部服务异常时 workflow 不崩；最终结合学情数据生成诊断、干预和推荐结果。

