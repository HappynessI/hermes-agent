# Hermes RAG Backend 接口文档

## 1. 接口用途

本接口用于在 Hermes 的一次 Turn 或一个 Agent 会话中，从 OpenViking 或 LLM-Wiki 检索外部知识，并把结构化证据注入模型上下文。

支持的内置后端：

| 后端名 | 范式 | 适用场景 |
|---|---|---|
| `openviking` | 直接检索 | 低延迟检索已索引的 OpenViking Resources |
| `llm-wiki` | Agentic 检索 | 需要模型多步搜索、阅读和导航冻结 Wiki |

RAG 后端与模型 Provider、Memory Provider 是三个不同概念：

- `provider`：主模型从哪里调用；
- `memory.provider`：会话长期记忆由谁管理；
- `knowledge.provider` / `rag_backend`：本次外部知识从哪里检索。

选择 `rag_backend="openviking"` 不会修改 `memory.provider`。

## 2. 推荐公开接口

业务调用方优先使用 `AIAgent` 接口，不需要直接操作 Provider。

### 2.1 单次请求选择后端

```python
result = agent.run_conversation(
    user_message="How long are deleted Wix Forms retained?",
    rag_backend="openviking",
)
```

新增参数：

| 参数 | 类型 | 默认值 | 功能 |
|---|---|---:|---|
| `rag_backend` | `str` | `""` | 覆盖当前 Turn 的 RAG 后端。允许 `openviking`、`llm-wiki`；空值按 session/profile/legacy 解析 |

返回值仍然是 Hermes 原有的 result dictionary。发生 RAG 选择或错误时，其中增加 `rag` 字段。

请求级选择只影响当前 Turn，不会：

- 修改 Agent 的 session 选择；
- 修改 Profile；
- 修改 Memory Provider；
- 修改另一个并发请求的选择。

### 2.2 设置会话后端

```python
agent.set_rag_backend("llm-wiki")

first = agent.run_conversation("Find the relevant policy")
second = agent.run_conversation("Summarize the exceptions")
```

接口：

```python
def set_rag_backend(self, backend: str) -> None
```

| `backend` 值 | 行为 |
|---|---|
| `openviking` | 后续 Turn 默认使用 OpenViking |
| `llm-wiki` | 后续 Turn 默认使用 LLM-Wiki |
| `""` | 清除 session 覆盖，恢复 Profile/legacy 解析 |
| 未知名称 | 抛出 `ValueError` |
| 已知但不可用 | 抛出 `ValueError` |

单次 `run_conversation(..., rag_backend=...)` 的优先级仍高于 session 选择。

## 3. CLI 接口

### 3.1 基本调用

```text
hermes --rag-backend openviking -z "Find the deployment guide"
```

```text
hermes --rag-backend llm-wiki -z "What does the Wiki say about alpha?"
```

参数：

| CLI 参数 | 是否必填 | 功能 |
|---|---|---|
| `--rag-backend` | 否 | 当前 one-shot 使用的后端，只允许 `openviking`、`llm-wiki` |
| `-z` / `--oneshot` | 是 | 本次用户问题 |
| `--usage-file` | 否 | 将用量和结构化 `rag` 结果写入 JSON 文件 |

### 3.2 输出诊断文件

```text
hermes \
  --rag-backend openviking \
  --usage-file rag-usage.json \
  -z "Find the deployment guide"
```

`rag-usage.json` 中会包含：

```json
{
  "rag": {
    "requested_backend": "openviking",
    "actual_backend": "openviking",
    "latency_ms": 848.9,
    "degraded": false,
    "error": "",
    "warnings": [],
    "evidence": [],
    "metadata": {},
    "trace_id": "..."
  }
}
```

## 4. Profile 配置接口

行为配置放在当前 Profile 的 `config.yaml` 中。API Key 等秘密继续沿用现有秘密配置，不应写入 `knowledge`。

### 4.1 通用配置

```yaml
knowledge:
  enabled: true
  provider: openviking
  top_k: 5
  evidence_char_budget: 12000
  allow_fallback: false
```

| 参数 | 类型 | 默认值 | 功能 |
|---|---|---:|---|
| `enabled` | `bool` | `true` | 是否初始化 Knowledge Manager |
| `provider` | `str` | `""` | Profile 默认后端；允许 `openviking`、`llm-wiki` 或空值 |
| `top_k` | `int` | `5` | 最终 Evidence 硬上限，必须大于 0 |
| `evidence_char_budget` | `int` | `12000` | Evidence 正文合计字符上限，必须大于 0 |
| `allow_fallback` | `bool` | `false` | 预留字段；当前版本不能通过它开启 fallback |
| `max_context_chars` | `int` | `12000` | 旧配置兼容别名，优先使用 `evidence_char_budget` |

后端选择顺序：

```text
run_conversation(rag_backend=...)
    > agent.set_rag_backend(...)
    > knowledge.provider
    > 支持 Knowledge 的 legacy Memory Provider
```

任一层明确选出的后端失败时，不会尝试下一后端。

### 4.2 OpenViking 配置

```yaml
knowledge:
  provider: openviking
  openviking:
    target_uri: viking://resources/converted_md
    score_threshold: 0.15
    timeout_seconds: 4.0
    request_timeout_seconds: 3.0
    full_read_limit: 2
    prefer_abstract: false
```

| 参数 | 默认值 | 功能与范围 |
|---|---:|---|
| `target_uri` | `""` | 限定资源子树，例如 `viking://resources/converted_md` |
| `score_threshold` | `0.15` | 相关度阈值，限制到 `[0, 1]` |
| `timeout_seconds` | `4.0` | 总检索超时，限制到 `0.25–60` 秒 |
| `request_timeout_seconds` | `3.0` | 单 HTTP 请求超时，限制到 `0.25–60` 秒 |
| `full_read_limit` | `2` | 最多完整读取多少候选正文，限制到 `0–100` |
| `prefer_abstract` | `false` | 已有摘要时是否优先使用摘要 |

自动 Turn 检索固定为 resource-only。OpenViking 已有的 `viking_search`、`viking_read`、`viking_browse` 和 `viking_add_resource` 显式工具仍受原 toolset 开关控制，但不是自动 RAG 的必要步骤。

### 4.3 LLM-Wiki 配置

```yaml
knowledge:
  provider: llm-wiki
  llm_wiki:
    root: C:/data/frozen-wiki
    manifest: C:/data/frozen-wiki/manifest.json
    model: ""
    model_provider: ""
    max_iterations: 20
```

| 参数 | 默认值 | 功能 |
|---|---:|---|
| `root` | `<HERMES_HOME>/wiki` | 已构建、只读的 Wiki 根目录 |
| `manifest` | `""` | 可选语料 manifest，用于语料指纹 |
| `model` | `""` | 内层导航 Agent 模型；空值沿用当前 Profile |
| `model_provider` | `""` | 内层模型 Provider；空值沿用当前 Profile |
| `max_iterations` | `20` | 内层 Agent 最大迭代数，有效最小值为 3 |

`root` 必须存在以下文件：

```text
SCHEMA.md
index.md
log.md
```

运行期只查询，不构建或修改 Wiki。

## 5. 直接 Manager 接口

需要脱离完整 Agent 使用 RAG 时，可以直接创建 Manager。

### 5.1 创建 Manager

```python
from agent.knowledge_base_manager import create_configured_knowledge_manager
from agent.retrieval_scope import ProviderScope

scope = ProviderScope(
    user_id="user-1",
    agent_id="default",
    session_id="session-1",
    project_id="project-1",
    task_id="task-1",
    workspace="workspace-1",
    platform="cli",
    profile="default",
    hermes_home="/home/user/.hermes",
)

manager = create_configured_knowledge_manager(
    config={
        "enabled": True,
        "provider": "openviking",
        "top_k": 5,
        "evidence_char_budget": 12000,
        "openviking": {
            "target_uri": "viking://resources/converted_md",
        },
    },
    scope=scope,
    active_memory_provider=None,
    init_kwargs={"platform": "cli"},
)
```

工厂签名：

```python
def create_configured_knowledge_manager(
    *,
    config: Mapping[str, Any] | None,
    scope: ProviderScope,
    active_memory_provider: Any = None,
    init_kwargs: Mapping[str, Any] | None = None,
    registry: Optional[KnowledgeProviderRegistry] = None,
) -> Optional[KnowledgeBaseManager]
```

| 参数 | 功能 |
|---|---|
| `config` | `knowledge` 配置节点，不是整个 config.yaml |
| `scope` | 默认隔离范围 |
| `active_memory_provider` | 当前 Memory Provider；只有兼容 Knowledge 协议时才可能共享 |
| `init_kwargs` | Provider 初始化参数 |
| `registry` | 自定义 Provider Registry；为空使用内置 Registry |

当 `enabled=false` 时返回 `None`。

### 5.2 执行检索

```python
try:
    result = manager.search(
        "How long are deleted Wix Forms retained?",
        backend="openviking",
        top_k=5,
        evidence_char_budget=12000,
        filters={"score_threshold": 0.2},
        trace_id="request-123",
        session_id="session-1",
        task_id="task-1",
    )
finally:
    manager.shutdown()
```

签名：

```python
def search(
    query: str,
    *,
    backend: str = "",
    top_k: Optional[int] = None,
    evidence_char_budget: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None,
    trace_id: str = "",
    scope: Optional[ProviderScope] = None,
    session_id: str = "",
    task_id: str = "",
) -> KnowledgeSearchResult
```

| 参数 | 默认值 | 功能 |
|---|---:|---|
| `query` | 必填 | 检索问题；空文本返回空结果 |
| `backend` | `""` | 请求级后端覆盖 |
| `top_k` | `None` | 使用 Manager 默认值；必须大于 0 |
| `evidence_char_budget` | `None` | 使用 Manager 默认值；必须大于 0 |
| `filters` | `None` | Provider 特定过滤；OpenViking 当前支持 `score_threshold` |
| `trace_id` | `""` | 为空时自动生成 UUID |
| `scope` | `None` | 完整覆盖 Manager scope |
| `session_id` | `""` | 单独覆盖 scope 的 session ID |
| `task_id` | `""` | 单独覆盖 scope 的 task ID |

`top_k <= 0` 或字符预算小于等于 0 时抛出 `ValueError`。

## 6. 数据协议

协议定义在 `agent/knowledge_provider.py`。

### 6.1 `KnowledgeRequest`

```python
@dataclass(frozen=True)
class KnowledgeRequest:
    query: str
    backend: str = ""
    top_k: int = 5
    scope: ProviderScope = field(default_factory=ProviderScope)
    evidence_char_budget: Optional[int] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    trace_id: str = ""
```

### 6.2 `KnowledgeEvidence`

```python
@dataclass(frozen=True)
class KnowledgeEvidence:
    evidence_id: str
    source_uri: str
    content: str
    score: Optional[float] = None
    title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
```

字段说明：

| 字段 | 功能 |
|---|---|
| `evidence_id` | Provider 内稳定证据 ID |
| `source_uri` | 可追踪来源 URI |
| `content` | 最终证据正文 |
| `score` | 可选相关度分数 |
| `title` | 可读标题 |
| `metadata` | 资源 ID、page type、source IDs 等 |
| `provenance` | backend、trace、配置/语料/Skill 指纹等 |

### 6.3 `KnowledgeSearchResult`

```python
@dataclass(frozen=True)
class KnowledgeSearchResult:
    content: str = ""
    provider: str = ""
    citations: List[KnowledgeCitation] = field(default_factory=list)
    evidence: List[KnowledgeEvidence] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    requested_backend: str = ""
    actual_backend: str = ""
    latency_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    trace_id: str = ""
    degraded: bool = False
    error: str = ""
```

`content` 和 `citations` 是兼容字段；新调用方应优先读取 `evidence`。

## 7. `result["rag"]` 返回格式

```json
{
  "requested_backend": "llm-wiki",
  "actual_backend": "llm-wiki",
  "latency_ms": 40331.88,
  "degraded": false,
  "error": "",
  "warnings": [],
  "evidence": [
    {
      "evidence_id": "llm-wiki:entities/wix-forms.md",
      "source_uri": "wiki://entities/wix-forms.md",
      "title": "Wix Forms",
      "content": "...",
      "score": null,
      "metadata": {
        "path": "entities/wix-forms.md",
        "source_ids": ["..."]
      },
      "provenance": {
        "backend": "llm-wiki",
        "trace_id": "..."
      }
    }
  ],
  "metadata": {
    "query_sha256": "...",
    "top_k": 5,
    "evidence_count": 1,
    "evidence_chars": 5846,
    "selection_source": "request",
    "fallback_used": false
  },
  "trace_id": "..."
}
```

通用字段：

| 字段 | 功能 |
|---|---|
| `requested_backend` | 请求显式传入的规范化后端；没有请求覆盖时可为空 |
| `actual_backend` | 实际执行检索的后端；加载前失败时可为空 |
| `latency_ms` | Provider 检索总延迟 |
| `degraded` | 是否发生可诊断的降级/失败 |
| `error` | 脱敏错误文本 |
| `warnings` | 截断、不足证据等非致命告警 |
| `evidence` | 最终注入模型的证据 |
| `metadata` | 预算、选择来源、指纹和 Provider 指标 |
| `trace_id` | 检索追踪 ID |

通用 metadata：

| 字段 | 功能 |
|---|---|
| `query_sha256` | 查询 SHA-256，避免诊断必须保存问题正文 |
| `top_k` | 本次硬上限 |
| `evidence_count` | 最终证据数 |
| `evidence_chars` | 最终证据字符数 |
| `selection_source` | `request`、`session`、`profile` 或 `legacy` |
| `fallback_used` | 正常公共返回契约中固定为 `false` |

Provider 在创建/初始化前失败时，`actual_backend` 为空，并且部分公共 metadata 可能尚未生成。

## 8. 两种后端的执行流程

### 8.1 OpenViking

```text
请求
  → 解析 OpenViking
  → resource-only 搜索
  → 候选数 max(top_k × 4, 20)
  → 阈值过滤和排序
  → 最多选择 top_k
  → 在总超时/full_read_limit 内读取正文或摘要
  → Provider 字符预算
  → 生成 Evidence
  → Manager 再执行公共 top-K/字符预算
  → 注入主模型
```

特点：

- 不需要额外导航模型；
- 延迟通常较低；
- 依赖 OpenViking 服务、索引和连接配置；
- 自动 RAG 不检索 Memory/Skills。

### 8.2 LLM-Wiki

```text
请求
  → 创建隔离子 Agent
  → 只启用 llm_wiki_search/read
  → 读取 SCHEMA/index/log
  → 多步搜索与阅读
  → 工具层执行 root/source/字符预算
  → 子 Agent 返回严格 JSON
  → 校验 citation 确实 read 过
  → 按真实 read trace 生成 Evidence
  → Manager 再执行公共预算
  → 注入主模型
```

特点：

- 可以多步导航；
- 需要额外模型调用和 token；
- 延迟、成本高于直接检索；
- 在线接口只读；
- 路径与 Evidence 由工具代码约束，不依赖模型自律。

## 9. 失败处理

当前实现禁止静默跨后端 fallback。

| 场景 | 行为 |
|---|---|
| 未配置 Knowledge，调用方也未选择 | 返回空 Knowledge 结果，保持旧行为 |
| 请求未知 backend | `degraded=true`，返回未知/不可用错误 |
| Provider 初始化失败 | 记录脱敏错误，不切换 Provider |
| Provider 查询失败 | `degraded=true`，保留请求/实际后端信息 |
| `top_k` 或字符预算非法 | 抛出 `ValueError` |
| LLM-Wiki 返回非严格 JSON | `degraded=true` |
| LLM-Wiki 引用未读页面 | `degraded=true` |
| Evidence 超出预算 | 截断并增加 warning/metadata 标记 |

调用方建议：

```python
result = agent.run_conversation(question, rag_backend="llm-wiki")
rag = result.get("rag") or {}

if rag.get("degraded"):
    # 记录 trace_id 和脱敏 error，决定是否提示用户重试。
    # 不要在调用方偷偷改用另一个后端，除非产品协议明确允许。
    handle_rag_failure(rag)
```

## 10. 扩展自定义 Provider

自定义 Provider 继承 `KnowledgeBaseProvider`：

```python
from agent.knowledge_provider import (
    KnowledgeBaseProvider,
    KnowledgeEvidence,
    KnowledgeRequest,
    KnowledgeSearchResult,
)


class MyKnowledgeProvider(KnowledgeBaseProvider):
    @property
    def knowledge_name(self) -> str:
        return "my-provider"

    def get_knowledge_tool_schemas(self):
        return []

    def retrieve_knowledge(self, request: KnowledgeRequest):
        return KnowledgeSearchResult(
            provider=self.knowledge_name,
            evidence=[
                KnowledgeEvidence(
                    evidence_id="doc-1",
                    source_uri="my://doc-1",
                    title="Document 1",
                    content="bounded evidence",
                    provenance={
                        "backend": self.knowledge_name,
                        "trace_id": request.trace_id,
                    },
                )
            ],
        )
```

注册工厂：

```python
from agent.knowledge_registry import KnowledgeProviderRegistry

registry = KnowledgeProviderRegistry()
registry.register("my-provider", lambda config: MyKnowledgeProvider())
```

Provider 应遵守：

1. `knowledge_name` 稳定且与 Registry 名称一致。
2. 不在 `configure_knowledge()` 中读取或持久化无关秘密。
3. 尽早执行 Provider 内预算；Manager 仍会二次约束。
4. `source_uri` 和 provenance 必须可追踪。
5. Provider 失败应返回 degraded result 或抛出可脱敏异常。
6. 不自行切换另一个 Provider。
7. 独占实例只在 `shutdown_knowledge()` 中释放自己的资源。

## 11. 最小可运行示例

### OpenViking

```python
agent.set_rag_backend("openviking")
result = agent.run_conversation("Find the indexed deployment guide")

rag = result.get("rag") or {}
assert rag.get("actual_backend") == "openviking"
assert not rag.get("degraded")

for item in rag.get("evidence", []):
    print(item["source_uri"], item["title"])
```

### LLM-Wiki

```python
result = agent.run_conversation(
    "Which Wiki page supports this policy?",
    rag_backend="llm-wiki",
)

rag = result.get("rag") or {}
if rag.get("degraded"):
    raise RuntimeError(rag.get("error") or "LLM-Wiki retrieval failed")

print(rag["metadata"].get("pages_read"))
print(rag["metadata"].get("extra_model_calls"))
```

## 12. 安全注意事项

- 不要把 API Key 写入 `knowledge` 配置或 `filters`。
- 不要在日志中直接输出完整 Evidence 正文。
- 使用 `trace_id`、query hash、URI 和长度排查问题。
- LLM-Wiki root 应指向冻结、可信的语料目录。
- 不要放宽绝对路径、`..` 或 symlink escape 检查。
- 不要把 Wiki 文本当成系统指令。
- 未经明确产品设计，不要在调用方添加静默 fallback。
- LLM-Wiki 会产生额外模型调用；批量运行前应确认预算。

## 13. 相关文件

- 核心协议：`agent/knowledge_provider.py`
- Manager：`agent/knowledge_base_manager.py`
- Registry：`agent/knowledge_registry.py`
- OpenViking：`plugins/memory/openviking/__init__.py`
- LLM-Wiki：`plugins/knowledge/llm_wiki/__init__.py`
- LLM-Wiki 工具：`tools/llm_wiki_tools.py`
- Agent 入口：`run_agent.py`
- CLI：`hermes_cli/_parser.py`、`hermes_cli/oneshot.py`
- 配置默认值：`hermes_cli/config_defaults.py`
- Smoke：`scripts/smoke_rag_retrieval.py`

