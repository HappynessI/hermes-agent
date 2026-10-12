# 测试版本与知识库接入

本分支同步了 HMBB Benchmark 实验实际调用的本地 Hermes 0.19.1 源码。知识库后端入口位于 `plugins/knowledge/`，独立 Utopia 服务源码位于 `backends/utopia/`。

## LLM-Wiki

在当前 Hermes Profile 的 `config.yaml` 中配置：

```yaml
knowledge:
  enabled: true
  provider: llm-wiki
  top_k: 8
  evidence_char_budget: 24000
  llm-wiki:
    root: ./work/wiki
    model: ""
    model_provider: ""
    max_iterations: 20
```

Wiki 根目录需要已有的 `SCHEMA.md`、`index.md`、`log.md` 和知识页面。检索阶段由内部 Hermes 模型执行，只能调用限定在 Wiki 根目录中的搜索和读取工具。检索返回的证据再交给外层模型；外层不需要开放 Wiki 检索工具。

```bash
python hermes --rag-backend llm-wiki -z "根据知识库回答问题"
```

## Utopia

```yaml
knowledge:
  enabled: true
  provider: utopia
  top_k: 8
  evidence_char_budget: 24000
  utopia:
    mcp_url: http://127.0.0.1:1516/api/v1/kbs/<kb-id>/mcp
    token_env: UTOPIA_READ_TOKEN
    timeout_seconds: 60
```

先在 Utopia 创建知识库并导入资料，再为该知识库创建只读 token，通过 `UTOPIA_READ_TOKEN` 环境变量提供给 Hermes。模型 API 凭据使用 Hermes Profile 的现有配置。

Utopia 接入先执行一次 `search_chunks` 检索；调用方启用 `knowledge` 工具集时，模型还可以继续搜索、读取文档或查询图谱。实验使用的外层调用参数包括：

```python
agent = AIAgent(
    # 模型与认证参数由调用方的配置提供。
    enabled_toolsets=["knowledge"],
    max_iterations=20,
    skip_memory=True,
    skip_knowledge=False,
)
result = agent.run_conversation(question, rag_backend="utopia")
```

Utopia 服务的常规部署方法见[服务 README](../backends/utopia/README.zh-CN.md)。复用测试用 Dockerfile 时，可从仓库根目录执行：

```bash
mkdir -p work/utopia-cargo-cache
docker build \
  --build-context cargo-cache=./work/utopia-cargo-cache \
  -f backends/utopia/Dockerfile.benchmark \
  -t hermes-utopia:test backends/utopia
UTOPIA_IMAGE=hermes-utopia:test \
  docker compose -f backends/utopia/docker-compose.yml --profile app up -d
```

`cargo-cache` 可以是空目录，也可以使用已有的 Cargo registry 缓存。数据库、索引和上传文件由各自部署保存，模型与 token 均在运行时配置。

## 验证

使用仓库统一测试入口：

```bash
bash scripts/run_tests.sh \
  tests/agent/test_knowledge_plugin_integration.py \
  tests/agent/test_knowledge_base_manager.py \
  tests/agent/test_llm_wiki_provider.py \
  tests/agent/test_utopia_provider.py \
  tests/plugins/memory/test_openviking_provider.py \
  tests/agent/test_turn_finalizer_iteration_limit_exit.py \
  tests/hermes_cli/test_argparse_flag_propagation.py \
  tests/hermes_cli/test_oneshot_usage_file.py -q
```

实验结果见 [HMBB_Benchmark](https://github.com/HappynessI/HMBB_Benchmark)。
