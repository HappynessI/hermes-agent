# 测试使用的知识库后端

本仓库包含 HMBB Benchmark 测试使用的 Hermes 源码，以及对应的知识库接入代码和 Utopia 服务源码。

| 组件 | 位置 | 作用 |
|---|---|---|
| Knowledge 接口与路由 | `../agent/knowledge_provider.py`、`../agent/knowledge_registry.py`、`../agent/knowledge_base_manager.py` | 选择后端，控制证据数量、字符预算与失败状态 |
| LLM-Wiki | `../plugins/knowledge/llm_wiki/`、`../tools/llm_wiki_tools.py`、`../skills/research/llm-wiki/` | 内部检索模型搜索、读取冻结 Wiki，再将证据交给客服模型 |
| Utopia 接入 | `../plugins/knowledge/utopia/` | 通过 MCP 检索；可向客服模型开放允许的只读工具 |
| Utopia 服务 | `utopia/` | 测试所用 Utopia 0.1.0 源码，保留 Apache-2.0 许可证 |
| Utopia 测试镜像 | `utopia/Dockerfile.benchmark` | 测试使用的 Docker 构建入口 |

Utopia 源码来自 [deeplethe/utopia](https://github.com/deeplethe/utopia)，采用本地测试版本，未跟随其后续 main 更新。配置和运行方法见[后端接入说明](../docs/testing-knowledge-backends.zh-CN.md)。
