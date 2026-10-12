# 本地测试版本的知识库接入

本分支同步了 HMBB Benchmark 实际使用的 Hermes 源码。当前代码入口和后端配置见[测试版本与知识库接入](testing-knowledge-backends.zh-CN.md)。

| 内容 | 位置 |
|---|---|
| Knowledge 接口、路由和证据预算 | `agent/knowledge_provider.py`、`agent/knowledge_registry.py`、`agent/knowledge_base_manager.py` |
| LLM-Wiki Provider 与私有检索工具 | `plugins/knowledge/llm_wiki/`、`tools/llm_wiki_tools.py` |
| LLM-Wiki 内置 Skill | `skills/research/llm-wiki/` |
| Utopia 只读 MCP Provider | `plugins/knowledge/utopia/` |
| Utopia 服务源码与测试用构建入口 | `backends/utopia/`、`backends/utopia/Dockerfile.benchmark` |
| Agent、CLI 与配置接入 | `agent/agent_init.py`、`agent/turn_context.py`、`agent/turn_finalizer.py`、`run_agent.py`、`hermes_cli/` |
| 后端注册和路由检查 | `tests/agent/test_knowledge_plugin_integration.py`、`tests/agent/test_knowledge_base_manager.py` |
| 后端与 CLI 检查 | `tests/agent/test_llm_wiki_provider.py`、`tests/agent/test_utopia_provider.py`、`tests/hermes_cli/` |

业务配置和实验结果由 [HMBB_Benchmark](https://github.com/HappynessI/HMBB_Benchmark) 维护。
