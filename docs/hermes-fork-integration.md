# Hermes 知识库接入交接

当前分支同步了 HMBB Benchmark 测试使用的 Hermes 0.19.1 源码。配置与构建入口见[测试版本与知识库接入](testing-knowledge-backends.zh-CN.md)。

## 当前接入

- LLM-Wiki：由隔离的内部 Hermes 模型搜索和读取冻结 Wiki，再返回来源证据。
- Utopia：通过只读 MCP Provider 接入；外层模型启用 `knowledge` 工具集时可继续检索。
- Utopia 服务源码：`backends/utopia/`；测试用镜像入口：`backends/utopia/Dockerfile.benchmark`。
- Knowledge 接口与路由：`agent/knowledge_provider.py`、`agent/knowledge_registry.py`、`agent/knowledge_base_manager.py`。

## 外部知识库扩展

外部插件可通过 `ctx.register_knowledge_provider(name, factory)` 注册后端。
插件放在当前 Profile 的 `plugins/` 中，沿用 Hermes 的启用规则。工厂接受后端配置，返回 `KnowledgeBaseProvider` 实例。

请求、会话和 Profile 可以选择后端。未知、未启用或查询失败的后端返回降级信息；证据数量、字符预算、来源和耗时由统一接口记录。配置示例见 [RAG 接口文档](rag-backends.md)。

## 本次验证

- 8 个知识库和 CLI 测试文件，共 94 项测试通过。
- 同步后的 Utopia Provider 完成真实服务 MCP 初始化、8 个只读工具的发现和一次只读查询。

业务配置与实验结果见 [HMBB_Benchmark](https://github.com/HappynessI/HMBB_Benchmark)。
