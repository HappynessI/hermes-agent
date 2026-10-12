import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent.knowledge_base_manager import (
    KnowledgeBaseManager,
    build_knowledge_context_block,
    create_configured_knowledge_manager,
    inject_knowledge_provider_tools,
)
from agent.knowledge_provider import (
    KnowledgeBaseProvider,
    KnowledgeCitation,
    KnowledgeEvidence,
    KnowledgeRequest,
    KnowledgeSearchResult,
)
from agent.knowledge_registry import (
    KnowledgeProviderRegistry,
    create_builtin_knowledge_registry,
)
from agent.retrieval_scope import ProviderScope
from agent.turn_context import compose_user_api_content


class FakeKnowledgeProvider(KnowledgeBaseProvider):
    def __init__(self, *, name="fake-kb", fail_search=False, available=True):
        self.name = name
        self.fail_search = fail_search
        self.available = available
        self.calls = []
        self.knowledge_configs = []
        self.resources = {}

    @property
    def knowledge_name(self):
        return self.name

    def knowledge_is_available(self):
        return self.available

    def initialize_knowledge(self, scope, **kwargs):
        self.calls.append(("initialize", scope, kwargs))

    def configure_knowledge(self, config):
        self.knowledge_configs.append(dict(config))

    def shutdown_knowledge(self):
        self.calls.append(("shutdown",))

    def search_knowledge(self, query, *, scope):
        self.calls.append(("search", query, scope))
        if self.fail_search:
            raise RuntimeError("offline")
        return KnowledgeSearchResult(
            content=f"doc for {query}",
            provider=self.knowledge_name,
            citations=[KnowledgeCitation(uri="doc://guide", resource_id="guide")],
        )

    def ingest_resource(self, resource, *, scope):
        resource_id = resource["id"]
        self.resources[(scope.user_id, scope.project_id, resource_id)] = resource
        return resource_id

    def update_resource(self, resource_id, resource, *, scope):
        self.resources[(scope.user_id, scope.project_id, resource_id)] = resource
        return resource_id

    def delete_resource(self, resource_id, *, scope):
        return self.resources.pop((scope.user_id, scope.project_id, resource_id))

    def rebuild_index(self, *, scope):
        return {"scope": scope.project_id, "rebuilt": True}

    def get_knowledge_tool_schemas(self):
        return [{"name": "kb_search", "description": "search", "parameters": {}}]

    def handle_knowledge_tool_call(self, tool_name, args, **kwargs):
        return json.dumps({"tool": tool_name, "query": args.get("query")})


def _scope(user="u1", session="s1", project="p1"):
    return ProviderScope(user_id=user, session_id=session, project_id=project)


def test_empty_and_disabled_knowledge_base_are_noops():
    assert KnowledgeBaseManager().search("question") == KnowledgeSearchResult()
    provider = FakeKnowledgeProvider()
    manager = KnowledgeBaseManager(enabled=False)
    assert manager.add_provider(provider) is False
    assert manager.get_all_tool_schemas() == []


def test_search_preserves_citation_and_runtime_scope_isolation():
    provider = FakeKnowledgeProvider()
    manager = KnowledgeBaseManager(scope=_scope())
    assert manager.add_provider(provider)

    first = manager.search("alpha", session_id="s-a")
    second = manager.search(
        "alpha", scope=_scope(user="u2", session="s-b", project="p2")
    )

    assert first.citations[0].uri == "doc://guide"
    assert provider.calls[0][2] == _scope(session="s-a")
    assert provider.calls[1][2] == _scope(user="u2", session="s-b", project="p2")
    assert provider.calls[0][2] != provider.calls[1][2]


def test_resource_lifecycle_is_scoped_and_does_not_write_memory():
    provider = FakeKnowledgeProvider()
    manager = KnowledgeBaseManager(scope=_scope())
    manager.add_provider(provider)

    assert manager.ingest({"id": "r1", "body": "v1"}) == "r1"
    assert manager.update("r1", {"id": "r1", "body": "v2"}) == "r1"
    assert manager.rebuild() == {"scope": "p1", "rebuilt": True}
    deleted = manager.delete("r1")

    assert deleted["body"] == "v2"
    assert provider.resources == {}


def test_provider_failure_degrades_without_raising():
    manager = KnowledgeBaseManager(scope=_scope())
    manager.add_provider(FakeKnowledgeProvider(fail_search=True))

    result = manager.search("question")

    assert result.degraded is True
    assert result.error == "offline"
    assert result.content == ""


def test_provider_error_is_redacted_before_returning_to_caller():
    class SecretFailureProvider(EvidenceProvider):
        def retrieve_knowledge(self, request):
            return KnowledgeSearchResult(
                provider=self.knowledge_name,
                degraded=True,
                error="OPENAI_API_KEY=sk-test-secret-value-123456789",
            )

    manager = KnowledgeBaseManager()
    manager.add_provider(SecretFailureProvider(name="llm-wiki"))

    result = manager.search("question", backend="llm-wiki")

    assert "sk-test-secret-value" not in result.error
    assert "***" in result.error or "REDACT" in result.error.upper()


def test_tool_routing_and_injection_are_independent():
    provider = FakeKnowledgeProvider()
    manager = KnowledgeBaseManager(scope=_scope())
    manager.add_provider(provider)
    agent = type(
        "Agent",
        (),
        {
            "_knowledge_base_manager": manager,
            "tools": [],
            "valid_tool_names": set(),
        },
    )()

    assert inject_knowledge_provider_tools(agent) == 1
    assert manager.has_tool("kb_search")
    assert (
        json.loads(manager.handle_tool_call("kb_search", {"query": "q"}))["query"]
        == "q"
    )
    assert agent.tools[0]["function"]["name"] == "kb_search"


def test_tool_injection_preserves_existing_memory_toolset_gate():
    provider = FakeKnowledgeProvider()
    manager = KnowledgeBaseManager(scope=_scope())
    manager.add_provider(provider)
    agent = type(
        "Agent",
        (),
        {
            "_knowledge_base_manager": manager,
            "tools": [],
            "valid_tool_names": set(),
            "enabled_toolsets": [],
            "disabled_toolsets": [],
        },
    )()

    assert inject_knowledge_provider_tools(agent) == 0
    assert agent.tools == []


def test_context_order_labels_and_budget_are_stable():
    provider = FakeKnowledgeProvider()
    manager = KnowledgeBaseManager(scope=_scope())
    manager.max_context_chars = 5
    manager.add_provider(provider)
    result = manager.search("long query")

    assert result.content == "doc f"
    assert result.metadata["truncated"] is True
    composed = compose_user_api_content("question", "remembered", "plugin", result)

    assert composed.index("<memory-context>") < composed.index("<knowledge-context>")
    assert composed.index("<knowledge-context>") < composed.index("plugin")
    assert "external knowledge" in build_knowledge_context_block(result)
    assert "conversation memory" in build_knowledge_context_block(result)


@pytest.mark.parametrize(
    ("memory", "knowledge", "has_memory", "has_knowledge"),
    [
        ("", "", False, False),
        ("remembered", "", True, False),
        ("", KnowledgeSearchResult(content="document"), False, True),
        ("remembered", KnowledgeSearchResult(content="document"), True, True),
    ],
)
def test_context_composition_supports_independent_capability_modes(
    memory, knowledge, has_memory, has_knowledge
):
    composed = compose_user_api_content("question", memory, "", knowledge)
    rendered = composed or ""

    assert ("<memory-context>" in rendered) is has_memory
    assert ("<knowledge-context>" in rendered) is has_knowledge
    if not has_memory and not has_knowledge:
        assert composed is None


def test_registry_normalizes_names_and_creates_fresh_providers():
    registry = KnowledgeProviderRegistry()
    registry.register(
        "LLM_WIKI", lambda config: FakeKnowledgeProvider(name=config["name"])
    )

    first = registry.create("llm-wiki", {"name": "llm-wiki"})
    second = registry.create("LLM_WIKI", {"name": "llm-wiki"})

    assert registry.names() == ["llm-wiki"]
    assert first.knowledge_name == "llm-wiki"
    assert second is not first
    with pytest.raises(ValueError, match="already registered"):
        registry.register("llm-wiki", lambda config: FakeKnowledgeProvider())
    with pytest.raises(ValueError, match="Unknown knowledge provider"):
        registry.create("missing")


def test_builtin_registry_constructs_real_providers(tmp_path, monkeypatch):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    for name in ("SCHEMA.md", "index.md", "log.md"):
        (wiki / name).write_text(f"# {name}\n", encoding="utf-8")
    monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://127.0.0.1:1933")
    registry = create_builtin_knowledge_registry()

    llm_wiki = registry.create("llm-wiki", {"root": str(wiki)})
    openviking = registry.create(
        "openviking",
        {"target_uri": "viking://resources/frozen"},
    )

    assert llm_wiki.knowledge_name == "llm-wiki"
    assert llm_wiki.knowledge_is_available() is True
    assert openviking.knowledge_name == "openviking"
    assert openviking.knowledge_is_available() is True
    assert openviking._knowledge_config["target_uri"] == (
        "viking://resources/frozen"
    )


class EvidenceProvider(FakeKnowledgeProvider):
    def retrieve_knowledge(self, request: KnowledgeRequest) -> KnowledgeSearchResult:
        self.calls.append(("retrieve", request))
        return KnowledgeSearchResult(
            provider=self.knowledge_name,
            evidence=[
                KnowledgeEvidence(
                    evidence_id=f"{self.knowledge_name}-{index}",
                    source_uri=f"doc://{self.knowledge_name}/{index}",
                    title=f"Document {index}",
                    content=value,
                    score=1.0 / index,
                    metadata={"chunk_id": str(index)},
                    provenance={"backend": self.knowledge_name},
                )
                for index, value in enumerate(("alpha", "bravo", "charlie"), start=1)
            ],
        )


def test_request_backend_overrides_session_and_profile_defaults():
    manager = KnowledgeBaseManager(default_backend="openviking")
    openviking = EvidenceProvider(name="openviking")
    llm_wiki = EvidenceProvider(name="llm-wiki")
    assert manager.add_provider(openviking)
    assert manager.add_provider(llm_wiki)
    manager.set_session_backend("llm_wiki")

    session_result = manager.search("question")
    explicit_result = manager.search("question", backend="OPENVIKING")

    assert session_result.actual_backend == "llm-wiki"
    assert explicit_result.requested_backend == "openviking"
    assert explicit_result.actual_backend == "openviking"
    assert len(llm_wiki.calls) == 1
    assert len(openviking.calls) == 1


def test_manager_enforces_top_k_and_character_budget_on_evidence():
    manager = KnowledgeBaseManager()
    manager.add_provider(EvidenceProvider(name="llm-wiki"))

    result = manager.search(
        "question",
        backend="llm-wiki",
        top_k=2,
        evidence_char_budget=7,
        trace_id="trace-1",
    )

    assert [item.content for item in result.evidence] == ["alpha", "br"]
    assert len(result.citations) == 2
    assert result.metadata["top_k_truncated"] is True
    assert result.metadata["evidence_budget_truncated"] is True
    assert result.metadata["evidence_count"] == 2
    assert result.metadata["evidence_chars"] == 7
    assert result.metadata["selection_source"] == "request"
    assert result.metadata["fallback_used"] is False
    assert len(result.metadata["query_sha256"]) == 64
    assert result.trace_id == "trace-1"
    assert result.requested_backend == result.actual_backend == "llm-wiki"


def test_unknown_explicit_backend_is_diagnostic_and_does_not_fallback():
    manager = KnowledgeBaseManager()
    provider = EvidenceProvider(name="openviking")
    manager.add_provider(provider)

    result = manager.search("question", backend="missing")

    assert result.degraded is True
    assert "missing" in result.error
    assert result.actual_backend == ""
    assert provider.calls == []


def test_concurrent_requests_do_not_share_backend_selection_state():
    manager = KnowledgeBaseManager(default_backend="openviking")
    manager.add_provider(EvidenceProvider(name="openviking"))
    manager.add_provider(EvidenceProvider(name="llm-wiki"))
    requested = ["openviking", "llm-wiki"] * 20

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda backend: manager.search("question", backend=backend),
                requested,
            )
        )

    assert [result.actual_backend for result in results] == requested
    assert manager.default_backend == "openviking"


def test_explicit_non_default_backend_is_loaded_lazily_without_changing_default():
    created = []
    registry = KnowledgeProviderRegistry()
    registry.register(
        "llm-wiki",
        lambda config: created.append(config) or EvidenceProvider(name="llm-wiki"),
    )
    manager = KnowledgeBaseManager(default_backend="openviking", scope=_scope())
    manager.add_provider(EvidenceProvider(name="openviking"), make_default=True)
    manager.configure_lazy_providers(
        registry,
        provider_configs={"llm-wiki": {"root": "wiki"}},
        init_kwargs={"platform": "cli"},
    )
    manager.initialize(platform="cli")

    result = manager.search("question", backend="llm-wiki")

    assert result.requested_backend == result.actual_backend == "llm-wiki"
    assert manager.default_backend == "openviking"
    assert created == [{"root": "wiki"}]
    llm_wiki = manager._registrations["llm-wiki"].provider
    assert llm_wiki.calls[0] == ("initialize", _scope(), {"platform": "cli"})


def test_concurrent_lazy_load_constructs_selected_backend_once():
    created = []
    registry = KnowledgeProviderRegistry()
    registry.register(
        "llm-wiki",
        lambda config: created.append(1) or EvidenceProvider(name="llm-wiki"),
    )
    manager = KnowledgeBaseManager(scope=_scope())
    manager.configure_lazy_providers(registry)
    manager.initialize()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda _: manager.search("question", backend="llm-wiki"),
                range(20),
            )
        )

    assert len(created) == 1
    assert {result.actual_backend for result in results} == {"llm-wiki"}
    assert manager.default_backend == ""


def test_configured_openviking_reuses_memory_instance_without_owning_lifecycle():
    shared = FakeKnowledgeProvider(name="openviking")
    registry = KnowledgeProviderRegistry()
    registry.register(
        "openviking",
        lambda config: pytest.fail("shared OpenViking must not be constructed twice"),
    )

    manager = create_configured_knowledge_manager(
        config={"provider": "openviking"},
        scope=_scope(),
        active_memory_provider=shared,
        init_kwargs={"platform": "cli"},
        registry=registry,
    )

    assert manager is not None
    assert manager.provider is shared
    assert shared.calls == []
    assert shared.knowledge_configs == [{}]
    manager.shutdown()
    assert shared.calls == []


def test_knowledge_provider_selection_does_not_switch_memory_provider():
    memory_provider = FakeKnowledgeProvider(name="openviking")
    llm_wiki = FakeKnowledgeProvider(name="llm-wiki")
    registry = KnowledgeProviderRegistry()
    registry.register("llm-wiki", lambda config: llm_wiki)

    manager = create_configured_knowledge_manager(
        config={"provider": "llm-wiki", "top_k": 2},
        scope=_scope(),
        active_memory_provider=memory_provider,
        init_kwargs={"platform": "cli"},
        registry=registry,
    )

    assert manager is not None
    assert manager.default_backend == "llm-wiki"
    assert manager.default_top_k == 2
    assert manager.available_backends == ["llm-wiki", "openviking"]
    assert manager.provider is llm_wiki
    assert memory_provider.calls == []
    assert llm_wiki.calls == [("initialize", _scope(), {"platform": "cli"})]
    manager.shutdown()
    assert memory_provider.calls == []
    assert llm_wiki.calls[-1] == ("shutdown",)


def test_unavailable_configured_backend_does_not_fall_back_to_memory_provider():
    memory_provider = EvidenceProvider(name="openviking")
    unavailable = FakeKnowledgeProvider(name="llm-wiki", available=False)
    registry = KnowledgeProviderRegistry()
    registry.register("llm-wiki", lambda config: unavailable)

    manager = create_configured_knowledge_manager(
        config={"provider": "llm-wiki"},
        scope=_scope(),
        active_memory_provider=memory_provider,
        registry=registry,
    )
    result = manager.search("question")

    assert result.degraded is True
    assert "llm-wiki" in result.error
    assert memory_provider.calls == []


def test_explicit_knowledge_tools_do_not_require_memory():
    provider = FakeKnowledgeProvider()
    manager = KnowledgeBaseManager(scope=_scope())
    manager.add_provider(provider)
    from types import SimpleNamespace
    agent = SimpleNamespace(_knowledge_base_manager=manager, tools=[],
                            valid_tool_names=set(), enabled_toolsets=['knowledge'],
                            disabled_toolsets=['memory'])
    assert inject_knowledge_provider_tools(agent) == 1
    assert agent.valid_tool_names == {'kb_search'}
    agent.tools=[]
    agent.disabled_toolsets=['knowledge']
    assert inject_knowledge_provider_tools(agent) == 0
