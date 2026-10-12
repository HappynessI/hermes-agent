"""Real profile plugin discovery through the shared knowledge boundary."""

import os
from pathlib import Path

import pytest
import yaml

from agent.knowledge_base_manager import create_configured_knowledge_manager
from agent.knowledge_registry import create_knowledge_registry
from agent.retrieval_scope import ProviderScope


@pytest.fixture
def plugin_profile(monkeypatch):
    from hermes_cli import plugins

    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    monkeypatch.delenv("HERMES_SAFE_MODE", raising=False)
    home = Path(os.environ["HERMES_HOME"])
    wiki = home / "wiki"
    wiki.mkdir()
    for name in ("SCHEMA.md", "index.md", "log.md"):
        (wiki / name).write_text("# Empty test wiki\n")
    return home, manager


def install(home, name="external-kb", *, enabled=True, registration=None):
    folder = home / "plugins" / name
    folder.mkdir(parents=True)
    (folder / "plugin.yaml").write_text(yaml.safe_dump({
        "name": name, "version": "0.1.0", "description": "Test backend",
    }))
    registration = registration or 'ctx.register_knowledge_provider("external-kb", Provider)'
    (folder / "__init__.py").write_text('''
from agent.knowledge_provider import KnowledgeBaseProvider, KnowledgeEvidence, KnowledgeSearchResult

class Provider(KnowledgeBaseProvider):
    knowledge_name = "external-kb"

    def __init__(self, config):
        self.config = config
        self.scope = None
        self.closed = False

    def initialize_knowledge(self, scope, **kwargs):
        self.scope = scope

    def get_knowledge_tool_schemas(self):
        return []

    def retrieve_knowledge(self, request):
        if self.config.get("fail"):
            raise RuntimeError("backend offline")
        return KnowledgeSearchResult(provider=self.knowledge_name, evidence=[
            KnowledgeEvidence(evidence_id="test", source_uri="doc://test", content=self.config["text"])
        ])

    def shutdown_knowledge(self):
        self.closed = True

def register(ctx):
    ''' + registration + '\n')
    config = {"plugins": {"enabled": [name] if enabled else []}}
    (home / "config.yaml").write_text(yaml.safe_dump(config))


def configured(home, **extra):
    return create_configured_knowledge_manager(
        config={"enabled": True, "provider": "llm-wiki",
                "llm_wiki": {"root": str(home / "wiki")},
                "external_kb": {"text": "verified business evidence", **extra}},
        scope=ProviderScope(profile="isolated-test", hermes_home=str(home)),
    )


def test_real_plugin_lazy_routing_scope_budget_and_lifecycle(plugin_profile):
    home, pm = plugin_profile
    install(home)
    manager = configured(home)
    manager.initialize()
    assert pm._plugins["external-kb"].enabled
    assert "external-kb" not in manager.available_backends
    result = manager.search("question", backend="EXTERNAL_KB", evidence_char_budget=8)
    assert result.requested_backend == result.actual_backend == "external-kb"
    assert not result.degraded
    assert len(result.evidence[0].content) <= 8
    assert result.evidence[0].source_uri == "doc://test"
    assert manager.default_backend == "llm-wiki"
    provider = manager._registrations["external-kb"].provider
    assert provider.scope.hermes_home == str(home)
    manager.set_session_backend("external-kb")
    assert manager.search("next").actual_backend == "external-kb"
    other = configured(home)
    other.search("question", backend="external-kb")
    assert other._registrations["external-kb"].provider is not provider
    manager.shutdown()
    other.shutdown()
    assert provider.closed


@pytest.mark.parametrize("mode", ["disabled", "safe", "failed-register"])
def test_non_active_plugins_cannot_supply_backends(plugin_profile, monkeypatch, mode):
    home, pm = plugin_profile
    install(home, enabled=mode != "disabled", registration=(
        'ctx.register_knowledge_provider("external-kb", Provider)\n    raise RuntimeError("registration failed")'
        if mode == "failed-register" else None
    ))
    if mode == "safe":
        monkeypatch.setenv("HERMES_SAFE_MODE", "1")
    result = configured(home).search("question", backend="external-kb")
    assert result.degraded
    assert result.requested_backend == "external-kb"
    assert result.actual_backend != "llm-wiki"
    assert not result.evidence
    assert not pm.get_knowledge_provider_factories()


def test_failed_backend_does_not_fall_back_to_wiki(plugin_profile):
    home, _ = plugin_profile
    install(home)
    result = configured(home, fail=True).search("question", backend="external-kb")
    assert result.degraded
    assert "backend offline" in result.error
    assert result.actual_backend != "llm-wiki"
    assert not result.evidence


def test_rediscovery_removes_disabled_factories_and_safe_mode_hides_them(plugin_profile, monkeypatch):
    home, pm = plugin_profile
    install(home)
    snapshot = create_knowledge_registry()
    assert "external-kb" in snapshot.names()
    monkeypatch.setenv("HERMES_SAFE_MODE", "1")
    assert "external-kb" not in create_knowledge_registry().names()
    monkeypatch.delenv("HERMES_SAFE_MODE")
    (home / "config.yaml").write_text("plugins:\n  enabled: []\n")
    pm.discover_and_load(force=True)
    assert "external-kb" not in create_knowledge_registry().names()
    assert "external-kb" in snapshot.names()


@pytest.mark.parametrize("registration", [
    'ctx.register_knowledge_provider("LLM_WIKI", Provider)',
    'ctx.register_knowledge_provider("external-kb", None)',
    'ctx.register_knowledge_provider("external-kb", Provider)\n    ctx.register_knowledge_provider("EXTERNAL_KB", Provider)',
])
def test_invalid_or_duplicate_registration_is_atomic(plugin_profile, registration):
    home, pm = plugin_profile
    install(home, registration=registration)
    assert "external-kb" not in create_knowledge_registry().names()
    assert pm._plugins["external-kb"].error
    assert not pm.get_knowledge_provider_factories()


def test_invalid_factory_result_is_degraded(plugin_profile):
    home, _ = plugin_profile
    install(home, registration='ctx.register_knowledge_provider("external-kb", lambda config: object())')
    result = configured(home).search("question", backend="external-kb")
    assert result.degraded
    assert not result.evidence


def test_other_profile_cannot_discover_this_profiles_plugin(plugin_profile, tmp_path, monkeypatch):
    from hermes_cli import plugins

    home, _ = plugin_profile
    install(home)
    assert "external-kb" in create_knowledge_registry().names()
    other_home = tmp_path / "other-profile"
    other_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(other_home))
    monkeypatch.setattr(plugins, "_plugin_manager", plugins.PluginManager())
    assert "external-kb" not in create_knowledge_registry().names()


def test_two_plugins_cannot_replace_each_others_backend(plugin_profile):
    home, pm = plugin_profile
    install(home, "first-kb")
    install(home, "second-kb")
    (home / "config.yaml").write_text("plugins:\n  enabled: [first-kb, second-kb]\n")
    registry = create_knowledge_registry()
    assert "external-kb" in registry.names()
    loaded = [pm._plugins[key] for key in ("first-kb", "second-kb")]
    assert sum(plugin.enabled for plugin in loaded) == 1
    assert sum("already registered" in (plugin.error or "") for plugin in loaded) == 1


def test_cli_accepts_external_backend_without_loading_it():
    from hermes_cli._parser import build_top_level_parser

    parser, _, _ = build_top_level_parser()
    args = parser.parse_args(["--rag-backend", "external-kb", "-z", "question"])
    assert args.rag_backend == "external-kb"
