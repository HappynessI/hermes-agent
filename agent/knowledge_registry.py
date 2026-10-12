"""Stateless discovery and construction for knowledge providers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Dict, List

from agent.knowledge_provider import KnowledgeBaseProvider


KnowledgeProviderFactory = Callable[[Mapping[str, Any]], KnowledgeBaseProvider]


def normalize_knowledge_provider_name(name: str) -> str:
    """Return the canonical external name used by config and requests."""
    return str(name or "").strip().lower().replace("_", "-")


class KnowledgeProviderRegistry:
    """Map stable provider names to factories without holding request state."""

    def __init__(self) -> None:
        self._factories: Dict[str, KnowledgeProviderFactory] = {}

    def register(self, name: str, factory: KnowledgeProviderFactory) -> None:
        normalized = normalize_knowledge_provider_name(name)
        if not normalized:
            raise ValueError("Knowledge provider name must not be empty")
        if not callable(factory):
            raise TypeError("Knowledge provider factory must be callable")
        if normalized in self._factories:
            raise ValueError(f"Knowledge provider '{normalized}' is already registered")
        self._factories[normalized] = factory

    def names(self) -> List[str]:
        return sorted(self._factories)

    def create(
        self,
        name: str,
        config: Mapping[str, Any] | None = None,
    ) -> KnowledgeBaseProvider:
        normalized = normalize_knowledge_provider_name(name)
        factory = self._factories.get(normalized)
        if factory is None:
            available = ", ".join(self.names()) or "none"
            raise ValueError(
                f"Unknown knowledge provider '{normalized or name}'. Available: {available}"
            )
        provider = factory(dict(config or {}))
        if not isinstance(provider, KnowledgeBaseProvider):
            raise TypeError(
                f"Knowledge provider factory '{normalized}' returned "
                f"{type(provider).__name__}, expected KnowledgeBaseProvider"
            )
        actual = normalize_knowledge_provider_name(provider.knowledge_name)
        if actual != normalized:
            raise ValueError(
                f"Knowledge provider factory '{normalized}' returned provider '{actual}'"
            )
        return provider


def create_builtin_knowledge_registry() -> KnowledgeProviderRegistry:
    """Create the runtime registry without importing provider plugins eagerly."""
    registry = KnowledgeProviderRegistry()

    def _load_openviking(_config: Mapping[str, Any]) -> KnowledgeBaseProvider:
        from plugins.memory import load_memory_provider

        provider = load_memory_provider("openviking")
        if not isinstance(provider, KnowledgeBaseProvider):
            raise RuntimeError("OpenViking knowledge provider is not available")
        provider.configure_knowledge(_config)
        return provider

    def _load_llm_wiki(config: Mapping[str, Any]) -> KnowledgeBaseProvider:
        from plugins.knowledge.llm_wiki import LlmWikiKnowledgeProvider

        return LlmWikiKnowledgeProvider(config)

    def _load_utopia(config: Mapping[str, Any]) -> KnowledgeBaseProvider:
        from plugins.knowledge.utopia import UtopiaKnowledgeProvider

        return UtopiaKnowledgeProvider(config)

    registry.register("utopia", _load_utopia)
    registry.register("openviking", _load_openviking)
    registry.register("llm-wiki", _load_llm_wiki)
    return registry


def create_knowledge_registry() -> KnowledgeProviderRegistry:
    """Combine built-ins with enabled plugins, without creating backend clients.

    Discovery uses the normal profile-scoped opt-in and safe-mode gates. Each
    manager gets a snapshot so later discovery cannot mutate an active session.
    """
    from hermes_cli.plugins import discover_plugins, get_plugin_manager

    registry = create_builtin_knowledge_registry()
    discover_plugins()
    for name, factory in get_plugin_manager().get_knowledge_provider_factories().items():
        registry.register(name, factory)
    return registry
