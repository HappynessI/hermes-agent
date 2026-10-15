import pytest

from agent.knowledge_base_manager import create_configured_knowledge_manager
from agent.knowledge_registry import create_builtin_knowledge_registry
from agent.retrieval_scope import ProviderScope


@pytest.mark.parametrize("config, expected", [({}, True), ({"prefetch": False}, False)])
def test_prefetch_configuration_preserves_knowledge_manager(config, expected):
    manager = create_configured_knowledge_manager(
        config=config,
        scope=ProviderScope(),
        registry=create_builtin_knowledge_registry(),
    )
    assert manager is not None
    assert manager.enabled
    assert manager.prefetch is expected


@pytest.mark.parametrize("value", ["false", 0, None])
def test_invalid_prefetch_configuration_is_rejected(value):
    with pytest.raises(ValueError, match="knowledge.prefetch"):
        create_configured_knowledge_manager(
            config={"prefetch": value}, scope=ProviderScope()
        )
