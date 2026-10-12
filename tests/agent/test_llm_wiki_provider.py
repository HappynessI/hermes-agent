import json

import pytest

from agent.knowledge_provider import KnowledgeRequest
from agent.retrieval_scope import ProviderScope
from plugins.knowledge.llm_wiki import (
    HermesLlmWikiRunner,
    LlmWikiKnowledgeProvider,
    LlmWikiRunResult,
    _strict_agent_payload,
)
from tools.llm_wiki_tools import (
    WikiReadRecord,
    WikiRunTrace,
    normalize_wiki_path,
    resolve_wiki_path,
    wiki_run_context,
)
from tools.registry import registry as tool_registry


def _make_wiki(root):
    root.mkdir()
    (root / "SCHEMA.md").write_text("# Schema\n", encoding="utf-8")
    (root / "index.md").write_text("# Index\n", encoding="utf-8")
    (root / "log.md").write_text("# Log\n", encoding="utf-8")
    (root / "concepts").mkdir()
    (root / "concepts" / "alpha.md").write_text(
        "---\n"
        "title: Alpha\n"
        "type: concept\n"
        "source_article_ids: [source-a]\n"
        "---\n\n"
        "Alpha evidence is documented here.\n",
        encoding="utf-8",
    )
    (root / "concepts" / "broad.md").write_text(
        "---\n"
        "title: Broad\n"
        "type: concept\n"
        "source_article_ids: [source-b, source-c]\n"
        "---\n\n"
        "Broad evidence.\n",
        encoding="utf-8",
    )


def test_wiki_path_normalization_accepts_legacy_prefix_and_rejects_escape(tmp_path):
    wiki = tmp_path / "wiki"
    _make_wiki(wiki)

    assert normalize_wiki_path("wiki/concepts/alpha.md") == "concepts/alpha.md"
    assert (
        resolve_wiki_path(
            wiki.resolve(),
            "wiki/concepts/alpha.md",
            require_file=True,
        )[1]
        == "concepts/alpha.md"
    )
    for unsafe in ("../secret.md", "/etc/passwd", "C:\\secret.md"):
        with pytest.raises(ValueError, match="Unsafe Wiki path"):
            normalize_wiki_path(unsafe)


def test_wiki_path_resolution_rejects_symlink_escape(tmp_path):
    wiki = tmp_path / "wiki"
    _make_wiki(wiki)
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    link = wiki / "concepts" / "outside.md"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="escapes configured root"):
        resolve_wiki_path(wiki.resolve(), "concepts/outside.md", require_file=True)


def test_read_tools_enforce_logical_source_and_character_budgets(tmp_path):
    wiki = tmp_path / "wiki"
    _make_wiki(wiki)

    with wiki_run_context(
        "task-1",
        wiki,
        top_k=1,
        evidence_char_budget=12,
    ) as state:
        orientation = json.loads(state.read("SCHEMA.md", offset=1, limit=20))
        evidence = json.loads(state.read("concepts/alpha.md", offset=1, limit=50))
        rejected = json.loads(state.read("concepts/broad.md", offset=1, limit=50))
        trace = state.snapshot()

    assert orientation["source_ids"] == []
    assert evidence["source_ids"] == ["source-a"]
    assert len(evidence["content"]) == 12
    assert "source budget" in rejected["error"]
    assert trace.source_ids == ("source-a",)
    assert [item.path for item in trace.evidence] == ["concepts/alpha.md"]
    assert trace.warnings


def test_strict_agent_payload_rejects_wrapped_or_incomplete_json():
    valid = _strict_agent_payload(
        '{"answer":"yes","answer_citations":[],"insufficient_evidence":false}'
    )
    assert valid["answer"] == "yes"
    with pytest.raises(ValueError, match="exactly one JSON"):
        _strict_agent_payload(
            'Result: {"answer":"yes","answer_citations":[],"insufficient_evidence":false}'
        )
    with pytest.raises(ValueError, match="missing fields"):
        _strict_agent_payload('{"answer":"yes"}')


def test_real_runner_loads_bundled_skill_and_uses_private_read_tools(
    tmp_path, monkeypatch
):
    wiki = tmp_path / "wiki"
    _make_wiki(wiki)
    captured = {}

    class FakeAgent:
        def run_conversation(self, prompt, *, task_id):
            captured["prompt"] = prompt
            for path in ("SCHEMA.md", "index.md", "log.md", "concepts/alpha.md"):
                payload = json.loads(
                    tool_registry.dispatch(
                        "llm_wiki_read",
                        {"path": path, "offset": 1, "limit": 100},
                        task_id=task_id,
                    )
                )
                assert "error" not in payload
            return {
                "final_response": json.dumps(
                    {
                        "answer": "Alpha is supported.",
                        "answer_citations": ["wiki/concepts/alpha.md"],
                        "insufficient_evidence": False,
                    }
                ),
                "provider_calls_total": 2,
                "input_tokens": 50,
                "output_tokens": 10,
            }

        def shutdown_memory_provider(self):
            captured["shutdown"] = True

        def close(self):
            captured["closed"] = True

    def fake_builder(**kwargs):
        captured["builder"] = kwargs
        return FakeAgent()

    monkeypatch.setattr(
        "hermes_cli.oneshot.build_noninteractive_agent",
        fake_builder,
    )
    request = KnowledgeRequest(
        query="What is alpha?",
        backend="llm-wiki",
        top_k=1,
        evidence_char_budget=500,
        trace_id="trace-real-path",
    )

    result = HermesLlmWikiRunner().run(request, root=wiki.resolve())

    assert result.answer_citations == ("concepts/alpha.md",)
    assert result.trace.pages_read[-1] == "concepts/alpha.md"
    assert result.trace.source_ids == ("source-a",)
    assert captured["builder"]["toolsets"] == ["llm-wiki-internal"]
    assert captured["builder"]["skip_memory"] is True
    assert captured["builder"]["skip_knowledge"] is True
    assert captured["shutdown"] is captured["closed"] is True


class _FakeRunner:
    def __init__(self, trace):
        self.trace = trace
        self.calls = []

    def run(self, request, *, root):
        self.calls.append((request, root))
        return LlmWikiRunResult(
            answer="Alpha is supported.",
            answer_citations=("concepts/alpha.md",),
            insufficient_evidence=False,
            trace=self.trace,
            model_calls=3,
            input_tokens=100,
            output_tokens=20,
        )


def test_provider_maps_only_actual_read_trace_to_structured_evidence(tmp_path):
    wiki = tmp_path / "wiki"
    _make_wiki(wiki)
    trace = WikiRunTrace(
        pages_searched=(".",),
        pages_read=("SCHEMA.md", "index.md", "concepts/alpha.md"),
        evidence=(
            WikiReadRecord(
                path="concepts/alpha.md",
                content="Alpha evidence.",
                source_ids=("source-a",),
                metadata={"title": "Alpha", "type": "concept"},
            ),
        ),
        source_ids=("source-a",),
        search_calls=1,
        read_calls=3,
        warnings=(),
    )
    runner = _FakeRunner(trace)
    provider = LlmWikiKnowledgeProvider({"root": str(wiki)}, runner=runner)
    scope = ProviderScope(session_id="s1", project_id="p1")
    provider.initialize_knowledge(scope)

    result = provider.retrieve_knowledge(
        KnowledgeRequest(
            query="What is alpha?",
            backend="llm-wiki",
            top_k=1,
            scope=scope,
            evidence_char_budget=100,
            trace_id="trace-1",
        )
    )

    assert result.provider == "llm-wiki"
    assert result.degraded is False
    assert len(result.evidence) == 1
    assert result.evidence[0].source_uri == "wiki://concepts/alpha.md"
    assert result.evidence[0].metadata["source_ids"] == ["source-a"]
    assert result.evidence[0].provenance["trace_id"] == "trace-1"
    assert result.metadata["extra_model_calls"] == 3
    assert result.metadata["agent_answer_citations"] == ["concepts/alpha.md"]
    assert runner.calls[0][1] == wiki.resolve()


def test_provider_returns_typed_degradation_for_runner_protocol_error(tmp_path):
    wiki = tmp_path / "wiki"
    _make_wiki(wiki)

    class BrokenRunner:
        def run(self, request, *, root):
            raise ValueError("invalid agent JSON")

    provider = LlmWikiKnowledgeProvider({"root": str(wiki)}, runner=BrokenRunner())
    provider.initialize_knowledge(ProviderScope())
    result = provider.retrieve_knowledge(KnowledgeRequest(query="question"))

    assert result.degraded is True
    assert result.error == "invalid agent JSON"
    assert result.evidence == []
