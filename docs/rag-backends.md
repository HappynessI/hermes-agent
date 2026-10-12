# RAG Backends

Hermes can retrieve indexed knowledge through built-in `openviking`,
`llm-wiki`, or a knowledge backend registered by an enabled external plugin. RAG selection is independent from `memory.provider`: choosing
OpenViking for knowledge retrieval never enables OpenViking conversational
memory or changes the configured Memory provider.

## Selection order

For each turn Hermes resolves exactly one backend in this order:

1. `rag_backend` passed with that request;
2. the backend selected for the live `AIAgent` session;
3. `knowledge.provider` in the active profile;
4. the legacy knowledge-capable Memory provider, when one is already active.

Hermes does not silently fall back to the other RAG backend. An unknown,
unavailable, or failed backend produces a structured degraded result naming
the requested backend and the error.

## Configuration

Behavioral settings belong in the active profile's `config.yaml`. Keep API
keys and other credentials in the existing secret configuration.

```yaml
knowledge:
  enabled: true
  provider: openviking       # built-in name, enabled plugin backend name, or ""
  top_k: 5
  evidence_char_budget: 12000
  allow_fallback: false

  openviking:
    target_uri: viking://resources/my-corpus
    score_threshold: 0.15
    timeout_seconds: 4.0
    request_timeout_seconds: 3.0
    full_read_limit: 2
    prefer_abstract: false

  llm_wiki:
    root: C:/data/frozen-wiki
    manifest: C:/data/frozen-wiki/manifest.json
    model: ""                # empty uses the active profile model
    model_provider: ""
    max_iterations: 20
```

An LLM-Wiki root must already contain `SCHEMA.md`, `index.md`, and `log.md`.
Online retrieval is read-only and never builds or updates the Wiki. Its private
agent can only call root-confined Markdown search/read tools; absolute paths,
`..`, non-Markdown files, and symlink escapes are rejected.

OpenViking uses the existing OpenViking connection and authentication setup.
When Memory and RAG both use OpenViking, Hermes shares the same provider
instance and Memory remains its sole lifecycle owner. Otherwise Knowledge owns
and closes its independent OpenViking instance.

## Request and session selection

Python callers can select a backend for one turn:

```python
result = agent.run_conversation(
    "What does the indexed corpus say about alpha?",
    rag_backend="llm-wiki",
)
print(result["rag"]["actual_backend"])
print(result["rag"]["evidence"])
```

Or set a session preference, which later explicit request arguments can still
override:

```python
agent.set_rag_backend("openviking")
result = agent.run_conversation("Find the deployment guide")
```

The one-shot CLI exposes the same request-scoped choice:

```text
hermes --rag-backend llm-wiki -z "What does the Wiki say about alpha?"
hermes --rag-backend openviking --usage-file rag-usage.json -z "Find the guide"
```

The optional usage JSON contains the structured `rag` object. It includes the
requested and actual backend, trace ID, total latency, warnings/error,
provenance-bearing evidence, a query hash, evidence count/characters, backend
and corpus fingerprints when available, stage timings, and LLM-Wiki's extra
model/token accounting.

## Budget and failure semantics

- `top_k` is a hard evidence cap. LLM-Wiki additionally enforces it as a
  logical-source cap while pages are being read.
- `evidence_char_budget` is enforced in provider tools and again at the common
  manager boundary before evidence enters the answer context.
- Provider errors are returned as degraded RAG metadata and secret-redacted.
- `requested_backend` and `actual_backend` are never rewritten to disguise a
  failure; `fallback_used` remains false in this implementation.
- LLM-Wiki accepts exactly one JSON response object from its private agent and
  rejects citations to pages that were not actually read.

No new HTTP service is introduced. Existing API surfaces can wrap
`AIAgent.run_conversation(..., rag_backend=...)` if an HTTP entry point is
needed later.

## Verification

Run the focused suite through the canonical test runner:

```text
bash scripts/run_tests.sh tests/agent/test_knowledge_plugin_integration.py \
  tests/agent/test_knowledge_base_manager.py \
  tests/agent/test_llm_wiki_provider.py \
  tests/plugins/memory/test_openviking_provider.py \
  tests/agent/test_turn_finalizer_iteration_limit_exit.py \
  tests/hermes_cli/test_argparse_flag_propagation.py \
  tests/hermes_cli/test_oneshot_usage_file.py -q
```

The previous fixed-500 experiment remains formally invalid because it had six
evidence-budget violations and two evidence-archive failures. Do not reuse it
as proof for this integration. Run a small real-backend smoke and protocol
preflight before any new paid fixed-500 evaluation.

## External knowledge backend plugins

Install a standalone plugin under the active profile's `plugins/<plugin-id>/`
with `plugin.yaml` and `__init__.py`. Enable its plugin ID using
`hermes plugins enable <plugin-id>`. A backend name and plugin ID may differ.
Normal plugin opt-in and safe-mode rules apply; merely copying files does not
activate a backend. For the isolated fork, use `hermes-fork` for these commands.

The plugin's entry point registers a factory:

```python
def register(ctx):
    from .provider import MyKnowledgeProvider
    ctx.register_knowledge_provider("my-kb", MyKnowledgeProvider)
```

`MyKnowledgeProvider(config)` must return a fresh instance implementing
`agent.knowledge_provider.KnowledgeBaseProvider`, including `knowledge_name`
and `get_knowledge_tool_schemas()` (return `[]` for retrieval-only providers).
Implement `retrieve_knowledge(request)` to return a `KnowledgeSearchResult`
with source-bearing `KnowledgeEvidence` objects, and lifecycle methods when
connections need initialization or closing. `initialize_knowledge` receives
`ProviderScope`; the adapter must enforce appropriate remote corpus/tenant
isolation, not merely record that scope. Factory config comes from
`knowledge.my_kb` or `knowledge.my-kb` in the active profile.

Factories are registered without creating clients. The selected default is
constructed at manager setup; alternative backends are constructed only when
selected. Registration becomes visible only after the plugin's `register`
function succeeds. Duplicate normalized names, including built-in names, are
rejected. Provider instance type and name are checked when the factory runs.

Each knowledge manager captures its registry for the session. Enable/disable
plugins before starting a new Hermes process; existing sessions keep their
snapshot. No model tools are added merely by registering a factory. No change
to `memory.provider` is required.

```text
hermes-fork --rag-backend my-kb --usage-file /tmp/my-kb-usage.json -z "Question"
hermes-fork --rag-backend llm-wiki --usage-file /tmp/wiki-usage.json -z "Question"
```

The CLI accepts external backend names without eager plugin loading during
argument parsing. An unknown, disabled, misconfigured or failed backend is
reported through degraded RAG metadata; it never silently queries LLM-Wiki.
These commands demonstrate the interface: an actual `my-kb` adapter must be
installed and enabled before it can return evidence.
