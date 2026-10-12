# Dual RAG Backend Integration Design

## Scope

This change extends Hermes' existing knowledge boundary so callers can select
either `openviking` or `llm-wiki` without changing the configured memory
provider. It does not change Memory, Skills evolution, PlanIR, Goal VERIFY, or
the default behavior of profiles that do not configure a knowledge provider.

## Implemented state

- `KnowledgeBaseProvider` is the typed resource/knowledge contract, including
  structured requests, evidence, provenance, diagnostics, and configuration.
- `KnowledgeBaseManager` resolves one backend per retrieval from registered,
  lazily initialized providers and enforces the final evidence budget.
- Agent initialization registers Knowledge independently from Memory while
  retaining the legacy dual-capability memory-provider adaptation as the last
  resolution option.
- OpenViking implements both contracts. Knowledge shares the active Memory
  instance only when both sides explicitly select OpenViking; otherwise the
  Knowledge manager owns an independent instance.
- LLM-Wiki is exposed as a read-only, agentic knowledge provider with a private
  search/read toolset. Its runtime adapter is separate from the evaluation
  harness and records only pages actually read as evidence.

## Target behavior

Backend resolution follows this order:

1. explicit request backend;
2. session backend;
3. profile `knowledge.provider`;
4. the legacy dual-capability memory-provider adaptation.

Both providers may be registered, but one retrieval selects exactly one. A
request must never mutate process-global provider state or `memory.provider`.
Fallback is disabled unless a future explicit option enables it. Every result
records requested and actual backend names.

## Runtime shape

The implementation extends the existing Knowledge modules:

```text
caller / Agent turn
        |
        v
KnowledgeRequest
        |
        v
KnowledgeBaseManager ---- KnowledgeProviderRegistry
        |                         |
        |              +----------+----------+
        |              |                     |
        v              v                     v
budget enforcement  OpenViking           LLM-Wiki
        |              adapter             adapter
        +------------------+------------------+
                           v
              KnowledgeSearchResult
              + structured evidence
```

The model-facing system prompt and tool schemas remain stable for a live
conversation. Request-level selection changes only the knowledge sidecar for
that turn, which is persisted through the existing `api_content` mechanism.
The first version does not add a new core model tool.

## Contract

`KnowledgeRequest` carries the query, selected backend, `top_k`, evidence
character budget, and `ProviderScope`. `KnowledgeEvidence` carries a stable
evidence ID, source URI, content, optional score/title, metadata, and
provenance. `KnowledgeSearchResult` retains the existing content/citations
fields for compatibility and adds evidence, backend identity, warnings,
latency, and trace metadata.

The manager, not prompts, enforces `top_k` and the evidence character budget.
Legacy providers that only return `content` remain supported during migration.

## OpenViking lifecycle

- If Memory and Knowledge both select OpenViking, register the same instance
  with Knowledge as non-owning. `MemoryManager` initializes and closes it.
- If only Knowledge selects OpenViking, create one independently owned instance
  and initialize/close it through the Knowledge lifecycle.
- Selecting OpenViking for Knowledge never writes `memory.provider`, enables
  memory writes, or changes memory scope.

## LLM-Wiki boundary

The adapter is query-only in this integration. It operates on a configured,
frozen Wiki root and never writes `log.md`, query pages, or source files during
retrieval. It must:

- resolve every reported path beneath the Wiki root;
- reject absolute paths, `..`, and symlink escapes;
- derive provenance from pages actually read rather than model assertions;
- validate exactly one structured result;
- enforce evidence budgets after deriving the actual source set;
- report its additional model calls, tokens, and latency.

Build/ingest remains a separate workflow. Evaluation dataset, gold, judge, and
statistics code stays in `hermes-wixqa-eval`.

## Public entry points

The core Python retrieval API is the source of truth. The first external
wrapper will be a CLI/one-shot backend option. An HTTP route, if required, must
wrap the same API in Hermes' existing server rather than start another server.

## Verification order

1. Contract, registry, selection, budgets, and concurrency tests with fakes.
2. OpenViking shared/independent lifecycle and Memory/RAG combination tests.
3. LLM-Wiki path, schema, provenance, and read-only tests.
4. Real registration-path tests under a temporary `HERMES_HOME`.
5. Live smoke for both providers.
6. 20-question protocol preflight, repeated latency panel, multi-format
   regression, then fixed-500 only after zero budget/archive failures.
