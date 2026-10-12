#!/usr/bin/env python3
"""Run one safe, read-only retrieval through the public Knowledge manager."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.knowledge_base_manager import create_configured_knowledge_manager
from agent.retrieval_scope import ProviderScope


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=("openviking", "llm-wiki"))
    parser.add_argument("--query", required=True)
    parser.add_argument("--wiki-root")
    parser.add_argument("--target-uri", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--evidence-char-budget", type=int, default=12000)
    parser.add_argument("--trace-id", default="rag-retrieval-smoke")
    return parser


def main() -> int:
    args = _parser().parse_args()
    backend_config: dict[str, object] = {}
    if args.backend == "openviking":
        backend_config["target_uri"] = args.target_uri
    else:
        if not args.wiki_root:
            raise SystemExit("--wiki-root is required for llm-wiki")
        backend_config["root"] = str(Path(args.wiki_root).expanduser().resolve())

    manager = create_configured_knowledge_manager(
        config={
            "enabled": True,
            "provider": args.backend,
            "top_k": args.top_k,
            "evidence_char_budget": args.evidence_char_budget,
            args.backend.replace("-", "_"): backend_config,
        },
        scope=ProviderScope(
            session_id=f"smoke-{args.backend}",
            project_id="rag-integration-smoke",
            platform="cli",
        ),
        init_kwargs={"platform": "cli"},
    )
    if manager is None:
        raise SystemExit("Knowledge manager is disabled")
    try:
        result = manager.search(
            args.query,
            backend=args.backend,
            trace_id=args.trace_id,
        )
        summary = {
            "requested_backend": result.requested_backend,
            "actual_backend": result.actual_backend,
            "trace_id": result.trace_id,
            "degraded": result.degraded,
            "error": result.error,
            "warnings": result.warnings,
            "latency_ms": result.latency_ms,
            "evidence_count": len(result.evidence),
            "evidence_chars": sum(len(item.content) for item in result.evidence),
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "source_uri": item.source_uri,
                    "title": item.title,
                    "score": item.score,
                    "content_chars": len(item.content),
                    "source_ids": item.metadata.get("source_ids", []),
                }
                for item in result.evidence
            ],
            "metrics": {
                key: value
                for key, value in result.metadata.items()
                if key
                in {
                    "query_sha256",
                    "top_k",
                    "evidence_count",
                    "evidence_chars",
                    "selection_source",
                    "fallback_used",
                    "backend_config_fingerprint",
                    "corpus_fingerprint",
                    "runtime_skill_fingerprint",
                    "candidate_count",
                    "search_latency_ms",
                    "evidence_read_latency_ms",
                    "extra_model_calls",
                    "extra_input_tokens",
                    "extra_output_tokens",
                    "insufficient_evidence",
                }
            },
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1 if result.degraded or result.error else 0
    finally:
        manager.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
