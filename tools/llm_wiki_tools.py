"""Read-only, root-confined tools for agentic LLM-Wiki retrieval."""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, Iterator, List, Optional

import yaml

from tools.registry import registry, tool_error


_TOOLSET = "llm-wiki-internal"
_MAX_SEARCH_RESULTS = 100
_MAX_READ_LINES = 2000
_MAX_PAGE_CHARS = 100_000


@dataclass(frozen=True)
class WikiReadRecord:
    path: str
    content: str
    source_ids: tuple[str, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WikiRunTrace:
    pages_searched: tuple[str, ...]
    pages_read: tuple[str, ...]
    evidence: tuple[WikiReadRecord, ...]
    source_ids: tuple[str, ...]
    search_calls: int
    read_calls: int
    warnings: tuple[str, ...]
    search_latency_ms: float = 0.0
    read_latency_ms: float = 0.0


def _frontmatter(content: str) -> Dict[str, Any]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    try:
        closing = next(
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        )
    except StopIteration:
        return {}
    try:
        value = yaml.safe_load("\n".join(lines[1:closing])) or {}
    except yaml.YAMLError:
        return {}
    return value if isinstance(value, dict) else {}


def _string_list(value: Any) -> List[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _source_ids(relative_path: str, metadata: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    article_id = metadata.get("article_id")
    if isinstance(article_id, str) and article_id.strip():
        values.append(article_id.strip())
    values.extend(_string_list(metadata.get("source_article_ids")))
    if not values:
        values.extend(_string_list(metadata.get("sources")))
    if not values and relative_path.startswith("raw/"):
        values.append(relative_path)
    return list(dict.fromkeys(values))


def normalize_wiki_path(raw_path: str) -> str:
    """Normalize a model-supplied Wiki path and reject escape syntax."""
    raw = str(raw_path or "").strip().replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    if raw == "wiki":
        raw = ""
    elif raw.startswith("wiki/"):
        raw = raw[5:]
    posix = PurePosixPath(raw or ".")
    if (
        not raw
        or posix.is_absolute()
        or PureWindowsPath(raw).is_absolute()
        or any(part in {"", ".."} for part in posix.parts)
    ):
        raise ValueError(f"Unsafe Wiki path: {raw_path!r}")
    return posix.as_posix()


def resolve_wiki_path(
    root: Path,
    raw_path: str,
    *,
    require_file: bool = False,
    require_directory: bool = False,
) -> tuple[Path, str]:
    relative = normalize_wiki_path(raw_path)
    candidate = (root / Path(*PurePosixPath(relative).parts)).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Wiki path escapes configured root: {raw_path!r}") from exc
    if require_file and (not candidate.is_file() or candidate.suffix.lower() != ".md"):
        raise ValueError(f"Wiki page is not a Markdown file: {raw_path!r}")
    if require_directory and not candidate.is_dir():
        raise ValueError(f"Wiki search path is not a directory: {raw_path!r}")
    return candidate, candidate.relative_to(root).as_posix()


class _WikiRunState:
    def __init__(
        self,
        root: Path,
        *,
        top_k: int,
        evidence_char_budget: int,
    ) -> None:
        self.root = root.resolve(strict=True)
        self.top_k = max(1, int(top_k))
        self.evidence_char_budget = max(1, int(evidence_char_budget))
        self._remaining_chars = self.evidence_char_budget
        self._source_ids: List[str] = []
        self._evidence: Dict[str, WikiReadRecord] = {}
        self._pages_searched: List[str] = []
        self._pages_read: List[str] = []
        self._warnings: List[str] = []
        self.search_calls = 0
        self.read_calls = 0
        self.search_latency_ms = 0.0
        self.read_latency_ms = 0.0
        self._lock = threading.RLock()

    def search(
        self,
        pattern: str,
        *,
        path: str,
        file_glob: str,
        limit: int,
    ) -> str:
        needle = str(pattern or "").strip()
        if not needle:
            return tool_error("pattern is required")
        started = time.perf_counter()
        search_root, search_relative = resolve_wiki_path(
            self.root,
            path or ".",
            require_directory=True,
        )
        glob = str(file_glob or "*.md").strip()
        if not glob or "/" in glob or "\\" in glob or ".." in glob:
            return tool_error("file_glob must be a filename-only pattern")
        bounded_limit = max(1, min(int(limit or 20), _MAX_SEARCH_RESULTS))
        matches: List[Dict[str, Any]] = []
        folded = needle.casefold()
        for candidate in sorted(search_root.rglob(glob)):
            if len(matches) >= bounded_limit:
                break
            if not candidate.is_file() or candidate.suffix.lower() != ".md":
                continue
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(self.root)
                content = resolved.read_text(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                continue
            for line_number, line in enumerate(content.splitlines(), start=1):
                if folded not in line.casefold():
                    continue
                matches.append({
                    "path": resolved.relative_to(self.root).as_posix(),
                    "line": line_number,
                    "excerpt": line[:500],
                })
                if len(matches) >= bounded_limit:
                    break
        with self._lock:
            self.search_calls += 1
            self.search_latency_ms += (time.perf_counter() - started) * 1000
            self._pages_searched.append(search_relative)
        return json.dumps(
            {"pattern": needle, "matches": matches, "count": len(matches)},
            ensure_ascii=False,
        )

    def read(self, path: str, *, offset: int, limit: int) -> str:
        started = time.perf_counter()
        candidate, relative = resolve_wiki_path(
            self.root,
            path,
            require_file=True,
        )
        content = candidate.read_text(encoding="utf-8", errors="replace")
        metadata = _frontmatter(content)
        source_ids = _source_ids(relative, metadata)
        start = max(1, int(offset or 1))
        bounded_limit = max(1, min(int(limit or 500), _MAX_READ_LINES))
        lines = content.splitlines()
        snippet = "\n".join(lines[start - 1 : start - 1 + bounded_limit])
        snippet = snippet[:_MAX_PAGE_CHARS]

        with self._lock:
            new_source_ids = [
                source_id
                for source_id in source_ids
                if source_id not in self._source_ids
            ]
            if len(self._source_ids) + len(new_source_ids) > self.top_k:
                return tool_error(
                    "Reading this page would exceed the configured logical "
                    f"source budget ({self.top_k}). Stop or choose a narrower page."
                )
            if source_ids:
                if self._remaining_chars <= 0:
                    return tool_error(
                        "The configured evidence character budget is exhausted."
                    )
                if len(snippet) > self._remaining_chars:
                    snippet = snippet[: self._remaining_chars]
                    self._warnings.append(
                        f"Evidence from {relative} was truncated to the character budget"
                    )
                self._remaining_chars -= len(snippet)
                self._source_ids.extend(new_source_ids)
                previous = self._evidence.get(relative)
                if previous is None:
                    self._evidence[relative] = WikiReadRecord(
                        path=relative,
                        content=snippet,
                        source_ids=tuple(source_ids),
                        metadata=metadata,
                    )
                elif snippet and snippet not in previous.content:
                    self._evidence[relative] = WikiReadRecord(
                        path=relative,
                        content=f"{previous.content}\n\n{snippet}",
                        source_ids=previous.source_ids,
                        metadata=previous.metadata,
                    )
            self.read_calls += 1
            self.read_latency_ms += (time.perf_counter() - started) * 1000
            self._pages_read.append(relative)

        return json.dumps(
            {
                "path": relative,
                "offset": start,
                "line_count": len(snippet.splitlines()),
                "source_ids": source_ids,
                "content": snippet,
            },
            ensure_ascii=False,
        )

    def snapshot(self) -> WikiRunTrace:
        with self._lock:
            return WikiRunTrace(
                pages_searched=tuple(self._pages_searched),
                pages_read=tuple(self._pages_read),
                evidence=tuple(self._evidence.values()),
                source_ids=tuple(self._source_ids),
                search_calls=self.search_calls,
                read_calls=self.read_calls,
                warnings=tuple(self._warnings),
                search_latency_ms=self.search_latency_ms,
                read_latency_ms=self.read_latency_ms,
            )


_RUNS: Dict[str, _WikiRunState] = {}
_RUNS_LOCK = threading.RLock()


@contextmanager
def wiki_run_context(
    task_id: str,
    root: Path,
    *,
    top_k: int,
    evidence_char_budget: int,
) -> Iterator[_WikiRunState]:
    key = str(task_id or "").strip()
    if not key:
        raise ValueError("task_id is required for an LLM-Wiki run")
    state = _WikiRunState(
        root,
        top_k=top_k,
        evidence_char_budget=evidence_char_budget,
    )
    with _RUNS_LOCK:
        if key in _RUNS:
            raise RuntimeError(f"LLM-Wiki task is already active: {key}")
        _RUNS[key] = state
    try:
        yield state
    finally:
        with _RUNS_LOCK:
            _RUNS.pop(key, None)


def _state_for(task_id: str) -> Optional[_WikiRunState]:
    with _RUNS_LOCK:
        return _RUNS.get(str(task_id or ""))


def _handle_search(args: dict, **kwargs: Any) -> str:
    state = _state_for(kwargs.get("task_id", ""))
    if state is None:
        return tool_error("No active LLM-Wiki retrieval context")
    try:
        return state.search(
            args.get("pattern", ""),
            path=args.get("path", "."),
            file_glob=args.get("file_glob", "*.md"),
            limit=args.get("limit", 20),
        )
    except (OSError, ValueError) as exc:
        return tool_error(str(exc))


def _handle_read(args: dict, **kwargs: Any) -> str:
    state = _state_for(kwargs.get("task_id", ""))
    if state is None:
        return tool_error("No active LLM-Wiki retrieval context")
    try:
        return state.read(
            args.get("path", ""),
            offset=args.get("offset", 1),
            limit=args.get("limit", 500),
        )
    except (OSError, ValueError) as exc:
        return tool_error(str(exc))


_SEARCH_SCHEMA = {
    "name": "llm_wiki_search",
    "description": "Search text only inside the configured read-only LLM-Wiki.",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "default": "."},
            "file_glob": {"type": "string", "default": "*.md"},
            "limit": {"type": "integer", "default": 20},
        },
        "required": ["pattern"],
    },
}

_READ_SCHEMA = {
    "name": "llm_wiki_read",
    "description": "Read a Markdown page inside the configured read-only LLM-Wiki.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "default": 1},
            "limit": {"type": "integer", "default": 500},
        },
        "required": ["path"],
    },
}


registry.register(
    name="llm_wiki_search",
    toolset=_TOOLSET,
    schema=_SEARCH_SCHEMA,
    handler=_handle_search,
    description=_SEARCH_SCHEMA["description"],
    emoji="🔎",
    max_result_size_chars=50_000,
)
registry.register(
    name="llm_wiki_read",
    toolset=_TOOLSET,
    schema=_READ_SCHEMA,
    handler=_handle_read,
    description=_READ_SCHEMA["description"],
    emoji="📖",
    max_result_size_chars=100_000,
)
