#!/usr/bin/env python3
"""Archive dense prior-memory embeddings outside publication-facing metadata."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_SOURCE_GLOBS = [
    "data/archive_local/cache/insq_*_meta.json",
    "data/archive_local/meta/insq_*_meta.json",
]
DEFAULT_OUTPUT = Path("data/archive_local/embeddings/insq_prior_memory_embeddings.jsonl")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _conversation_id_from_path(path: Path) -> str:
    name = path.name
    for suffix in ("_meta_checkpoint.json", "_meta.json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _iter_embedding_records(meta_path: Path) -> Iterable[Dict[str, Any]]:
    meta = _load_json(meta_path)
    conversation_id = str(meta.get("conversation_id") or _conversation_id_from_path(meta_path))
    if not conversation_id.startswith("insq_") and meta_path.name.startswith("insq_"):
        conversation_id = f"insq_{conversation_id}"

    for memory in meta.get("prior_memory", []) or []:
        if not isinstance(memory, dict) or "embedding" not in memory:
            continue
        yield {
            "conversation_id": conversation_id,
            "memory_id": memory.get("id"),
            "memory": memory.get("memory"),
            "embedding": memory.get("embedding"),
        }


def archive_embeddings(source_globs: List[str], output_jsonl: Path) -> Dict[str, int]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    records: List[Dict[str, Any]] = []

    for pattern in source_globs:
        for meta_file in sorted(glob.glob(pattern)):
            for record in _iter_embedding_records(Path(meta_file)):
                key = (
                    record.get("conversation_id"),
                    record.get("memory_id"),
                    record.get("memory"),
                )
                if key in seen:
                    continue
                seen.add(key)
                records.append(record)

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "records": len(records),
        "conversations": len({record["conversation_id"] for record in records}),
    }


def strip_embeddings(meta_globs: List[str]) -> Dict[str, int]:
    files_changed = 0
    embeddings_removed = 0

    for pattern in meta_globs:
        for meta_file in sorted(glob.glob(pattern)):
            path = Path(meta_file)
            meta = _load_json(path)
            changed = False
            for memory in meta.get("prior_memory", []) or []:
                if isinstance(memory, dict) and "embedding" in memory:
                    memory.pop("embedding", None)
                    changed = True
                    embeddings_removed += 1
            if changed:
                with path.open("w", encoding="utf-8") as handle:
                    json.dump(meta, handle, ensure_ascii=False, indent=2)
                files_changed += 1

    return {"files_changed": files_changed, "embeddings_removed": embeddings_removed}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract dense prior-memory embeddings from local archived metadata into "
            "an ignored JSONL sidecar. Optionally strip embeddings from public metadata."
        )
    )
    parser.add_argument(
        "--source-glob",
        action="append",
        default=None,
        help="Metadata glob to scan for prior_memory embeddings. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Ignored sidecar JSONL path for archived embeddings.",
    )
    parser.add_argument(
        "--strip-public-glob",
        action="append",
        default=[],
        help="Optional public metadata glob whose prior_memory.embedding fields should be removed.",
    )
    args = parser.parse_args()

    source_globs = args.source_glob or DEFAULT_SOURCE_GLOBS
    archive_summary = archive_embeddings(source_globs, args.output_jsonl)
    strip_summary = strip_embeddings(args.strip_public_glob)
    print(
        json.dumps(
            {
                "output_jsonl": str(args.output_jsonl),
                "archive": archive_summary,
                "strip": strip_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
