#!/usr/bin/env python3
"""Build an utterance-level label table with explicit label provenance.

The script is intentionally conservative about transcript text. By default it
uses the annotation release task context only. Full local transcripts are joined
only when --conversation-root is provided.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


ASPECT_KEYS = {
    "informativeness",
    "novelty",
    "relevance",
    "implication_scope",
    "scope",
    "CIG",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def corpus_from_conversation_id(conversation_id: str) -> str:
    return str(conversation_id).split("_", 1)[0]


def normalize_aspect(aspect: str) -> str:
    aspect_key = str(aspect).strip()
    aspect_lower = aspect_key.lower()
    if aspect_lower in {"scope", "implication_scope", "implication scope", "imsc"}:
        return "Scope"
    if aspect_lower in {"informativeness", "info", "cig"}:
        return "CIG"
    if aspect_lower in {"novelty", "novo"}:
        return "Novelty"
    if aspect_lower in {"relevance", "relv"}:
        return "Relevance"
    return aspect_key


def task_context_rows(release: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for task in release.get("tasks", []):
        for utt in task.get("target_utterances", []) or []:
            rows.append(
                {
                    "corpus": task.get("corpus_id") or corpus_from_conversation_id(task.get("conversation_id", "")),
                    "conversation_id": task.get("conversation_id"),
                    "segment_id": task.get("segment_id"),
                    "task_id": task.get("task_id"),
                    "utterance_id": str(utt.get("utterance_id")),
                    "utterance_index": utt.get("utterance_index"),
                    "utterance_speaker": utt.get("utterance_speaker"),
                    "role": utt.get("role"),
                    "utterance_text": utt.get("utterance_text"),
                    "skipped": utt.get("skipped"),
                    "context_source": "annotation_release_task",
                }
            )
    return pd.DataFrame(rows).drop_duplicates(
        subset=["conversation_id", "utterance_id", "task_id"], keep="first"
    )


def human_annotation_rows(release: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    task_lookup = {
        task.get("task_id"): task
        for task in release.get("tasks", [])
    }
    for ann in release.get("annotations", []) or []:
        task = task_lookup.get(ann.get("base_task_id")) or task_lookup.get(ann.get("task_id")) or {}
        corpus = task.get("corpus_id") or corpus_from_conversation_id(ann.get("conversation_id", ""))
        for utt in ann.get("target_utterances", []) or []:
            labels = utt.get("labels", {}) or {}
            for aspect, rating in labels.items():
                if aspect not in ASPECT_KEYS:
                    continue
                rows.append(
                    {
                        "corpus": corpus,
                        "conversation_id": ann.get("conversation_id"),
                        "segment_id": task.get("segment_id"),
                        "task_id": ann.get("task_id"),
                        "base_task_id": ann.get("base_task_id"),
                        "utterance_id": str(utt.get("utterance_id")),
                        "utterance_index": utt.get("utterance_index"),
                        "aspect": normalize_aspect(aspect),
                        "label_value": rating,
                        "label_source": "human_individual",
                        "source_id": ann.get("annotator_id"),
                        "model": None,
                        "context_type": ann.get("task_type"),
                        "label_kind": "raw_rating",
                        "n_raters": None,
                    }
                )
    return pd.DataFrame(rows)


def human_consensus_rows(path: Optional[Path]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    df = pd.read_csv(path)
    rows: List[Dict[str, Any]] = []
    for row in df.to_dict("records"):
        rows.append(
            {
                "corpus": row.get("corpus"),
                "conversation_id": row.get("conv_id"),
                "segment_id": None,
                "task_id": None,
                "base_task_id": None,
                "utterance_id": str(row.get("utterance_id")),
                "utterance_index": None,
                "aspect": normalize_aspect(str(row.get("aspect"))),
                "label_value": row.get("y_hard"),
                "label_source": "human_consensus",
                "source_id": "human_consensus",
                "model": None,
                "context_type": None,
                "label_kind": "consensus_hard",
                "n_raters": row.get("n_raters"),
                "y_cont": row.get("y_cont"),
                "w_conf": row.get("w_conf"),
                "lo": row.get("lo"),
                "hi": row.get("hi"),
                "p1": row.get("p1"),
                "p2": row.get("p2"),
                "p3": row.get("p3"),
                "p4": row.get("p4"),
            }
        )
    return pd.DataFrame(rows)


def iter_llm_prediction_rows(path: Path) -> Iterable[Dict[str, Any]]:
    data = load_json(path)
    tasks = data if isinstance(data, list) else data.get("tasks", [])
    for task in tasks:
        conversation_id = task.get("conversation_id")
        corpus = corpus_from_conversation_id(conversation_id or "")
        segment_id = task.get("segment_id")
        predictions = task.get("predictions", {}) or {}
        for model, by_context in predictions.items():
            if not isinstance(by_context, dict):
                continue
            for context_type, ratings in by_context.items():
                if not isinstance(ratings, list):
                    continue
                for item in ratings:
                    if not isinstance(item, dict):
                        continue
                    utterance_id = item.get("utterance_id")
                    utterance_index = item.get("utterance_index")
                    for aspect, value in item.items():
                        if aspect not in ASPECT_KEYS:
                            continue
                        yield {
                            "corpus": corpus,
                            "conversation_id": conversation_id,
                            "segment_id": segment_id,
                            "task_id": task.get("task_id"),
                            "base_task_id": task.get("task_id"),
                            "utterance_id": str(utterance_id) if utterance_id is not None else None,
                            "utterance_index": utterance_index,
                            "aspect": normalize_aspect(aspect),
                            "label_value": value,
                            "label_source": "llm",
                            "source_id": model,
                            "model": model,
                            "context_type": context_type,
                            "label_kind": "model_prediction",
                            "n_raters": None,
                        }


def llm_prediction_rows(paths: List[Path]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        rows.extend(iter_llm_prediction_rows(path))
    return pd.DataFrame(rows)


def load_conversation_context(conversation_root: Optional[Path], conversations: Iterable[str]) -> pd.DataFrame:
    if not conversation_root:
        return pd.DataFrame()

    rows: List[pd.DataFrame] = []
    for conversation_id in sorted(set(filter(None, conversations))):
        corpus = corpus_from_conversation_id(conversation_id)
        path = conversation_root / corpus / f"{conversation_id}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        keep = [
            c for c in [
                "utterance_id",
                "utterance_index",
                "utterance_speaker",
                "role",
                "utterance_text",
            ]
            if c in df.columns
        ]
        if not keep:
            continue
        sub = df[keep].copy()
        sub["corpus"] = corpus
        sub["conversation_id"] = conversation_id
        if "utterance_id" in sub.columns:
            sub["utterance_id"] = sub["utterance_id"].astype(str)
        rows.append(sub)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["context_source"] = "local_full_conversation"
    return out


def merge_context(labels: pd.DataFrame, task_context: pd.DataFrame, full_context: pd.DataFrame) -> pd.DataFrame:
    context = task_context.copy()
    if not full_context.empty:
        context = pd.concat([full_context, task_context], ignore_index=True, sort=False)
        context = context.drop_duplicates(
            subset=["conversation_id", "utterance_id"], keep="first"
        )

    if labels.empty:
        return context

    join_cols = ["conversation_id", "utterance_id"]
    context_cols = [
        c for c in [
            "conversation_id",
            "utterance_id",
            "utterance_index",
            "utterance_speaker",
            "role",
            "utterance_text",
            "skipped",
            "context_source",
        ]
        if c in context.columns
    ]
    merged = labels.merge(
        context[context_cols].drop_duplicates(subset=join_cols),
        on=join_cols,
        how="left",
        suffixes=("", "_context"),
    )
    if "utterance_index_context" in merged.columns:
        if "utterance_index" in merged.columns:
            merged["utterance_index"] = merged["utterance_index"].where(
                merged["utterance_index"].notna(),
                merged["utterance_index_context"],
            )
        else:
            merged["utterance_index"] = merged["utterance_index_context"]
        merged = merged.drop(columns=["utterance_index_context"])
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an integrated utterance-label table with explicit human/LLM provenance."
    )
    parser.add_argument(
        "--annotation-release",
        type=Path,
        default=Path("data/archive_local/release_export/annotations_release.json"),
    )
    parser.add_argument("--human-final-labels", type=Path, default=None)
    parser.add_argument("--llm-ratings", type=Path, nargs="*", default=[])
    parser.add_argument("--conversation-root", type=Path, default=None)
    parser.add_argument(
        "--corpus-filter",
        choices=["fora", "insq"],
        default=None,
        help="Optionally keep only one corpus in the integrated output.",
    )
    parser.add_argument(
        "--exclude-text",
        action="store_true",
        help="Drop utterance text and speaker names from the output.",
    )
    parser.add_argument("--include-unlabeled-utterances", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/release/integrated_labels.csv"))
    args = parser.parse_args()

    release = load_json(args.annotation_release)
    task_context = task_context_rows(release)

    label_frames = [
        human_annotation_rows(release),
        human_consensus_rows(args.human_final_labels),
        llm_prediction_rows(args.llm_ratings),
    ]
    labels = pd.concat([df for df in label_frames if not df.empty], ignore_index=True, sort=False)

    conversations = set(task_context.get("conversation_id", []))
    if not labels.empty and "conversation_id" in labels:
        conversations.update(labels["conversation_id"].dropna().astype(str).tolist())
    full_context = load_conversation_context(args.conversation_root, conversations)

    out = merge_context(labels, task_context, full_context)
    if args.corpus_filter and "corpus" in out.columns:
        out = out[out["corpus"].astype(str) == args.corpus_filter].copy()
    if args.exclude_text:
        out = out.drop(columns=[c for c in ["utterance_text", "utterance_speaker"] if c in out.columns])
    if args.include_unlabeled_utterances and not full_context.empty:
        labelled_keys = out[["conversation_id", "utterance_id"]].drop_duplicates()
        unlabelled = full_context.merge(labelled_keys, on=["conversation_id", "utterance_id"], how="left", indicator=True)
        unlabelled = unlabelled[unlabelled["_merge"] == "left_only"].drop(columns=["_merge"])
        if not unlabelled.empty:
            unlabelled["label_source"] = "unlabeled"
            out = pd.concat([out, unlabelled], ignore_index=True, sort=False)

    preferred = [
        "corpus",
        "conversation_id",
        "segment_id",
        "task_id",
        "base_task_id",
        "utterance_id",
        "utterance_index",
        "utterance_speaker",
        "role",
        "utterance_text",
        "aspect",
        "label_value",
        "label_source",
        "source_id",
        "model",
        "context_type",
        "label_kind",
        "n_raters",
        "y_cont",
        "w_conf",
        "lo",
        "hi",
        "p1",
        "p2",
        "p3",
        "p4",
        "skipped",
        "context_source",
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    out = out[cols]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    summary = {
        "output": str(args.output),
        "rows": int(len(out)),
        "label_source_counts": out.get("label_source", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(),
        "conversation_text_joined": bool(args.conversation_root),
    }
    summary_path = args.output.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
