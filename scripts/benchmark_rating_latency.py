#!/usr/bin/env python3
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import pandas as pd


def _load_segment_metadata(corpus: str, episode_id: str):
    """Prefer release metadata; fall back to archived checkpoints if present."""
    release_meta = Path("data") / "raw" / corpus / f"{episode_id}_meta.json"
    if release_meta.exists():
        with release_meta.open() as f:
            meta = json.load(f)
        segments = meta.get("segmentation", {}).get("segments")
        if segments:
            return meta, segments

    archive_meta = Path("data") / "archive_local" / "processed_segments" / "openai" / f"{episode_id}_meta_checkpoint.json"
    if archive_meta.exists():
        with archive_meta.open() as f:
            archived = json.load(f)
        with release_meta.open() as f:
            meta = json.load(f)
        return meta, archived["segmentation"]["segments"]

    raise FileNotFoundError(
        f"No segment metadata found for {episode_id}. Expected {release_meta} "
        f"or {archive_meta}."
    )


def _mean_std(vals):
    if not vals:
        return None, None
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


async def run_one_model(
    backend,
    dialogue_df,
    topic,
    segments_meta,
    segment_ids,
    prior_window=5,
    gemini_thinking_budget=128,
):
    from dialogue_rating import _get_llm, rate_text
    from memory.multiparty_memory import generate_formatted_memories
    from task_utility import slice_to_text

    llm = _get_llm(backend, use_async=True)
    provider, model = backend
    rows = []
    for sidx in segment_ids:
        start, end = segments_meta[sidx]["intervals"]
        current_dialogue = slice_to_text(dialogue_df, start, end)
        immediate_prior = slice_to_text(dialogue_df, max(0, start - prior_window), start)
        retrieved = segments_meta[sidx].get("retreived_memories", []) or []
        memory = generate_formatted_memories(retrieved, start)

        input_obj = {
            "topic": topic,
            "prior_knowledge": (
                "Prior dialogue memory:\n"
                f"{memory}\n\n"
                "Immediate prior dialogue:\n"
                f"{immediate_prior}\n\n"
            ),
            "current_dialogue": current_dialogue,
            "tgt_utts": current_dialogue,
            "start": str(start),
            "end": str(max(start, end - 1)),
            "total": str(max(0, end - start)),
            "context_type": "memory",
        }

        t0 = time.perf_counter()
        ok = True
        err = None
        items = 0
        try:
            ratings = await rate_text(
                llm=llm,
                input_obj=input_obj,
                backend=backend,
                mode="full",
                thinking_budget=gemini_thinking_budget if provider == "gemini" else 1024,
            )
            if isinstance(ratings, list):
                items = len(ratings)
            else:
                ok = False
                err = "non_list_response"
        except Exception as e:
            ok = False
            err = f"{type(e).__name__}: {str(e)[:200]}"
        dt = time.perf_counter() - t0
        rows.append(
            {
                "segment_index": sidx,
                "latency_s": dt,
                "ok": ok,
                "items": items,
                "error": err,
            }
        )

    ok_lat = [r["latency_s"] for r in rows if r["ok"]]
    mean, std = _mean_std(ok_lat)
    return {
        "n_success": len(ok_lat),
        "n_total": len(rows),
        "mean_latency_s": mean,
        "std_latency_s": std,
        "segment_results": rows,
    }


async def main():
    parser = argparse.ArgumentParser(description="Benchmark rating latency in memory mode.")
    parser.add_argument("--corpus", choices=["fora", "insq"], required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--segments", default="0,1,2")
    parser.add_argument("--prior-window", type=int, default=5)
    parser.add_argument("--gemini-thinking-budget", type=int, default=128)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    segment_ids = [int(x.strip()) for x in args.segments.split(",") if x.strip()]

    meta, seg_meta = _load_segment_metadata(args.corpus, args.episode_id)
    dialogue = pd.read_csv(f"data/raw/{args.corpus}/{args.episode_id}.csv")

    models = [
        ("openai", "gpt-5"),
        ("openai", "gpt-5-mini"),
        ("gemini", "gemini-2.5-pro"),
        ("gemini", "gemini-2.5-flash"),
    ]

    results = {}
    for backend in models:
        key = f"{backend[0]}:{backend[1]}"
        results[key] = await run_one_model(
            backend=backend,
            dialogue_df=dialogue,
            topic=meta.get("topic", ""),
            segments_meta=seg_meta,
            segment_ids=segment_ids,
            prior_window=args.prior_window,
            gemini_thinking_budget=args.gemini_thinking_budget,
        )

    out = {
        "protocol": {
            "corpus": args.corpus,
            "episode_id": args.episode_id,
            "mode": "memory",
            "segments": segment_ids,
            "chatgpt_reasoning_effort": "minimal",
            "gemini_thinking_budget": args.gemini_thinking_budget,
            "prior_window": args.prior_window,
        },
        "results": results,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
