from __future__ import annotations

import asyncio
import glob
import json
import logging
import os

from typing import Callable, Dict, List, Literal, Tuple
from memory.multiparty_memory import MultipartyMemory, process_conversation_segment_memory, get_retreived_memories_from_result, generate_formatted_memories

import pandas as pd
import tiktoken

from dialogue_rating import evaluate_ratings, rate_text
from llm.gemini_utility import call_gemini_async, call_gemini_sync
from llm.openai_utility import call_openai_async, _is_gpt5
from prompts import (
    ASPECT_AWARE_SUMMARY_PROMPT,
    DIRECT_SUMMARY_PROMPT,
    RECURSIVE_SUMMARY_PROMPT,
    MEMORY_SUMMARY_PROMPT,
    SummaryClass,
)
from util import load_json
from task_utility import remove_code_blocks, slice_to_text

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUMMARY_MIN_WORDS = 210
SUMMARY_MAX_WORDS = 350
TOP_K_CONTEXT = 5
#
# MODEL_CONFIG = {
#     "memory": ["gemini", "gemini-2.5-flash"],
#     "intro": ["gemini", "gemini-2.5-pro"],
#     "sum": ["gemini", "gemini-2.5-pro"],
#     "rate": ["gemini", "gemini-2.5-pro"]
# }
MODEL_CONFIG = {
    "memory": ["openai", "gpt-5-mini"],
    "memory_embedding": ["huggingface", "Qwen/Qwen3-Embedding-0.6B"],
    "intro": ["gemini", "gemini-2.5-pro"],
    "sum": ["openai", "gpt-5"],
    "rate": ["openai", "gpt-5"]
}




logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

enc = tiktoken.encoding_for_model("gpt-4o-mini")

_Backend = Literal["openai", "gemini"]


def _get_llm(backend: _Backend = "gemini", *, use_async: bool = False):
    """Return a callable for the requested LLM backend."""
    if backend == "openai":
        return call_openai_async
    return call_gemini_async if use_async else call_gemini_sync


def _validate_summary(response_text) -> bool:
    """Validate that the LLM response contains a summary of the expected length."""
    try:
        summary_json = json.loads(remove_code_blocks(response_text))
        summary = summary_json["summary"]
    except (json.JSONDecodeError, KeyError, AttributeError):
        logger.debug("Unable to decode summary JSON.")
        return False

    word_count = len(summary.split())
    if not SUMMARY_MIN_WORDS <= word_count <= SUMMARY_MAX_WORDS:
        logger.debug("Summary word‑count %d outside expected range.", word_count)
        return False
    return True





# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

async def generate_summary_variants(
    info: Dict[str, str], memory: MultipartyMemory, backend: list, methods=None
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Generate multiple summary variants for a dialogue segment."""
    if methods is None:
        methods = ["direct", "recursive", "aspect", "memory"]
    prompts = {}
    if "direct" in methods:
        prompts["direct"] = DIRECT_SUMMARY_PROMPT.format(
            topic=info["topic"],
            prior_dialogue=info["prior_dialogue"],
            current_dialogue=info["current_dialogue"],
        )
    if "recursive" in methods:
        prompts["recursive"] = RECURSIVE_SUMMARY_PROMPT.format(
            topic=info["topic"],
            prior_summary=info["prior_summary"],
            current_dialogue=info["current_dialogue"],
        )
    if "aspect" in methods:
        prompts["aspect"] = ASPECT_AWARE_SUMMARY_PROMPT.format(
            topic=info["topic"],
            prior_dialogue=info["prior_dialogue"],
            current_dialogue=info["current_dialogue"],
        )

    if "memory" in methods:
        formatted_memories = generate_formatted_memories(info["prior_memory"])
        prompts["memory"] = MEMORY_SUMMARY_PROMPT.format(
            topic=info["topic"],
            formatted_memories=formatted_memories,
            current_dialogue=info["current_dialogue"],
        )

    llm = _get_llm(backend[0], use_async=True)
    if backend[0] == "gemini":
        cfg = {"response_mime_type": "application/json", "response_schema": SummaryClass}
    else:
        cfg = {"verbosity": "low", "format": "json_object"}

    tasks = {
        name: asyncio.create_task(
            llm(prompt, model=backend[1], generation_config=cfg, valid_func=_validate_summary, allow_forced_return=True)
        )
        for name, prompt in prompts.items()
    }

    responses = await asyncio.gather(*tasks.values())
    if backend[0] == "gemini":
        summaries = {
            name: resp.parsed.summary
            for name, resp in zip(tasks.keys(), responses)
            if resp is not None
        }
    else:
        summaries = {
            name: json.loads(remove_code_blocks(resp)).get("summary", "")
            for name, resp in zip(tasks.keys(), responses)
            if resp is not None
        }
    return prompts, summaries


def pick_best_summary(
    reference_ratings: Dict[str, int],
    candidate_ratings: Dict[str, Dict[str, int]],
) -> Tuple[Tuple[str, float], Dict[str, float]]:
    """Select the highest‑scoring summary variant."""
    best: Tuple[str, float] = ("", float("-inf"))
    all_scores: Dict[str, float] = {}

    for method, rating in candidate_ratings.items():
        score = evaluate_ratings(rating, reference_ratings)
        all_scores[method] = score
        if score > best[1]:
            best = (method, score)

    logger.info("Best summary method: %s (%.3f)", best[0], best[1])
    return best, all_scores


# ---------------------------------------------------------------------------
# Recursive summarisation pipeline
# ---------------------------------------------------------------------------

async def quality_checked_recursive_summaries(
        dialogue: pd.DataFrame,
        memory: MultipartyMemory,
        backend: dict,
        meta_path: str,
        checkpoint_path: str | None = None,
        overwrite: bool = False,
) -> Dict:
    """Generate quality‑checked summaries for each dialogue segment."""

    meta = load_json(meta_path)

    conversation_id = meta["conversation_id"]

    topic: str = meta["topic"]
    intro_summary: str = meta.get("intro_summary", None)

    segments: List[Dict] = meta["segmentation"]["segments"]

    first_segment_start = segments[0]["intervals"][0]
    prior_dialogue_full = dialogue[dialogue.utterance_index < first_segment_start]

    if "backend_config" in meta:
        for key, val in MODEL_CONFIG.items():
            if val != meta["backend_config"].get(key, None):
                overwrite = True
                break
    else:
        overwrite = True

    meta["backend_config"] = backend

    if (("prior_memory" not in meta or not meta["prior_memory"]) or overwrite) and first_segment_start > 0:
        memory, result = process_conversation_segment_memory(prior_dialogue_full, meta, memory=memory, run_id=conversation_id, context_k=TOP_K_CONTEXT)
        meta["prior_memory"] = memory.get_all(run_id=conversation_id, limit=10000).get("results", None)
        query_filter = memory.vector_store._create_filter({})
        all_vectors = memory.vector_store.client.scroll(
            collection_name=memory.vector_store.collection_name,
            scroll_filter=query_filter,
            limit=10000,
            with_payload=True,
            with_vectors=True,
        )
        vector_counts = 0
        for i, mem in enumerate(meta["prior_memory"]):
            mem_id = mem["id"]
            for v in all_vectors[0]:
                if v.id == mem_id:
                    meta["prior_memory"][i]["embedding"] = v.vector
                    vector_counts += 1
                    break
        if vector_counts == len(meta["prior_memory"]):
            logger.info(f"Storing {len(meta['prior_memory'])} memories and vectors in meta")
            with open(meta_path, "w") as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=2)

    else:
        prior_memory_json = meta["prior_memory"]
        memory = MultipartyMemory(backend=backend["memory"][0])
        embedding_dict = {m["memory"]: m["embedding"] for m in prior_memory_json if 'embedding' in m}
        for m in prior_memory_json:
            m['metadata']["run_id"] = conversation_id
            if "actor_id" not in m['metadata'] and "speaker" in m['metadata']:
                m['metadata']['actor_id'] = m['metadata']['speaker']
            memory._create_memory(m["memory"], embedding_dict, m['metadata'])



    logger.info("Processing '%s' with %d segments.", topic, len(segments))
    prior_summary = intro_summary

    for idx, segment in enumerate(segments):
        start, end = segment["intervals"]
        conf_start, conf_end = segment["confidence"]

        logger.info(
            "Segment %d/%d: %d–%d (conf %.2f–%.2f)",
            idx + 1,
            len(segments),
            start,
            end,
            conf_start,
            conf_end,
        )

        segment["prior_summary"] = prior_summary

        current_dialogue_text = slice_to_text(dialogue, start, end)
        prior_dialogue_text = slice_to_text(dialogue, 0, start)

        # The context is all utterances before the current segment
        context_start = max(0, start - TOP_K_CONTEXT)
        context_df = dialogue.iloc[context_start:start]
        segment_df = dialogue.iloc[start:end]

        # --- REFACTORED SECTION ---
        # A single call now handles both memory retrieval and updating, avoiding redundant claim extraction.
        memory, result = process_conversation_segment_memory(segment_df, meta, memory=memory, run_id=conversation_id, context_df=context_df, context_k=TOP_K_CONTEXT)
        actions = [r.get("results", []) if r is not None else [] for r in result]
        retrieved_memories = get_retreived_memories_from_result(result, start)
        prompt_input = {
            "topic": topic,
            "prior_summary": prior_summary,
            "prior_dialogue": prior_dialogue_text,
            "current_dialogue": current_dialogue_text,
            "prior_memory": retrieved_memories  # Use the retrieved memories
        }
        # --- END REFACTORED SECTION ---

        # 1. Generate candidate summaries

        _, summaries = await generate_summary_variants(prompt_input, memory, backend=backend["sum"], methods=["memory"])

        llm = _get_llm(backend["rate"][0], use_async=True)

        # 2. Build pseudo‑reference rating

        if "full_context_rating" in segments and segments["full_context_rating"] and not overwrite:
            reference_ratings = segment["full_context_rating"]
        else:
            reference_input = {
                "topic": topic,
                "prior_knowledge": prior_dialogue_text,
                "current_dialogue": current_dialogue_text,
                "start": str(start),
                "end": str(end - 1),
                "total": str(end - start),
            }
            reference_ratings = await rate_text(llm, reference_input, backend=backend["rate"])
        #

        # 3. Score candidate summaries
        candidate_summaries = {}

        for sum_method in ["direct", "recursive", "aspect", "memory"]:
            if sum_method in summaries:
                sum = summaries.get(sum_method)
                if sum_method == "recursive":
                    sum_method = "prior"
                candidate_summaries[f"{sum_method}_summary"] = sum

        if "summary_ratings" not in segment or "no_summary" not in segment["summary_ratings" ]:
            candidate_summaries["no_summary"] = "No prior knowledge or prior dialogue information available."

        keys = list(candidate_summaries.keys())
        rating_coroutines = [
            rate_text(
                llm,
                {
                    "topic": topic,
                    "prior_knowledge": candidate_summaries[k],
                    "current_dialogue": current_dialogue_text,
                    "start": str(start),
                    "end": str(end - 1),
                    "total": str(end - start),
                },
                backend=backend["rate"],
            )
            for k in keys
        ]

        variant_ratings = dict(zip(keys, await asyncio.gather(*rating_coroutines)))



        if "no_summary" not in candidate_summaries and "no_summary" in segment["summary_ratings"]:
            candidate_summaries["no_summary"] = "No prior knowledge or prior dialogue information available."
            variant_ratings["no_summary"] = segment["summary_ratings"]["no_summary"]

        (best_method, best_score), score_map = pick_best_summary(
            reference_ratings, variant_ratings
        )

        # 4. Update state & checkpoint (Memory update is already done)
        prior_summary = summaries.get("recursive", prior_summary)
        segment.update(
            {
                "best_sum_method": best_method,
                "best_sum_score": best_score,
                "best_context_sum": candidate_summaries[best_method],
                "summary_ratings": variant_ratings,
                "summary_scores": score_map,
                "full_context_rating": reference_ratings,
                "memory_actions": actions,
                "retreived_memories": retrieved_memories,
                "summaries": summaries
            }
        )

        if checkpoint_path:
            with open(checkpoint_path, "w") as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=2)

    return meta


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def get_recursive_summaries(
    dialogue: pd.DataFrame,
    memory: MultipartyMemory,
    backend: dict[str, any],
    meta_path: str,
    checkpoint_path: str | None = None,
    *,
    overwrite: bool = False,
) -> Dict:
    """Synchronous wrapper around ``quality_checked_recursive_summaries``."""
    return asyncio.run(
        quality_checked_recursive_summaries(
            dialogue,
            memory=memory,
            backend=backend,
            meta_path=meta_path,
            checkpoint_path=checkpoint_path,
            overwrite=overwrite,
        )
    )


async def _generate_intro_summaries(
    dialogue_files: List[str],
    *,
    backend: _Backend = "openai",
    use_async: bool = False,
) -> List[Dict]:
    tasks: List[Dict] = []
    for fp in dialogue_files:
        dialogue_df = pd.read_csv(fp)
        intro_df = dialogue_df[dialogue_df.phase == 0]

        intro_dialogue = "\n\n".join(
            f"{r.utterance_speaker} ({r.role}): {r.utterance_text}" for _, r in intro_df.iterrows()
        )

        tasks.append(
            {
                "conversation_id": str(int(intro_df["conversation_id"].iloc[0])),
                "topic": intro_df["title"].iloc[0],
                "prompt": DIRECT_SUMMARY_PROMPT.format(
                    prior_dialogue=intro_dialogue, topic=intro_df["title"].iloc[0]
                ),
            }
        )

    llm = _get_llm(backend, use_async=use_async)
    cfg = {"response_mime_type": "application/json", "response_schema": SummaryClass}

    if use_async:
        responses = await asyncio.gather(
            *[llm(prompt=t["prompt"], generation_config=cfg) for t in tasks]
        )
    else:
        responses = [
            llm(prompt=t["prompt"], generation_config=cfg) for t in tasks
        ]

    for task, resp in zip(tasks, responses):
        task.pop("prompt", None)
        task["intro_summary"] = resp.parsed.summary

    return tasks


def generate_intro_summaries(*, backend: _Backend = "gemini", use_async: bool = False) -> None:
    dialogue_files = glob.glob("../data/raw/*.csv")
    metas = asyncio.run(
        _generate_intro_summaries(dialogue_files, backend=backend, use_async=use_async)
    )

    for m in metas:
        fname = f"insq_{m['conversation_id']}_meta.json"
        with open(os.path.join("../data/meta", fname), "w") as fh:
            json.dump(m, fh, ensure_ascii=False, indent=2)


def generate_segment_summaries(backend, overwrite: bool = False) -> None:
    dialogue_files = glob.glob("../data/raw/insq/*.csv")
    memory = MultipartyMemory(backend=backend["memory"][0])
    for idx, df_path in enumerate(dialogue_files, start=1):
        logger.info("Summarising (%d/%d): %s", idx, len(dialogue_files), df_path)

        cov_id = int(df_path.split("_")[-1].replace(".csv", ""))

        if cov_id not in [17957]:
            continue

        meta_path = df_path.replace(".csv", "_meta.json")
        checkpoint_path = meta_path.replace(".json", "_checkpoint.json").replace("/raw/insq/", f"/processed_segments/openai/")
        meta_path = checkpoint_path
        output_path = checkpoint_path

        # if os.path.exists(output_path):
        #     exist_output = load_json(output_path)
        #     segments = exist_output["segmentation"]["segments"]
        #     if all("memory_actions" in seg for seg in segments):
        #         logger.info(f"Episode '{cov_id}' already summarised. Skipping.")
        #         print(f"Completed and skipped dialogue: {df_path}")
        #         continue
        dialogue_df = pd.read_csv(df_path)
        memory.reset()
        updated_meta = get_recursive_summaries(
            dialogue_df,
            memory=memory,
            backend=backend,
            meta_path=meta_path,
            checkpoint_path=checkpoint_path,
            overwrite=overwrite
        )

        with open(output_path, "w") as fh:
            json.dump(updated_meta, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    generate_segment_summaries(backend=MODEL_CONFIG, overwrite=False)
