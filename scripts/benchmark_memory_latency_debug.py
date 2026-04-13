#!/usr/bin/env python3
import argparse
import json
import logging
import os
import signal
import statistics
import tempfile
import time
import uuid
from copy import deepcopy
from pathlib import Path

import pandas as pd

# Must be set before importing mem0 modules.
os.environ.setdefault("MEM0_TELEMETRY", "False")


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


class AlarmTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise AlarmTimeout("operation timed out")


signal.signal(signal.SIGALRM, _alarm_handler)


def call_with_alarm(fn, timeout_s: int):
    signal.alarm(timeout_s)
    try:
        return fn()
    finally:
        signal.alarm(0)


def setup_logger(log_file: str):
    logger = logging.getLogger("memory_latency_debug")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def patch_gemini(thinking_budget: int):
    from google.genai import types as gtypes
    import mem0.llms.gemini as mem0_gemini

    original = mem0_gemini.GeminiLLM.generate_response

    def patched(self, messages, response_format=None, tools=None, tool_choice="auto"):
        system_instruction, contents = self._reformat_messages(messages)
        cfg = {
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "thinking_config": gtypes.ThinkingConfig(thinking_budget=thinking_budget),
        }
        if system_instruction:
            cfg["system_instruction"] = system_instruction
        if response_format is not None and response_format.get("type") == "json_object":
            cfg["response_mime_type"] = "application/json"
            if "schema" in response_format:
                cfg["response_schema"] = response_format["schema"]
        generation_config = gtypes.GenerateContentConfig(**cfg)
        response = self.client.models.generate_content(
            model=self.config.model,
            contents=contents,
            config=generation_config,
        )
        return self._parse_response(response, tools)

    mem0_gemini.GeminiLLM.generate_response = patched
    return original


def patch_openai(timeout_s: int):
    import mem0.llms.openai as mem0_openai

    original = mem0_openai.OpenAILLM.generate_response

    def patched(self, messages, response_format=None, tools=None, tool_choice="auto", **kwargs):
        params = self._get_supported_params(messages=messages, **kwargs)
        params.update({"model": self.config.model, "messages": messages})
        if response_format:
            params["response_format"] = response_format
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice
        response = self.client.chat.completions.create(**params, timeout=timeout_s)
        return self._parse_response(response, tools)

    mem0_openai.OpenAILLM.generate_response = patched
    return original


def patch_timed_add(logger, llm_timeout_s: int):
    from mem0.memory.utils import remove_code_blocks
    import memory.multiparty_memory as mm
    from memory.multiparty_memory import MultipartyMemory

    original = MultipartyMemory._add_to_vector_store

    def timed_add(self, target_utterance, context, metadata, filters, output_memory=False):
        turn_id = target_utterance.get("turn_id", target_utterance.get("index", "unknown"))
        speaker = target_utterance.get("speaker", target_utterance.get("name", "unknown"))
        total_t0 = time.perf_counter()

        extraction_prompt = self.config.custom_fact_extraction_prompt
        formatted_context = mm.format_context_for_llm(context)
        formatted_target = mm.format_context_for_llm([target_utterance])
        full_extraction_prompt = (
            f"{extraction_prompt}\n\n### Task\n\n**Context**\n{formatted_context}\n\n"
            f"**Target utterance**\n{formatted_target}\n"
            "**Output** (List of extracted memory json objects)\n"
        )

        extraction_elapsed = 0.0
        extracted = []
        for attempt in range(1, 4):
            try:
                logger.info(
                    "stage=claim_extraction event=start turn_id=%s speaker=%s attempt=%d",
                    turn_id,
                    speaker,
                    attempt,
                )
                t0 = time.perf_counter()
                response = call_with_alarm(
                    lambda: self.llm.generate_response(
                        messages=[{"role": "user", "content": full_extraction_prompt}],
                        response_format={"type": "json_object"},
                    ),
                    timeout_s=llm_timeout_s,
                )
                dt = time.perf_counter() - t0
                extraction_elapsed += dt
                extracted = json.loads(remove_code_blocks(response)).get("memories", [])
                if not isinstance(extracted, list):
                    extracted = []
                logger.info(
                    "stage=claim_extraction event=success turn_id=%s speaker=%s elapsed=%.3f claims=%d",
                    turn_id,
                    speaker,
                    dt,
                    len(extracted),
                )
                break
            except Exception as exc:
                logger.warning(
                    "stage=claim_extraction event=error turn_id=%s speaker=%s attempt=%d error=%s",
                    turn_id,
                    speaker,
                    attempt,
                    type(exc).__name__,
                )
                time.sleep(1)

        if not extracted:
            return {
                "results": [],
                "old_memories": {},
                "timing": {
                    "extraction": extraction_elapsed,
                    "retrieval_matching": 0.0,
                    "integration": 0.0,
                    "total": time.perf_counter() - total_t0,
                },
            }

        retrieval_t0 = time.perf_counter()
        embeddings_map = {}
        found = []
        for fact in extracted:
            claim = fact.get("claim")
            if not claim:
                continue
            emb = self.embedding_model.embed(claim, memory_action="add")
            embeddings_map[claim] = emb
            found.extend(self.vector_store.search(query=claim, vectors=emb, limit=5, filters=filters))
        unique = {}
        for mem in found:
            m = {"id": mem.id, "claim": mem.payload["data"], **mem.payload}
            m.pop("data", None)
            unique[mem.id] = m
        existing = list(unique.values())
        update_prompt = self.config.custom_update_memory_prompt.replace(
            "{{retrieved_old_memory_dict}}", json.dumps(existing, indent=2)
        ).replace("{{new_retrieved_claims_list}}", json.dumps(extracted, indent=2))

        resolved = []
        for attempt in range(1, 4):
            try:
                logger.info(
                    "stage=retrieval_matching event=start turn_id=%s speaker=%s attempt=%d",
                    turn_id,
                    speaker,
                    attempt,
                )
                response = call_with_alarm(
                    lambda: self.llm.generate_response(
                        messages=[{"role": "user", "content": update_prompt}],
                        response_format={"type": "json_object"},
                    ),
                    timeout_s=llm_timeout_s,
                )
                unresolved = json.loads(remove_code_blocks(response)).get("memory_updates", [])
                resolved = mm._resolve_conflicting_updates(unresolved)
                report = mm.validate_memory_updates(existing, extracted, resolved, fix=True)
                if report.get("ok", False):
                    break
            except Exception as exc:
                logger.warning(
                    "stage=retrieval_matching event=error turn_id=%s speaker=%s attempt=%d error=%s",
                    turn_id,
                    speaker,
                    attempt,
                    type(exc).__name__,
                )
                time.sleep(1)
        retrieval_elapsed = time.perf_counter() - retrieval_t0
        logger.info(
            "stage=retrieval_matching event=done turn_id=%s speaker=%s elapsed=%.3f updates=%d",
            turn_id,
            speaker,
            retrieval_elapsed,
            len(resolved),
        )

        integration_t0 = time.perf_counter()
        processed = []
        for update in resolved:
            action = update.get("action")
            source = update.get("source")
            target = update.get("target")
            logical_relation = update.get("logical_relation")
            if not source:
                continue
            try:
                md = deepcopy(metadata)
                md.update(
                    {
                        "speaker": source.get("speaker"),
                        "target_speaker": source.get("target_speaker"),
                        "turn_id": source.get("turn_id"),
                        "actor_id": source.get("speaker"),
                        "logical_relation": logical_relation,
                    }
                )
                claim = source["claim"]
                embedding = {claim: embeddings_map[claim]} if claim in embeddings_map else {
                    claim: self.embedding_model.embed(claim, memory_action="update")
                }
                memory_id = None
                if action == "ADD":
                    memory_id = self._create_memory(data=claim, existing_embeddings=embedding, metadata=md)
                elif action == "UPDATE" and target:
                    self._update_memory(
                        memory_id=target["id"],
                        data=claim,
                        existing_embeddings=embedding,
                        metadata=md,
                    )
                    memory_id = target["id"]
                processed.append(
                    {
                        "id": memory_id,
                        "memory": claim,
                        "event": action,
                        "logical_relation": logical_relation,
                        "previous_memory": target,
                    }
                )
            except Exception as exc:
                logger.warning(
                    "stage=integration event=error turn_id=%s speaker=%s error=%s",
                    turn_id,
                    speaker,
                    type(exc).__name__,
                )
        integration_elapsed = time.perf_counter() - integration_t0
        total_elapsed = time.perf_counter() - total_t0
        logger.info(
            "stage=integration event=done turn_id=%s speaker=%s elapsed=%.3f total=%.3f",
            turn_id,
            speaker,
            integration_elapsed,
            total_elapsed,
        )
        return {
            "results": processed,
            "old_memories": unique,
            "timing": {
                "extraction": extraction_elapsed,
                "retrieval_matching": retrieval_elapsed,
                "integration": integration_elapsed,
                "total": total_elapsed,
            },
        }

    MultipartyMemory._add_to_vector_store = timed_add
    return original


def summarize(values):
    if not values:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def main():
    parser = argparse.ArgumentParser(description="Run memory ops latency benchmark with debug logs.")
    parser.add_argument("--backend", choices=["openai", "gemini", "both"], default="both")
    parser.add_argument("--corpus", choices=["fora", "insq"], default="insq")
    parser.add_argument("--episode-id", default="insq_2228")
    parser.add_argument("--segments", default="0,1,2")
    parser.add_argument("--top-k-context", type=int, default=5)
    parser.add_argument("--llm-timeout-s", type=int, default=130)
    parser.add_argument("--openai-timeout-s", type=int, default=120)
    parser.add_argument("--gemini-thinking-budget", type=int, default=128)
    parser.add_argument("--out-json", default="/tmp/memory_ops_latency_budget128_minimal_final.json")
    parser.add_argument("--log-file", default="/tmp/memory_ops_latency_debug.log")
    args = parser.parse_args()

    logger = setup_logger(args.log_file)
    logger.info("benchmark_start backend=%s", args.backend)

    import mem0.llms.gemini as mem0_gemini
    import mem0.llms.openai as mem0_openai
    from memory.multiparty_memory import MultipartyMemory, process_conversation_segment_memory

    segments = [int(x.strip()) for x in args.segments.split(",") if x.strip()]
    meta, segment_data = _load_segment_metadata(args.corpus, args.episode_id)
    dialogue = pd.read_csv(f"data/raw/{args.corpus}/{args.episode_id}.csv")

    original_gemini = patch_gemini(args.gemini_thinking_budget)
    original_openai = patch_openai(args.openai_timeout_s)
    original_add = patch_timed_add(logger, args.llm_timeout_s)

    try:
        backends = ["openai", "gemini"] if args.backend == "both" else [args.backend]
        all_results = {}
        for backend in backends:
            os.environ["MEM0_QDRANT_PATH"] = os.path.join(
                tempfile.gettempdir(),
                f"mem0_qdrant_{backend}_{uuid.uuid4().hex}",
            )
            logger.info("backend_start backend=%s qdrant_path=%s", backend, os.environ["MEM0_QDRANT_PATH"])
            memory = MultipartyMemory(backend=backend)
            op = {k: [] for k in ["extraction", "retrieval_matching", "integration", "total"]}
            for sidx in segments:
                start, end = segment_data[sidx]["intervals"]
                context_df = dialogue.iloc[max(0, start - args.top_k_context):start]
                segment_df = dialogue.iloc[start:end]
                logger.info("segment_start backend=%s segment=%d rows=%d", backend, sidx, len(segment_df))
                memory, results = process_conversation_segment_memory(
                    segment_df,
                    meta,
                    memory=memory,
                    run_id=args.episode_id,
                    context_df=context_df,
                    context_k=args.top_k_context,
                )
                for row in results:
                    timing = (row or {}).get("timing", {})
                    for key in op:
                        if key in timing:
                            op[key].append(float(timing[key]))
                logger.info("segment_done backend=%s segment=%d samples_total=%d", backend, sidx, len(op["total"]))
            all_results[backend] = {key: summarize(vals) for key, vals in op.items()}
            logger.info("backend_done backend=%s samples_total=%d", backend, all_results[backend]["total"]["n"])

        output = {
            "protocol": {
                "episode_id": args.episode_id,
                "segments": segments,
                "chatgpt_reasoning_effort": "minimal",
                "gemini_thinking_budget": args.gemini_thinking_budget,
                "top_k_context": args.top_k_context,
            },
            "results": all_results,
            "log_file": args.log_file,
        }
        with open(args.out_json, "w") as f:
            json.dump(output, f, indent=2)
        logger.info("benchmark_done out_json=%s", args.out_json)
    finally:
        MultipartyMemory._add_to_vector_store = original_add
        mem0_openai.OpenAILLM.generate_response = original_openai
        mem0_gemini.GeminiLLM.generate_response = original_gemini


if __name__ == "__main__":
    main()
