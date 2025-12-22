import concurrent
import json
import logging
import os
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from mem0 import Memory
from mem0.configs.base import MemoryConfig, LlmConfig, EmbedderConfig, VectorStoreConfig
from mem0.memory.main import _build_filters_and_metadata
from mem0.memory.utils import remove_code_blocks
from memory.memory_validation import validate_memory_updates

from llm.gemini_utility import is_rate_limited

from google.api_core.exceptions import ResourceExhausted, GoogleAPICallError, InternalServerError

from memory.multiparty_prompts import (
    MULTIPARTY_CLAIM_EXTRACTION_PROMPT,
    MULTIPARTY_MEMORY_UPDATE_PROMPT,
)

logger = logging.getLogger(__name__)

from google.api_core.exceptions import (
    ResourceExhausted,
    InternalServerError,
    GoogleAPICallError,
    ServiceUnavailable,
    TooManyRequests,  # available in google.api_core
)


MAX_TRIES = 3


def _resolve_conflicting_updates(actions: List[Dict]) -> List[Dict]:
    """
    Resolves conflicts where multiple actions target the same memory_id.
    The strategy is "last one wins": if multiple updates/supersedes are
    planned for the same target, only the last one in the list is kept.
    """
    target_tracker = {}
    resolved_actions = []

    # First pass: identify the last action for each target_id
    for i, action_item in enumerate(actions):
        if action_item.get("action") in ["UPDATE"]:
            target = action_item.get("target")
            if target and "id" in target:
                target_id = target["id"]
                # Store the index of the last seen action for this target_id
                target_tracker[target_id] = i

    # Second pass: build the final list of actions
    for i, action_item in enumerate(actions):
        action_type = action_item.get("action")
        if action_type in ["UPDATE"]:
            target_id = action_item.get("target", {}).get("id")
            # If this action is the last one designated for its target, keep it
            if target_id and target_tracker.get(target_id) == i:
                resolved_actions.append(action_item)
            elif target_id:
                logging.warning(
                    f"Conflict detected. Discarding action '{action_type}' for target_id '{target_id}' "
                    f"in favor of a later action."
                )
        else:
            # Keep all other actions like ADD and NONE
            resolved_actions.append(action_item)

    return resolved_actions


def format_context_for_llm(messages: List[Dict]) -> str:
    """
    Formats a list of message dicts into a string for the LLM context.
    Each message must have 'index' and 'content'.
    'speaker' is used if present, otherwise defaults to the 'role'.
    """
    if not messages:
        return "No Context"

    response = ""
    for msg in messages:
        speaker = msg.get("name", msg.get("role", "unknown"))
        response += f"{msg.get('index', '-')}. {speaker}: {msg['content']}\n"
    return response.strip()


class MultipartyMemory(Memory):
    """
    A memory class specialized for multi-party conversations.
    It uses custom, structured prompts to extract and consolidate facts
    while maintaining attribution to the correct speaker.
    """

    def __init__(self, backend: str = "gemini", custom_config: dict = None):
        """

        :type backend: object
        """
        # Centralize the custom prompts, as they are essential for both backends.
        common_config_args = {
            "custom_fact_extraction_prompt": MULTIPARTY_CLAIM_EXTRACTION_PROMPT,
            "custom_update_memory_prompt": MULTIPARTY_MEMORY_UPDATE_PROMPT,
        }

        if backend == "openai":
            # For OpenAI, we can often rely on the default LlmConfig and EmbedderConfig
            # from mem0, but we still need to pass our custom prompts.

            llm_config = LlmConfig(
                provider="openai",
                config={
                    # Using a modern, capable, and cost-effective model
                    "api_key": os.environ.get("OPENAI_API_KEY"),
                    "model": "gpt-5-mini",
                    "max_tokens": 50000,
                    "reasoning_effort": "minimal",
                    "verbosity": "low"
                }
            )

        elif backend == "gemini":
            # For Gemini, we must explicitly define the provider and models.
            # We will create the necessary config objects.

            llm_config = LlmConfig(
                provider="gemini",
                config={
                    # Using a modern, capable, and cost-effective model
                    "api_key": os.environ.get("GEMINI_API_KEY"),
                    "model": "gemini-2.5-flash",
                    "max_tokens": 50000,
                    "temperature": 0.1,
                    "thinking_budget": 0
                }
            )
        else:
            raise ValueError(f"Unsupported backend: '{backend}'. Please choose from ['openai', 'gemini'].")

        embedder_config = EmbedderConfig(
            provider="huggingface",
            config={
                "model": "Qwen/Qwen3-Embedding-0.6B",
                "embedding_dims": 1024,
                "model_kwargs": {
                    "model_kwargs": {"attn_implementation": "eager", "device_map": "auto"},
                    "tokenizer_kwargs": {"padding_side": "left"}
                }
            }
        )

        vector_store_config = VectorStoreConfig(
            provider="qdrant",
            config={
                "embedding_model_dims": 1024,
            }
        )

        # Assemble the final MemoryConfig object, including our common prompts
        config = MemoryConfig(
            llm=llm_config,
            embedder=embedder_config,
            vector_store=vector_store_config,
            **common_config_args
        )



        # If a custom_config dict is provided, we can smartly update our generated config
        if custom_config:
            # This is an advanced feature allowing runtime overrides, e.g., changing the model
            config = MemoryConfig(**config.model_dump(), **custom_config)

        # Initialize the parent Memory class with the fully constructed config object
        super().__init__(config)
        logging.info(f"MultipartyMemory initialized with '{backend}' backend.")

    def add(self, target_utterance: Dict[str, Any], context: List[Dict[str, Any]], run_id: str, metadata: Optional[Dict[str, Any]] = None, output_memory: bool = False):
        """
        Adds memory from a target utterance, considering its conversational context.
        Args:
        target_utterance (Dict): The single, new message to process.
        Must contain 'index', 'speaker', 'content'.
        context (List[Dict]): A list of previous messages for context.
        run_id (str): The identifier for the current conversation session.
        metadata (Dict, optional): Additional metadata to store with all created memories.
        """
        # In a multi-party context, agent_id/user_id are less relevant than run_id.
        # We scope everything to the conversation's run_id.
        if not run_id:
            raise ValueError("'run_id' is required for MultipartyMemory.")

        if not all(k in target_utterance for k in ["index", "name", "content"]):
            raise ValueError("target_utterance must contain 'index', 'name', and 'content'.")

        # Build the necessary metadata and filters for this operation

        processed_metadata, effective_filters = _build_filters_and_metadata(
            run_id=run_id,
            input_metadata=metadata,
        )

        # The core logic is now in a single, dedicated thread
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                self._add_to_vector_store,
                target_utterance,
                context,
                processed_metadata,
                effective_filters,
                output_memory
            )

        vector_store_result = future.result()
        return vector_store_result


    def _add_to_vector_store(
            self,
            target_utterance: Dict,
            context: List[Dict],
            metadata: Dict,
            filters: Dict,
            output_memory: bool=False
    ):
        """
        The main pipeline for multi-party memory processing.
        1. Fact Extraction: Extracts atomic facts from the target utterance.
        2. Memory Search: Finds relevant existing memories.
        3. Memory Consolidation: Decides whether to ADD, UPDATE, or SUPERSEDE.
        4. Memory Persistence: Executes the decisions from the consolidation step.
        """

        # --- 1. Fact Extraction Step ---
        # ... (This part remains the same)

        extraction_prompt = self.config.custom_fact_extraction_prompt
        formatted_context = format_context_for_llm(context)
        formatted_target = format_context_for_llm([target_utterance])

        full_extraction_prompt = f"{extraction_prompt}\n\n### Task\n\n**Context**\n{formatted_context}\n\n**Target utterance**\n{formatted_target}\n**Output** (List of extracted memory json objects)\n"



        complete_extraction = False
        extraction_attempts = 0
        new_retrieved_claims = []

        while not complete_extraction and extraction_attempts < 3:
            try:
                extraction_attempts += 1
                response = self.llm.generate_response(
                    messages=[{"role": "user", "content": full_extraction_prompt}],
                    response_format={"type": "json_object"},
                )
                new_retrieved_claims = json.loads(remove_code_blocks(response)).get("memories", [])
                if not isinstance(new_retrieved_claims, list): new_retrieved_claims = []
                complete_extraction = True
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                logging.error(f"Failed extraction attempt: Error parsing fact extraction response: {e}\nResponse: {e}")
                time.sleep(10)


        if not new_retrieved_claims:
            logging.info("No new facts retrieved from input. No memory updates to perform.")
            return None

        # --- 2. Memory Search Step ---

        embeddings_map = {}
        all_found_memories = []

        # Loop through each new fact to embed and search individually

        for fact in new_retrieved_claims:

            claim = fact.get("claim")
            if not claim:
                continue

            # Embed one claim at a time
            embedding = self.embedding_model.embed(claim, memory_action="add")
            embeddings_map[claim] = embedding

            # Search for relevant memories for this specific claim
            search_results = self.vector_store.search(
            query=claim,
            vectors=embedding,
            limit=5,
            filters=filters,
            )

            all_found_memories.extend(search_results)

        # Consolidate all found memories into a unique set
        unique_memories = {}

        for mem in all_found_memories:
            # Reconstruct the full memory object for the consolidation prompt
            memory_obj = {"id": mem.id, "claim": mem.payload["data"], **mem.payload}
            if 'data' in memory_obj: del memory_obj['data']  # 'data' is now 'claim'
            unique_memories[mem.id] = memory_obj

        existing_memories_list = list(unique_memories.values())
        logging.info(f"Found {len(existing_memories_list)} potentially relevant existing memories.")

        # --- 3. Memory Consolidation Step ---

        update_prompt_template = self.config.custom_update_memory_prompt
        existing_memories_json = json.dumps(existing_memories_list, indent=2)
        new_facts_json = json.dumps(new_retrieved_claims, indent=2)
        full_update_prompt = update_prompt_template.replace(
            "{{retrieved_old_memory_dict}}", existing_memories_json
        ).replace(
            "{{new_retrieved_claims_list}}", new_facts_json
        )

        complete_memory_action_detection = False
        detection_attempts = 0
        actions_to_perform = {}

        while not complete_memory_action_detection and detection_attempts < 3:
            try:
                detection_attempts += 1
                response = self.llm.generate_response(
                    messages=[{"role": "user", "content": full_update_prompt}],
                    response_format={"type": "json_object"},
                )
                actions_to_perform = json.loads(remove_code_blocks(response))
                unresolved_updates = actions_to_perform.get("memory_updates", [])
                resolved_updates = _resolve_conflicting_updates(unresolved_updates)
                report = validate_memory_updates(existing_memories_list, new_retrieved_claims, resolved_updates, fix=True)
                if report.get("ok", False):
                    complete_memory_action_detection = True
                else:
                    print(report)
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                logging.error(f"Error parsing memory update response: {e}\nResponse: {response}")
                if detection_attempts < 3:
                    time.sleep(10)
                else:
                    logging.error(f"Error parsing memory update response after max attempts.")



        # --- 4. Memory Persistence Step ---
        processed_results = []

        for update_item in resolved_updates:
            action = update_item.get("action")
            source_fact = update_item.get("source")
            target_memory = update_item.get("target")
            logical_relation = update_item.get("logical_relation")
            if not source_fact:
                logging.warning(f"Skipping update item due to missing 'source': {update_item}")
                continue
            try:
                claim_metadata = deepcopy(metadata)
                claim_metadata.update({
                    "speaker": source_fact.get("speaker"),
                    "target_speaker": source_fact.get("target_speaker"),
                    "turn_id": source_fact.get("turn_id"),
                    "actor_id": source_fact.get("speaker"),
                    "logical_relation": logical_relation
                })
                claim = source_fact["claim"]

        # Initialize a dictionary to hold the embedding for this specific operation
                embedding_for_operation = {}
                if action == "ADD":
                    # For ADD, the claim is from the original new facts,
                    # so its embedding should be in our pre-computed map.
                    embedding = embeddings_map.get(claim)
                    if embedding:
                        embedding_for_operation = {claim: embedding}

                # For UPDATE, the claim might be a new, consolidated string from the LLM.
                # It won't be in embeddings_map, so we must handle it separately.
                # If the embedding wasn't found for ADD for some reason, we'll embed here too.
                if not embedding_for_operation:

                    # This branch will primarily be hit by UPDATE actions or if the
                    # map lookup for ADD unexpectedly fails.
                    logging.info(f"Generating on-demand embedding for consolidated claim: '{claim}'")
                    embedding = self.embedding_model.embed(claim, memory_action="update")
                    embedding_for_operation = {claim: embedding}

                memory_id = None

                if action == "ADD":
                    memory_id = self._create_memory(
                        data=claim,
                        existing_embeddings=embedding_for_operation,
                        metadata=claim_metadata,
                    )

                elif action == "UPDATE":
                    if not target_memory: continue

                    self._update_memory(
                        memory_id=target_memory["id"],
                        data=claim,
                        existing_embeddings=embedding_for_operation,
                        metadata=claim_metadata,
                    )
                    memory_id=target_memory["id"]

                elif action == "NONE":
                    logging.info(f"NONE for claim: '{claim}'")

                processed_results.append({
                    "id": memory_id,
                    "memory": claim,
                    "event": action,
                    "logical_relation": logical_relation,
                    "previous_memory": target_memory,
                })

            except Exception as e:
                logging.error(f"Error processing memory action: {update_item}, Error: {e}")


        result = {"results": processed_results}
        if output_memory:
            result["old_memories"] = unique_memories
        return result

    def _extract_claims(self, target_utterance: Dict, context: List[Dict]) -> List[Dict]:
        """Helper to extract claims from a single utterance."""
        extraction_prompt = self.config.custom_fact_extraction_prompt
        formatted_context = format_context_for_llm(context)
        formatted_target = format_context_for_llm([target_utterance])
        full_extraction_prompt = f"{extraction_prompt}\n\n### Task\n\n**Context**\n{formatted_context}\n\n**Target utterance**\n{formatted_target}\n**Output** (List of extracted memory json objects)\n"

        response = self.llm.generate_response(
            messages=[{"role": "user", "content": full_extraction_prompt}],
            response_format={"type": "json_object"},
        )
        try:
            extracted = json.loads(remove_code_blocks(response)).get("memories", [])
            return extracted if isinstance(extracted, list) else []
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Error parsing fact extraction response: {e}\nResponse: {response}")
            return []

    def relevent_memory_retreival(
            self,
            upcoming_segment: List[Dict[str, Any]],
            context: List[Dict[str, Any]],
            *,  # Enforce keyword-only arguments
            run_id: str,
            k: int = 3,
            use_claim_extraction: bool = False,
    ):
        """
        Generates a contextual briefing for an upcoming conversation segment.
        """
        if not run_id:
            raise ValueError("'run_id' is required to retrieve memories.")

        if not upcoming_segment:
            return {"retrieval_data": []}

        queries = []
        if use_claim_extraction:
            logging.info("Using high-precision strategy: Extracting claims before searching.")
            for utterance in upcoming_segment:
                current_context = context + upcoming_segment[:upcoming_segment.index(utterance)]
                extracted_claims = self._extract_claims(utterance, current_context)
                if extracted_claims:
                    queries.extend([claim['claim'] for claim in extracted_claims if 'claim' in claim])
        else:
            logging.info("Using fast strategy: Searching with raw utterance content.")
            queries = [utterance.get("content", "") for utterance in upcoming_segment if utterance.get("content")]

        if not queries:
            return {"retrieval_data": []}

        search_results = self.search_many(queries=queries, run_id=run_id, limit=k)

        unique_memories = {}
        for result_group in search_results:
            for mem in result_group.get("results", []):
                if mem.get('id') is not None:
                    unique_memories[mem["id"]] = mem

        if not unique_memories:
            return {"retrieval_data": []}

        return {"retrieval_data": list(unique_memories.values())}


    def search_many(self, queries: List[str], run_id: str, limit: int) -> List[Dict]:
        """
        Performs multiple search queries in a loop and returns a list of their results.
        This is a helper method for the briefing generation.
        """
        all_results = []
        logging.info(f"Performing {len(queries)} separate searches...")
        for query in queries:
            if not query:
                continue
            try:
                # This still re-embeds, for maximum efficiency, this could also be refactored
                # to accept pre-computed embeddings. For now, we leave it for other use cases.
                result = self.search(query=query, run_id=run_id, limit=limit)
                all_results.append(result)
            except Exception as e:
                logging.error(f"An error occurred during search for query: '{query}'. Error: {e}")
                all_results.append({"results": []})
        return all_results

def generate_formatted_memories(retrieval_memory, cutpoint=-1):
    for mem in retrieval_memory:
        if "metadata" in mem:
            meta = mem.get("metadata", {})
            mem["target_speaker"] = meta.get("target_speaker")
            mem["turn_id"] = meta.get("turn_id")
            mem["speaker"] = meta.get("actor_id")

    def sort_key(mem):
        turn_id = mem.get('turn_id')
        try:
            return int(turn_id) if turn_id is not None else float('inf')
        except (ValueError, TypeError):
            return float('inf')

    retrieval_memory.sort(key=sort_key)

    formatted_retrieved_memories = ""
    for mem in retrieval_memory:
        turn_id = mem.get('turn_id', '-1')
        if cutpoint > 0 and int(turn_id) > cutpoint:
            continue
        speaker = mem.get('speaker', 'Unknown')
        claim = mem.get('memory', mem.get('claim', ''))
        formatted_retrieved_memories += f"- (Turn #{turn_id}) {speaker}: \"{claim}\"\n"
    return formatted_retrieved_memories

def process_conversation_segment_memory(segment_df, meta, memory, run_id, context_df=None, context_k=5):
    """
    Example function to process a whole conversation and populate memory.
    """

    if context_df is None or context_df.empty:
        dialogue_df = segment_df
        start = 0
    else:
        dialogue_df = pd.concat([context_df, segment_df], ignore_index=True)
        start = min(context_k, len(context_df))

    dialogue_df = dialogue_df.sort_values(by='utterance_index').reset_index(drop=True)
    results = []

    # Use the new efficient function


    utt_count = 0

    for i in range(start, len(dialogue_df)):

        row = dialogue_df.iloc[i]
        print(f"Processing utterance {utt_count + 1}/{len(segment_df)}...")

        # The context is recent k utterances before the current one
        context_start = max(0, i - context_k)
        context_df = dialogue_df.iloc[context_start:i]
        context_list = [
            {
                "index": r.utterance_index,
                "name": r.utterance_speaker + f"({r.role})",
                "content": r.utterance_text
            }
            for _, r in context_df.iterrows()
        ]
        target_utterance = {
            "index": row.utterance_index,
            "name": row.utterance_speaker + f"({row.role})",
            "content": row.utterance_text
        }

        process_complete = False
        attempt = 0
        result = None

        while not process_complete and attempt < MAX_TRIES:
            try:
                attempt += 1
                result = memory.add(
                    target_utterance=target_utterance,
                    context=context_list,
                    run_id=run_id,
                    metadata={"topic": meta.get("topic")},
                    output_memory=True
                )
                process_complete = True

            except (ResourceExhausted, TooManyRequests) as e:
                # This is the specific 429/quota case -> sleep 60s
                wait_seconds = 60
                print(f"[Gemini] 429/RESOURCE_EXHAUSTED on attempt {attempt}/{MAX_TRIES}. "
                      f"Retrying in {wait_seconds}s.")
                time.sleep(wait_seconds)

            except (ServiceUnavailable, InternalServerError) as e:
                # Transient server-side errors: shorter, growing delay
                wait_seconds = 10 * attempt
                print(f"[Gemini] transient {e.__class__.__name__} on attempt {attempt}/{MAX_TRIES}. "
                      f"Retrying in {wait_seconds}s.")
                time.sleep(wait_seconds)

            except GoogleAPICallError as e:
                # Catch-all for Google API errors; check if it’s actually a 429
                if is_rate_limited(e):
                    wait_seconds = 60
                    print(f"[Gemini] quota hit (429) on attempt {attempt}/{MAX_TRIES}. "
                          f"Retrying in {wait_seconds}s.")
                    time.sleep(wait_seconds)
                else:
                    # Not retryable / unexpected Google API error -> surface it
                    print(f"[Gemini] non-retryable GoogleAPICallError: {e}")
                    break

            except Exception as e:
                # Truly unexpected (coding bug, bad input, etc.)
                print(f"Unexpected error in memory.add: {e}")
                break

        # Only append successful results if you prefer; otherwise keep None to preserve position
        if process_complete:
            results.append(result)
        else:
            # Optional: record a failure sentinel or re-raise
            results.append(None)
        utt_count += 1

    return memory, results


def get_retreived_memories_from_result(results, cutoff):

    old_memories = {}
    retrieved_memories = []
    actions = []
    for r in results:
        if not r:
            continue
        else:
            old_memories.update(r["old_memories"])
            actions.extend(r["results"])

    for k, v in old_memories.items():
        if int(v["turn_id"]) >= cutoff:
            mem_action = [a for a in actions if a["id"] == k]
            if len(mem_action) > 0:
                action = mem_action[0]
                if action["event"] == "ADD":
                    continue
                elif action["event"] == "UPDATE":
                    source_mem = action["previous_memory"]
                    source_id = source_mem["id"]
                    if source_id not in old_memories:
                        retrieved_memories.append(source_mem)
        retrieved_memories.append(v)
    return retrieved_memories

