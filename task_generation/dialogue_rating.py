from __future__ import annotations

import os, io, json, re, glob, asyncio, tempfile, shutil
from typing import Dict, Any, List, Tuple, Optional, Literal, Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import defaultdict

from sklearn.metrics import cohen_kappa_score, mean_absolute_error

# --- Your existing imports (assumed available) ---
from prompts import (
    build_eval_prompt,
    build_eval_prompt_using_external_ratings,
    EVAL_CLAIMS_WITH_MEMORY_PROMPT_FULL,
    FullRatingClass,
    InfomrativeRatingClass,  # keeping your exact class names
    MixedRatingClass,
)
from task_utility import remove_code_blocks, slice_to_text, strip_html_regex
from util import load_json
from llm.openai_utility import call_openai_async
from llm.gemini_utility import call_gemini_async, call_gemini_sync
from memory.multiparty_memory import generate_formatted_memories


# ---------------------------------------------------------------------------
#  Globals / Settings
# ---------------------------------------------------------------------------

_BackendName = Literal["openai", "gemini"]
_Backend = Tuple[_BackendName, str]  # (provider, model)

SELECTED_CONVIDS: Dict[str, List[int]] = {
    "insq": [20148, 24367, 5180, 9437, 16966, 9973, 17957, 23135, 2228, 26062],
    "fora": [2187, 2188, 2239, 2246, 2359, 2361, 2409, 2429, 2700, 2733],
}

MAX_CONCURRENCY = int(os.getenv("RATING_MAX_CONCURRENCY", "16"))  # cap async tasks
RETRY_ATTEMPTS = 5


# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------

def atomic_write_json(path: str, data: Any) -> None:
    """Write JSON atomically to avoid file corruption on crashes."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_json_", dir=d, text=True)
    try:
        with io.open(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        shutil.move(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def restructure_ratings(ratings: Dict[str, Dict[str, List[Dict[str, Any]]]]
                        ) -> Dict[str, List[Dict[str, Any]]]:
    """
    Collapse a ratings object of the form:
        {
          "mode_1": {
            "method_1": [{"utterance_index": int, "aspect_A": int, "context_type": str}, ...],
            "method_2": [...],
            ...
          },
          "mode_2": { ... },  # same methods, same utterances, disjoint aspects
          ...
        }

    into:
        {
          "method_1": [{"utterance_index": int, "aspect_A": int, "aspect_B": int, ..., "context_type": str}, ...],
          "method_2": [...],
          ...
        }

    Notes:
    - Aspects are any keys other than {"utterance_index", "context_type"}.
    - If multiple modes provide different `context_type` values for the same utterance,
      they are deduplicated; if >1 unique remains, they are joined with "|".
    - Utterances are sorted by `utterance_index` in each method.
    """

    # method -> utterance_index -> aggregated dict
    agg: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    # track context types across modes per (method, utterance_index)
    ctx: Dict[str, Dict[int, set]] = defaultdict(lambda: defaultdict(set))

    # discover all methods to preserve stable method set/order
    # (order of dicts in Python 3.7+ preserves insertion; we iterate modes then methods)
    method_order: List[str] = []
    seen_method = set()

    for mode_name, mode_payload in ratings.items():
        if not isinstance(mode_payload, dict):
            continue
        for method_name, records in mode_payload.items():
            if method_name not in seen_method:
                method_order.append(method_name)
                seen_method.add(method_name)

            if not isinstance(records, list):
                continue

            for rec in records:
                if "utterance_index" not in rec:
                    continue
                uidx = int(rec["utterance_index"])
                # create base container
                if uidx not in agg[method_name]:
                    agg[method_name][uidx] = {"utterance_index": uidx}

                # merge aspects (anything that's not utterance_index/context_type)
                for k, v in rec.items():
                    if k in ("utterance_index", "context_type"):
                        continue
                    agg[method_name][uidx][k] = v

                # collect context_type (may differ per mode)
                ctype = rec.get("context_type")
                if ctype is not None:
                    ctx[method_name][uidx].add(str(ctype))

    # finalize: flatten to requested structure with sorted utterances & resolved context_type
    result: Dict[str, List[Dict[str, Any]]] = {}
    for method_name in method_order:
        uidx_map = agg.get(method_name, {})
        ordered_uidx = sorted(uidx_map.keys())
        merged_list: List[Dict[str, Any]] = []
        for uidx in ordered_uidx:
            item = dict(uidx_map[uidx])  # copy
            # resolve context_type
            cset = ctx.get(method_name, {}).get(uidx, set())
            if len(cset) == 1:
                item["context_type"] = next(iter(cset))
            elif len(cset) > 1:
                # If conflicting, join deterministically
                item["context_type"] = "|".join(sorted(cset))
            # else: no context_type found anywhere; omit the key
            merged_list.append(item)
        result[method_name] = merged_list

    return result


def _extract_text(output: Any) -> str:
    """Get plain text from various SDK outputs and strip code fences."""
    s = getattr(output, "text", output)
    s = s if isinstance(s, str) else str(s)
    # Strip ```json ... ``` or ``` ... ```
    m = re.search(r"```(?:json)?\s*(.*?)```", s, flags=re.S)
    return m.group(1).strip() if m else s.strip()


def _get_llm(backend: _Backend, *, use_async: bool = True) -> Callable:
    """Return an LLM callable for the requested backend."""
    provider, _model = backend
    if provider == "openai":
        # OpenAI path is async
        return call_openai_async
    # Gemini path supports both
    return call_gemini_async if use_async else call_gemini_sync


def _gemini_config_for_mode(mode: str):
    """Schema’d responses for Gemini by rating mode."""
    if mode == "info":
        return {
            "response_mime_type": "application/json",
            "response_schema": list[InfomrativeRatingClass],
            "temperature": 0.0,
        }
    if mode == "mix":
        return {
            "response_mime_type": "application/json",
            "response_schema": list[MixedRatingClass],
            "temperature": 0.0,
        }
    # default "full"
    return {
        "response_mime_type": "application/json",
        "response_schema": list[FullRatingClass],
        "temperature": 0.0,
    }


def _openai_config_for_mode(_: str):
    """OpenAI JSON-ish response config (uses your utility’s args)."""
    return {"reason_effort": "medium", "verbosity": "low", "format": "json_object"}


def _make_prompt(mode: str, input_obj: Dict[str, Any]) -> str:
    """Render the correct evaluation prompt."""

    if mode == "info":
        dimensions=["informativeness"]
    elif mode == "mix":
        dimensions=["novelty", "relevance", "implication_scope"]
    elif mode in ["informativeness", "novelty", "relevance", "implication_scope"]:
        dimensions = [mode]
    else:
        # full
        dimensions = ["informativeness", "novelty", "relevance", "implication_scope"]

    if mode == "ext":
        prompt_template = build_eval_prompt_using_external_ratings(dimensions=dimensions, context_type=input_obj["context_type"],
                                        include_example_block=True)
    else:
        prompt_template = build_eval_prompt(dimensions=dimensions, context_type=input_obj["context_type"],
                                        include_example_block=True)
    return prompt_template.format(
        topic=input_obj["topic"],
        context=input_obj["prior_knowledge"],
        target=input_obj["current_dialogue"],
        start=input_obj["start"],
        end=input_obj["end"],
        total=input_obj["total"],
    )


def create_rating_valid_func(start_index: int, last_index: int):
    """Validate utterance-level rating JSON list with required keys and index range."""
    allowed = {"utterance_index", "informativeness", "novelty", "relevance", "implication_scope", "context_type"}

    def _validate(resp_text: str) -> bool:
        try:
            rating_json = json.loads(remove_code_blocks(resp_text))
            if not isinstance(rating_json, list) or not rating_json:
                return False
            # basic keys / contiguous indices check
            seen = []
            for r in rating_json:
                if "utterance_index" not in r:
                    return False
                for k in r.keys():
                    if k not in allowed:
                        return False
                seen.append(int(r["utterance_index"]))
            # expect contiguous coverage for [start_index, last_index]
            if seen[0] != int(start_index) or seen[-1] != int(last_index):
                return False
            return True
        except Exception:
            return False

    return _validate


def create_claim_valid_func(expected_claim_ids: List[str]):
    """
    Validate claim-level ratings JSON.
    Expect a list of objects:
      { "claim_id": str, "informativeness": int, "novelty": int, "relevance": int, "implication_scope": int }
    """
    allowed = {"id", "informativeness", "novelty", "relevance", "implication_scope"}

    expected = set(map(str, expected_claim_ids))

    def _validate(resp_text: str) -> bool:
        try:
            obj = json.loads(remove_code_blocks(resp_text))
            if not isinstance(obj, list) or not obj:
                return False
            got = set()
            for item in obj:
                if not isinstance(item, dict):
                    return False
                if "id" not in item:
                    return False
                for k in item.keys():
                    if k not in allowed:
                        return False
                got.add(str(item["id"]))
            # must cover all expected claims (allow extra keys only if they’re in allowed set)
            return expected.issubset(got)
        except Exception:
            return False

    return _validate


async def _call_rater(
    llm_callable: Callable,
    backend: _Backend,
    prompt: str,
    valid_func,
    attempts: int = RETRY_ATTEMPTS,
) -> Any:
    """Generic retry wrapper for model calls, returning parsed JSON."""
    provider, model = backend
    if provider == "gemini":
        gen_config = _gemini_config_for_mode("full")  # default; the caller can render prompt per mode already
    else:
        gen_config = _openai_config_for_mode("full")

    last_err = None
    text = None

    for i in range(1, attempts + 1):
        try:
            output = await llm_callable(
                prompt,
                model=model,
                generation_config=gen_config,
                valid_func=valid_func,
            )
            text = _extract_text(output)
            return json.loads(text)
        except Exception as e:
            last_err = e
            if i < attempts:
                await asyncio.sleep(1.5 ** i)
            else:
                raise ValueError(f"Bad JSON from evaluator after {attempts} attempts.\nLast error: {last_err}\nRaw: {text}")


# ---------------------------------------------------------------------------
#  Metrics (unchanged logic, minor tidy)
# ---------------------------------------------------------------------------

def evaluate_ratings(
    prediction: List[Dict[str, Any]],
    reference: List[Dict[str, Any]],
    aspects: List[str] = ["informativeness", "novelty", "relevance", "implication_scope"],
    distance_metric: str = "cohen_kappa",
) -> float:
    pred_vec: List[int] = []
    ref_vec: List[int] = []

    if len(prediction) != len(reference):
        return 0.0

    try:
        pred_by_idx = {int(p["utterance_index"]): p for p in prediction}
        for utt in reference:
            uidx = int(utt.get("utterance_index"))
            p = pred_by_idx.get(uidx)
            if p is None:
                return 0.0
            for a in aspects:
                ref_vec.append(int(utt.get(a)))
                pred_vec.append(int(p.get(a)))
        if distance_metric == "cohen_kappa":
            return float(cohen_kappa_score(ref_vec, pred_vec))
        return float(mean_absolute_error(ref_vec, pred_vec))
    except Exception:
        return 0.0


def euclidean_distance(v1: Dict[str, int], v2: Dict[str, int]) -> float:
    a = np.array(list(v1.values()), dtype=float)
    b = np.array(list(v2.values()), dtype=float)
    return float(np.linalg.norm(a - b, ord=2))


# ---------------------------------------------------------------------------
#  Knowledge builders & utterance-level predictions (fixed)
# ---------------------------------------------------------------------------

def build_knowledge_dict(info: Dict[str, Any], methods: Optional[List[str]]) -> Dict[str, str]:
    if not methods:
        methods = ["full", "summary", "memory", "short_prior", "no_knowledge"]

    immediate_prior = info.get("immediate_prior") or info.get("immeidate_prior") or ""
    summary = info.get("summary") or ""
    full_prior = info.get("full_prior") or ""
    memory = info.get("memory") or ""

    kd: Dict[str, str] = {}
    if "full" in methods:
        kd["full"] = f"Full Prior Dialogue:\n{full_prior}\n\n"

    if "summary" in methods:
        kd["summary"] = (
            "Prior dialogue summary:\n"
            f"{summary}\n\n"
            "Immediate prior dialogue:\n"
            f"{immediate_prior}\n\n"
        )

    if "memory" in methods:
        kd["memory"] = (
            "Prior dialogue memory:\n"
            f"{memory}\n\n"
            "Immediate prior dialogue:\n"
            f"{immediate_prior}\n\n"
        )

    if "short_prior" in methods:
        kd["short_prior"] = (
            "No other prior knowledge.\n"
            "Immediate prior dialogue:\n"
            f"{immediate_prior}\n\n"
        )
    if "no_knowledge" in methods:
        kd["no_knowledge"] = (
            "No other prior knowledge.\n"
            "No prior dialogue:\n"
        )

    return kd


async def rate_text(
    llm: Callable,
    input_obj: Dict[str, Any],
    backend: _Backend,
    mode: str = "full",
):
    """Utterance-level rating for a segment, returning a list of ratings."""
    # Render prompt
    prompt = _make_prompt(mode, input_obj)

    # Backend-specific generation config
    provider, _ = backend
    if provider == "gemini":
        gen_config = _gemini_config_for_mode(mode)
    else:
        gen_config = _openai_config_for_mode(mode)

    # Retry with validation
    attempts = RETRY_ATTEMPTS
    start_i = int(input_obj["start"])
    end_i = int(input_obj["end"])
    valid_func = create_rating_valid_func(start_i, end_i)

    last_err = None
    txt = None
    for i in range(1, attempts + 1):
        try:
            output = await llm(
                prompt,
                model=backend[1],
                generation_config=gen_config,
                valid_func=valid_func,
            )
            txt = _extract_text(output)
            data = json.loads(txt)
            return data
        except Exception as e:
            last_err = e
            if i < attempts:
                await asyncio.sleep(1.5 ** i)
            else:
                raise ValueError(f"Bad JSON from evaluator after {attempts} attempts.\nLast error: {last_err}\nRaw: {txt}")


async def generate_rating_variants(
    info: Dict[str, Any],
    backend: _Backend,
    methods: List[str] | None = None,
    split_aspects: bool = False,
    use_external_ratings: bool = False,
) -> Tuple[Dict[str, str], Dict[str, List[Any]]]:
    """
    Generate multiple rating variants for a dialogue segment under different knowledge views.
    Runs ALL (mode × method) requests concurrently (bounded by a single global semaphore)
    so different modes don't block each other.
    Returns (knowledge_dict, per-method aggregated ratings).
    """
    knowledge_dict = build_knowledge_dict(info, methods)
    llm = _get_llm(backend, use_async=True)

    start, end = info["interval"]
    start_s = str(start)
    end_s = str(max(start, end - 1))
    total_s = str(max(0, end - start))

    if use_external_ratings:
        task_types = ["ext"]
    elif not split_aspects:
        task_types = ["info", "mix"]
    else:
        task_types = ["informativeness", "novelty", "relevance", "implication_scope"]


    # Single global limiter so modes don't serialize each other
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _rate_one(mode: str, name: str, knowledge: str):
        """
        Returns (mode, name, response_or_list)
        """
        payload = {
            "topic": info.get("topic", ""),
            "prior_knowledge": knowledge,
            "current_dialogue": info.get("tgt_utts", ""),
            "start": start_s,
            "end": end_s,
            "total": total_s,
            "context_type": name,
        }
        async with sem:
            try:
                resp = await rate_text(llm, payload, backend=backend, mode=mode)
            except Exception:
                resp = None
        # Normalize failure to []
        return mode, name, ([] if (resp is None) else resp)

    # Fire ALL (mode × method) tasks at once
    tasks = [
        asyncio.create_task(_rate_one(mode, name, knowledge))
        for mode in task_types
        for name, knowledge in knowledge_dict.items()
    ]

    # Gather without blocking per-mode
    results = await asyncio.gather(*tasks, return_exceptions=False)

    # Rebuild ratings_by_mode -> {mode: {method: [items]}}
    ratings_by_mode: Dict[str, Dict[str, Any]] = {m: {} for m in task_types}
    for mode, name, resp in results:
        ratings_by_mode.setdefault(mode, {})[name] = resp

    # Collapse/aggregate aspects across modes into per-method lists
    result = restructure_ratings(ratings_by_mode)
    return knowledge_dict, result

def load_human_data(path, tasks):

    utts_info = []
    for corpus in ["fora", "insq"]:
        for task in tasks:
            if corpus not in task["task_id"]:
                continue
            utts = task["target_utterances"]
            for utt in utts:
                if utt["skipped"]:
                    continue
                text = utt["utterance_text"]
                length = len(text.split(" "))
                conv_id = corpus + "_" + str(utt['conversation_id']).split("_")[-1]
                utts_info.append(
                    {'corpus': corpus, "conversation_id": conv_id, 'utterance_index': utt['utterance_index'],
                     'utterance_id': str(utt['utterance_id']), "text": text, "length": length})
    utts_df = pd.DataFrame(utts_info)
    hm_labels = pd.read_csv(path)
    final_labels = hm_labels.astype({'utterance_id': 'str'})
    final_df = hm_labels.pivot_table(
        index=["corpus", "conv_id", "utterance_id"],  # keep utterance identifiers
        columns="aspect",  # spread aspects across columns
        values=["n_raters", "y_hard", "w_conf", "p1", "p2", "p3", "p4", "y_cont", "lo", "hi",
                "mask_t1", "mask_t2", "mask_t3"],  # values to pivot
        aggfunc="first"  # if duplicates, take first
    )

    # Flatten the multi-level column index (e.g. ('y_hard','Novelty') → 'y_hard_Novelty')
    final_df.columns = [f"{val}_{aspect}" for val, aspect in final_df.columns]

    # Reset index so corpus/conv_id/utterance_id are normal columns
    final_df = final_df.reset_index()
    final_df = pd.merge(final_df, utts_df[["utterance_id", 'utterance_index', "text", "length"]], how="left",
                        on=["utterance_id"])
    hard_df = final_df[['corpus', 'conv_id', 'utterance_id', 'utterance_index', 'y_hard_CIG',
                        'y_hard_Novelty', 'y_hard_Relevance', 'y_hard_Scope', "text", "length"]]

    hard_df = hard_df.rename(
        columns={'y_hard_CIG': 'info', 'y_hard_Novelty': 'novo', 'y_hard_Relevance': "relv", "y_hard_Scope": "imsc"})

    return hard_df

def load_external_gpt_data(path, tasks):

    gpt_preds = load_json(path)
    gpt_labels = {}
    for i, id in enumerate(gpt_preds['ids']):
        if id not in gpt_labels:
            gpt_labels[str(id)] = {}
        gpt_labels[id]['info'] = gpt_preds["gpt-5_full"]["info"][i]
        gpt_labels[id]['novo'] = gpt_preds["gpt-5_full"]["novo"][i]
        gpt_labels[id]['relv'] = gpt_preds["gpt-5_full"]["relv"][i]
        gpt_labels[id]['imsc'] = gpt_preds["gpt-5_full"]["imsc"][i]
    utts_info = []
    for corpus in ["fora", "insq"]:
        for task in tasks:
            if corpus not in task["task_id"]:
                continue
            utts = task["target_utterances"]
            for utt in utts:
                if utt["skipped"]:
                    continue
                text = utt["utterance_text"]
                length = len(text.split(" "))
                conv_id = corpus + "_" + str(utt['conversation_id']).split("_")[-1]
                if str(utt['utterance_id']) not in gpt_labels:
                    continue
                gpt_label = gpt_labels[str(utt['utterance_id'])]
                utts_info.append(
                    {'corpus': corpus, "conv_id": conv_id, 'utterance_index': utt['utterance_index'],
                     'utterance_id': str(utt['utterance_id']), "text": text, "length": length,
                     "info": gpt_label["info"], "novo": gpt_label["novo"], "relv": gpt_label["relv"],
                     "imsc": gpt_label["imsc"]})
    utts_df = pd.DataFrame(utts_info)
    return utts_df





# ---------------------------------------------------------------------------
#  Utterance-level driver (minor fixes + atomic write)
# ---------------------------------------------------------------------------

async def predict_utterances_ratings(all_tasks,
                                     backend: _Backend, output,
                                     split_aspects: bool = False,
                                     external_ratings: str = None,
                                     on_segment: bool = True):
    tasks: List[Dict[str, Any]] = []

    ex_df = None
    use_external_ratings = False
    if external_ratings == "hm":
        ex_df = load_human_data("../analysis/final_labels.csv", all_tasks)
        use_external_ratings = True
    elif external_ratings == "gpt_ex":
        ex_df = load_external_gpt_data("../analysis/agg_gpt_preds.json", all_tasks)



    for task in all_tasks:
        conv_id_prefix, conv_num = task["conversation_id"].split("_")
        if conv_id_prefix in SELECTED_CONVIDS and int(conv_num) in SELECTED_CONVIDS[conv_id_prefix]:
            tasks.append(task)

    # Map conversation csv files
    dfs_dict: Dict[str, str] = {}
    for file in glob.glob("../data/raw/insq/*.csv") + glob.glob("../data/raw/fora/*.csv"):
        cid = os.path.basename(file).split(".")[0]
        dfs_dict[cid] = file

    provider, model = backend

    erro_count = 0
    for task in tqdm(tasks, total=len(tasks), desc="Utterance-level", unit="task"):
        conv_id = task["conversation_id"]

        task.setdefault("predictions", {})

        if not external_ratings:
            model_string = model
        else:
            model_string = f"{model}_{external_ratings}"

        task["predictions"].setdefault(model_string, {})
        if task["predictions"][model_string] and "full" in task["predictions"][model_string] and len(task["predictions"][model_string]["full"]) > 0:  # already filled
            continue
        else:
            print(f'catch incomplete task: {task["task_id"]}')

        segment_id = task["segment_id"]
        segments = load_json(f"../data/processed_segments/{provider}/{conv_id}_meta_checkpoint.json")
        segment = segments["segmentation"]["segments"][segment_id]
        start, end = segment["intervals"]
        start = task["target_utterances"][0]["utterance_index"]
        immediate_prior_dialogue = task["prior_history"].split("<br><br>")

        dialogue = pd.read_csv(dfs_dict[conv_id])



        prior_dialogue_text = slice_to_text(dialogue, 0, start)


        retrieved = segment.get("retreived_memories", [])
        memory = generate_formatted_memories(retrieved, start)

        topic = task["topic"] + (f" (goal: {task['goal']})" if "goal" in task else "")

        if on_segment:
            current_dialogue_text = slice_to_text(dialogue, start, end, ex_df)
            immediate_prior_dialogue_text = slice_to_text(dialogue, max(start - len(immediate_prior_dialogue), 0),
                                                          start)
            info = {
                "topic": topic,
                "tgt_utts": current_dialogue_text,
                "summary": strip_html_regex(task.get("summary", "")),
                "immeidate_prior": immediate_prior_dialogue_text,  # keep misspelling for backward compat
                "full_prior": prior_dialogue_text,
                "memory": memory,
                "interval": [start, end],
            }
            _, ratings = await generate_rating_variants(info, backend, split_aspects=split_aspects, use_external_ratings=use_external_ratings)
            if len(ratings["full"]) == 0:
                erro_count += 1

        else:
            ratings = []
            for i, utterance in enumerate(task["target_utterances"]):
                if utterance["skipped"]:
                    ratings.append(None)
                    continue
                immediate_prior_dialogue_text = slice_to_text(dialogue, max(start - len(immediate_prior_dialogue), 0),
                                                              start + i)
                current_dialogue_text = slice_to_text(dialogue, start+i, start+i+1, ex_df)
                info = {
                    "topic": topic,
                    "tgt_utts": current_dialogue_text,
                    "summary": strip_html_regex(task.get("summary", "")),
                    "immeidate_prior": immediate_prior_dialogue_text,  # keep misspelling for backward compat
                    "full_prior": prior_dialogue_text,
                    "memory": memory,
                    "interval": [start + i, start + i + 1],
                }

                _, utt_ratings = await generate_rating_variants(info, backend, split_aspects=split_aspects,
                                                            use_external_ratings=use_external_ratings)
                ratings.append(utt_ratings)




        task["predictions"][model_string] = ratings

        atomic_write_json(output, all_tasks)
    print("unfilled error cases: {}".format(erro_count))


# ---------------------------------------------------------------------------
#  CLAIM-LEVEL RATING (COMPLETED)
# ---------------------------------------------------------------------------

async def predict_claims_ratings(backend: _Backend):
    """
    For each selected task:
      - Walk segment utterances
      - If there are memory 'actions' tied to an utterance, build a claims prompt
      - Ask the model to rate each claim (with existing-memory and immediate context)
      - Save to task["claim_predictions"][model][<utterance_index>] = List[ratings]
    """
    all_tasks = load_json("../data/ratings/tasks_ratings.json")
    tasks: List[Dict[str, Any]] = []

    for task in all_tasks:
        conv_id_prefix, conv_num = task["conversation_id"].split("_")
        if conv_id_prefix in SELECTED_CONVIDS and int(conv_num) in SELECTED_CONVIDS[conv_id_prefix]:
            tasks.append(task)

    # Map conversation csv files
    dfs_dict: Dict[str, str] = {}
    for file in glob.glob("../data/raw/insq/*.csv") + glob.glob("../data/raw/fora/*.csv"):
        cid = os.path.basename(file).split(".")[0]
        dfs_dict[cid] = file

    provider, model = backend
    llm = _get_llm(backend, use_async=True)

    # Concurrency semaphore for claim calls
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    for task in tqdm(tasks, total=len(tasks), desc="Claim-level", unit="task"):
        conv_id = task["conversation_id"]
        segment_id = task["segment_id"]

        # ensure output bucket
        task.setdefault("claim_predictions", {})
        task["claim_predictions"].setdefault(model, {})

        # skip if already done
        # (remove this guard if you want to recompute)
        if task["claim_predictions"][model]:
            continue

        segments = load_json(f"../data/processed_segments/{provider}/{conv_id}_meta_checkpoint.json")
        segment = segments["segmentation"]["segments"][segment_id]

        start, end = segment["intervals"]
        dialogue = pd.read_csv(dfs_dict[conv_id])
        current_dialogue = dialogue.iloc[start:end]

        existing_memory_list = segment.get("retreived_memories", []) or []
        existing_memory: Dict[str, Any] = {mem["id"]: mem for mem in existing_memory_list}

        memory_actions: List[List[Dict[str, Any]]] = segment.get("memory_actions", []) or []
        topic = task["topic"] + (f" (goal: {task['goal']})" if "goal" in task else "")

        # We will gather async claim calls per utterance
        claim_call_futures: List[asyncio.Task] = []
        claim_meta: List[Tuple[int, List[str]]] = []  # (utterance_index, expected_claim_ids)

        for abs_idx, row in current_dialogue.iterrows():
            local_idx = abs_idx - start
            utterance_index = int(abs_idx)  # the absolute utt index in the whole dialogue

            if local_idx >= len(memory_actions):
                continue

            utterance_actions = memory_actions[local_idx] or []
            if not utterance_actions:
                continue

            # Build claims block & ensure "previous_memories" become part of existing_memory
            claim_lines: List[str] = []
            expected_claim_ids: List[str] = []
            speaker = row["utterance_speaker"]

            for action in utterance_actions:
                prev_mem = action.get("previous_memories")
                if prev_mem and isinstance(prev_mem, dict):
                    pmid = prev_mem.get("id")
                    if pmid and pmid not in existing_memory:
                        existing_memory[pmid] = prev_mem

                claim_id = str(action.get("id"))
                expected_claim_ids.append(claim_id)

                # You can enrich speaker/claim formatting as needed
                claim_lines.append(
                    f"- claim_id: {claim_id}; speaker: {speaker}; claim: {action.get('memory', '')}"
                )

            if not expected_claim_ids:
                continue

            existing_memory_string = generate_formatted_memories(list(existing_memory.values()), utterance_index)
            immediate_prior = slice_to_text(dialogue, max(abs_idx - 3, 0), abs_idx)

            prompt = EVAL_CLAIMS_WITH_MEMORY_PROMPT_FULL.format(
                topic=topic,
                dialogue_context=immediate_prior,
                existing_memories=existing_memory_string,
                claims="\n".join(claim_lines),
            )

            valid_func = create_claim_valid_func(expected_claim_ids)

            async def _one_claim_call(pmt: str, vfunc, u_idx: int, exp_ids: List[str]):
                async with sem:
                    # We want a JSON array (list) back
                    # For Gemini, set schema to a generic list of dicts; for OpenAI, request json_object but return list.
                    # We'll reuse _call_rater with an injected generation_config via prompt; simpler to inline here:

                    provider, mdl = backend
                    gen_config = _gemini_config_for_mode("full") if provider == "gemini" else _openai_config_for_mode("full")

                    # Retry loop
                    last_err, txt = None, None
                    for i in range(1, RETRY_ATTEMPTS + 1):
                        try:
                            out = await llm(
                                pmt,
                                model=mdl,
                                generation_config=gen_config,
                                valid_func=vfunc,
                            )
                            txt = _extract_text(out)
                            data = json.loads(txt)
                            # Normalize to list
                            if isinstance(data, dict) and "ratings" in data:
                                data = data["ratings"]
                            if not isinstance(data, list):
                                raise ValueError("Expected a list of claim ratings.")
                            return u_idx, data
                        except Exception as e:
                            last_err = e
                            if i < RETRY_ATTEMPTS:
                                await asyncio.sleep(1.5 ** i)
                            else:
                                # Return empty on failure but keep pipeline alive
                                return u_idx, []

            claim_call_futures.append(asyncio.create_task(_one_claim_call(prompt, valid_func, utterance_index, expected_claim_ids)))
            claim_meta.append((utterance_index, expected_claim_ids))

        if not claim_call_futures:
            # Nothing to rate in this segment
            atomic_write_json("../data/ratings/tasks_ratings.json", all_tasks)
            continue

        results = await asyncio.gather(*claim_call_futures, return_exceptions=True)

        # Stitch results per utterance index
        out_bucket: Dict[str, List[Dict[str, Any]]] = task["claim_predictions"][model]
        for res in results:
            if isinstance(res, Exception) or res is None:
                continue
            u_idx, ratings = res
            out_bucket[str(u_idx)] = ratings  # save under string key for JSON stability

        # Persist after each task to allow resuming
        atomic_write_json("../data/ratings/tasks_ratings.json", all_tasks)


# ---------------------------------------------------------------------------
#  Entrypoint
# ---------------------------------------------------------------------------

def main():
    # Example: run claim-level ratings with OpenAI
    models = ["gpt-5"]
    for i in range(1,2):
        print("Iteration {}".format(i))
        for m in models:
            print("Model: {}".format(m))
            tasks = load_json(f"../data/ratings/tasks_ratings_gptex.json")
            asyncio.run(predict_utterances_ratings(tasks, ("openai", m), output=f"../data/ratings/tasks_ratings_gptex.json", external_ratings="gpt_ex", split_aspects=True, on_segment=True))
            # asyncio.run(predict_claims_ratings(("openai", m)))


if __name__ == "__main__":
    main()