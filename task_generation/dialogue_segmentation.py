# ── Dialogue segmentation (OpenAI or Gemini) ────────────────────────────
from __future__ import annotations

import glob
from copy import deepcopy
import os, re, json, asyncio, numpy as np
from typing import List, Tuple, Literal
import pandas as pd
from prompts import SEG_SYSTEM_MSG, SEG_SCHEMA, SegmentClass
from llm.gemini_utility import *
from llm.openai_utility import *

from util import parse_json_markdown, load_json

def generate_segments_validation(last_utterance_index):
    def validate_segments(response):
        try:
            resp_text = response
            segments = json.loads(resp_text)

            last_seg = segments[-1]
            last_bound = last_seg['utterances_interval'][1]
            if last_bound < last_utterance_index -1:
                return False
            elif last_bound > last_utterance_index:
                return False
            else:
                return True
        except Exception:
            return False
    return validate_segments

def get_llm(backend: Literal["openai", "gemini"], use_async: bool = False):
    if backend == "openai":
        if use_async:
            return call_openai_async
        else:
            return call_openai

    else:
        if use_async:
            return call_gemini_async
        else:
            return call_gemini_sync


# ── 0. Choose your giant-context models here ────────────────────────────
OPENAI_MODEL      = "gpt-4o"   # or "gpt-4o" etc.
GEMINI_MODEL      = "gemini-2.5-pro-preview-05-06"
# GEMINI_MODEL      = "gemini-2.5-flash-preview-05-20"
MAX_TOKENS_OUT    = 512             # adjust for your model



# ── 2.  Backend-specific SDK setups ─────────────────────────────────────
import openai
from google import genai
from google.genai import types
from dotenv import load_dotenv
import logging

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_org_key = os.getenv("OPENAI_ORG_KEY")
openai_client = openai.OpenAI(api_key=openai_api_key, organization=openai_org_key)

gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# -----------------------------------------------------------------------
# 3. Helpers
# -----------------------------------------------------------------------
def build_segmentation_prompt(dialogue: List[Tuple[int, str, str]],
                              topic: str,
                              prior_summary: str) -> str:
    """
    Fills in the dialogue lines plus the topic/prior summary into the user prompt.
    """
    # number each utterance
    lines = [f"{i}. {spk}: {txt}"  for i, spk, txt in dialogue]
    dialogue_block = "\n".join(lines)

    return (
        f"Goal: You are an expert dialogue analyst. Your task is to segment an interaction phase into coherent subtopic segments of around 500-800 words or no loner that 15 speaker turns each."
        f"Dialogue:\n\n{dialogue_block}\n\n"
        f"Return the segmented list into JSON format as: \n{SEG_SCHEMA}\n"
    )


# -----------------------------------------------------------------------
# 4. Diversity runs → voting
# -----------------------------------------------------------------------
async def get_interval_runs(dialogue: List[Tuple[int, str, str]],
                            p: int,
                            backend: str,
                            topic: str = None,
                            context: str = None,
                            last_utterance_index: int = None,
                            max_temp: float = 0.0,
                            use_async: bool = False):
    prompt = build_segmentation_prompt(dialogue, topic, context)
    sysprompt = SEG_SYSTEM_MSG.format(
                topic=topic,
                prior_summary=(context or "(none yet)")
            )

    llm = get_llm(backend, use_async=use_async)

    configs = []
    temps  = []
    for i in range(p):
        if max_temp == 0.0:
            temps.append(0.0)
            configs.append(dict(response_mime_type="application/json", response_schema=list[SegmentClass], temperature= 0.0))
        else:
            temp = (max_temp / (p-1)) * i
            temps.append(temp)
            configs.append(dict(response_mime_type="application/json", response_schema=list[SegmentClass], temperature= temp))

    is_async = asyncio.iscoroutinefunction(llm)
    if is_async:
        if last_utterance_index:
            runs = await asyncio.gather(*[llm(prompt,generation_config=configs[i], valid_func=generate_segments_validation(last_utterance_index)) for i in range(p)])
        else:
            runs = await asyncio.gather(
                *[llm(prompt, generation_config=configs[i])
                  for i in range(p)])
    else:
        runs = [llm(prompt, generation_config=configs[i]) for i in range(p)]
    if backend == "gemini":
        runs = [run.parsed for run in runs]     # length p
    else:
        runs = [json.loads(run) for run in runs]
    return runs


def accumulate_votes(runs: List[List[Tuple[int,int]]], dialogue_len: int, start_index: int=0) -> np.ndarray:
    votes = np.zeros(dialogue_len, float)
    for run in runs:
        intervals = [seg['utterances_interval'] for seg in run]
        for b, _ in intervals:
            index = b - start_index
            # skip first boundary (=1)
            for idx, w in ((index,1), (index-1,0.5), (index+1,0.5)):
                if 0 <= idx < dialogue_len -1:
                    votes[idx] += w
    return votes / len(runs)             # 0-1 scores


def pick_boundaries(scores: np.ndarray,
                    tau: float,
                    min_len: int,
                    max_len: int,
                    start_index: int = 0) -> Tuple[List[int], List[float]]:
    """
    Returns 'bounds' as segment *starts* (absolute indices).
    Final segment ends at start_index + len(scores) - 1 (end-inclusive) or +len(scores) (end-exclusive).

    Rules:
      • Prefer local peaks >= tau (3-point neighborhood).
      • Enforce: each segment length >= min_len.
      • Enforce: distance between consecutive starts <= max_len.
      • Only use fallback (best score, rightmost tie) when remaining tail > max_len.
      • Never add a boundary that would leave the final segment < min_len.
    """
    n = len(scores)
    if n == 0:
        return [start_index], [1.0]

    # Local peaks (exclude edges for 3-pt neighborhood)
    if n >= 3:
        is_peak = (scores[1:-1] >= tau) & (scores[1:-1] >= scores[:-2]) & (scores[1:-1] >= scores[2:])
        peaks = np.nonzero(is_peak)[0] + 1
    else:
        peaks = np.array([], dtype=int)

    bounds: List[int] = [start_index]
    confs:  List[float] = [1.0]

    last = 0  # local index for last boundary
    while True:
        remaining = (n - 1) - last  # distance from last start to final utterance (inclusive index space)

        # If we cannot start another segment of at least min_len, stop.
        if remaining < min_len:
            break

        # Any next boundary must leave >= min_len for the final segment:
        # so candidate <= (n - 1) - min_len
        cap_max = (n - 1) - min_len
        # Also, never use the very last utterance as a boundary
        hard_max = min(cap_max, n - 2)

        # Window to search for the next boundary start
        wmin = last + min_len
        if wmin > hard_max:
            break  # placing any boundary would violate the tail-min_len rule

        if remaining <= max_len:
            # Tail is already within max_len: only cut if there is a true peak >= tau
            cand_peaks = peaks[(peaks >= wmin) & (peaks <= hard_max)]
            if cand_peaks.size == 0:
                break  # no qualifying peak -> stop; last segment runs to the end
            # Pick the highest-scoring peak (rightmost on ties)
            scores_at = scores[cand_peaks]
            best = np.where(scores_at == scores_at.max())[0][-1]
            chosen = int(cand_peaks[best])
            chosen_score = float(scores[chosen])

        else:
            # remaining > max_len: we MUST place a boundary within [last+min_len, last+max_len],
            # still respecting the tail-min_len cap.
            wmax = min(last + max_len, hard_max)
            if wmin > wmax:
                break  # defensive

            cand_peaks = peaks[(peaks >= wmin) & (peaks <= wmax)]
            if cand_peaks.size > 0:
                scores_at = scores[cand_peaks]
                best = np.where(scores_at == scores_at.max())[0][-1]  # rightmost best peak
                chosen = int(cand_peaks[best])
                chosen_score = float(scores[chosen])
            else:
                # Fallback only in this branch: best score in window, rightmost tie
                segment = scores[wmin:wmax + 1]
                maxval = float(segment.max())
                rel_idx = np.where(segment == maxval)[0]
                chosen = wmin + int(rel_idx[-1])
                chosen_score = maxval

        # Append boundary
        bounds.append(chosen + start_index)
        confs.append(chosen_score)
        last = chosen

    return bounds, confs


def to_segments(bounds: List[int], confidences: List[int], last_index: int) :
    bounds.append(last_index)
    confidences.append(1)
    segments = []
    for i in range(len(bounds) - 1):
        segments.append({
            "intervals": [bounds[i], bounds[i+1]],
            "confidence": [confidences[i], confidences[i+1]]
        })
    return segments

def _calculate_error_margin(wordcount, min_wordcount, max_wordcount):
    """
    Helper function to calculate the error margin of a segment's word count.
    The error is zero if the word count is within the desired min/max range.
    Otherwise, it's the absolute difference from the nearest boundary.
    """
    if wordcount < min_wordcount:
        return min_wordcount - wordcount
    elif wordcount > max_wordcount:
        return wordcount - max_wordcount
    return 0

def merge_segments(segments, dialogue, min_wordcount=450, max_wordcount=750, max_utterances=25):
    """
    Merges dialogue segments based on word count, utterance count, and a strict
    error margin criterion to minimize deviation from the ideal word count range.

    Args:
        segments (list): A list of segment dictionaries. Each dict must have
                         an "intervals" key with a (start, end) tuple.
        dialogue (list): A list of utterance tuples, where each tuple is
                         (timestamp, speaker, text).
        min_wordcount (int): The minimum desired word count for a merged segment.
        max_wordcount (int): The maximum desired word count for a merged segment.
        max_utterances (int): The maximum number of utterances allowed in a
                              merged segment.

    Returns:
        list: A new list of merged segment dictionaries.
    """
    if not segments:
        return []

    # First, pre-process each segment to calculate its initial word and utterance count.
    for seg in segments:
        start, end = seg["intervals"]
        # Filter utterances that fall within the segment's time interval
        segment_utterances = [utt for utt in dialogue if utt[0] >= start and utt[0] < end]
        seg_text = [utt[2] for utt in segment_utterances]
        seg["wordcount"] = sum(len(str(s).split()) for s in seg_text)
        seg["utterance_count"] = len(segment_utterances)

    merged = []
    buffer = None  # This holds the segment we are currently building

    for seg in segments:
        # If there's no buffer, start with the current segment
        if buffer is None:
            buffer = deepcopy(seg)
            continue

        # Calculate the potential properties of a new segment if we merge
        tentative_wc = buffer["wordcount"] + seg["wordcount"]
        tentative_uc = buffer["utterance_count"] + seg["utterance_count"]

        # Calculate the error margin for the buffer as it is, and for the potential merged segment
        error_margin_buffer = _calculate_error_margin(buffer["wordcount"], min_wordcount, max_wordcount)
        error_margin_merged = _calculate_error_margin(tentative_wc, min_wordcount, max_wordcount)

        # --- Stricter Merging Criteria ---
        # 1. The merged segment must have a smaller error margin than the buffer alone.
        # 2. The total number of utterances must not exceed the maximum allowed.
        if error_margin_merged < error_margin_buffer and tentative_uc <= max_utterances:
            # If criteria are met, merge the current segment into the buffer
            buffer["intervals"] = (buffer["intervals"][0], seg["intervals"][1])
            buffer["confidence"] = (buffer["confidence"][0], seg["confidence"][1])
            buffer["wordcount"] = tentative_wc
            buffer["utterance_count"] = tentative_uc
        else:
            # If criteria are not met, the buffer is considered complete.
            merged.append(buffer)
            # The current segment becomes the start of a new buffer.
            buffer = deepcopy(seg)

    # After the loop, don't forget to add the last segment still in the buffer
    if buffer is not None:
        merged.append(buffer)

    return merged

# -----------------------------------------------------------------------
# 5. Public API
# -----------------------------------------------------------------------
async def segment_dialogue(
    dialogue: List[Tuple[int, str, str]],
    *,
    use_async: bool=True,
    p: int = 5,
    tau: float = .6,
    min_len: int = 3,
    max_len: int = 20,
    backend: Literal["openai","gemini"] = "openai",
    topic: str = None,
    context: str = None,
):
    start_index = dialogue[0][0]
    last_index = start_index + len(dialogue) -1
    runs  = await get_interval_runs(dialogue, p, backend, topic, context, use_async=use_async, last_utterance_index=last_index)
    scores = accumulate_votes(runs, len(dialogue), start_index = start_index)
    bounds, confidences = pick_boundaries(scores, tau, min_len, max_len, start_index=dialogue[0][0])
    segments =  to_segments(bounds, confidences, last_index)
    segments = merge_segments(segments, dialogue)
    segments_dict = {"segments": segments, "boundary_conf_mean": np.mean(confidences),
                     "segments_num": len(segments),
                     "segments_intervals_len_mean": np.mean([s["intervals"][1] - s["intervals"][0] for s in segments])}
    return segments_dict

async def  segment_insq_dialogues(p=5, update_csv=False):
    dialogue_files = glob.glob("../data/raw/*.csv")
    for i, dialogue_file in enumerate(dialogue_files):
        print("Processing dialogue file: {}".format(dialogue_file))
        print(f"Progress {i} / {len(dialogue_files)}")
        meta_file = dialogue_file.replace(".csv", "_meta.json").replace("raw", "meta")
        meta = load_json(meta_file)
        dialogue = pd.read_csv(dialogue_file)
        discussion = dialogue.loc[dialogue.phase == 1]
        discussion = [(int(r.utterance_index), f"{r.utterance_speaker} ({r.role}): ", r.utterance_text) for i, r in discussion.iterrows()]
        topic = meta["topic"]
        intro_sum = meta["intro_summary"]
        segmentation = await segment_dialogue(discussion, p=p, backend="openai", topic=topic, context=intro_sum, use_async=True)
        meta["segmentation"] = segmentation
        segments = segmentation["segments"]
        with open(meta_file.replace("/meta", "/cache"), "w") as f:
            json.dump(meta, f)
        if update_csv:
            for seg_idx, segment in enumerate(segments):
                start, end = segment["intervals"]
                # mark rows whose utterance_index falls inside this interval
                mask = dialogue["utterance_index"].between(start, end, inclusive="both")
                dialogue.loc[mask, "segment"] = seg_idx
            dialogue.to_csv(dialogue_file.replace("/raw", "/cache"), index=False)


# -----------------------------------------------------------------------
# 6. Demo
# -----------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(segment_insq_dialogues(update_csv=True))