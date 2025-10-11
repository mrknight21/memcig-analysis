import numpy as np
import pandas as pd
import glob
import json
from llm.openai_utility import call_openai
from util import load_json
from task_generation.dialogue_segmentation import segment_dialogue
import asyncio

phase_prompt_template = '''
You are a phase segmenter for a community discussion transcript from:
"City of Durham Neighborhood Improvement Services" and
"Be Connected Durham & Beyond Conversations."

PHASE DEFINITIONS (keys must match EXACTLY):
{
  "opening": "Purpose, agenda, ground rules, housekeeping, facilitator opens.",
  "discussion": "Discussion for the vision, engagement, and improvement of the community.",
  "wrap-up": "Closing thoughts, final questions, next steps, thanks, sign-off."
}

TASK
Given a transcript already split into utterances U[0..N-1] (zero-based indexing), identify ONE contiguous block for each phase and return the start index (inclusive) and end index (exclusive).

RULES
- Output JSON ONLY, no extra text.
- Indices are integers in [1, N].
- Each phase is optional. If a phase is absent, use null instead of a range.
- No overlaps across phases. If content interleaves, choose the best single contiguous block per phase.
- If a phase appears multiple times, pick the longest contiguous block that best fits the definition (tie → earliest).
- Minimal valid span is length 1 (e.g., [i, i+1]).
- All utterance have to be covered.
- The phase should followed the order of opening -> discussion -> wrap-up.

INPUT
Transcript (array of utterances): <dialogue_string>

OUTPUT FORMAT (JSON):
{
  "opening": [int, int] | null,
  "discussion": [int, int] | null,
  "wrap-up": [int, int] | null,
}
'''

phases_code = {"opening":0, "discussion":1, "wrap-up":2}

def mark_phase_range(df, start: int, end: int, value: int = 1, inplace: bool = True):
    d = df if inplace else df.copy()
    ui = pd.to_numeric(d['utterance_index'], errors='coerce')
    d.loc[(ui >= start) & (ui < end), 'phase'] = value
    return None if inplace else d

def segment_phases():
    df_files = glob.glob("../data/raw/fora/*.csv")
    for f in df_files:
        if "196_2733" not in f:
            continue
        cov_id = int(f.split("_")[-1].replace(".csv", ""))
        print("processing {} discussion".format(cov_id))
        df = pd.read_csv(f)
        with open(f.replace(".csv", "_meta.json"), "r") as file:
            meta = json.load(file)
        dialogue_string = "\n".join(
        f"{row.utterance_index}. {row.utterance_speaker} ({row.role}): {row.utterance_text}"
        for _, row in df.iterrows())
        prompt = phase_prompt_template.replace("<dialogue_string>", dialogue_string)

        attempts = 3
        complete = False
        while attempts > 0 and not complete:
            try:
                attempts -= 1
                phases = call_openai(prompt, "gpt-5")
                phases = json.loads(phases)
                for phase, code in phases_code.items():
                    if phase not in phases:
                        break

                for phase, indexes in phases.items():

                    code = phases_code[phase]
                    if indexes:
                        mark_phase_range(df, indexes[0], indexes[1], code)
                complete = True
            except Exception as e:
                print(e)
                print(f"Try again with {str(attempts)} left.")
        meta["phases"] = phases
        meta["goal"] = "Gather community residents' visions, priorities, experience of engagement and needs to inform City of Durham planning in infrastructure and community engagement improvements."

        # df.to_csv(f, index=False)
        with open(f.replace(".csv", "_meta.json"), "w") as file:
            json.dump(meta, file)
        print(f"updated {f} with phases!")

async def  segment_fora_dialogues(p=5, update_csv=True):
    dialogue_files = glob.glob("../data/raw/fora/*.csv")
    for i, dialogue_file in enumerate(dialogue_files):
        print("Processing dialogue file: {}".format(dialogue_file))
        print(f"Progress {i} / {len(dialogue_files)}")
        if "196_2733" not in dialogue_file:
            continue
        meta_file = dialogue_file.replace(".csv", "_meta.json")
        meta = load_json(meta_file)
        dialogue = pd.read_csv(dialogue_file)
        if 'Unnamed: 0' in dialogue.columns:
            print("cleaned columns")
            dialogue = dialogue.drop(columns=['Unnamed: 0'])
        if dialogue.utterance_index[0] != 0:
            print("correct utterance index")
            dialogue.utterance_index = range(0, len(dialogue))
        discussion = dialogue.loc[dialogue.phase == 1]
        discussion = [(int(r.utterance_index), f"{r.utterance_speaker} ({r.role}): ", r.utterance_text) for i, r in discussion.iterrows()]
        topic = meta["goal"]
        opening = "\n".join(
        f"{row.utterance_index}. {row.utterance_speaker} ({row.role}): {row.utterance_text}"
        for _, row in dialogue.loc[dialogue.phase == 0].iterrows())
        segmentation = await segment_dialogue(discussion, p=p, backend="openai", topic=topic, context=opening, use_async=True)
        meta["segmentation"] = segmentation
        segments = segmentation["segments"]
        with open(meta_file, "w") as f:
            json.dump(meta, f)
        if update_csv:
            for seg_idx, segment in enumerate(segments):
                start, end = segment["intervals"]
                # mark rows whose utterance_index falls inside this interval
                mask = dialogue["utterance_index"].between(start, end, inclusive="both")
                dialogue.loc[mask, "segment"] = seg_idx
            dialogue.to_csv(dialogue_file, index=False)


if __name__ == "__main__":
    # segment_phases()
    asyncio.run(segment_fora_dialogues(update_csv=True))