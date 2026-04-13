import argparse
from pathlib import Path
from typing import Iterable, List, Dict, Set, Tuple
import pandas as pd
import os
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from util import load_json

ratings_file = "data/archive_local/ratings/tasks_ratings.json"
segment_dir = "data/archive_local/processed_segments/openai"


def read_all_csvs(input_dir: Path, corpus: str = "insq") -> Dict[Path, pd.DataFrame]:
    csvs = {}
    for p in sorted(input_dir.glob(f"{corpus}_*.csv")):
        df = pd.read_csv(p)
        # sanity: required column
        assert "utterance_text" in df.columns, f"Missing 'utterance_text' in {p}"
        csvs[p] = df
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {input_dir}")
    return csvs

def episode_iterative_features(dialogue, segments, ratings, corpus):
    dialogue_len = len(dialogue)
    claim_count = [0] * dialogue_len
    mem_delta = [0] * dialogue_len
    mem_delta_info = [0] * dialogue_len
    mem_delta_novo = [0] * dialogue_len
    mem_delta_relv = [0] * dialogue_len
    mem_delta_imsc = [0] * dialogue_len
    mem_delta_triad = [0] * dialogue_len


    mem_rating_used = [False] * dialogue_len

    for i, segment in enumerate(segments):
        start, end = segment["intervals"]
        claim_preds = []
        segment_rating = [r for r in ratings if r["segment_id"] == i]
        rating_index = []
        if len(segment_rating) > 0:
            segment_rating = segment_rating[0]
            rating_start, rating_end = segment_rating["target_utterances"][0]["utterance_index"], segment_rating["target_utterances"][-1]["utterance_index"]
            rating_index = range(rating_start, rating_end+1)
            claim_preds = segment_rating["claim_predictions"]["gpt-5"]

        seg_memory_actions = segment["memory_actions"]
        local_index = 0
        for j in range(start, end):
            try:
                memory_actions = seg_memory_actions[local_index]
                claim_count[j] = len(memory_actions)
                mem_delta[j] = len([a for a in memory_actions if a["event"] != "NONE"])

                if j in rating_index:
                    mem_rating_used[j] = True
                    utt_rating = claim_preds.get(str(j), [])
                    if len(utt_rating) > 0:
                        assert len(utt_rating) == len(memory_actions)
                        for k, r in enumerate(utt_rating):
                            r["event"] = memory_actions[k]["event"]
                        mem_delta_info[j] = len([r for r in utt_rating if r["informativeness"] > 2 and r["event"] != "NONE"])
                        mem_delta_novo[j] = len([r for r in utt_rating if r["novelty"] > 2 and r["event"] != "NONE"])
                        mem_delta_relv[j] = len([r for r in utt_rating if r["relevance"] > 2 and r["event"] != "NONE"])
                        mem_delta_imsc[j] = len([r for r in utt_rating if r["implication_scope"] > 2 and r["event"] != "NONE"])
                        mem_delta_triad[j] = len([r for r in utt_rating if r["novelty"] > 2 and r["relevance"] > 2 and r["implication_scope"] > 2  and r["event"] != "NONE"])
                local_index += 1
            except Exception as e:
                print(f"unmatch memory action items vs utterances: segment {i}")
                break

    dialogue["claim_count"] = claim_count
    dialogue["mem_delta"] = mem_delta
    dialogue["mem_rating_used"] = mem_rating_used
    dialogue["mem_delta_info"] = mem_delta_info
    dialogue["mem_delta_novo"] = mem_delta_novo
    dialogue["mem_delta_relv"] = mem_delta_relv
    dialogue["mem_delta_imsc"] = mem_delta_imsc
    dialogue["mem_delta_triad"] = mem_delta_triad

    return dialogue



def generate_memory_feats(input_dir, output_dir, corpus: str = "insq", ratings_path: Path = Path(ratings_file), segment_meta_dir: Path = Path(segment_dir)):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Reading CSVs...")
    csvs = read_all_csvs(input_dir, corpus=corpus)
    ratings = load_json(ratings_path)

    print("Augmenting episodes with episode-level memory features...")
    for path, df in csvs.items():
        print(f"  Processing: {path.name}")
        conv_id = path.name.split(".")[0]
        corpus = conv_id.split("_")[0]
        eps_ratings = [r for r in ratings if r["conversation_id"] == conv_id and "claim_predictions" in r]

        meta_path = segment_meta_dir / f"{conv_id}_meta_checkpoint.json"
        if not os.path.isfile(meta_path):
            continue
        else:
            meta = load_json(meta_path)
            eps_segments = meta["segmentation"]["segments"]

        enriched = episode_iterative_features(df, eps_segments, eps_ratings, corpus)
        out_path = output_dir / path.name
        enriched.to_csv(out_path, index=False)

    print("Done. Augmented CSVs written to:", str(output_dir))


def parse_args():
    ap = argparse.ArgumentParser(description="Compute memory based features for debate episodes.")
    ap.add_argument("--input_dir", type=Path, default=Path("data/raw/insq"), help="Folder containing episode CSVs")
    ap.add_argument("--output_dir", type=Path, default=Path("data/archive_local/feat/insq"), help="Folder to write augmented CSVs")
    ap.add_argument("--corpus", choices=["insq", "fora"], default="insq")
    ap.add_argument("--ratings_path", type=Path, default=Path(ratings_file))
    ap.add_argument("--segment_meta_dir", type=Path, default=Path(segment_dir))

    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    generate_memory_feats(
        args.input_dir,
        args.output_dir,
        corpus=args.corpus,
        ratings_path=args.ratings_path,
        segment_meta_dir=args.segment_meta_dir,
    )
