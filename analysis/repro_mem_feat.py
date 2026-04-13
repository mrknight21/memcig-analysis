
import pandas as pd
import json
import sys
import os

# Add scripts to path
sys.path.append(os.path.abspath("scripts"))
# We can't easily import from scripts/mem_features.py if it has global variables or imports that fail.
# But let's try importing the function.
try:
    from scripts.mem_features import episode_iterative_features
except ImportError:
    # Copy paste the function if import fails due to structure
    print("Import failed, defining function locally.")
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
                # Check if target_utterances exists and is not empty
                if "target_utterances" in segment_rating and segment_rating["target_utterances"]:
                    rating_start, rating_end = segment_rating["target_utterances"][0]["utterance_index"], segment_rating["target_utterances"][-1]["utterance_index"]
                    rating_index = range(rating_start, rating_end+1)
                
                if "claim_predictions" in segment_rating and "gpt-5" in segment_rating["claim_predictions"]:
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
                            # assert len(utt_rating) == len(memory_actions) 
                            # Commenting out assert to see if it passes or prints mismatch
                            if len(utt_rating) != len(memory_actions):
                                print(f"Mismatch at {j}: rating {len(utt_rating)} vs actions {len(memory_actions)}")
                            
                            for k, r in enumerate(utt_rating):
                                if k < len(memory_actions):
                                    r["event"] = memory_actions[k]["event"]
                            
                            mem_delta_info[j] = len([r for r in utt_rating if r.get("informativeness", 0) > 2 and r.get("event") != "NONE"])
                            mem_delta_novo[j] = len([r for r in utt_rating if r.get("novelty", 0) > 2 and r.get("event") != "NONE"])
                            mem_delta_relv[j] = len([r for r in utt_rating if r.get("relevance", 0) > 2 and r.get("event") != "NONE"])
                            mem_delta_imsc[j] = len([r for r in utt_rating if r.get("implication_scope", 0) > 2 and r.get("event") != "NONE"])
                            mem_delta_triad[j] = len([r for r in utt_rating if r.get("novelty", 0) > 2 and r.get("relevance", 0) > 2 and r.get("implication_scope", 0) > 2  and r.get("event") != "NONE"])
                    local_index += 1
                except Exception as e:
                    print(f"unmatch memory action items vs utterances: segment {i} error {e}")
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

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def repro():
    # INSQ is the redistributable transcript corpus in the public release.
    # This helper still depends on archived local ratings/checkpoints.
    target_ids = ["insq_2228"]
    ratings_path = "data/archive_local/ratings/tasks_ratings.json"
    print(f"Loading {ratings_path}...")
    all_ratings = load_json(ratings_path)
    
    for conv_id in target_ids:
        print(f"\nProcessing {conv_id}...")
        csv_path = f"data/raw/insq/{conv_id}.csv"
        meta_path = f"data/archive_local/processed_segments/openai/{conv_id}_meta_checkpoint.json"
        
        if not os.path.exists(csv_path):
            print(f"CSV not found: {csv_path}")
            continue
            
        print(f"Loading {csv_path}...")
        df = pd.read_csv(csv_path)
        
        print(f"Loading {meta_path}...")
        if not os.path.exists(meta_path):
            print(f"Meta checkpoint not found: {meta_path}")
            continue
            
        meta = load_json(meta_path)
        segments = meta["segmentation"]["segments"]
        
        ratings = [r for r in all_ratings if r["conversation_id"] == conv_id and "claim_predictions" in r]
        print(f"Found {len(ratings)} ratings for {conv_id}")
        
        try:
            enriched = episode_iterative_features(df, segments, ratings, "fora")
            print("Success! Saving...")
            out_path = f"data/archive_local/feat/insq/{conv_id}.csv"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            enriched.to_csv(out_path, index=False)
            print("Saved.")
        except Exception as e:
            print(f"Crashed on {conv_id}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    repro()
