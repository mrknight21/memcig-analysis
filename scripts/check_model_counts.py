
import json
import os
from collections import Counter

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def check_counts():
    base_dir = "data/archive_local/ratings"
    runs = [0, 1, 2]
    
    for run_idx in runs:
        filename = f"run_{run_idx}.json"
        path = os.path.join(base_dir, filename)
        
        if not os.path.exists(path):
            print(f"Run {run_idx}: File not found ({path})")
            continue
            
        data = load_json(path)
        print(f"\n--- Checking Run {run_idx} ({len(data)} segments) ---")
        
        model_counts = Counter()
        model_sets = {} # map frozenset of models to count
        
        discrepancies = []
        
        # First pass: gather stats
        all_model_keys = set()
        for entry in data:
            preds = entry.get("predictions", {})
            models = sorted(list(preds.keys()))
            model_counts[len(models)] += 1
            
            model_set = frozenset(models)
            if model_set not in model_sets:
                model_sets[model_set] = 0
            model_sets[model_set] += 1
            
            all_model_keys.update(models)
            
            # Store for detailed check
            conv_id = entry.get("conversation_id")
            utt_idx = entry.get("utterance_index")
            entry_id = f"{conv_id}_{utt_idx}"
            
        # Determine majority set
        if not model_sets:
            print("  No predictions found.")
            continue
            
        majority_set = max(model_sets, key=model_sets.get)
        majority_count = model_sets[majority_set]
        
        print(f"  Most common model set ({majority_count}/{len(data)}):")
        print(f"    {sorted(list(majority_set))}")
        
        if len(model_sets) > 1:
            print(f"  WARNING: Found {len(model_sets)} different sets of models.")
            for mset, count in model_sets.items():
                if mset != majority_set:
                    print(f"    Set with {len(mset)} models (count {count}):")
                    print(f"      Missing: {sorted(list(majority_set - mset))}")
                    print(f"      Extra: {sorted(list(mset - majority_set))}")
                    
            # List specific segments with issues
            print("  Segments with deviant model sets:")
            for entry in data:
                preds = entry.get("predictions", {})
                models = frozenset(preds.keys())
                if models != majority_set:
                    conv_id = entry.get("conversation_id")
                    utt_idx = entry.get("utterance_index")
                    print(f"    {conv_id} (idx {utt_idx}): has {len(models)} models")
        else:
            print("  OK: All segments have the exact same set of models.")

if __name__ == "__main__":
    check_counts()
