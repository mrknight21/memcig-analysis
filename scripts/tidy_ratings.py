
import json
import os
import glob

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def merge_ratings(run_index):
    base_dir = "data/archive_local/ratings"
    # Patterns for this run index
    # Note: tasks_ratings_0.json, tasks_ratings_gemini_0.json, tasks_ratings_oss_0.json
    patterns = [
        f"tasks_ratings_{run_index}.json",
        f"tasks_ratings_gemini_{run_index}.json",
        f"tasks_ratings_oss_{run_index}.json"
    ]
    
    merged_map = {} # Key: (conversation_id, utterance_index) -> entry dict
    
    print(f"Merging run {run_index}...")
    
    for filename in patterns:
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            print(f"  Skipping {filename} (not found)")
            continue
            
        print(f"  Loading {filename}...")
        data = load_json(path)
        print(f"    Found {len(data)} entries.")
        
        for entry in data:
            # Create a unique key for the segment
            conv_id = entry.get("conversation_id")
            # Use segment_id as the primary identifier for segments
            seg_id = entry.get("segment_id")
            
            # Filter 1: Must have valid IDs
            if not conv_id or seg_id is None:
                # print(f"Skipping missing ID: {conv_id}, {seg_id}")
                continue
            
            seg_id = str(seg_id)
            
            # Filter 2: Must have predictions
            preds = entry.get("predictions", {})
            if not preds:
                # print(f"Skipping empty preds: {conv_id}, {seg_id}")
                continue
                
            key = (conv_id, seg_id)
            
            if key not in merged_map:
                # Initialize with this entry
                merged_map[key] = entry
            else:
                # Merge predictions
                existing_preds = merged_map[key].get("predictions", {})
                
                # Update existing with new. 
                existing_preds.update(preds)
                merged_map[key]["predictions"] = existing_preds
                
                # Also merge claim_predictions if present
                if "claim_predictions" in entry:
                    if "claim_predictions" not in merged_map[key]:
                        merged_map[key]["claim_predictions"] = {}
                    merged_map[key]["claim_predictions"].update(entry["claim_predictions"])

    # Convert map back to list
    merged_list = list(merged_map.values())
    print(f"  Total merged entries: {len(merged_list)}")
    
    output_path = os.path.join(base_dir, f"run_{run_index}.json")
    save_json(merged_list, output_path)
    print(f"  Saved to {output_path}")

def main():
    for i in range(3):
        merge_ratings(i)

if __name__ == "__main__":
    main()
