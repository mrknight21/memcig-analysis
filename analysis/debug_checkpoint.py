
import json
import os

def check_checkpoint(conv_id):
    path = f"data/archive_local/processed_segments/openai/{conv_id}_meta_checkpoint.json"
    if not os.path.exists(path):
        path = f"data/raw/insq/{conv_id}_meta.json"
    print(f"Checking {path}...")
    
    if not os.path.exists(path):
        print("File not found.")
        return

    try:
        with open(path, 'r') as f:
            meta = json.load(f)
        print("JSON loaded successfully.")
        
        segments = meta.get("segmentation", {}).get("segments")
        if segments is None:
            print("Error: 'segments' key missing or None.")
        else:
            print(f"Found {len(segments)} segments.")
            for i, seg in enumerate(segments):
                if "intervals" not in seg:
                    print(f"Segment {i} missing 'intervals'")
                if "memory_actions" not in seg:
                    print(f"Segment {i} missing 'memory_actions'")
                
                intervals = seg.get("intervals")
                if intervals and len(intervals) != 2:
                    print(f"Segment {i} invalid intervals: {intervals}")
                    
                mem_actions = seg.get("memory_actions")
                if mem_actions is None:
                    print(f"Segment {i} memory_actions is None")
            
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_checkpoint("insq_2228")
