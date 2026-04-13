
import json
import os

def debug_qwen():
    path = "data/archive_local/ratings/run_0.json"
    if not os.path.exists(path):
        print("run_0.json not found")
        return

    data = json.load(open(path))
    
    qwen_keys = [k for k in data[0]['predictions'].keys() if 'Qwen' in k]
    if not qwen_keys:
        print("No Qwen models found in first entry.")
        # Search other entries
        for i, entry in enumerate(data):
            keys = [k for k in entry['predictions'].keys() if 'Qwen' in k]
            if keys:
                print(f"Found Qwen in entry {i}: {keys}")
                qwen_keys = keys
                break
    
    if not qwen_keys:
        print("No Qwen models found in any entry.")
        return

    qwen_model = qwen_keys[0]
    print(f"Inspecting model: {qwen_model}")
    
    found_valid = False
    for i, entry in enumerate(data):
        preds = entry['predictions'].get(qwen_model)
        if not preds:
            continue
            
        print(f"\nEntry {i} ({entry.get('conversation_id')}): Type {type(preds)}")
        if isinstance(preds, list):
            print(f"  List length: {len(preds)}")
            for j, item in enumerate(preds):
                if item is None:
                    print(f"    Item {j}: None")
                else:
                    print(f"    Item {j}: Keys {list(item.keys())}")
                    # Check if 'full' has content
                    if 'full' in item and item['full']:
                        print(f"      'full' has {len(item['full'])} predictions.")
                        print(f"      Sample: {item['full'][0]}")
                        found_valid = True
        elif isinstance(preds, dict):
             print(f"  Dict keys: {list(preds.keys())}")
             if 'full' in preds and preds['full']:
                 found_valid = True
        
        if found_valid:
            break

if __name__ == "__main__":
    debug_qwen()
