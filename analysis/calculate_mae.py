
import pandas as pd
import json
import os
import glob
import numpy as np
import sys

# Add script dir to path to import correlation_analysis functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from correlation_analysis import load_features, load_human_labels

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_human_labels_all(labels_path):
    print(f"Loading human labels from {labels_path}")
    try:
        df = pd.read_csv(labels_path)
        # Do NOT filter for CIG. Keep all aspects.
        
        # Normalize IDs
        # Need to import normalize_id or redefine it.
        # Simple version:
        if 'conv_id' in df.columns:
            df['conversation_id'] = df['conv_id'].apply(lambda x: x.split('_')[-1] if '_' in str(x) else str(x))
        
        if 'utterance_id' in df.columns:
            df['utterance_id'] = df['utterance_id'].astype(str)
            
        return df
    except Exception as e:
        print(f"Error reading labels: {e}")
        return pd.DataFrame()

def calculate_mae():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    feat_dir = os.path.join(project_root, "data", "feat")
    labels_path = os.path.join(project_root, "analysis", "final_labels.csv")
    ratings_dir = os.path.join(project_root, "data", "ratings")
    
    # 1. Load Human Labels (ALL aspects)
    print("Loading human labels...")
    human_df = load_human_labels_all(labels_path)
    print(f"Loaded {len(human_df)} human labels.")
    print(f"Aspects found: {human_df['aspect'].unique()}")
    
    # 2. Load Features (for ID mapping)
    print("Loading features for ID mapping...")
    feat_df = load_features(feat_dir)
    
    # Ensure types
    human_df['conversation_id'] = human_df['conversation_id'].astype(str)
    human_df['utterance_id'] = human_df['utterance_id'].astype(str)
    feat_df['conversation_id'] = feat_df['conversation_id'].astype(str)
    feat_df['utterance_id'] = feat_df['utterance_id'].astype(str)
    feat_df['utterance_index'] = feat_df['utterance_index'].astype(str)
    
    # Merge human labels with features to get utterance_index
    merged_human = pd.merge(human_df, feat_df[['conversation_id', 'utterance_id', 'utterance_index']], 
                            on=['conversation_id', 'utterance_id'], how='inner')
    
    print(f"Merged human labels with features: {len(merged_human)} rows.")
    
    # 3. Load Predictions (Keep runs separate)
    runs = [0, 1, 2]
    aspect_map = {
        'CIG': 'informativeness',
        'Novelty': 'novelty',
        'Relevance': 'relevance',
        'Scope': 'implication_scope'
    }
    
    # Store predictions: list of dicts
    # [ { 'run': 0, 'conv_id': ..., 'model': ..., 'context': ..., 'scores': {...} }, ... ]
    all_preds = []
    
    for run_idx in runs:
        path = os.path.join(ratings_dir, f"run_{run_idx}.json")
        if not os.path.exists(path):
            continue
            
        data = load_json(path)
        for entry in data:
            conv_id = entry.get("conversation_id")
            preds_dict = entry.get("predictions", {})
            for model, contexts_dict in preds_dict.items():
                # Filter out GPT-5_True
                if model == "gpt-5_True":
                    continue
                
                # Handle Qwen structure: List of Utterances -> Context -> Prediction
                if isinstance(contexts_dict, list):
                    for i, utt_contexts in enumerate(contexts_dict):
                        if not utt_contexts:
                            continue
                        for context_name, val in utt_contexts.items():
                            preds_list = val if isinstance(val, list) else [val]
                            for p in preds_list:
                                utt_idx = str(p.get("utterance_index"))
                                if not utt_idx or utt_idx == 'None': pass
                                
                                row = {
                                    'run': run_idx,
                                    'conversation_id': conv_id,
                                    'utterance_index': utt_idx,
                                    'model': model,
                                    'context': context_name
                                }
                                for ma in aspect_map.values():
                                    row[ma] = float(p.get(ma)) if p.get(ma) is not None else np.nan
                                all_preds.append(row)
                    continue

                if not isinstance(contexts_dict, dict):
                    continue

                # Iterate over all contexts (full, summary, memory, etc.)
                for context_name, utt_preds in contexts_dict.items():
                    if not isinstance(utt_preds, list):
                        continue
                        
                    for p in utt_preds:
                        utt_idx = str(p.get("utterance_index"))
                        row = {
                            'run': run_idx,
                            'conversation_id': conv_id,
                            'utterance_index': utt_idx,
                            'model': model,
                            'context': context_name
                        }
                        for ma in aspect_map.values():
                            row[ma] = float(p.get(ma)) if p.get(ma) is not None else np.nan
                        all_preds.append(row)

    pred_df = pd.DataFrame(all_preds)
    
    # Normalize ID
    def normalize_id(conv_id):
        if conv_id.startswith("insq_"):
            return conv_id.replace("insq_", "")
        elif conv_id.startswith("fora_"):
            parts = conv_id.split('_')
            if parts[-1].isdigit():
                return parts[-1]
        return conv_id

    pred_df['conversation_id'] = pred_df['conversation_id'].apply(normalize_id)
    
    # 4. Merge Human and Model
    results = []
    models = pred_df['model'].unique()
    contexts = pred_df['context'].unique()
    print(f"Models found: {models}")
    print(f"Contexts found: {contexts}")
    
    for model in models:
        model_subset = pred_df[pred_df['model'] == model]
        
        for context in contexts:
            context_subset = model_subset[model_subset['context'] == context]
            if context_subset.empty:
                continue

            for human_aspect, model_aspect in aspect_map.items():
                human_subset = merged_human[merged_human['aspect'] == human_aspect].copy()
                
                # Merge on (conv_id, utt_index)
                # Note: context_subset has multiple rows per (conv_id, utt_index) due to runs
                # We merge, which will duplicate human rows for each run
                merged = pd.merge(human_subset, context_subset[['run', 'conversation_id', 'utterance_index', model_aspect]], 
                                  on=['conversation_id', 'utterance_index'], how='inner')
                
                if not merged.empty:
                    merged = merged.dropna(subset=['y_cont', model_aspect])
                    
                    # Group by Corpus and Run first
                    # Calculate MAE per run
                    run_maes = []
                    
                    # We need to group by corpus to report per-corpus results
                    for corpus, corpus_group in merged.groupby('corpus'):
                        
                        # Calculate MAE for each run in this corpus
                        per_run_stats = []
                        for run_idx, run_group in corpus_group.groupby('run'):
                            mae = np.mean(np.abs(run_group['y_cont'] - run_group[model_aspect]))
                            per_run_stats.append(mae)
                        
                        if per_run_stats:
                            avg_mae = np.mean(per_run_stats)
                            std_mae = np.std(per_run_stats)
                            results.append({
                                'corpus': corpus,
                                'model': model,
                                'context': context,
                                'aspect': human_aspect,
                                'mae': avg_mae,
                                'mae_std': std_mae,
                                'n_runs': len(per_run_stats),
                                'n_samples': len(corpus_group) // len(per_run_stats) # Approx samples per run
                            })
    
    # 5. Output Results
    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("No results calculated. Check ID matching.")
        return

    results_df = results_df.sort_values(['corpus', 'aspect', 'model', 'context'])
    
    print("\nMAE Results (Average across runs):")
    print(results_df)
    
    out_path = os.path.join(project_root, "analysis", "mae_results_detailed.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")

if __name__ == "__main__":
    calculate_mae()
