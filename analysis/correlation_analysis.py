
import pandas as pd
import numpy as np
import glob
import os
from scipy.stats import spearmanr, pearsonr
import argparse

def normalize_id(val):
    """Extract the last numeric sequence from an ID string."""
    import re
    s = str(val)
    # Find all numeric sequences
    nums = re.findall(r'\d+', s)
    if nums:
        return nums[-1] # Return the last one (e.g. 2187 from fora_..._2187)
    return s

def load_features(feat_dir):
    all_dfs = []
    feat_files = glob.glob(os.path.join(feat_dir, "*", "*.csv"))
    print(f"Found {len(feat_files)} feature files in {feat_dir}")

    for f in feat_files:
        try:
            df = pd.read_csv(f)
            # Normalize IDs
            if 'conversation_id' in df.columns:
                df['conversation_id'] = df['conversation_id'].apply(normalize_id)
            if 'utterance_id' in df.columns:
                df['utterance_id'] = df['utterance_id'].astype(str)
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not all_dfs:
        return pd.DataFrame()
    
    return pd.concat(all_dfs, ignore_index=True)

def load_human_labels(labels_path):
    print(f"Loading human labels from {labels_path}")
    try:
        df = pd.read_csv(labels_path)
        # Filter for CIG aspect
        df = df[df['aspect'] == 'CIG'].copy()
        
        # Normalize IDs
        if 'conv_id' in df.columns:
            df['conversation_id'] = df['conv_id'].apply(normalize_id)
        
        if 'utterance_id' in df.columns:
            df['utterance_id'] = df['utterance_id'].astype(str)
            
        return df
    except Exception as e:
        print(f"Error reading labels: {e}")
        return pd.DataFrame()

def load_gpt5_predictions(ratings_dir):
    import json
    all_preds = []
    
    # Load all 3 files
    for i in range(3):
        path = os.path.join(ratings_dir, f"tasks_ratings_{i}.json")
        if not os.path.exists(path):
            print(f"Warning: {path} not found.")
            continue
            
        print(f"Loading {path}...")
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            for task in data:
                conv_id = task.get('conversation_id')
                if not conv_id:
                    continue
                
                # Normalize conv_id (e.g. insq_2228 -> 2228)
                conv_id = normalize_id(conv_id)
                
                # Extract predictions
                # Path: predictions -> gpt-5 -> full -> list of dicts
                preds = task.get('predictions', {}).get('gpt-5', {}).get('full', [])
                
                for p in preds:
                    utt_idx = p.get('utterance_index')
                    info_score = p.get('informativeness')
                    
                    if utt_idx is not None and info_score is not None:
                        all_preds.append({
                            'conversation_id': conv_id,
                            'utterance_index': int(utt_idx),
                            'gpt5_informativeness': float(info_score)
                        })
                        
        except Exception as e:
            print(f"Error reading {path}: {e}")
            
    if not all_preds:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_preds)
    # Average across runs if multiple predictions exist for the same utterance
    df = df.groupby(['conversation_id', 'utterance_index'], as_index=False)['gpt5_informativeness'].mean()
    
    return df

def calculate_correlations(merged_df, target_col, feature_cols):
    results = []
    
    for feature in feature_cols:
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(merged_df[feature]):
            continue
            
        # Drop NaNs for pair
        valid_data = merged_df[[target_col, feature]].dropna()
        
        if len(valid_data) < 10: # Minimum sample size
            continue
            
        spearman_corr, spearman_p = spearmanr(valid_data[target_col], valid_data[feature])
        pearson_corr, pearson_p = pearsonr(valid_data[target_col], valid_data[feature])
        
        results.append({
            'proxy': feature,
            'spearman_r': spearman_corr,
            'spearman_p': spearman_p,
            'pearson_r': pearson_corr,
            'pearson_p': pearson_p,
            'n': len(valid_data)
        })
            
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Correlation analysis: Human/GPT-5 vs Proxies.")
    parser.add_argument("--feat_dir", type=str, default="data/archive_local/feat", help="Directory containing feature CSVs")
    parser.add_argument("--labels_path", type=str, default="analysis/final_labels.csv", help="Path to human labels CSV")
    parser.add_argument("--ratings_dir", type=str, default="data/archive_local/ratings", help="Directory containing GPT-5 ratings JSONs")
    parser.add_argument("--target_source", type=str, default="human", choices=["human", "gpt5"], help="Target for correlation: 'human' or 'gpt5'")
    parser.add_argument("--filter_by_human", action="store_true", help="If true, filter samples to match human annotations")
    args = parser.parse_args()

    # Handle paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    def resolve_path(path_str: str) -> str:
        return path_str if os.path.isabs(path_str) else os.path.join(project_root, path_str)

    feat_dir = resolve_path(args.feat_dir)
    labels_path = resolve_path(args.labels_path)
    ratings_dir = resolve_path(args.ratings_dir)
    
    # Load features
    feat_df = load_features(feat_dir)
    if feat_df.empty:
        print("Error: No features loaded.")
        return
    print(f"Loaded {len(feat_df)} feature rows.")

    # Load target data
    if args.target_source == "human":
        target_df = load_human_labels(labels_path)
        target_col = 'y_cont'
        merge_on = ['conversation_id', 'utterance_id']
        print(f"Loaded {len(target_df)} human label rows.")
    else:
        target_df = load_gpt5_predictions(ratings_dir)
        target_col = 'gpt5_informativeness'
        merge_on = ['conversation_id', 'utterance_index']
        print(f"Loaded {len(target_df)} GPT-5 prediction rows.")
        
        # Filter by human labels if requested
        if args.filter_by_human:
            human_df = load_human_labels(labels_path)
            if not human_df.empty:
                # Ensure types match for merge
                human_df['conversation_id'] = human_df['conversation_id'].astype(str)
                human_df['utterance_id'] = human_df['utterance_id'].astype(str)
                target_df['conversation_id'] = target_df['conversation_id'].astype(str)
                target_df['utterance_index'] = target_df['utterance_index'].astype(str)
                
                # Filter target_df to keep only rows present in human_df
                # Note: human_df uses 'utterance_id', gpt5 uses 'utterance_index'.
                # We need to map them.
                # Actually, looking at previous steps, 'utterance_id' in human labels IS the index for INSQ?
                # Let's check debug output from Step 99:
                # Human: insq_2228, 2243 (utterance_id)
                # Feature: 10612, 10612 (utterance_id), 0 (utterance_index)
                
                # Wait, for INSQ, utterance_id seems to be a global ID, not index.
                # GPT-5 predictions use 'utterance_index'.
                # Feature files have both 'utterance_id' and 'utterance_index'.
                # We should filter AFTER merging with features, because features link id <-> index.
                pass

    if target_df.empty:
        print("Error: No target data loaded.")
        return

    # Merge
    # Ensure merge keys match types
    for key in merge_on:
        feat_df[key] = feat_df[key].astype(str)
        target_df[key] = target_df[key].astype(str)
    
    merged_df = pd.merge(target_df, feat_df, on=merge_on, how='inner')
    print(f"Merged data size: {len(merged_df)}")
    
    # Apply filter if requested (now that we have both ID and Index from features)
    if args.filter_by_human and args.target_source == "gpt5":
        human_df = load_human_labels(labels_path)
        if not human_df.empty:
            print("Filtering by human labels...")
            human_df['conversation_id'] = human_df['conversation_id'].astype(str)
            human_df['utterance_id'] = human_df['utterance_id'].astype(str)
            
            # Create a set of valid (conv_id, utt_id) pairs
            valid_pairs = set(zip(human_df['conversation_id'], human_df['utterance_id']))
            
            # Filter merged_df
            # merged_df should have 'utterance_id' from feat_df
            merged_df = merged_df[merged_df.apply(lambda x: (str(x['conversation_id']), str(x['utterance_id'])) in valid_pairs, axis=1)]
            print(f"Filtered data size: {len(merged_df)}")
    
    if merged_df.empty:
        print("No matching rows found after merge/filter. Check IDs.")
        return

    # Define proxies (all numeric columns from features, excluding IDs and target)
    exclude_cols = ['conversation_id', 'utterance_id', 'utterance_index', 'title', 'phase', 'role', 'utterance_text', 'segment', 'utterance_speaker', 'conv_id', 'aspect', 'y_hard', 'w_conf', 'p1', 'p2', 'p3', 'p4', 'y_cont', 'lo', 'hi', 'mask_t1', 'mask_t2', 'mask_t3', 'n_raters', 'gpt5_informativeness']
    feature_cols = [c for c in feat_df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(feat_df[c])]
    
    print(f"Calculating correlations for {len(feature_cols)} proxies against {target_col}...")
    
    corr_df = calculate_correlations(merged_df, target_col, feature_cols)
    
    if corr_df.empty:
        print("No correlations calculated.")
        return

    # Rank by absolute Spearman correlation
    corr_df['abs_spearman'] = corr_df['spearman_r'].abs()
    ranked_df = corr_df.sort_values(by='abs_spearman', ascending=False)

    print(f"\nTop Correlations with {args.target_source.upper()} Target ({target_col}):")
    print(ranked_df[['proxy', 'spearman_r', 'pearson_r', 'n']].head(20).to_markdown(index=False))
    
    # Save results
    output_filename = f"{args.target_source}_correlation_results.csv"
    output_path = os.path.join(base_dir, output_filename)
    ranked_df.to_csv(output_path, index=False)
    print(f"\nFull results saved to {output_path}")

if __name__ == "__main__":
    main()
