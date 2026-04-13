
import pandas as pd
import glob
import os
import sys

# Add script dir to path to import correlation_analysis functions if needed, 
# but simpler to just copy/paste the load functions or import them.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from correlation_analysis import load_features, load_human_labels

def debug_missing_mem():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    feat_dir = os.path.join(project_root, "data", "feat")
    labels_path = os.path.join(project_root, "analysis", "final_labels.csv")
    
    # Load data
    print("Loading features...")
    feat_df = load_features(feat_dir)
    print(f"Features: {len(feat_df)}")
    
    print("Loading labels...")
    labels_df = load_human_labels(labels_path)
    print(f"Labels: {len(labels_df)}")
    
    # Ensure types
    feat_df['conversation_id'] = feat_df['conversation_id'].astype(str)
    feat_df['utterance_id'] = feat_df['utterance_id'].astype(str)
    labels_df['conversation_id'] = labels_df['conversation_id'].astype(str)
    labels_df['utterance_id'] = labels_df['utterance_id'].astype(str)
    
    # Merge
    merged_df = pd.merge(labels_df, feat_df, on=['conversation_id', 'utterance_id'], how='left')
    print(f"Merged (Left Join on Labels): {len(merged_df)}")
    
    # Check memory columns
    mem_cols = ['mem_delta', 'mem_delta_info', 'mem_delta_relv']
    other_cols = ['token_count', 'tfidf_sum']
    
    print("\nMissing Values Count:")
    print(merged_df[mem_cols + other_cols].isna().sum())
    
    # Inspect rows where mem_delta is NaN
    missing_mem = merged_df[merged_df['mem_delta'].isna()]
    if not missing_mem.empty:
        print(f"\n{len(missing_mem)} rows missing memory features.")
        print("Sample of missing rows:")
        print(missing_mem[['conversation_id', 'utterance_id', 'token_count']].head())
        
        # Check if they are from specific conversations
        print("\nMissing by Conversation ID:")
        print(missing_mem['conversation_id'].value_counts())
        
    # Check if they are 0s that got dropped? 
    # The correlation script drops NaNs. 
    # Let's check if there are 0s in the original feature file that might be treated as missing if the user thought 0 meant missing, 
    # but the script only drops NaNs.
    # Wait, did I see NaNs in the feature files?
    
    # Let's check if the feature files themselves have NaNs for these columns
    print("\nChecking raw feature dataframe for NaNs in memory columns:")
    print(feat_df[mem_cols].isna().sum())
    
    # Check if 'mem_rating_used' is relevant?
    if 'mem_rating_used' in feat_df.columns:
         print("\nmem_rating_used distribution:")
         print(feat_df['mem_rating_used'].value_counts(dropna=False))

if __name__ == "__main__":
    debug_missing_mem()
