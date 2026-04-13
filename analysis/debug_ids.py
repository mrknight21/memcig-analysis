
import pandas as pd
import glob
import os

def check_ids():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    feat_dir = os.path.join(project_root, "data", "feat")
    labels_path = os.path.join(project_root, "analysis", "final_labels.csv")
    
    print("--- Human Labels ---")
    df_labels = pd.read_csv(labels_path)
    print(df_labels[['conv_id', 'utterance_id']].head())
    print(df_labels[['conv_id', 'utterance_id']].dtypes)
    
    print("\n--- Feature Files ---")
    feat_files = glob.glob(os.path.join(feat_dir, "*", "*.csv"))
    if feat_files:
        df_feat = pd.read_csv(feat_files[0])
        print(f"File: {feat_files[0]}")
        print(df_feat[['conversation_id', 'utterance_id']].head())
        print(df_feat[['conversation_id', 'utterance_id']].dtypes)
    else:
        print("No feature files found.")

if __name__ == "__main__":
    check_ids()
