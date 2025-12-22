
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Iterable, List, Dict
from util import load_json
import ast

def _to_float_list(x):
    """Coerce a cell into a list[float]. Handles lists, strings like '[...]', or NaN."""
    x = eval(x)
    if isinstance(x, list):
        return [float(v) for v in x]
    if pd.isna(x):
        return []
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, (list, tuple)):
                return [float(v) for v in parsed]
        except Exception:
            return []
    # last resort: try iterating
    try:
        return [float(v) for v in x]
    except Exception:
        return []

def top_quartile_mean(vals):
    """Mean of the top 25% largest values. Returns NaN for empty."""
    if not vals:
        return np.nan
    n = len(vals)
    k = max(1, int(np.ceil(0.25 * n)))  # at least 1 element
    # largest first (so for negatives, 'largest' = least negative)
    return float(np.mean(sorted(vals, reverse=True)[:k]))

def read_all_csvs(input_dir: Path) -> Dict[Path, pd.DataFrame]:
    csvs = {}
    for p in sorted(input_dir.glob("*.csv")):
        df = pd.read_csv(p)
        # sanity: required column
        assert "utterance_text" in df.columns, f"Missing 'utterance_text' in {p}"
        csvs[p] = df
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {input_dir}")
    return csvs


def merge_entropy_feat(input_dir, output_dir):
    insq_ent = pd.read_csv("../data/entropy/insq_meta_llama_Llama_32_3B_c-1_meta.csv")
    fora_ent = pd.read_csv("../data/entropy/fora_meta_llama_Llama_32_3B_c-1_meta.csv")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Reading CSVs...")
    csvs = read_all_csvs(input_dir)

    print("Augmenting episodes with episode-level entropy features...")
    for path, df in csvs.items():
        print(f"  Processing: {path.name}")
        session_id = path.name.split(".")[0]
        corpus = session_id.split("_")[0]
        conv_id = int(session_id.split("_")[1])

        if corpus == "insq":
            ent_sub = insq_ent[insq_ent.conv_id == conv_id]
        else:
            ent_sub = fora_ent[fora_ent.conv_id == session_id]
        if len(ent_sub) == 0:
            print(f"missing entropy computation: {session_id}")
            continue

        # enriched = episode_iterative_features(df, eps_segments, eps_ratings, corpus)
        enriched = df.merge(ent_sub[["index", 'tokens_h', 'sum_h', 'sent_avg_h', 'norm_sent_avg_h']], how="left",
                          left_on='utterance_index', right_on='index')
        enriched["top_quatile_avg_ent"] = (
            enriched["tokens_h"]
            .apply(_to_float_list)
            .apply(top_quartile_mean)
        )
        out_path = output_dir / path.name
        enriched.to_csv(out_path, index=False)

    print("Done. Augmented CSVs written to:", str(output_dir))


def parse_args():
    ap = argparse.ArgumentParser(description="Compute memory based features for debate episodes.")
    ap.add_argument("--input_dir", type=Path, required=True, help="Folder containing episode CSVs")
    ap.add_argument("--output_dir", type=Path, required=True, help="Folder to write augmented CSVs")

    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    merge_entropy_feat(args.input_dir, args.output_dir)