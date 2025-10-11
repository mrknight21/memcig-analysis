import pandas as pd
import numpy as np
import re
import glob
import json
import os
from util import load_json
from scipy.stats import ttest_ind
from statsmodels.api import OLS, add_constant
from statsmodels.stats.sandwich_covariance import cov_cluster

ACT_ALL = ['prob','conf','inst','intp','supp','util']
MTV_ALL = ['info_mtv','social_mtv','coord_mtv']

seg_files = glob.glob("../data/processed_segments/openai/*.json")
insq_files = glob.glob("../data/whow/insq/*.xlsx")
fora_files = glob.glob("../data/whow/fora/*.csv")

dfs_dict= {}
for file in glob.glob("../data/raw/insq/*.csv") + glob.glob("../data/raw/fora/*.csv"):
    cid = os.path.basename(file).split(".")[0]
    dfs_dict[cid] = file

def find_whow(corpus, conv_id):
    if corpus == "insq":
        for f in insq_files:
            f_conv_id = f.split("/")[-1].split("_")[1]
            if f_conv_id == str(conv_id):
                return pd.read_excel(f, index_col=0)
    else:
        for f in fora_files:
            f_conv_id = f.split("/")[-1].split("_")[-1].replace(".csv", "")
            if f_conv_id == str(conv_id):
                return pd.read_csv(f, index_col=0)
    return None

def aggregate_sequence_dfs():
    for seg_file in seg_files:

        seq_data = []

        segment_info = load_json(seg_file)

        corpus = "insq" if "insq" in seg_file else "fora"
        conv_id = segment_info['conversation_id'] if "insq" in seg_file else segment_info['conversation_id'].split("_")[-1]

        whow_df = find_whow(corpus, conv_id)
        dialogue_df = pd.read_csv(dfs_dict[f"{corpus}_{conv_id}"])

        if whow_df is None or dialogue_df is None:
            continue

        segments = segment_info['segmentation']["segments"]
        for segment in segments:
            start, end = segment["intervals"]
            full_ratings = segment["full_context_rating"]
            seg_dialogue = dialogue_df.iloc[start:end]
            assert len(full_ratings) == len(seg_dialogue)



            if corpus == "insq":
                id_start, id_end = int(seg_dialogue.iloc[0].utterance_id), int(seg_dialogue.iloc[-1].utterance_id)
                whow_sub = [utt.to_dict() for i, utt in whow_df.iterrows() if int(utt.id.split("_")[0]) in range(id_start, id_end + 1)]
            else:
                whow_sub = [utt.to_dict() for i, utt in whow_df.iterrows() if int(utt.id.split("_")[-2]) in range(start + 1, end + 1)]

            local_index = 0
            for utt_index, row in seg_dialogue.iterrows():
                role = row.role
                text = row.utterance_text
                rating = full_ratings[local_index]
                utt_index = row.utterance_index
                utt_id = row.utterance_id
                utt_record = {"corpus_id": corpus, "conv_id": conv_id,
                              "utterance_id": utt_id, "utt_index": utt_index,
                              "role": role, "text": text,
                              "info": rating["informativeness"],
                              "novo": rating["novelty"],
                              "relv": rating["relevance"],
                              "imsc": rating["implication_scope"],
                              "info_mtv": None, "social_mtv": None,"coord_mtv": None,
                              "prob": None, "conf": None, "inst": None, "intp": None,
                              "supp": None, "util": None}

                if role in ["mod", "moderator", "host"]:
                    if corpus == "insq":
                        mod_utts = [utt for utt in whow_sub if utt['id'].split("_")[0] == str(utt_id)]
                        utt_record['info_mtv'] = any([utt["informational motive(gpt)"] for utt in mod_utts])
                        utt_record['social_mtv'] = any([utt["social motive(gpt)"] for utt in mod_utts])
                        utt_record['coord_mtv'] = any([utt["coordinative motive(gpt)"] for utt in mod_utts])
                        utt_record['prob'] = any([utt["dialogue act(gpt)"] == 0 for utt in mod_utts])
                        utt_record['conf'] = any([utt["dialogue act(gpt)"] == 1 for utt in mod_utts])
                        utt_record['inst'] = any([utt["dialogue act(gpt)"] == 2 for utt in mod_utts])
                        utt_record['intp'] = any([utt["dialogue act(gpt)"] == 3 for utt in mod_utts])
                        utt_record['supp'] = any([utt["dialogue act(gpt)"] == 4 for utt in mod_utts])
                        utt_record['util'] = any([utt["dialogue act(gpt)"] == 5 for utt in mod_utts])
                    else:
                        mod_utts = [utt for utt in whow_sub if int(utt['id'].split("_")[-2]) == int(utt_index) + 1]
                        utt_record['info_mtv'] = any([utt["informational motive"] for utt in mod_utts])
                        utt_record['social_mtv'] = any([utt["social motive"] for utt in mod_utts])
                        utt_record['coord_mtv'] = any([utt["coordinative motive"] for utt in mod_utts])
                        utt_record['prob'] = any([utt["dialogue_act"] == "Probing" for utt in mod_utts])
                        utt_record['conf'] = any([utt["dialogue_act"] == "Confronting" for utt in mod_utts])
                        utt_record['inst'] = any([utt["dialogue_act"] == "Instruction" for utt in mod_utts])
                        utt_record['intp'] = any([utt["dialogue_act"] == "Interpretation" for utt in mod_utts])
                        utt_record['supp'] = any([utt["dialogue_act"] == "Supplement" for utt in mod_utts])
                        utt_record['util'] = any([utt["dialogue_act"] == "All Utility" for utt in mod_utts])

                seq_data.append(utt_record)
                local_index += 1

        seq_df = pd.DataFrame(seq_data)
        seq_df.to_csv(f"../data/whow_sequences/{corpus}_{conv_id}.csv", index=False)

def stack_episodes(dfs, episode_ids=None, infer_conv_from='utterance_id'):
    """
    Combine multiple episode-level DataFrames into a single conversation-level DataFrame.

    Parameters
    ----------
    dfs : List[pd.DataFrame] or Dict[str, pd.DataFrame]
        Each df is one episode (one conversation).
        If a dict, keys are used as conv_ids automatically.
    episode_ids : Optional[List[str]]
        Explicit conv_id for each df in `dfs` (ignored if `dfs` is a dict).
        If None, we will try:
          1) existing 'conv_id' column if present and single-valued per df;
          2) infer from `infer_conv_from` by dropping the last underscore token;
          3) fall back to epi_001, epi_002, ...
    infer_conv_from : str
        Column name to infer conv_id from (default 'utterance_id') by removing the last "_token".

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with a valid 'conv_id' column and aligned columns.
    """
    # Accept dict input
    if isinstance(dfs, dict):
        episode_ids = list(dfs.keys())
        dfs = list(dfs.values())
    else:
        episode_ids = episode_ids or [None]*len(dfs)

    # ---- 1) unified schema (union of all columns) ----
    all_cols = set()
    for d in dfs:
        all_cols |= set(d.columns)
    all_cols = list(all_cols)

    combined = []
    auto_id_counter = 1

    for i, d in enumerate(dfs):
        df = d.copy()

        # Align schema
        for c in all_cols:
            if c not in df.columns:
                df[c] = np.nan
        df = df[all_cols]

        # ---- 2) ensure conv_id ----
        conv_id = episode_ids[i]
        if conv_id is None:
            if 'conv_id' in df.columns and df['conv_id'].notna().any():
                # use existing if single-valued, else coerce to single value
                uniq = df['conv_id'].dropna().unique()
                conv_id = str(uniq[0]) if len(uniq) >= 1 else None

        if conv_id is None and infer_conv_from in df.columns:
            # try to infer from utterance_id like "EGI_..._2733_16" -> drop last token
            def _infer(u):
                s = str(u).split('_')
                return '_'.join(s[:-1]) if len(s) > 1 else None
            inferred = df[infer_conv_from].dropna().map(_infer)
            conv_id = inferred.mode().iloc[0] if not inferred.dropna().empty else None

        if conv_id is None:
            conv_id = f"epi_{auto_id_counter:03d}"
            auto_id_counter += 1

        df['conv_id'] = conv_id

        # ---- 3) normalize types for key analysis cols if present ----
        # turn index
        if 'utt_index' in df.columns:
            df['utt_index'] = pd.to_numeric(df['utt_index'], errors='coerce')
        # role/text
        if 'role' in df.columns:
            df['role'] = df['role'].astype(str)
        if 'text' in df.columns:
            df['text'] = df['text'].astype(str)

        # outcomes
        for c in ['info','novo','relv','imsc']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # moderator binary labels → {0,1} floats
        act_cols  = [c for c in ['prob','conf','inst','intp','supp','util'] if c in df.columns]
        mtv_cols  = [c for c in ['info_mtv','social_mtv','coord_mtv'] if c in df.columns]
        for c in act_cols + mtv_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0).clip(0,1)

        combined.append(df)

    # ---- 4) concat and sort ----
    out = pd.concat(combined, ignore_index=True)

    # optional: ensure unique utterance_id
    if 'utterance_id' in out.columns:
        dup = out['utterance_id'].duplicated(keep=False)
        if dup.any():
            # append conv_id to duplicates to make unique but traceable
            out.loc[dup, 'utterance_id'] = (
                out.loc[dup, 'utterance_id'].astype(str) + '::' + out.loc[dup, 'conv_id'].astype(str)
            )

    # sort by conv_id then utt_index if available
    sort_cols = ['conv_id'] + (['utt_index'] if 'utt_index' in out.columns else [])
    out = out.sort_values(sort_cols, kind='mergesort').reset_index(drop=True)

    return out

def build_adjacency_asof(df, moderator_roles=('moderator','mod','chair','host','facilitator','mc')):
    d = df.copy()

    # Normalize & coerce
    d['role_norm'] = d['role'].astype(str).str.lower().str.strip()
    d['is_mod'] = d['role_norm'].isin(set(moderator_roles)).astype(int)
    d['utt_index'] = pd.to_numeric(d['utt_index'], errors='coerce')

    # Ensure expected mod feature columns exist (0/1 floats)
    for c in ACT_ALL + MTV_ALL:
        if c not in d.columns:
            d[c] = 0.0
        d[c] = pd.to_numeric(d[c], errors='coerce').fillna(0.0).clip(0,1)

    # Moderator table (unique per conv_id, utt_index)
    mod = (d[d['is_mod']==1]
           .dropna(subset=['utt_index'])
           .sort_values(['conv_id','utt_index'])
           [['conv_id','utt_index'] + ACT_ALL + MTV_ALL]
           .copy())
    if not mod.empty:
        mod = mod.groupby(['conv_id','utt_index'], as_index=False)[ACT_ALL+MTV_ALL].max()

    # Participant table
    part = (d[(d['is_mod']==0) & d['utt_index'].notna()]
              .sort_values(['conv_id','utt_index'])
              .copy())

    if mod.empty or part.empty:
        # No detected moderators for at least some convs: still return part with prev_* = 0
        out = part.copy()
        for c in ACT_ALL + MTV_ALL:
            out[f'prev_{c}'] = 0.0
        return out

    # Per-conversation asof merge
    out_frames = []
    for conv, g_part in part.groupby('conv_id', sort=False):
        g_mod = mod[mod['conv_id']==conv]
        g_mod = g_mod.rename(columns={'utt_index': 'mod_utt_index'})
        if g_mod.empty:
            g = g_part.copy()
            for c in ACT_ALL + MTV_ALL:
                g[f'prev_{c}'] = 0.0
            out_frames.append(g)
            continue

        g = pd.merge_asof(
            g_part.sort_values('utt_index'),
            g_mod.sort_values('mod_utt_index'),
            by='conv_id', left_on='utt_index', right_on='mod_utt_index',
            direction='backward', allow_exact_matches=True
        )
        # 3) Distance (turns) from the preceding moderator; NaN if none exists yet
        g['lag_from_prev_mod'] = g['utt_index'] - g['mod_utt_index']
        # (optional) make it a nullable integer
        g['lag_from_prev_mod'] = g['lag_from_prev_mod'].astype('Int64')

        # prev_* features (robust creation)
        for c in ACT_ALL + MTV_ALL:
            src = g[f"{c}_y"] if c in g.columns else pd.Series(np.nan, index=g.index)
            g[f'prev_{c}'] = pd.to_numeric(src, errors='coerce').fillna(0.0).clip(0,1)

        # Keep participant columns + prev_* only
        g = g.drop(columns=[f"{c}_y" for c in (ACT_ALL + MTV_ALL) if c in g.columns], errors='ignore')
        g = g.drop(columns=[f"{c}_x" for c in (ACT_ALL + MTV_ALL) if c in g.columns], errors='ignore')
        desired = list(g_part.columns) + [f'prev_{c}' for c in (ACT_ALL + MTV_ALL)]
        existing = [c for c in desired if c in g.columns]
        g = g.loc[:, existing]
        out_frames.append(g)

    out = pd.concat(out_frames, ignore_index=True)

    # Make sure outcomes are numeric
    for col in ['info','novo','relv','imsc']:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')

    return out

def summarize_adjacency(part):
    part = part.copy()
    part['info'] = pd.to_numeric(part['info'], errors='coerce')
    baseline = part['info'].mean()

    # Only the intended binary prev_* features (act + motivation)
    candidate_prev = [f'prev_{c}' for c in (ACT_ALL + MTV_ALL)]
    prev_cols = [c for c in candidate_prev if c in part.columns]

    records = []
    for key in prev_cols:
        x = part[[key, 'info']].dropna(subset=['info']).copy()
        x[key] = pd.to_numeric(x[key], errors='coerce').fillna(0).clip(0,1)

        g = (x.groupby(key)['info']
               .agg(mean='mean',
                    sem=lambda s: s.std(ddof=1)/np.sqrt(len(s)) if len(s)>1 else np.nan,
                    count='count')
               .reset_index().rename(columns={key:'level'}))
        g['level'] = g['level'].astype(int)
        g['feature'] = key.replace('prev_','')
        g['lift_vs_overall'] = g['mean'] - baseline

        x1 = x.loc[x[key]==1, 'info']; x0 = x.loc[x[key]==0, 'info']
        if len(x1) >= 2 and len(x0) >= 2:
            _, pval = ttest_ind(x1, x0, equal_var=False, nan_policy='omit')
        else:
            pval = np.nan

        g['p_value'] = np.where(g['level']==1, pval, np.nan)
        g['sig'] = (g['level']==1) & (g['p_value'].notna()) & (g['p_value']<0.05)
        g['lift_vs_overall_marked'] = [
            f"{v:.3f}{'*' if s else ''}" for v, s in zip(g['lift_vs_overall'], g['sig'])
        ]
        records.append(g)

    out = pd.concat(records, ignore_index=True) if records else pd.DataFrame(
        columns=['feature','level','mean','sem','count','lift_vs_overall','lift_vs_overall_marked','p_value']
    )
    out = out.sort_values(['feature','level']).reset_index(drop=True)
    return baseline, out[['feature','level','mean','sem','count','lift_vs_overall','lift_vs_overall_marked','p_value']]

def main():
    seq_files = glob.glob("../data/whow_sequences/*.csv")
    seq_dfs = [pd.read_csv(f) for f in seq_files]
    seq_dfs = [df for df in seq_dfs if len(df) > 100]
    df_all = stack_episodes(seq_dfs)

    part = build_adjacency_asof(df_all, moderator_roles=('moderator', 'mod', 'host'))

    print("Rows in part:", len(part))
    print("Any nonzero prev_* counts?",
          {c: int(part[c].sum()) for c in part.columns if c.startswith('prev_')})

    print("Rows in part:", len(part))
    print("Prev feature columns:",
          [c for c in part.columns if c in [f'prev_{x}' for x in (ACT_ALL + MTV_ALL)]])

    baseline, adj_tab = summarize_adjacency(part)  # your Welch t-test version with stars
    print(f"Overall participant informativeness mean: {baseline:.3f}")

    adj_1 = (adj_tab.query("level==1")
             .sort_values("lift_vs_overall", ascending=False)
             .reset_index(drop=True))
    adj_1[['feature', 'count', 'mean', 'sem', 'lift_vs_overall_marked', 'p_value']].head(20)




if __name__ == "__main__":
    main()
