#!/usr/bin/env python3
"""
Debate Informativeness Feature Pipeline

Given a folder of CSV files (one per episode) with columns:
  title, conversation_id, phase, utterance_id, utterance_index,
  utterance_speaker, role, nontext, utterance_text, segment

This script creates additional feature columns that capture several
informativeness-related signals, computed iteratively through the
conversation order within each episode:

Features per utterance
----------------------
Token/length basics
- token_count: total tokens (spaCy)
- content_token_count: content tokens (non-stopword alphabetic lemmas)

TF–IDF (episode-level, utterance-as-document)
- For each conversation_id (episode), we fit a TF–IDF on all its utterances
- tfidf_mean: mean TF–IDF weight of nonzero terms in the utterance
- tfidf_sum: sum TF–IDF weights in the utterance
- tfidf_max: max TF–IDF weight in the utterance

Lexical Specificity (rarity; episode-level IDF)
- specificity_mean_idf: mean IDF value (from the episode's vectorizer) of the utterance's content lemmas
- specificity_median_idf: median IDF value (episode-level)

Novelty (within-episode, cumulative)
- novel_word_count: # unique content lemmas not seen before in this episode
- novel_word_density: novel_word_count / max(1, content_token_count)
- seen_vocab_size_so_far: size of the cumulative lemma set before this utterance
- cumulative_new_lemma_count: cumulative count of first-seen lemmas up to this utterance

Entity Novelty (within-episode, cumulative)
- entity_count: total named entity mentions in utt (selected labels)
- novel_entity_count: # entity surface forms not seen previously in this episode
- novel_entity_ratio: novel_entity_count / max(1, entity_count)
- novel_entity_density_token: novel_entity_count / max(1, token_count)
- seen_entity_size_so_far: size of the cumulative entity set before this utt
- cumulative_new_entity_count: cumulative count of first-seen entities so far

Outputs
-------
- Writes augmented CSVs to an output directory (mirrors input filenames)
- Also writes a per-file JSON containing **episode-level IDF vocabularies** keyed by conversation_id for reproducibility

Usage
-----
-----
python debate_informativeness_pipeline.py \
  --input_dir /path/to/episodes \
  --output_dir /path/to/augmented \
  [--language en_core_web_sm] [--min_df 2] [--max_df 0.95]

Notes
-----
- Requires: pandas, numpy, scikit-learn, spacy
- spaCy model: en_core_web_sm (or another English model)
  Install via: python -m spacy download en_core_web_sm

Author: ChatGPT (GPT-5 Thinking)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Dict, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import spacy
    from spacy.language import Language
except Exception as e:
    raise RuntimeError(
        "spaCy is required. Please `pip install spacy` and download an English model, "
        "e.g., `python -m spacy download en_core_web_sm`."
    ) from e


# -------- Utility: text processing (spaCy) --------
ENTITY_LABELS = {
    "PERSON", "ORG", "GPE", "LOC", "NORP", "EVENT", "WORK_OF_ART",
    "LAW", "PRODUCT", "FAC", "LANGUAGE"  # include LANGUAGE optionally
}


def load_nlp(model_name: str) -> Language:
    """Load a spaCy pipeline.
    - First try to load the requested model (e.g., en_core_web_sm).
    - If it's missing, attempt an in-code download.
    - If that also fails (offline env), fall back to a minimal English pipeline
      with a sentencizer and *no lemmatizer*; in that case we degrade to using
      lowercased alphabetic tokens as "lemmas".
    """
    try:
        nlp = spacy.load(model_name)
        # Ensure sentencizer is available for efficient streaming
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer", first=True)
        return nlp
    except Exception:
        # Try to download the small English model programmatically
        try:
            from spacy.cli import download as spacy_download
            spacy_download(model_name)
            nlp = spacy.load(model_name)
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer", first=True)
            return nlp
        except Exception:
            # Last-resort fallback for offline environments
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer", first=True)
            # Note: no lemmatizer available; content_lemmas() will use lowercased forms
            return nlp


def iter_spacy_docs(nlp: Language, texts: Iterable[str], batch_size: int = 128):
    """Yield spaCy docs efficiently over texts."""
    for doc in nlp.pipe(texts, batch_size=batch_size):
        yield doc


def content_lemmas(doc) -> List[str]:
    out = []
    for tok in doc:
        if tok.is_space or tok.is_punct or tok.like_num:
            continue
        if tok.is_stop:
            continue
        # use lowercase lemma; fall back to lowercased form
        lemma = (tok.lemma_ or tok.text).lower()
        if not lemma.isalpha():
            continue
        out.append(lemma)
    return out


def entity_strings(doc) -> List[str]:
    ents = []
    for ent in doc.ents:
        if ent.label_ in ENTITY_LABELS:
            ents.append(ent.text.strip().lower())
    return ents


# -------- Custom analyzer for TfidfVectorizer (uses spaCy lemmas) --------
class SpacyLemmaAnalyzer:
    def __init__(self, nlp: Language):
        self.nlp = nlp

    def __call__(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return content_lemmas(doc)


# -------- Build corpus-level TF–IDF --------

def build_vectorizer(all_texts: List[str], nlp: Language, min_df=2, max_df=0.95) -> TfidfVectorizer:
    analyzer = SpacyLemmaAnalyzer(nlp)
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        min_df=min_df,
        max_df=max_df,
        lowercase=False,
        dtype=np.float64,
        norm=None,
    )
    vectorizer.fit(all_texts)
    return vectorizer


def build_episode_vectorizers(df: pd.DataFrame, nlp: Language, min_df=2, max_df=0.95) -> Dict[str, TfidfVectorizer]:
    """Build a TF–IDF vectorizer **per conversation_id** using utterances as documents."""
    vecs: Dict[str, TfidfVectorizer] = {}
    for conv_id, g in df.groupby("conversation_id"):
        texts = g["utterance_text"].fillna("").astype(str).tolist()
        vecs[str(conv_id)] = build_vectorizer(texts, nlp, min_df=min_df, max_df=max_df)
    return vecs


# -------- Feature calculations --------

def tfidf_stats_for_text(text: str, vectorizer: TfidfVectorizer) -> Tuple[float, float, float, List[float]]:
    vec = vectorizer.transform([text])
    data = vec.data
    if data.size == 0:
        return 0.0, 0.0, 0.0, []
    tfidf_mean = float(data.mean())
    tfidf_sum = float(data.sum())
    tfidf_max = float(data.max())
    # collect IDFs for the present features (columns in vec)
    # mapping indices -> idf
    idf = vectorizer.idf_
    idfs_present = [float(idf[j]) for j in vec.indices]
    return tfidf_mean, tfidf_sum, tfidf_max, idfs_present


def episode_iterative_features(df: pd.DataFrame, nlp: Language, min_df: int, max_df: float) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Compute features while using **episode-level TF–IDF** (per conversation_id)."""
    df = df.sort_values(["conversation_id", "utterance_index"]).reset_index(drop=True)

    # Build per-episode vectorizers and capture IDF maps for export
    episode_vecs = build_episode_vectorizers(df, nlp, min_df=min_df, max_df=max_df)
    episode_idf_maps: Dict[str, Dict[str, float]] = {}
    for conv_id, vec in episode_vecs.items():
        vocab = vec.vocabulary_
        idf = vec.idf_.tolist()
        idx_to_term = {idx: term for term, idx in vocab.items()}
        episode_idf_maps[conv_id] = {idx_to_term[i]: float(idf[i]) for i in range(len(idf)) if i in idx_to_term}

    docs = list(iter_spacy_docs(nlp, df["utterance_text"].fillna("")))

    token_counts = []
    content_token_counts = []
    entity_counts = []

    novel_word_counts = []
    novel_word_densities = []
    seen_vocab_sizes = []
    cumulative_new_lemma_counts = []

    novel_entity_counts = []
    novel_entity_ratios = []
    novel_entity_densities_token = []
    seen_entity_sizes = []
    cumulative_new_entity_counts = []

    tfidf_means = []
    tfidf_sums = []
    tfidf_maxs = []
    specificity_means = []
    specificity_medians = []

    current_conv = None
    seen_lemmas: Set[str] = set()
    seen_entities: Set[str] = set()
    cum_new_lemmas = 0
    cum_new_entities = 0

    for i, (row, doc) in enumerate(zip(df.itertuples(index=False), docs)):
        conv_id = str(row.conversation_id)
        if conv_id != current_conv:
            current_conv = conv_id
            seen_lemmas.clear()
            seen_entities.clear()
            cum_new_lemmas = 0
            cum_new_entities = 0

        # token stats
        toks = [t for t in doc if not t.is_space]
        token_count = len(toks)
        lemmas = content_lemmas(doc)
        content_count = len(lemmas)

        ents = entity_strings(doc)
        ent_set = set(ents)
        entity_count = len(ents)

        # novelty: words (lemmas)
        lemma_set = set(lemmas)
        new_lemmas = {l for l in lemma_set if l not in seen_lemmas}
        novel_word_count = len(new_lemmas)
        novel_word_density = novel_word_count / max(1, content_count)
        seen_vocab_size = len(seen_lemmas)
        cum_new_lemmas += novel_word_count
        seen_lemmas.update(lemma_set)

        # novelty: entities
        new_entities = {e for e in ent_set if e not in seen_entities}
        novel_entity_count = len(new_entities)
        novel_entity_ratio = novel_entity_count / max(1, len(ent_set))
        novel_entity_density_token = novel_entity_count / max(1, token_count)
        seen_entity_size = len(seen_entities)
        cum_new_entities += novel_entity_count
        seen_entities.update(ent_set)

        # episode-level TF–IDF & specificity (IDF from this episode)
        vec = episode_vecs[conv_id]
        tfidf_mean, tfidf_sum, tfidf_max, idfs_present = tfidf_stats_for_text(row.utterance_text or "", vec)
        if idfs_present:
            spec_mean = float(np.mean(idfs_present))
            spec_median = float(np.median(idfs_present))
        else:
            spec_mean = 0.0
            spec_median = 0.0

        token_counts.append(token_count)
        content_token_counts.append(content_count)
        entity_counts.append(entity_count)

        novel_word_counts.append(novel_word_count)
        novel_word_densities.append(novel_word_density)
        seen_vocab_sizes.append(seen_vocab_size)
        cumulative_new_lemma_counts.append(cum_new_lemmas)

        novel_entity_counts.append(novel_entity_count)
        novel_entity_ratios.append(novel_entity_ratio)
        novel_entity_densities_token.append(novel_entity_density_token)
        seen_entity_sizes.append(seen_entity_size)
        cumulative_new_entity_counts.append(cum_new_entities)

        tfidf_means.append(tfidf_mean)
        tfidf_sums.append(tfidf_sum)
        tfidf_maxs.append(tfidf_max)
        specificity_means.append(spec_mean)
        specificity_medians.append(spec_median)

    out = df.copy()
    out["token_count"] = token_counts
    out["content_token_count"] = content_token_counts
    out["entity_count"] = entity_counts

    out["tfidf_mean"] = tfidf_means
    out["tfidf_sum"] = tfidf_sums
    out["tfidf_max"] = tfidf_maxs

    out["specificity_mean_idf"] = specificity_means
    out["specificity_median_idf"] = specificity_medians

    out["novel_word_count"] = novel_word_counts
    out["novel_word_density"] = novel_word_densities
    out["seen_vocab_size_so_far"] = seen_vocab_sizes
    out["cumulative_new_lemma_count"] = cumulative_new_lemma_counts

    out["novel_entity_count"] = novel_entity_counts
    out["novel_entity_ratio"] = novel_entity_ratios
    out["novel_entity_density_token"] = novel_entity_densities_token
    out["seen_entity_size_so_far"] = seen_entity_sizes
    out["cumulative_new_entity_count"] = cumulative_new_entity_counts

    return out, episode_idf_maps


# -------- I/O and orchestration --------

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


def collect_all_texts(csvs: Dict[Path, pd.DataFrame]) -> List[str]:
    texts = []
    for df in csvs.values():
        texts.extend(df["utterance_text"].fillna("").astype(str).tolist())
    return texts


def save_idf_vocab_map(idf_map: Dict[str, Dict[str, float]], out_path: Path):
    """Save a JSON mapping conversation_id -> {term: idf} for this file."""
    payload = {"idf_by_conversation_id": idf_map}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def process_folder(input_dir: Path, output_dir: Path, model_name: str, min_df: int, max_df: float):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading spaCy model...", model_name)
    nlp = load_nlp(model_name)

    print("Reading CSVs...")
    csvs = read_all_csvs(input_dir)

    print("Augmenting episodes with episode-level TF–IDF features...")
    for path, df in csvs.items():
        print(f"  Processing: {path.name}")
        enriched, idf_map = episode_iterative_features(df, nlp, min_df=min_df, max_df=max_df)
        out_path = output_dir / path.name
        enriched.to_csv(out_path, index=False)
        # Save per-file IDF map keyed by conversation_id
        save_idf_vocab_map(idf_map, output_dir / f"{path.stem}_idf_by_episode.json")

    print("Done. Augmented CSVs written to:", str(output_dir))


# -------- CLI --------

def parse_args():
    ap = argparse.ArgumentParser(description="Compute informativeness features for debate episodes.")
    ap.add_argument("--input_dir", type=Path, required=True, help="Folder containing episode CSVs")
    ap.add_argument("--output_dir", type=Path, required=True, help="Folder to write augmented CSVs")
    ap.add_argument("--language", type=str, default="en_core_web_sm", help="spaCy model name")
    ap.add_argument("--min_df", type=int, default=2, help="TfidfVectorizer min_df")
    ap.add_argument("--max_df", type=float, default=0.95, help="TfidfVectorizer max_df")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_folder(args.input_dir, args.output_dir, args.language, args.min_df, args.max_df)
