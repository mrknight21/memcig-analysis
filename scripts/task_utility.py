"""
segment_selector.py
-------------------
End‑to‑end utilities for choosing the most cost‑effective conversation
segments to annotate.

Dependencies:
    numpy
    pandas
    scikit‑learn      (MinMaxScaler)
    scipy             (cosine distance for diversity filtering)

Author: Bryan Chen (2025‑07‑24)
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple, Dict, Set
import re
import unicodedata
from nltk.corpus import stopwords
import spacy
import re, html as htmlmod


STOP_WORDS = set(stopwords.words('english'))


# Load the spaCy model
try:
    nlp = spacy.load("en_core_web_trf")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...\n"
          "Please run: python -m spacy download en_core_web_sm")
    nlp = None


_ARTICLES = r"(?:\b(?:the|a|an)\s+)?"
_SEP = r"(?:\s+|[-–—]\s*)"  # spaces or hyphens (with optional following spaces)

def remove_code_blocks(content: str) -> str:
    """
    Removes enclosing code block markers ```[language] and ``` from a given string.

    Remarks:
    - The function uses a regex pattern to match code blocks that may start with ``` followed by an optional language tag (letters or numbers) and end with ```.
    - If a code block is detected, it returns only the inner content, stripping out the markers.
    - If no code block markers are found, the original content is returned as-is.
    """
    pattern = r"^```[a-zA-Z0-9]*\n([\s\S]*?)\n```$"
    match = re.match(pattern, content.strip())
    return match.group(1).strip() if match else content.strip()

def _plural_piece(tok: str) -> str:
    """Lightweight plural tolerance without over-generalizing."""
    t = tok.lower()
    esc = re.escape(t)
    # y -> ies (city/cities), but keep y form too
    if len(t) > 3 and t.endswith("y") and (len(t) < 2 or t[-2] not in "aeiou"):
        base = re.escape(t[:-1])
        return rf"(?:{base}y|{base}ies)"
    # default: optional trailing 's'
    if len(t) >= 3 and not t.endswith("ss"):
        return rf"{esc}s?"
    return esc

def _percent_piece(tok: str) -> str:
    """Handle common percent spellings if the token is exactly 'percent'."""
    return r"(?:percent|per\s*cent|%)" if tok.lower() == "percent" else re.escape(tok)

def _token_piece(tok: str, is_last: bool) -> str:
    """Compose the regex for one token: plural + percent + optional possessive (if last)."""
    core = _percent_piece(tok) if tok.lower() == "percent" else _plural_piece(tok)
    # Allow possessive only on the last token (e.g., Donohue's / Donohue’s)
    if is_last:
        core = rf"{core}(?:['’]s)?"
    return core

def _compile_pattern(keyword: str) -> Optional[re.Pattern]:
    kw = keyword.strip()
    if not kw:
        return None
    # split on spaces or hyphens in the keyword so we can accept either at match time
    tokens = re.split(r"[\s-]+", kw)
    pieces = [_token_piece(tok, is_last=(i == len(tokens) - 1)) for i, tok in enumerate(tokens)]
    core = _SEP.join(pieces)
    # use outer "word-ish" boundaries so we don't cut through alphanum
    pat = rf"(?<!\w){_ARTICLES}{core}(?!\w)"
    return re.compile(pat, flags=re.IGNORECASE)

def _apply_highlights(text: str, sorted_keywords: List[str]) -> str:
    """
    Highlight occurrences of each keyword in `text` using <strong>…</strong>,
    avoiding nested/overlapping tags. Matching is robust to:
      - leading articles (the/a/an)
      - hyphen vs space between tokens
      - simple plural variants
      - possessive on the last token
      - 'percent' vs 'per cent' vs '%'
    """
    # 1) gather matches
    all_matches = []
    for kw in sorted_keywords:
        pat = _compile_pattern(kw)
        if not pat:
            continue
        for m in pat.finditer(text):
            all_matches.append(m)

    if not all_matches:
        return text

    # 2) remove overlaps (prefer longer; then earlier start)
    all_matches.sort(key=lambda m: (len(m.group(0)), -m.start()), reverse=True)
    kept = []
    for m in all_matches:
        s, e = m.start(), m.end()
        if not any(s < km.end() and e > km.start() for km in kept):
            kept.append(m)

    kept.sort(key=lambda m: m.start())

    # 3) stitch with <strong>
    parts = []
    last = 0
    for m in kept:
        parts.append(text[last:m.start()])
        parts.append(f"<strong>{m.group(0)}</strong>")
        last = m.end()
    parts.append(text[last:])
    return "".join(parts)

# ────────────────────────────────────────────────────────────────────────────
#  1.  Per‑segment feature extraction
# ────────────────────────────────────────────────────────────────────────────
def get_segment_task_quality_information(
    task: Dict,
    segment: Dict,
    sum_type: str = "llm",
) -> Dict[str, float]:
    """
    Extracts quality & cost features for a single (task, segment) pair.

    Parameters
    ----------
    task : dict
        Annotation‑task meta data.
    segment : dict
        Segment‑level evaluation metrics.
    sum_type : str, default "llm"
        Summary variant to compare against "no_summary".

    Returns
    -------
    qa_info : dict
        Dictionary with all numeric features used downstream.
    """
    sum_key = f"{sum_type}_summary"

    qa_info = {
        # ↓ Cost‑reducing indicators
        "skipped_token_ratio": task["skipped_ratio"]["tokens"],
        "skipped_utterances_ratio": task["skipped_ratio"]["utterances"],
        # ↓ Benefit of having a summary
        "context_benefit": segment["summary_scores"][sum_key]
        - segment["summary_scores"]["no_summary"],
        "sum_score": segment["summary_scores"][sum_key],
        # ↓ Cost proxies
        "valid_utterance_count": len(
            [utt for utt in task["target_utterances"] if not utt["skipped"]]
        ),
        "total_reading_time": task["total_reading_time"],
        "non_mod_speaker_num": task["non_mod_speaker_num"],
    }

    # Per‑aspect stats
    aspect_info: Dict[str, float] = {}
    overall_means = []
    full_ctx_scores = segment["full_context_rating"]

    for aspect in ["implication_scope", "informativeness", "novelty", "relevance"]:
        scores = [s[aspect] for s in full_ctx_scores]
        aspect_mean = float(np.mean(scores))
        aspect_std = float(np.std(scores))
        aspect_info[f"{aspect}_mean"] = aspect_mean
        aspect_info[f"{aspect}_std"] = aspect_std
        overall_means.append(aspect_mean)

    aspect_info["overall_mean"] = float(np.mean(overall_means))
    aspect_info["overall_std"] = float(np.std(overall_means))  # ← fixed bug

    qa_info.update(aspect_info)
    return qa_info


# ────────────────────────────────────────────────────────────────────────────
# 2.  Normalisation helpers
# ────────────────────────────────────────────────────────────────────────────
def _minmax_normalise(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    In‑place Min‑Max scaling of selected columns to [0, 1].
    """
    if not columns:
        return df
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def normalise_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits positive‑direction and negative‑direction variables,
    then rescales both groups independently to [0, 1].
    """
    df = df.copy()

    positive_cols = [
        # larger ⇒ better
        "context_benefit",
        "sum_score",
        "overall_mean",
        "overall_std",
        "participant_count",
        "sum_score",
        "segment_confidence"

    ]
    negative_cols = [
        # larger ⇒ worse (cost)
        "read_time_dev",
        "utt_count_dev",
        "skipped_token_ratio",        # invert later
        "skipped_utterances_ratio",   # invert later
    ]

    _minmax_normalise(df, positive_cols)
    _minmax_normalise(df, negative_cols)
    return df


# ────────────────────────────────────────────────────────────────────────────
# 3.  Worth‑score computation
# ────────────────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = {
    #Positive
    # "w_ctx": 0.2,
    "w_overall": 0.2,
    "w_uncert": 0.2,
    "w_spk": 0.2,
    "w_sum": 0.2,
    "w_seg": 0.2,
    ##Negative
    "w_dev_time": 0.3,
    "w_dev_len": 0.2,
    "w_skip": 0.5,
    "alpha": 1.0,  # cost scaling
}


def compute_worth_scores(
    df: pd.DataFrame, weights: Dict[str, float] | None = None
) -> pd.DataFrame:
    """
    Adds `SEG_QUALITY`, `SEG_COST`, and `worth_score` columns to *df*.

    Returns a new DataFrame (original remains untouched).
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    df = df.copy()

    df["SEG_QUALITY"] = (
        # w["w_ctx"] * df["context_benefit"]
        + w["w_overall"] * df["overall_mean"]
        + w["w_sum"] * df["sum_score"]
        + w["w_uncert"] * df["overall_std"]
        + w["w_spk"] * df["participant_count"]
        + w["w_seg"] * df["segment_confidence"]
    )

    df["SEG_COST"] = (
        w["w_dev_time"] * df["read_time_dev"]
        + w["w_dev_len"] * df["utt_count_dev"]
        + w["w_skip"] * ( df["skipped_token_ratio"] + df["skipped_utterances_ratio"] / 2)
    )

    df["worth_score"] = df["SEG_QUALITY"] / (1.0 + w["alpha"] * df["SEG_COST"])
    return df


def select_top_k(
    df: pd.DataFrame,
    k: int
) -> pd.DataFrame:
    """
    Returns the *k* highest‑value segments, optionally diversity‑filtered.
    """
    df_sorted = df.sort_values("worth_score", ascending=False)
    return df_sorted.head(k)


# ────────────────────────────────────────────────────────────────────────────
# 5.  End‑to‑end helper
# ────────────────────────────────────────────────────────────────────────────
def rank_segments(
    segment_quality_info,
    k: int | None = None,
    ensure_diversity: bool = True,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    High‑level convenience wrapper: from raw tuples to ranked DataFrame.

    Each tuple → (task_dict, segment_dict [, optional_embedding_vector])

    If *k* is provided, returns only the selected segments; otherwise returns
    the full ranked list with worth scores.
    """


    df = pd.DataFrame(segment_quality_info)
    df = normalise_features(df)
    df = compute_worth_scores(df, weights=weights)

    if k is not None:
        df = select_top_k(df, k)

    return df.reset_index(drop=True)

def strip_html_regex(s: str) -> str:
    s = re.sub(r"(?is)<(script|style)\b.*?>.*?</\1>", "", s)  # drop script/style
    s = re.sub(r"(?s)<!--.*?-->", "", s)                      # drop comments
    s = re.sub(r"<[^>]+>", " ", s)                            # drop tags
    s = htmlmod.unescape(s)
    return " ".join(s.split())

def slice_to_text(dialogue: pd.DataFrame, start: int, end: int, external_rating= None) -> str:
    """Format a slice of the dialogue DataFrame into plain numbered text."""
    if external_rating is not None:
        tgt_utterances =[]
        for _, row in dialogue.iloc[start:end].iterrows():
            utt_string = f"{row.utterance_index}. {row.utterance_speaker} ({row.role}): {row.utterance_text}."
            utt_rating = external_rating[external_rating.utterance_id == str(row["utterance_id"])]
            if len(utt_rating) > 0:
                utt_rating = utt_rating.iloc[0]
                utt_string += f"\n External rating: Novelty: {utt_rating.novo}, Relevance: {utt_rating.relv}, Implication Scope: {utt_rating.imsc}."
            tgt_utterances.append(utt_string)
        return "\n\n".join(tgt_utterances)

    else:
        return "\n\n".join(
            f"{row.utterance_index}. {row.utterance_speaker} ({row.role}): {row.utterance_text}"
            for _, row in dialogue.iloc[start:end].iterrows()
        )

# assumes `nlp` and `STOP_WORDS` are available in your module

def extract_keywords(content: str) -> List[str]:
    """
    Identifies and cleans keywords from text, collapsing near-duplicates
    into canonical representatives for robust cross-article matching.
    """
    if nlp is None:
        return ["spaCy model not loaded."]

    doc_dialogue = nlp(content)

    # -------------------------
    # 1) Raw candidate phrases
    # -------------------------
    raw_keywords: Set[str] = set()
    MAX_KEYWORD_WORDS = 7
    entity_labels_to_include = {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "LOC"}

    # Named entities
    for ent in doc_dialogue.ents:
        if len(ent.text.split()) <= MAX_KEYWORD_WORDS and ent.label_ in entity_labels_to_include:
            raw_keywords.add(ent.text)

    # Base noun phrases
    for chunk in doc_dialogue.noun_chunks:
        toks = [t for t in chunk if not t.is_space]
        # keep phrases with at least 2 non-stop tokens
        non_stop = [t for t in toks if not t.is_stop]
        if 2 <= len(non_stop) and len(toks) <= MAX_KEYWORD_WORDS:
            raw_keywords.add(chunk.text)

    # -------------------------
    # 2) Normalization helpers
    # -------------------------
    LEADING_DETS = {"the", "a", "an"}
    DROP_FILLERS = {"about", "approximately", "around"}  # e.g., "about seventy percent"

    # quick map for common number words you actually see in your data
    NUM_WORDS = {
        "zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5",
        "six":"6","seven":"7","eight":"8","nine":"9","ten":"10",
        "twenty":"20","thirty":"30","forty":"40","fifty":"50",
        "sixty":"60","seventy":"70","eighty":"80","ninety":"90",
        # extend if needed
    }

    blacklist = {
        'a lot', 'the point', 'a problem', 'the issue', 'the view',
        'the relationship', 'the impact', 'the real debate', 'the person',
        'many different ways', 'absolutely nothing'
    }

    def basic_clean(s: str) -> str:
        # normalize unicode quotes/dashes
        s = unicodedata.normalize("NFKC", s)
        s = s.replace("’", "'").replace("–", "-").replace("—", "-")
        s = s.strip()
        # strip trailing dashes/punct
        s = re.sub(r"[-\s]*[-,.:;]+$", "", s)
        # normalize hyphens to spaces and collapse
        s = re.sub(r"[-]+", " ", s)
        s = re.sub(r"\s+", " ", s)
        return s

    def normalize_phrase(s: str) -> Tuple[str, Tuple[str, ...]]:
        """
        Returns (canonical_string, signature_tokens)
        - canonical_string: order-preserving, lemmatized (PROPN kept as surface), lowercased
        - signature_tokens: order-insensitive token set (sorted) for clustering
        """
        s0 = basic_clean(s).lower()

        # remove possessive 's anywhere it appears as a token
        s0 = re.sub(r"\b([a-z]+)'s\b", r"\1", s0)

        # tokenize with spaCy to decide what to keep
        doc = nlp(s0)

        # drop leading determiner if phrase > 1 token
        toks = [t for t in doc if not t.is_space]
        if len(toks) > 1 and toks and toks[0].lemma_ in LEADING_DETS:
            toks = toks[1:]

        # build normalized token list
        keep_pos = {"NOUN", "PROPN", "ADJ", "NUM"}
        out = []
        for t in toks:
            if t.is_stop:
                # drop common filler modifiers like "about"
                if t.lemma_ in DROP_FILLERS:
                    continue
                # keep 'of' only when it connects two nouns (common in org names)
                if t.lemma_ == "of":
                    out.append("of")
                continue

            if t.pos_ not in keep_pos:
                # keep 'of' already handled; ignore other POS (verbs, adv, etc.)
                continue

            if t.pos_ == "NUM":
                # normalize number words (best-effort)
                out.append(NUM_WORDS.get(t.lemma_, t.text))
            elif t.pos_ == "PROPN":
                # keep surface (already lowercased by s0); avoids weird lemmatization on names
                out.append(t.text)
            else:
                # lemmatize common nouns/adjectives
                out.append(t.lemma_)

        # trim if we removed everything
        out = [w for w in out if w]

        # remove trailing stopwords if any slipped through
        while len(out) > 1 and out[-1] in STOP_WORDS:
            out.pop()

        canonical = " ".join(out).strip()

        # signature: content token set (ignoring order); drop 'of' to merge academy variants
        sign_tokens = tuple(sorted(w for w in out if w != "of"))

        return canonical, sign_tokens

    # -------------------------
    # 3) Filter + normalize all
    # -------------------------
    normalized_items = []
    for kw in raw_keywords:
        # early basic cleaning (for blacklist comparison)
        base = basic_clean(kw).lower()
        if not base or base in blacklist:
            continue

        canonical, signature = normalize_phrase(kw)
        if not canonical:
            continue

        # skip if everything is stop or too short (but allow all-caps acronyms)
        if len(canonical) < 3 and canonical.upper() != canonical:
            continue

        # discard phrases that become all-stop after normalization
        doc_c = nlp(canonical)
        if all(t.is_stop for t in doc_c if not t.is_space):
            continue

        normalized_items.append((kw, canonical, signature))

    if not normalized_items:
        return []

    # -------------------------
    # 4) Cluster near-duplicates
    # -------------------------
    def jaccard(a: Tuple[str, ...], b: Tuple[str, ...]) -> float:
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 1.0
        return len(sa & sb) / max(1, len(sa | sb))

    JACCARD_THRESHOLD = 0.66  # merges "national academy" with "national academy sciences"

    clusters: List[Dict] = []
    for orig, can, sig in normalized_items:
        placed = False
        # exact signature match first
        for c in clusters:
            if c["signature"] == sig:
                c["items"].append((orig, can))
                # choose longest canonical as representative
                if len(can) > len(c["rep"]):
                    c["rep"] = can
                placed = True
                break
        if placed:
            continue

        # otherwise, merge by Jaccard on signatures (order-insensitive)
        best_i, best_sim = -1, 0.0
        for i, c in enumerate(clusters):
            sim = jaccard(sig, c["signature"])
            if sim > best_sim:
                best_i, best_sim = i, sim
        if best_sim >= JACCARD_THRESHOLD:
            clusters[best_i]["items"].append((orig, can))
            if len(can) > len(clusters[best_i]["rep"]):
                clusters[best_i]["rep"] = can
            # union signature for robustness
            clusters[best_i]["signature"] = tuple(sorted(set(clusters[best_i]["signature"]) | set(sig)))
        else:
            clusters.append({"rep": can, "items": [(orig, can)], "signature": sig})

    # -------------------------
    # 5) Prune substrings
    # -------------------------
    reps = [c["rep"] for c in clusters if c["rep"]]
    reps = sorted(set(reps), key=len, reverse=True)

    final_keywords: List[str] = []
    for k in reps:
        # drop if it's a strict substring of an already kept longer phrase
        if any(k != s and k in s for s in final_keywords):
            continue
        final_keywords.append(k)

    return final_keywords

