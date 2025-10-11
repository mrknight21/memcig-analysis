from __future__ import annotations

import os
from typing import Dict, Literal, Optional, Tuple, List

import numpy as np
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.metrics import cohen_kappa_score, accuracy_score, mean_squared_error, f1_score
from collections import defaultdict

# -------------------------------
# Config
# -------------------------------
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "flask_mongo_db")
COMPLETION_CODE = os.getenv("COMPLETION_CODE", "1234567890")
PORT = os.getenv("PORT", "8000")
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
MODE = os.getenv("MODE", "DEBUG")

# rater filter
BLACKLIST = {"aso", "maychill", "test"}                    # annotator ids to always skip
AGREEMENT_THRESHOLD = float(os.getenv("AGREE_THRESH", 0.3))  # mean QWK below this → drop
MIN_OVERLAP = int(os.getenv("MIN_OVERLAP", 3))             # min shared items to compute QWK
MIN_RATERS_TO_DROP = int(os.getenv("MIN_RATERS_TO_DROP", 2))  # only drop if conv has ≥ this many raters

# aspects
INFO_ASPECT_CANON = "CIG"  # canonical name for informativeness
MIX_ASPECTS = ("Novelty", "Relevance", "Scope")

# Mongo
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# ============================================================
# Mongo access
# ============================================================

# ============================================================
# Krippendorff's alpha (ordinal by default)
# ============================================================

def krippendorff_alpha(
    df: pd.DataFrame,
    item_col: str = "utterance_id",
    rater_col: str = "rater_id",
    rating_col: str = "rating",
    categories: Tuple[int, ...] = (1, 2, 3, 4),
    metric: Literal["ordinal", "interval", "nominal"] = "ordinal",
) -> float:
    """
    Krippendorff's alpha for possibly-incomplete rectangular data.
    Works with >=2 raters, missing data, and K ordered categories.

    metric:
      - 'ordinal'  : δ^2 uses cumulative (midpoint) ranks from global marginals
      - 'interval' : δ^2 = (i - j)^2 (scaled to [0,1] by (K-1)^2)
      - 'nominal'  : δ^2 = 1 if i!=j else 0
    """
    # build coincidence matrix O[K,K]
    cats = np.array(categories, dtype=int)
    K = len(cats)
    cat_to_idx = {c: i for i, c in enumerate(cats)}

    O = np.zeros((K, K), dtype=float)

    # group by item; compute per-item contributions
    for _, g in df.groupby(item_col):
        # keep valid ratings in the allowed categories
        vals = [cat_to_idx[int(v)] for v in g[rating_col].values if int(v) in cat_to_idx]
        if len(vals) < 2:
            continue
        counts = np.bincount(vals, minlength=K).astype(float)
        R = counts.sum()
        if R < 2:
            continue
        denom = R - 1.0
        # off-diagonals: n_i * n_j ; diagonal: n_i * (n_i-1)
        # normalized by (R-1)
        O += (counts[:, None] * counts[None, :]) / denom
        O[np.diag_indices_from(O)] -= counts / denom  # fix diagonal to n_i*(n_i-1)/(R-1)

    n_pairs = O.sum()
    if n_pairs <= 0:
        return np.nan

    m = O.sum(axis=1)                # category marginals
    p = m / n_pairs                  # global category proportions

    # distance matrix D[K,K]
    if metric == "nominal":
        D = np.ones((K, K), dtype=float)
        np.fill_diagonal(D, 0.0)
    elif metric == "interval":
        v = cats.astype(float)
        D = (v.reshape(-1, 1) - v.reshape(1, -1)) ** 2
        D /= (K - 1) ** 2  # scale to [0,1]
    elif metric == "ordinal":
        # ordinal δ^2 uses cumulative midpoints from marginals (Krippendorff 2013)
        # s_c = cumulative(p) - 0.5 * p (midpoint of each category on [0,1])
        cum = np.cumsum(p)
        s = cum - 0.5 * p
        D = (s.reshape(-1, 1) - s.reshape(1, -1)) ** 2
    else:
        raise ValueError("Unknown metric; choose 'ordinal', 'interval', or 'nominal'.")

    # observed disagreement
    Do = float((O * D).sum() / n_pairs)

    # expected coincidence matrix E from marginals
    n = m.sum()
    if n <= 1:
        return np.nan
    E = np.outer(m, m) / (n - 1.0)
    np.fill_diagonal(E, (m * (m - 1.0)) / (n - 1.0))

    De = float((E * D).sum() / n_pairs)
    if De == 0:
        return np.nan

    return 1.0 - (Do / De)


def compute_corpus_alphas(
    df_all: pd.DataFrame,
    outlier_report: Optional[pd.DataFrame] = None,
    use_filtered: bool = True,
    metric: Literal["ordinal", "interval", "nominal"] = "ordinal",
    min_raters_per_item: int = 2,
) -> pd.DataFrame:
    """
    Compute Krippendorff's alpha per corpus × aspect.
    If `use_filtered` and an outlier_report is provided, raters marked dropped per conv_id are removed.
    Returns columns: [corpus, aspect, items_used, ratings_used, raters_unique, alpha]
    """
    df = df_all.copy()

    # Optionally drop (conv_id, rater_id) pairs flagged as outliers
    if use_filtered and outlier_report is not None and not outlier_report.empty:
        drop_keys = (
            outlier_report[outlier_report["dropped"] == True]
            .assign(key=lambda d: d["conv_id"].astype(str) + "||" + d["rater_id"].astype(str))
            ["key"]
            .tolist()
        )
        if drop_keys:
            key_series = df["conv_id"].astype(str) + "||" + df["rater_id"].astype(str)
            df = df[~key_series.isin(set(drop_keys))].copy()

    # keep only the four aspects we care about (CIG ≡ Informativeness)
    aspect_map = {"CIG": "Informativeness", "Novelty": "Novelty", "Relevance": "Relevance", "Scope": "Scope"}
    df = df[df["aspect"].isin(aspect_map.keys())].copy()

    rows = []
    for corpus, dcor in df.groupby("corpus", sort=False):
        for asp_key, asp_pretty in aspect_map.items():
            sub = dcor[dcor["aspect"] == asp_key].copy()
            if sub.empty:
                rows.append({
                    "corpus": corpus, "aspect": asp_pretty,
                    "items_used": 0, "ratings_used": 0, "raters_unique": 0, "alpha": np.nan
                })
                continue

            # only items with ≥ min raters
            sub_items = (
                sub.groupby("utterance_id")
                   .filter(lambda g: g["rater_id"].nunique() >= min_raters_per_item)
            )
            if sub_items.empty:
                rows.append({
                    "corpus": corpus, "aspect": asp_pretty,
                    "items_used": 0, "ratings_used": 0, "raters_unique": sub["rater_id"].nunique(), "alpha": np.nan
                })
                continue

            alpha = krippendorff_alpha(
                sub_items,
                item_col="utterance_id",
                rater_col="rater_id",
                rating_col="rating",
                categories=(1, 2, 3, 4),
                metric=metric,
            )
            rows.append({
                "corpus": corpus,
                "aspect": asp_pretty,
                "items_used": int(sub_items["utterance_id"].nunique()),
                "ratings_used": int(len(sub_items)),
                "raters_unique": int(sub_items["rater_id"].nunique()),
                "alpha": float(alpha) if alpha == alpha else np.nan,  # keep NaN if undefined
            })

    result = pd.DataFrame(rows).sort_values(["corpus", "aspect"]).reset_index(drop=True)
    return result


def get_annotations():
    """Return a cursor over annotation documents."""
    return db["annotations"].find({})

# ============================================================
# Aggregation helpers (from your previous module)
# ============================================================

def entropy_nats(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    p /= p.sum()
    return float(-(p * np.log(p)).sum())

def dirichlet_smooth(counts: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    if counts.shape[0] != 4:
        raise ValueError("counts must be length-4 for classes 1..4")
    smoothed = counts + alpha
    return smoothed / smoothed.sum()

def argmax_with_tiebreak(
    p: np.ndarray,
    mode: Literal["lower", "upper", "prior"] = "lower",
    priors: Optional[np.ndarray] = None
) -> int:
    p = np.asarray(p, dtype=float)
    m = p.max()
    ties = np.flatnonzero(np.isclose(p, m))
    if len(ties) == 1:
        return int(ties[0] + 1)
    if mode == "upper":
        return int(ties[-1] + 1)
    if mode == "prior" and priors is not None:
        prior_vals = np.asarray(priors, dtype=float)[ties]
        j = ties[int(np.argmax(prior_vals))]
        return int(j + 1)
    return int(ties[0] + 1)

def round_half(
    mean_val: float,
    half: Literal["down", "up", "even", "soft", "prior"] = "down",
    p_soft: Optional[np.ndarray] = None,
    soft_tiebreak: Literal["lower", "upper", "prior"] = "lower",
    priors: Optional[np.ndarray] = None
) -> int:
    lo = int(np.floor(mean_val))
    frac = float(mean_val - lo)
    if frac < 0.5:
        return int(np.clip(lo, 1, 4))
    if frac > 0.5:
        return int(np.clip(lo + 1, 1, 4))
    if half == "down":
        return int(np.clip(lo, 1, 4))
    if half == "up":
        return int(np.clip(lo + 1, 1, 4))
    if half == "even":
        cand = lo if (lo % 2 == 0) else lo + 1
        return int(np.clip(cand, 1, 4))
    if half == "soft" and p_soft is not None:
        return argmax_with_tiebreak(p_soft, mode=soft_tiebreak, priors=priors)
    if half == "prior" and priors is not None:
        a, b = priors[lo - 1], priors[lo]
        return int(np.clip(lo if a >= b else lo + 1, 1, 4))
    return int(np.clip(lo, 1, 4))

# --------- rater calibration ----------

def _build_quantile_map(rater_counts: np.ndarray, global_counts: np.ndarray) -> Dict[int, int]:
    rater_counts = np.asarray(rater_counts, dtype=float)
    global_counts = np.asarray(global_counts, dtype=float)
    if rater_counts.sum() == 0:
        return {1: 1, 2: 2, 3: 3, 4: 4}
    cdf_r = np.cumsum(rater_counts) / rater_counts.sum()
    cdf_g = np.cumsum(global_counts) / max(global_counts.sum(), 1.0)
    mapping: Dict[int, int] = {}
    last = 1
    for k in range(1, 5):
        idx = int(np.argmin(np.abs(cdf_g - cdf_r[k - 1]))) + 1
        idx = max(idx, last)
        mapping[k] = idx
        last = idx
    return mapping

def apply_quantile_link(
    df: pd.DataFrame,
    aspect_col: str = "aspect",
    rater_col: str = "rater_id",
    rating_col: str = "rating"
) -> pd.Series:
    global_counts_by_aspect: Dict[str, np.ndarray] = {}
    for a, sub in df.groupby(aspect_col):
        counts = sub[rating_col].value_counts().reindex([1, 2, 3, 4], fill_value=0).values
        global_counts_by_aspect[a] = counts
    calibrated = []
    for (a, r), sub in df.groupby([aspect_col, rater_col]):
        r_counts = sub[rating_col].value_counts().reindex([1, 2, 3, 4], fill_value=0).values
        mapping = _build_quantile_map(r_counts, global_counts_by_aspect[a])
        mapped = sub[rating_col].map(mapping).astype(int)
        calibrated.append(mapped)
    return pd.concat(calibrated).reindex(df.index)

def apply_centering(
    df: pd.DataFrame,
    aspect_col: str = "aspect",
    rater_col: str = "rater_id",
    rating_col: str = "rating"
) -> pd.Series:
    g_stats: Dict[str, Tuple[float, float]] = {}
    for a, sub in df.groupby(aspect_col):
        g_mean = float(sub[rating_col].mean())
        g_std = float(sub[rating_col].std(ddof=0) or 1.0)
        g_stats[a] = (g_mean, g_std)
    out = pd.Series(index=df.index, dtype=float)
    for (a, r), sub in df.groupby([aspect_col, rater_col]):
        x = sub[rating_col].astype(float).values
        m = float(x.mean()); s = float(x.std(ddof=0) or 1.0)
        z = (x - m) / s
        g_mean, g_std = g_stats[a]
        y = g_mean + z * g_std
        out.loc[sub.index] = np.clip(np.round(y), 1, 4)
    return out.astype(int)

# --------- aggregation core ----------

def aggregate_annotations(
    df: pd.DataFrame,
    utter_col: str = "utterance_id",
    aspect_col: str = "aspect",
    rater_col: str = "rater_id",
    rating_col: str = "rating",
    do_calibration: Optional[Literal["quantile", "center", None]] = "quantile",
    dirichlet_alpha: float = 0.5,
    hard_strategy: Literal["two_mean_then_halfrule", "median_then_soft"] = "two_mean_then_halfrule",
    halfrule: Literal["down", "up", "even", "soft", "prior"] = "down",
    soft_tiebreak: Literal["lower", "upper", "prior"] = "lower",
    include_continuous: bool = True,
    include_interval_masks: bool = False
) -> pd.DataFrame:
    if not {"utterance_id", "aspect", "rater_id", "rating"}.issubset(df.columns):
        raise ValueError("Input df must have columns: utterance_id, aspect, rater_id, rating")

    data = df.copy()
    data["rating"] = pd.to_numeric(data["rating"], errors="coerce").astype("Int64")
    data = data.dropna(subset=["rating"])
    data["rating"] = data["rating"].astype(int)
    if (data["rating"] < 1).any() or (data["rating"] > 4).any():
        raise ValueError("Ratings must be integers in {1,2,3,4}.")

    # calibration
    if do_calibration == "quantile":
        data["rating_calib"] = apply_quantile_link(data, "aspect", "rater_id", "rating")
    elif do_calibration == "center":
        data["rating_calib"] = apply_centering(data, "aspect", "rater_id", "rating")
    else:
        data["rating_calib"] = data["rating"].astype(int)

    # global priors over calibrated ratings
    counts_all = data["rating_calib"].value_counts().reindex([1, 2, 3, 4], fill_value=0).values.astype(float)
    priors = counts_all / max(counts_all.sum(), 1.0)

    rows = []
    for (utt, asp), sub in data.groupby(["utterance_id", "aspect"], sort=False):
        ratings = sub["rating_calib"].astype(int).values
        counts = np.bincount(ratings, minlength=5)[1:5]
        p_soft = dirichlet_smooth(counts, alpha=dirichlet_alpha)
        H = entropy_nats(p_soft)
        w_conf = float(1.0 - H / np.log(4.0))
        y_cont = float(np.dot(p_soft, np.arange(1, 5))) if include_continuous else None

        if len(ratings) == 2 and hard_strategy == "two_mean_then_halfrule":
            mean_val = float(np.mean(ratings))
            y_hard = round_half(mean_val, half=halfrule, p_soft=p_soft,
                                soft_tiebreak=soft_tiebreak, priors=priors)
        else:
            med = np.median(ratings)
            y_hard = int(med) if float(med).is_integer() else argmax_with_tiebreak(p_soft, mode=soft_tiebreak, priors=priors)

        row = {
            "utterance_id": utt,
            "aspect": asp,
            "n_raters": int(len(ratings)),
            "y_hard": int(y_hard),
            "w_conf": w_conf,
            "p1": float(p_soft[0]), "p2": float(p_soft[1]),
            "p3": float(p_soft[2]), "p4": float(p_soft[3]),
        }
        if include_continuous:
            row["y_cont"] = y_cont

        if include_interval_masks:
            lo, hi = int(ratings.min()), int(ratings.max())
            mask_t = [1, 1, 1]
            for k in range(lo, hi):
                mask_t[k - 1] = 0
            row.update({"lo": lo, "hi": hi, "mask_t1": mask_t[0], "mask_t2": mask_t[1], "mask_t3": mask_t[2]})
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["utterance_id", "aspect"]).reset_index(drop=True)

# ============================================================
# Outlier removal (per conversation, across aspects if mix)
# ============================================================

def _pairwise_qwk_for_aspect(df_aspect: pd.DataFrame, r1: str, r2: str) -> Optional[float]:
    """Compute QWK for two raters on this aspect given a conv subset."""
    a = df_aspect[df_aspect["rater_id"] == r1][["utterance_id", "rating"]].set_index("utterance_id")
    b = df_aspect[df_aspect["rater_id"] == r2][["utterance_id", "rating"]].set_index("utterance_id")
    shared = a.join(b, how="inner", lsuffix="_a", rsuffix="_b")
    if len(shared) < MIN_OVERLAP:
        return None
    return float(cohen_kappa_score(shared["rating_a"], shared["rating_b"], weights="quadratic"))

def drop_outlier_annotators_per_conv(
    df: pd.DataFrame, conv_id: str, task_type: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For a single conversation (conv_id), compute mean pairwise QWK per annotator
    (averaging across aspects if task_type='mix'), drop annotators below threshold.
    Returns filtered df for this conv and a small report dataframe.
    """
    sub = df[df["conv_id"] == conv_id].copy()
    raters = list(sub["rater_id"].unique())
    report_rows = []

    if len(raters) < MIN_RATERS_TO_DROP:
        # not enough raters to safely drop anyone
        for r in raters:
            report_rows.append({"conv_id": conv_id, "rater_id": r, "mean_qwk": np.nan, "dropped": False, "n_pairs": 0})
        return sub, pd.DataFrame(report_rows)

    for r in raters:
        kappas = []
        pairs = 0
        aspects_qwks_dict = {}
        for s in raters:
            if s == r:
                continue
            if task_type == "mix":
                # average QWK across aspects (where overlap is sufficient)
                aspect_qwks = []
                for asp in MIX_ASPECTS:
                    if asp not in aspects_qwks_dict:
                        aspects_qwks_dict[asp] = []
                    df_asp = sub[sub["aspect"] == asp]
                    if df_asp.empty:
                        continue
                    q = _pairwise_qwk_for_aspect(df_asp, r, s)
                    if q is not None:
                        aspect_qwks.append(q)
                        aspects_qwks_dict[asp].append(q)
                if len(aspect_qwks) > 0:
                    aspect_mean = float(np.mean(aspect_qwks))
                    kappas.append(aspect_mean)
                    pairs += 1
            else:
                # info: single aspect (CIG)
                df_asp = sub[sub["aspect"] == INFO_ASPECT_CANON]
                q = _pairwise_qwk_for_aspect(df_asp, r, s)
                if q is not None:
                    kappas.append(q)
                    pairs += 1

        mean_qwk = float(np.mean(kappas)) if len(kappas) > 0 else np.nan

        if task_type == "mix":
            aspect_means = {asp: np.mean(qwks) for asp, qwks in aspects_qwks_dict.items()}
        else:
            aspect_means = {"info": mean_qwk}

        if not np.isnan(mean_qwk):
            if (mean_qwk < AGREEMENT_THRESHOLD) :
                dropped = True
            elif any([avg < AGREEMENT_THRESHOLD - 0.5 for asp, avg in aspect_means.items()]):
                dropped = True
            else:
                dropped = False
        else:
            dropped = False
        # dropped = (mean_qwk < AGREEMENT_THRESHOLD) if not np.isnan(mean_qwk) else False
        report_rows.append({"conv_id": conv_id, "rater_id": r, "mean_qwk": mean_qwk, "dropped": dropped, "n_pairs": pairs})

    report = pd.DataFrame(report_rows)
    # If base rule leaves at least 2, we're done
    kept_now = report.loc[~report["dropped"], "rater_id"].tolist()
    if len(kept_now) >= 2:
        to_drop = set(report.loc[report["dropped"], "rater_id"])
        filtered = sub[~sub["rater_id"].isin(to_drop)].copy()
        return filtered, report

    # Otherwise, enforce "keep top-2" policy by mean_qwk (then n_pairs, then rater_id)
    # Treat NaN mean_qwk as very low; prefer more overlap
    tmp = report.copy()
    tmp["score"] = tmp["mean_qwk"].fillna(-np.inf)
    tmp = tmp.sort_values(
        by=["score", "n_pairs", "rater_id"],
        ascending=[False, False, True],
        kind="mergesort"  # stable
    )
    keepers = set(tmp["rater_id"].head(min(2, len(tmp))).tolist())

    # Update dropped flags: keep exactly the chosen keepers
    report["dropped"] = ~report["rater_id"].isin(keepers)
    filtered = sub[sub["rater_id"].isin(keepers)].copy()
    return filtered, report

# ============================================================
# Parsing Mongo docs -> tidy DataFrame
# ============================================================

def _canonicalize_aspect(key: str, task_type: str) -> Optional[str]:
    k = key.strip().lower()
    if task_type == "mix":
        if k in {"novelty"}: return "Novelty"
        if k in {"relevance"}: return "Relevance"
        if k in {"scope", "implication_scope", "implicationscope"}: return "Scope"
        return None
    # info-like
    if k in {"informativeness", "informativness", "info", "cig"}:
        return INFO_ASPECT_CANON
    return None

def parse_mongo_annotations(raw: List[dict]) -> pd.DataFrame:
    """
    Parse Mongo 'annotations' docs into tidy rows:
    columns = [conv_id, utterance_id, aspect, rater_id, rating, task_type]
    """
    rows = []
    for doc in raw:
        annotator = str(doc.get("annotator") or doc.get("worker") or "").strip()
        if annotator in BLACKLIST or annotator == "":
            continue

        task_type = (doc.get("task_type") or doc.get("type") or "info").strip().lower()
        # conv_id: try explicit field, else derive from task_id like 'insq_2228_0_info'
        conv_id = doc.get("conv_id") or doc.get("conversation_id")
        if conv_id is None:
            tid = str(doc.get("task_id") or "")
            parts = tid.split("_")
            conv_id = "_".join(parts[:2]) if len(parts) >= 2 else tid

        tuts = doc.get("target_utterances") or doc.get("utterances") or []
        for tu in tuts:
            if tu.get("skipped") is True:
                continue
            labels = tu.get("labels", {}) or {}
            utt_id = tu.get("utterance_id") or tu.get("utt_id") or tu.get("id")
            if utt_id is None:
                continue
            for k, v in labels.items():
                aspect = _canonicalize_aspect(k, task_type)
                if aspect is None:
                    continue
                try:
                    rating = int(v)
                except Exception:
                    continue
                if rating < 1 or rating > 4:
                    continue
                rows.append({
                    "corpus": str(conv_id.split("_")[0]),
                    "conv_id": str(conv_id),
                    "task_type": str(task_type),
                    "utterance_id": str(utt_id),
                    "aspect": aspect,
                    "rater_id": annotator,
                    "rating": rating,
                    "task_type": "mix" if task_type == "mix" else "info"
                })
    if not rows:
        return pd.DataFrame(columns=["conv_id","utterance_id","aspect","rater_id","rating","task_type"])
    df = pd.DataFrame(rows)
    return df

# ============================================================
# End-to-end: outlier removal → calibration → aggregation
# ============================================================

def build_final_labels(
    df_all: pd.DataFrame,
    do_calibration: Literal["quantile","center",None] = "quantile",
    dirichlet_alpha: float = 0.5,
    hard_strategy: Literal["two_mean_then_halfrule","median_then_soft"] = "two_mean_then_halfrule",
    halfrule: Literal["down","up","even","soft","prior"] = "down",
    soft_tiebreak: Literal["lower","upper","prior"] = "lower",
    include_continuous: bool = True,
    include_interval_masks: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      final_labels: per (conv_id, utterance_id, aspect) aggregated labels
      outlier_report: per (conv_id, rater_id) mean_qwk, dropped flag
    """
    final_rows = []
    reports = []

    for conv_id, df_conv in df_all.groupby("conv_id", sort=False):
        corpus = conv_id.split("_")[0]
        for task_type in ["info", "mix"]:

            # 1) Outlier removal for this conversation
            df_conv_type = df_conv[df_conv.task_type == task_type]

            df_kept, report = drop_outlier_annotators_per_conv(df_conv_type, conv_id, task_type)
            reports.append(report)

            if df_kept.empty:
                continue

            # 2) Aggregate (within this conversation to keep utterance_id uniqueness)
            agg_conv = aggregate_annotations(
                df_kept[["utterance_id","aspect","rater_id","rating"]].copy(),
                do_calibration=do_calibration,
                dirichlet_alpha=dirichlet_alpha,
                hard_strategy=hard_strategy,
                halfrule=halfrule,
                soft_tiebreak=soft_tiebreak,
                include_continuous=include_continuous,
                include_interval_masks=include_interval_masks
            )
            agg_conv.insert(0, "conv_id", conv_id)
            agg_conv.insert(0, "corpus", corpus)
            final_rows.append(agg_conv)

    final_labels = pd.concat(final_rows, ignore_index=True) if final_rows else pd.DataFrame(
        columns=["corpus, ""conv_id","utterance_id","aspect","n_raters","y_hard","w_conf","p1","p2","p3","p4","y_cont","lo","hi","mask_t1","mask_t2","mask_t3"]
    )
    outlier_report = pd.concat(reports, ignore_index=True) if reports else pd.DataFrame(
        columns=["corpus", "conv_id","rater_id","mean_qwk","dropped","n_pairs"]
    )
    return final_labels, outlier_report

# ============================================================
# Example / main
# ============================================================

def evaluate_against_seed_gt(df):
    aspect_decoder = {"CIG": "info", "Novelty": "novo", "Relevance": "relv", "Scope": "imsc"}
    seed_df = pd.read_csv("../data/human_annotated/insq_2228.csv")
    seed_df = seed_df[seed_df.segment >= 0]
    seed_utt_ids = seed_df.utterance_id.unique().tolist()

    for corpus in ["insq", "fora"]:
        if corpus == "fora":
            continue
        print("Comparing with seed from corpus: {}".format(corpus))
        corpus_df = df[df.corpus == corpus]
        ratings = {"seed": {"info": [], "novo": [], "relv": [], "imsc": []},
                   "agg": {"info": [], "novo": [], "relv": [], "imsc": []}}
        for i, row in corpus_df.iterrows():
            utt_id = row["utterance_id"]
            if int(utt_id) not in seed_utt_ids:
                continue

            aspect = row["aspect"]
            aspect_code = aspect_decoder[aspect]
            ratings["agg"][aspect_code].append(row["y_hard"])

            seed_answer = seed_df.loc[seed_df.utterance_id == int(utt_id)][aspect_code].tolist()[0]
            ratings["seed"][aspect_code].append(seed_answer)

        for att in ["info", "novo", "relv", "imsc"]:
            print("Aspect {}".format(att))
            print(f"  mean squared error: "
                  f"{mean_squared_error(ratings['seed'][att], ratings['agg'][att]):.3f}")
            print(f"  Cohen κ (quadr.):  "
                  f"{cohen_kappa_score(ratings['seed'][att], ratings['agg'][att], weights='quadratic'):.3f}")
            print(f"  Accuracy:  "
                  f"{accuracy_score(ratings['seed'][att], ratings['agg'][att]):.3f}")

def _loo_consensus_for_group(
    df_grp: pd.DataFrame,
    rater_col: str,
    rating_col: str,
    dirichlet_alpha: float,
    hard_strategy: Literal["two_mean_then_halfrule","median_then_soft"],
    halfrule: Literal["down","up","even","soft","prior"],
    priors: Optional[np.ndarray],
    soft_tiebreak: Literal["lower","upper","prior"],
) -> Dict[str, Tuple[float, int]]:
    """
    For a single (utterance_id, aspect) group with columns [rater_id, rating_calib],
    compute leave-one-out consensus for each rater.

    Returns: dict[rater_id] -> (y_cont_loo, y_hard_loo)
    """
    # all calibrated integer ratings in {1..4}
    ratings = df_grp[rating_col].astype(int).values
    raters  = df_grp[rater_col].astype(str).values

    # counts for full group (we'll subtract one rating per LOO)
    base_counts = np.bincount(ratings, minlength=5)[1:5]

    out = {}
    for r_id, r_val in zip(raters, ratings):
        counts = base_counts.copy()
        counts[r_val - 1] -= 1
        if counts.sum() <= 0:
            # cannot form a consensus without at least 1 other rating
            continue

        p_soft = dirichlet_smooth(counts, alpha=dirichlet_alpha)
        y_cont = float(np.dot(p_soft, np.arange(1, 5)))

        # hard consensus
        if counts.sum() == 2 and hard_strategy == "two_mean_then_halfrule":
            # reconstruct the two ratings to apply your halfrule path
            # expand counts back to values
            vals = []
            for k, c in enumerate(counts, start=1):
                if c > 0:
                    vals += [k] * int(c)
            mean_val = float(np.mean(vals))
            y_hard = round_half(mean_val, half=halfrule, p_soft=p_soft,
                                soft_tiebreak=soft_tiebreak, priors=priors)
        else:
            # median-if-integer else tie-broken argmax over p_soft
            vals = []
            for k, c in enumerate(counts, start=1):
                if c > 0:
                    vals += [k] * int(c)
            med = np.median(vals)
            if float(med).is_integer():
                y_hard = int(med)
            else:
                y_hard = argmax_with_tiebreak(p_soft, mode=soft_tiebreak, priors=priors)

        out[str(r_id)] = (y_cont, int(y_hard))
    return out

def _apply_outlier_filter(df_all: pd.DataFrame, outlier_report: Optional[pd.DataFrame]) -> pd.DataFrame:
    if outlier_report is None or outlier_report.empty:
        return df_all.copy()
    drop_keys = (
        outlier_report[outlier_report["dropped"] == True]
        .assign(key=lambda d: d["conv_id"].astype(str) + "||" + d["rater_id"].astype(str))
        ["key"].tolist()
    )
    if not drop_keys:
        return df_all.copy()
    key_series = df_all["conv_id"].astype(str) + "||" + df_all["rater_id"].astype(str)
    return df_all[~key_series.isin(set(drop_keys))].copy()

def _calibrate(df: pd.DataFrame, mode: Optional[str]) -> pd.DataFrame:
    df = df.copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["rating"])
    df["rating"] = df["rating"].astype(int)
    if mode == "quantile":
        df["rating_calib"] = apply_quantile_link(df, "aspect", "rater_id", "rating")
    elif mode == "center":
        df["rating_calib"] = apply_centering(df, "aspect", "rater_id", "rating")
    else:
        df["rating_calib"] = df["rating"].astype(int)
    return df

def compute_human_variation(
    df_all: pd.DataFrame,
    outlier_report: Optional[pd.DataFrame],
    do_calibration: Literal["quantile","center",None] = "quantile",
    dirichlet_alpha: float = 0.5,
    hard_strategy: Literal["two_mean_then_halfrule","median_then_soft"] = "two_mean_then_halfrule",
    halfrule: Literal["down","up","even","soft","prior"] = "down",
    soft_tiebreak: Literal["lower","upper","prior"] = "lower",
) -> pd.DataFrame:
    """
    Human LOO baselines per corpus × aspect:
      - human_mse_mean/std: rater int rating vs LOO continuous consensus
      - human_f1_macro_mean/std: rater class vs LOO hard class
      Also: human_acc_mean, human_qwk_mean for quick reference.
    """
    # 1) filter out dropped annotators
    df = _apply_outlier_filter(df_all, outlier_report)
    if df.empty:
        return pd.DataFrame(columns=["corpus","aspect","n_raters","n_items","human_mse_mean","human_mse_std","human_f1_macro_mean","human_f1_macro_std","human_acc_mean","human_qwk_mean"])

    # 2) calibration
    df = _calibrate(df, do_calibration)

    # global priors for halfrule="prior"/soft tiebreak
    counts_all = df["rating_calib"].value_counts().reindex([1,2,3,4], fill_value=0).values.astype(float)
    priors = counts_all / max(counts_all.sum(), 1.0)

    rows = []
    # work per corpus × aspect to avoid cross-corpus prior leakage
    for (corpus, aspect), dca in df.groupby(["corpus","aspect"], sort=False):
        if dca.empty:
            continue

        # collect per-rater lists to later compute macro-F1 properly
        f1_by_rater = []
        acc_by_rater = []
        qwk_by_rater = []
        mse_by_rater = []

        # LOO computed per (utterance_id, aspect)
        for (utt, asp), g in dca.groupby(["utterance_id","aspect"], sort=False):
            # need at least 2 raters to form LOO consensus for anyone
            if g["rater_id"].nunique() < 2:
                continue

            loo = _loo_consensus_for_group(
                g[["rater_id","rating_calib"]],
                rater_col="rater_id",
                rating_col="rating_calib",
                dirichlet_alpha=dirichlet_alpha,
                hard_strategy=hard_strategy,
                halfrule=halfrule,
                priors=priors,
                soft_tiebreak=soft_tiebreak,
            )

            # accumulate per-rater comparisons on this item
            for _, row in g.iterrows():
                rid = str(row["rater_id"])
                if rid not in loo:
                    continue
                y_cont_loo, y_hard_loo = loo[rid]
                y_r = int(row["rating_calib"])

                mse_by_rater.append(((y_r - y_cont_loo) ** 2, rid))
                acc_by_rater.append((int(y_r == y_hard_loo), rid))
                qwk_by_rater.append(((y_r, y_hard_loo), rid))  # compute later per-rater as 2-class vectors
                f1_by_rater.append(((y_r, y_hard_loo), rid))

        if not mse_by_rater:
            rows.append({
                "corpus": corpus, "aspect": aspect,
                "n_raters": int(dca["rater_id"].nunique()),
                "n_items": int(dca["utterance_id"].nunique()),
                "human_mse_mean": np.nan, "human_mse_std": np.nan,
                "human_f1_macro_mean": np.nan, "human_f1_macro_std": np.nan,
                "human_acc_mean": np.nan, "human_qwk_mean": np.nan,
            })
            continue

        # roll up per-rater
        by_rater_true = defaultdict(list)
        by_rater_pred = defaultdict(list)
        by_rater_mse  = defaultdict(list)
        by_rater_acc  = defaultdict(list)

        for (val, rid) in mse_by_rater:
            by_rater_mse[rid].append(val)
        for (val, rid) in acc_by_rater:
            by_rater_acc[rid].append(val)
        for ((y_r, y_h), rid) in f1_by_rater:
            by_rater_true[rid].append(y_h)
            by_rater_pred[rid].append(y_r)

        # compute per-rater metrics
        per_mse = [float(np.mean(v)) for v in by_rater_mse.values() if len(v) > 0]
        per_acc = [float(np.mean(v)) for v in by_rater_acc.values() if len(v) > 0]
        per_f1  = []
        per_qwk = []
        for rid in by_rater_true.keys():
            y_true = np.array(by_rater_true[rid], dtype=int)
            y_pred = np.array(by_rater_pred[rid], dtype=int)
            # macro-F1 on 4 classes
            try:
                per_f1.append(float(f1_score(y_true, y_pred, average="macro", labels=[1,2,3,4], zero_division=0)))
            except Exception:
                # if a rater has only one class predicted/true, F1 can be undefined; skip
                continue
            try:
                per_qwk.append(float(cohen_kappa_score(y_true, y_pred, weights="quadratic")))
            except Exception:
                pass

        rows.append({
            "corpus": corpus,
            "aspect": aspect,
            "n_raters": int(dca["rater_id"].nunique()),
            "n_items": int(dca["utterance_id"].nunique()),
            "human_mse_mean": float(np.mean(per_mse)) if per_mse else np.nan,
            "human_mse_std": float(np.std(per_mse)) if per_mse else np.nan,
            "human_f1_macro_mean": float(np.mean(per_f1)) if per_f1 else np.nan,
            "human_f1_macro_std": float(np.std(per_f1)) if per_f1 else np.nan,
            "human_acc_mean": float(np.mean(per_acc)) if per_acc else np.nan,
            "human_qwk_mean": float(np.mean(per_qwk)) if per_qwk else np.nan,
            "human_qwk_std": float(np.std(per_qwk)) if per_qwk else np.nan,  # <-- add this line
        })

    return pd.DataFrame(rows).sort_values(["corpus","aspect"]).reset_index(drop=True)

def compute_pairwise_baselines(
    df_all: pd.DataFrame,
    outlier_report: Optional[pd.DataFrame],
    do_calibration: Literal["quantile","center",None] = "quantile",
) -> pd.DataFrame:
    """
    Optional: average pairwise rater-vs-rater metrics on shared items (per corpus × aspect).
    """
    df = _apply_outlier_filter(df_all, outlier_report)
    df = _calibrate(df, do_calibration)
    out = []
    for (corpus, aspect), dca in df.groupby(["corpus","aspect"], sort=False):
        raters = list(dca["rater_id"].unique())
        pairs = []
        for i in range(len(raters)):
            for j in range(i+1, len(raters)):
                r1, r2 = raters[i], raters[j]
                a = dca[dca["rater_id"] == r1][["utterance_id","rating_calib"]].set_index("utterance_id")
                b = dca[dca["rater_id"] == r2][["utterance_id","rating_calib"]].set_index("utterance_id")
                shared = a.join(b, how="inner", lsuffix="_a", rsuffix="_b")
                if len(shared) < 3:
                    continue
                y1 = shared["rating_calib_a"].astype(int).values
                y2 = shared["rating_calib_b"].astype(int).values
                mse = float(np.mean((y1 - y2)**2))
                acc = float(np.mean(y1 == y2))
                try:
                    f1 = float(f1_score(y2, y1, average="macro", labels=[1,2,3,4], zero_division=0))
                except Exception:
                    f1 = np.nan
                try:
                    qwk = float(cohen_kappa_score(y1, y2, weights="quadratic"))
                except Exception:
                    qwk = np.nan
                pairs.append((mse, acc, f1, qwk))
        if pairs:
            mses, accs, f1s, qwks = map(np.array, zip(*pairs))
            out.append({
                "corpus": corpus, "aspect": aspect,
                "pairwise_mse_mean": float(np.nanmean(mses)),
                "pairwise_f1_macro_mean": float(np.nanmean(f1s)),
                "pairwise_acc_mean": float(np.nanmean(accs)),
                "pairwise_qwk_mean": float(np.nanmean(qwks)),
                "n_pairs": int(len(pairs)),
            })
        else:
            out.append({
                "corpus": corpus, "aspect": aspect,
                "pairwise_mse_mean": np.nan,
                "pairwise_f1_macro_mean": np.nan,
                "pairwise_acc_mean": np.nan,
                "pairwise_qwk_mean": np.nan,
                "n_pairs": 0,
            })
    return pd.DataFrame(out).sort_values(["corpus","aspect"]).reset_index(drop=True)

def rater_vs_gold_scores(
    df_all: pd.DataFrame,
    final_labels: pd.DataFrame,
    outlier_report: pd.DataFrame | None,
    do_calibration: Literal["quantile","center",None] = "center",
    use_loo: bool = True,         # True = unbiased LOO gold; False = includes rater in gold
    dirichlet_alpha: float = 0.0, # must match how you built y_cont/y_hard if you recompute LOO
    hard_strategy: Literal["two_mean_then_halfrule","median_then_soft"] = "two_mean_then_halfrule",
    halfrule: Literal["down","up","even","soft","prior"] = "down",
    soft_tiebreak: Literal["lower","upper","prior"] = "prior",
) -> pd.DataFrame:
    """
    Returns one row per rater × corpus × aspect with:
      n_items, mse (soft vs y_cont), mae, macro_f1 (hard vs y_hard), qwk, acc.
    If use_loo=True, gold is recomputed per item excluding the rater (preferred).
    Otherwise we use provided final_labels (may include the rater's vote).
    """
    # 1) Filter dropped raters for fairness
    df = _apply_outlier_filter(df_all, outlier_report)
    if df.empty:
        return pd.DataFrame(columns=["rater_id","corpus","aspect","n_items","mse","mae","macro_f1","qwk","acc"])
    # 2) Calibrate like in your pipeline
    df = _calibrate(df, do_calibration)

    # gold (biased variant)
    gold_cols = ["corpus","utterance_id","aspect","y_cont","y_hard"]
    gold = final_labels[gold_cols].drop_duplicates().copy()

    rows = []

    # Precompute global priors for LOO halfrule/soft tie-breaking
    counts_all = df["rating_calib"].value_counts().reindex([1,2,3,4], fill_value=0).values.astype(float)
    priors = counts_all / max(counts_all.sum(), 1.0)

    # Work per corpus × aspect
    for (corpus, aspect), dca in df.groupby(["corpus","aspect"], sort=False):
        # Merge biased gold once (used if use_loo=False)
        g_slice = gold[(gold.corpus==corpus) & (gold.aspect==aspect)][["utterance_id","y_cont","y_hard"]]

        for rater, dr in dca.groupby("rater_id", sort=False):
            # only items the rater labeled
            items = dr[["utterance_id","rating_calib"]].copy().rename(columns={"rating_calib":"rater_rating"})
            if use_loo:
                # build LOO gold per item *for this rater*
                loo_cont = []
                loo_hard = []
                rater_vals = []
                # build groups per item on the full set of ratings for this aspect/corpus
                for utt, g in dca[dca["utterance_id"].isin(items["utterance_id"])].groupby("utterance_id", sort=False):
                    # need >=2 raters to form LOO for this rater
                    if g["rater_id"].nunique() < 2:
                        continue
                    # compute LOO consensus for this item
                    loo_map = _loo_consensus_for_group(
                        g[["rater_id","rating_calib"]],
                        rater_col="rater_id",
                        rating_col="rating_calib",
                        dirichlet_alpha=dirichlet_alpha,
                        hard_strategy=hard_strategy,
                        halfrule=halfrule,
                        priors=priors,
                        soft_tiebreak=soft_tiebreak,
                    )
                    if str(rater) not in loo_map:
                        continue
                    y_cont_loo, y_hard_loo = loo_map[str(rater)]
                    # rater's own rating on this item
                    rr = int(dr.loc[dr["utterance_id"]==utt, "rating_calib"].values[0])
                    loo_cont.append((utt, y_cont_loo))
                    loo_hard.append((utt, y_hard_loo))
                    rater_vals.append((utt, rr))
                if not rater_vals:
                    continue
                df_r = pd.DataFrame(rater_vals, columns=["utterance_id","rater_rating"])
                df_c = pd.DataFrame(loo_cont, columns=["utterance_id","y_cont"])
                df_h = pd.DataFrame(loo_hard, columns=["utterance_id","y_hard"])
                eval_df = df_r.merge(df_c, on="utterance_id").merge(df_h, on="utterance_id")
            else:
                # biased: compare to provided gold
                eval_df = items.merge(g_slice, on="utterance_id", how="inner")

            if eval_df.empty:
                continue

            # Soft metrics: rater integer vs y_cont
            y_r = eval_df["rater_rating"].astype(float).values
            y_c = eval_df["y_cont"].astype(float).values
            mse = float(np.mean((y_r - y_c)**2))
            mae = float(np.mean(np.abs(y_r - y_c)))

            # Hard metrics: rater integer vs y_hard (class)
            y_h = eval_df["y_hard"].astype(int).values
            acc = float(np.mean(y_r.astype(int) == y_h))
            try:
                macro_f1 = float(f1_score(y_h, y_r.astype(int), average="macro", labels=[1,2,3,4], zero_division=0))
            except Exception:
                macro_f1 = np.nan
            try:
                qwk = float(cohen_kappa_score(y_h, y_r.astype(int), weights="quadratic"))
            except Exception:
                qwk = np.nan

            rows.append({
                "rater_id": rater,
                "corpus": corpus,
                "aspect": aspect,
                "n_items": int(len(eval_df)),
                "mse": mse,
                "mae": mae,
                "macro_f1": macro_f1,
                "qwk": qwk,
                "acc": acc,
                "gold_variant": "LOO" if use_loo else "biased",
            })

    return pd.DataFrame(rows).sort_values(["corpus","aspect","rater_id","gold_variant"]).reset_index(drop=True)

def human_range_table(human_var: pd.DataFrame) -> pd.DataFrame:
    """
    Input: human_var from compute_human_variation (must contain human_qwk_mean and human_qwk_std).
    Output: one row per corpus × aspect with mean and [lower, upper] = mean ± 1 SD.
    """
    cols_needed = {"corpus","aspect","human_qwk_mean","human_qwk_std"}
    if not cols_needed.issubset(human_var.columns):
        raise ValueError(f"human_var must have columns {cols_needed}")
    t = human_var[["corpus","aspect","human_qwk_mean","human_qwk_std"]].copy()
    t["lower"] = t["human_qwk_mean"] - t["human_qwk_std"]
    t["upper"] = t["human_qwk_mean"] + t["human_qwk_std"]
    # Clip to [-1, 1] just in case (QWK is bounded)
    t["lower"] = t["lower"].clip(-1.0, 1.0)
    t["upper"] = t["upper"].clip(-1.0, 1.0)
    return t.rename(columns={
        "human_qwk_mean": "mean_qwk",
        "human_qwk_std": "sd_qwk"
    }).sort_values(["corpus","aspect"]).reset_index(drop=True)

def model_range_table(models_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: models_df with multiple model versions and their QWKs, columns:
           ['model','corpus','aspect','qwk']  (you can name 'model' anything; it’s just grouped over)
    Output: mean and [lower, upper] = mean ± 1 SD across model versions for each slice.
            Also includes min/max so you can eyeball spread if you want.
    """
    cols_needed = {"model","corpus","aspect","qwk"}
    if not cols_needed.issubset(models_df.columns):
        raise ValueError(f"models_df must have columns {cols_needed}")

    g = models_df.groupby(["corpus","aspect"], sort=False)["qwk"]
    out = g.agg(
        mean_qwk="mean",
        sd_qwk="std",
        min_qwk="min",
        max_qwk="max",
        n_models="count",
    ).reset_index()

    out["lower"] = out["mean_qwk"] - out["sd_qwk"]
    out["upper"] = out["mean_qwk"] + out["sd_qwk"]
    out["lower"] = out["lower"].clip(-1.0, 1.0)
    out["upper"] = out["upper"].clip(-1.0, 1.0)
    return out.sort_values(["corpus","aspect"]).reset_index(drop=True)

def main():
    # 0) Load raw docs
    annotations_cur = get_annotations()
    raw_annotations = [doc for doc in annotations_cur if str(doc.get("annotator","")).strip() not in BLACKLIST]

    # 1) Parse to tidy df
    df_all = parse_mongo_annotations(raw_annotations)
    if df_all.empty:
        print("No annotations found after parsing/blacklist.")
        return

    # 2) Build final labels with outlier removal + calibration + aggregation
    final_labels, outlier_report = build_final_labels(
        df_all,
        do_calibration=None,
        dirichlet_alpha=0.0,
        hard_strategy="two_mean_then_halfrule",
        halfrule="down",
        soft_tiebreak="prior",
        include_continuous=True,
        include_interval_masks=True
    )

    # 2b) Krippendorff's alpha per corpus × aspect
    alphas_clean = compute_corpus_alphas(df_all, outlier_report=outlier_report, use_filtered=True, metric="ordinal")
    # (optional) also see raw (no outlier removal)
    alphas_raw = compute_corpus_alphas(df_all, outlier_report=None, use_filtered=False, metric="ordinal")

    print("\n=== Krippendorff's alpha (raw, ordinal) by corpus × aspect ===")
    print(alphas_raw)

    print("\n=== Krippendorff's alpha (cleaned, ordinal) by corpus × aspect ===")
    print(alphas_clean)

    evaluate_against_seed_gt(final_labels)

    # 3) Human baselines (average human variation)
    human_var = compute_human_variation(
        df_all,
        outlier_report=outlier_report,
        do_calibration=None,            # match your aggregation choice
        dirichlet_alpha=0.0,
        hard_strategy="two_mean_then_halfrule",
        halfrule="down",
        soft_tiebreak="prior",
    )
    print("\n=== Human LOO baselines (per corpus × aspect) ===")
    print(human_var)

    # (Optional) Pairwise baselines
    pairwise = compute_pairwise_baselines(
        df_all, outlier_report=outlier_report, do_calibration="center"
    )
    print("\n=== Pairwise rater-vs-rater baselines (per corpus × aspect) ===")
    print(pairwise)

    # 4) (Optional) Save to Mongo or csv
    # db["final_labels"].drop({})
    # db["final_labels"].insert_many(final_labels.to_dict(orient="records"))
    final_labels.to_csv("final_labels.csv", index=False)
    # outlier_report.to_csv("outlier_report.csv", index=False)

    # Unbiased (recommended)
    per_rater_loo = rater_vs_gold_scores(
        df_all, final_labels, outlier_report,
        do_calibration=None, use_loo=True,
        dirichlet_alpha=0.0, hard_strategy="two_mean_then_halfrule",
        halfrule="down", soft_tiebreak="prior"
    )
    print("rater_vs_gold_scores with LOO")
    print(
        per_rater_loo
        .groupby(["corpus", "aspect"])[["qwk", "macro_f1", "mse", "mae"]]
        .mean()
    )

    # Biased (gold includes the rater)
    print("rater_vs_gold_scores without LOO")
    per_rater_biased = rater_vs_gold_scores(
        df_all, final_labels, outlier_report,
        do_calibration=None, use_loo=False,
        dirichlet_alpha=0.0, hard_strategy="two_mean_then_halfrule",
        halfrule="down", soft_tiebreak="prior"
    )
    print(per_rater_biased.groupby(["corpus", "aspect"])[["qwk", "macro_f1", "mse", "mae"]]
        .mean())

    # 1) Human acceptable range (use this to judge models)
    human_bounds = human_range_table(human_var)
    print("\n=== Human (LOO) acceptable range: mean ± 1 SD ===")
    print(human_bounds[["corpus", "aspect", "mean_qwk", "lower", "upper"]])
if __name__ == "__main__":
    main()
