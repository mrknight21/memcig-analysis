from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


ASPECT_MAP = {
    "CIG": "informativeness",
    "Novelty": "novelty",
    "Relevance": "relevance",
    "Scope": "implication_scope",
}
CONTEXT_ORDER = ["full", "memory", "no_knowledge", "short_prior", "summary"]
CONTEXT_LABELS = {
    "full": "Full",
    "memory": "Memory",
    "no_knowledge": "No Knowledge",
    "short_prior": "Short Prior",
    "summary": "Summary",
}
MODEL_ORDER = [
    "gpt-5",
    "gpt-5-mini",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "vllm:Qwen/Qwen3-4B-Instruct-2507",
]
MODEL_LABELS = {
    "gpt-5": "GPT-5",
    "gpt-5-mini": "GPT-5-Mini",
    "gemini-2.5-pro": "Gemini-2.5-Pro",
    "gemini-2.5-flash": "Gemini-2.5-Flash",
    "vllm:Qwen/Qwen3-4B-Instruct-2507": "Qwen3-4B",
}
EXCLUDED_MODELS = {
    "gpt-5_True",
    "gpt-5_hm",
    "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
}


def normalize_id(value: Any) -> str:
    nums = re.findall(r"\d+", str(value))
    return nums[-1] if nums else str(value)


def load_feature_mapping(feature_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(feature_dir.glob("*.csv")):
        df = pd.read_csv(
            path,
            usecols=lambda col: col in {"conversation_id", "utterance_id", "utterance_index"},
        )
        df["conversation_id"] = df["conversation_id"].apply(normalize_id).astype(str)
        df["utterance_id"] = df["utterance_id"].astype(str)
        df["utterance_index"] = df["utterance_index"].astype(str)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No feature CSVs found in {feature_dir}")
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        ["conversation_id", "utterance_id", "utterance_index"]
    )


def load_human_labels(labels_path: Path, feature_map: pd.DataFrame) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)
    labels["conversation_id"] = labels["conv_id"].apply(normalize_id).astype(str)
    labels["utterance_id"] = labels["utterance_id"].astype(str)
    merged = labels.merge(feature_map, on=["conversation_id", "utterance_id"], how="inner")
    merged["utterance_index"] = merged["utterance_index"].astype(str)
    return merged


def _iter_prediction_items(
    run_idx: int,
    conversation_id: str,
    model: str,
    contexts: Any,
) -> Iterable[Dict[str, Any]]:
    if model in EXCLUDED_MODELS:
        return

    if isinstance(contexts, dict):
        for context, items in contexts.items():
            if context not in CONTEXT_ORDER or not isinstance(items, list):
                continue
            for pred in items:
                yield from _prediction_row(run_idx, conversation_id, model, context, pred)

    elif isinstance(contexts, list):
        # OSS/Qwen runs are stored as one dict per target utterance:
        # [{"full": {...}, "memory": {...}, ...}, ...]
        for utterance_contexts in contexts:
            if not isinstance(utterance_contexts, dict):
                continue
            for context, value in utterance_contexts.items():
                if context not in CONTEXT_ORDER:
                    continue
                items = value if isinstance(value, list) else [value]
                for pred in items:
                    yield from _prediction_row(run_idx, conversation_id, model, context, pred)


def _prediction_row(
    run_idx: int,
    conversation_id: str,
    model: str,
    context: str,
    pred: Any,
) -> Iterable[Dict[str, Any]]:
    if not isinstance(pred, dict) or pred.get("utterance_index") is None:
        return
    row: Dict[str, Any] = {
        "run": run_idx,
        "conversation_id": str(conversation_id),
        "utterance_index": str(pred["utterance_index"]),
        "model": model,
        "context": context,
    }
    for model_col in ASPECT_MAP.values():
        row[model_col] = float(pred[model_col]) if pred.get(model_col) is not None else np.nan
    yield row


def load_predictions(ratings_dir: Path, runs: List[int]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run_idx in runs:
        path = ratings_dir / f"run_{run_idx}.json"
        data = json.loads(path.read_text())
        for task in data:
            conversation_id = normalize_id(task.get("conversation_id"))
            for model, contexts in task.get("predictions", {}).items():
                rows.extend(_iter_prediction_items(run_idx, conversation_id, model, contexts))
    return pd.DataFrame(rows)


def calculate_mae(
    labels_path: Path,
    feature_dir: Path,
    ratings_dir: Path,
    runs: List[int],
) -> pd.DataFrame:
    feature_map = load_feature_mapping(feature_dir)
    human_df = load_human_labels(labels_path, feature_map)
    pred_df = load_predictions(ratings_dir, runs)

    results: List[Dict[str, Any]] = []
    for model in sorted(
        pred_df["model"].unique(),
        key=lambda item: MODEL_ORDER.index(item) if item in MODEL_ORDER else 999,
    ):
        for context in CONTEXT_ORDER:
            context_df = pred_df[(pred_df["model"] == model) & (pred_df["context"] == context)]
            if context_df.empty:
                continue
            for human_aspect, model_col in ASPECT_MAP.items():
                human_subset = human_df[human_df["aspect"] == human_aspect]
                merged = human_subset.merge(
                    context_df[["run", "conversation_id", "utterance_index", model_col]],
                    on=["conversation_id", "utterance_index"],
                    how="inner",
                ).dropna(subset=["y_cont", model_col])
                if merged.empty:
                    continue
                for corpus, corpus_group in merged.groupby("corpus"):
                    run_maes = []
                    run_ns = []
                    for _, run_group in corpus_group.groupby("run"):
                        run_maes.append(
                            float(np.mean(np.abs(run_group["y_cont"] - run_group[model_col])))
                        )
                        run_ns.append(len(run_group))
                    results.append(
                        {
                            "corpus": corpus,
                            "model": model,
                            "context": context,
                            "aspect": human_aspect,
                            "mae": float(np.mean(run_maes)),
                            "mae_std": float(np.std(run_maes)),
                            "n_runs": len(run_maes),
                            "n_samples": int(np.mean(run_ns)),
                        }
                    )

    results_df = pd.DataFrame(results)
    results_df["model_sort"] = results_df["model"].map(
        {model: idx for idx, model in enumerate(MODEL_ORDER)}
    ).fillna(999)
    results_df["context_sort"] = results_df["context"].map(
        {context: idx for idx, context in enumerate(CONTEXT_ORDER)}
    )
    results_df["aspect_sort"] = results_df["aspect"].map(
        {"CIG": 0, "Novelty": 1, "Relevance": 2, "Scope": 3}
    )
    return results_df.sort_values(
        ["corpus", "model_sort", "context_sort", "aspect_sort"]
    ).drop(columns=["model_sort", "context_sort", "aspect_sort"])


def write_latex_table(results_df: pd.DataFrame, output_path: Path) -> None:
    aspects = list(ASPECT_MAP.keys())
    corpora = ["fora", "insq"]
    values: Dict[tuple, float] = {}

    for _, row in results_df.iterrows():
        values[(row.model, row.context, row.corpus, row.aspect)] = row.mae

    for model in MODEL_ORDER:
        for context in CONTEXT_ORDER:
            for corpus in corpora:
                aspect_values = [
                    values.get((model, context, corpus, aspect), np.nan) for aspect in aspects
                ]
                if any(pd.notna(aspect_values)):
                    values[(model, context, corpus, "Mean")] = float(np.nanmean(aspect_values))

    columns = [(corpus, aspect) for corpus in corpora for aspect in aspects + ["Mean"]]
    minima = {
        column: np.nanmin(
            [
                values.get((model, context, column[0], column[1]), np.nan)
                for model in MODEL_ORDER
                for context in CONTEXT_ORDER
            ]
        )
        for column in columns
    }

    def fmt(value: float, best: float) -> str:
        if pd.isna(value):
            return "--"
        rendered = f"{value:.3f}"
        return rf"\textbf{{{rendered}}}" if abs(value - best) < 0.0005 else rendered

    lines = [
        r"\begin{table*}[h]",
        r"\centering",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccc|ccccc}",
        r"\toprule",
        r"& & \multicolumn{5}{c}{\textbf{Fora}} & \multicolumn{5}{c}{\textbf{Insq}} \\",
        r"\cmidrule(lr){3-7} \cmidrule(lr){8-12}",
        r"\textbf{Model} & \textbf{Context} & \textbf{CIG} & \textbf{Nov.} & \textbf{Rel.} & \textbf{Scope} & \textbf{Mean} & \textbf{CIG} & \textbf{Nov.} & \textbf{Rel.} & \textbf{Scope} & \textbf{Mean} \\",
        r"\midrule",
    ]

    for model_index, model in enumerate(MODEL_ORDER):
        first_model_row = True
        for context in CONTEXT_ORDER:
            row = [MODEL_LABELS[model] if first_model_row else "", CONTEXT_LABELS[context]]
            for corpus in corpora:
                for aspect in aspects + ["Mean"]:
                    value = values.get((model, context, corpus, aspect), np.nan)
                    row.append(fmt(value, minima[(corpus, aspect)]))
            lines.append(" & ".join(row) + r" \\")
            first_model_row = False
        if model_index != len(MODEL_ORDER) - 1:
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\caption{MAE results for different models, contexts, and aspects across FORA and INSQ corpora. The last column in each corpus group shows the mean MAE across the four aspects. The lowest MAE in each column is highlighted in bold.}",
            r"\label{tab:mae_results}",
            r"\end{table*}",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate model/context MAE tables.")
    parser.add_argument("--labels", type=Path, default=Path("analysis/final_labels.csv"))
    parser.add_argument("--features", type=Path, default=Path("data/archive_local/feat"))
    parser.add_argument("--ratings", type=Path, default=Path("data/archive_local/ratings"))
    parser.add_argument("--runs", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out-csv", type=Path, default=Path("analysis/mae_results_detailed.csv"))
    parser.add_argument("--out-tex", type=Path, default=Path("results/mae_model_context_table.tex"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_df = calculate_mae(args.labels, args.features, args.ratings, args.runs)
    results_df.to_csv(args.out_csv, index=False)
    write_latex_table(results_df, args.out_tex)
    print(f"Saved {len(results_df)} rows to {args.out_csv}")
    print(f"Saved LaTeX table to {args.out_tex}")
    print(results_df.groupby("model")["n_runs"].agg(["min", "max", "count"]).to_string())


if __name__ == "__main__":
    main()
