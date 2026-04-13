# Scripts

This folder contains the reusable pipeline scripts used by the project.

## Main Pipeline

- `dialogue_segmentation.py`: segment raw conversation transcripts into subtopic spans.
- `dialogue_summarisation.py`: build full, summary, and memory-based prior-context representations.
- `dialogue_rating.py`: run LLM-based utterance ratings for CIG, novelty, relevance, and implication scope.
- `aggregate_annotation.py`: aggregate anonymized human annotations into consensus labels.
- `build_integrated_label_release.py`: build a release-friendly utterance-label table with explicit label provenance.
- `archive_memory_embeddings.py`: one-off helper to move dense prior-memory vectors from local metadata into the ignored `data/archive_local/embeddings/` sidecar.

## Release Data Builder

Build the labels-only release table:

```bash
python scripts/build_integrated_label_release.py \
  --annotation-release data/archive_local/release_export/annotations_release.json \
  --human-final-labels analysis/final_labels.csv \
  --exclude-text \
  --output data/release/integrated_labels_labels_only.csv
```

Build a local full-conversation table after obtaining transcripts locally:

```bash
python scripts/build_integrated_label_release.py \
  --annotation-release data/archive_local/release_export/annotations_release.json \
  --human-final-labels analysis/final_labels.csv \
  --llm-ratings data/archive_local/ratings/tasks_ratings_gemini_1.json \
  --conversation-root data/raw \
  --output data/release/integrated_labels_with_local_text.csv
```

The full-conversation output is for local analysis unless transcript redistribution is permitted.

## Latency Appendix

- `benchmark_memory_latency_debug.py`: benchmark memory-operation latency with per-operation logs.
- `benchmark_rating_latency.py`: benchmark rating inference latency in memory mode.

INSQ can be run directly from the release metadata:

```bash
python scripts/benchmark_rating_latency.py \
  --corpus insq \
  --episode-id insq_2228 \
  --segments 0,1,2 \
  --out-json /tmp/insq_rating_latency.json
```

```bash
python scripts/benchmark_memory_latency_debug.py \
  --corpus insq \
  --episode-id insq_2228 \
  --segments 0,1,2 \
  --out-json /tmp/insq_memory_latency.json \
  --log-file /tmp/insq_memory_latency.log
```

The generated LaTeX appendix table is stored in `results/latency_tables.tex`.

## Notes

Most scripts assume they are run from the repository root with `PYTHONPATH=.`. INSQ transcript and enriched metadata files live under `data/raw/insq/`. Development-only intermediates such as ratings, WHoW annotations, feature caches, archived embeddings, and annotation task exports live under the ignored `data/archive_local/` folder. Some model-running scripts require external API keys or local model infrastructure.
