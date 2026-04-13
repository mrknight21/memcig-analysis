# Data Manifest

This directory contains local research data and a smaller publication-oriented release artifact.

## Included Release Artifact

The files in `release/` are the intended publication-facing annotation dump. The canonical public annotation table is `integrated_labels_labels_only.csv`.

- `integrated_labels_labels_only.csv`: human individual labels plus human consensus labels in one utterance-level table
- `integrated_labels_labels_only.summary.json`: row counts for the integrated labels-only table
- `fora/labels_only.csv`: FORA-derived labels only; no FORA transcripts are included
- `fora/README.md`: redirect to the upstream FORA corpus and access process
- `insq/labels_only.csv`: INSQ-derived labels only
- `raw/insq/`: redistributable INSQ transcript files plus enriched pipeline metadata used by the project

The `raw/insq/*_meta.json` files are not just raw conversation metadata. They include study-generated artifacts such as segment boundaries, prior summaries, semantic-memory items, GPT-5 ratings under multiple summary/context variants, summary comparison scores, and full-context GPT-5 ratings. Dense embedding vectors were removed from public memory entries to keep the repository lightweight. If you have the local research archive, `scripts/archive_memory_embeddings.py` can place those vectors in the ignored `data/archive_local/embeddings/` sidecar; otherwise the pipeline regenerates embeddings when rebuilding memory stores. These model-generated annotations are included to support reproduction of the paper analyses, but human labels should be read from `release/integrated_labels_labels_only.csv`.

`annotations_release_long.csv` was an older flattened raw export of individual annotator ratings. It overlaps with `integrated_labels_labels_only.csv`, which is richer and includes both individual and consensus labels, so the long export has been moved to the ignored local archive. `annotations_release.json` is also a local/export artifact that includes task context and has been moved to the ignored local archive. Because it may contain FORA-derived text, it should not be included in the public archive until FORA redistribution approval is confirmed or the FORA context is removed.

## What The Release Contains

- `80` base annotated tasks
- anonymized annotator ids in the form `annotator_###`
- utterance-level labels for `info` and `mix` annotation variants
- labels-only tables with no utterance text or speaker names
- INSQ transcript text and enriched pipeline metadata under `raw/insq/`
- explicit label provenance fields:
- `label_source=human_individual` for anonymized annotator-level labels
- `label_source=human_consensus` for aggregated human labels from `final_labels.csv`
- `label_source=llm` for optional model-prediction rows generated from rating outputs

## Integrating Labels With Conversation Text

Use `scripts/build_integrated_label_release.py` to rebuild an integrated utterance-level table.

Labels-only release table:

```bash
python scripts/build_integrated_label_release.py \
  --annotation-release data/archive_local/release_export/annotations_release.json \
  --human-final-labels analysis/final_labels.csv \
  --exclude-text \
  --output data/release/integrated_labels_labels_only.csv
```

Local full-conversation table, including optional LLM prediction rows:

```bash
python scripts/build_integrated_label_release.py \
  --annotation-release data/archive_local/release_export/annotations_release.json \
  --human-final-labels analysis/final_labels.csv \
  --llm-ratings data/archive_local/ratings/tasks_ratings_gemini_1.json \
  --conversation-root data/raw \
  --output data/release/integrated_labels_with_local_text.csv
```

The `--conversation-root` option joins local transcript text from `data/raw/<corpus>/<conversation_id>.csv`. INSQ transcript text may be used from the included `data/raw/insq/` folder. Because FORA redistribution is still pending approval, full-text outputs containing FORA transcript content should be treated as local/private until permission is confirmed.

For FORA specifically, the public release should include only `data/release/fora/labels_only.csv` and `data/release/fora/README.md`. Users who need FORA transcript text should obtain it from the upstream corpus repository:

https://github.com/schropes/fora-corpus

## What Was Removed

The release dump excludes direct annotation-platform identifiers such as:

- raw annotator handles
- submission ids
- study ids
- session ids
- submission timestamps

## Other Data Folders

Other folders under `data/` are working research artifacts from development, feature extraction, or analysis. They may depend on local preprocessing steps, outside corpora, or service-specific infrastructure. These local artifacts have been moved to the ignored `data/archive_local/` folder when retained for audit/debugging.

Examples include:

- `raw/`
- `archive_local/processed_segments/`
- `archive_local/feat/`
- `archive_local/ratings/`
- `archive_local/entropy/`
- `archive_local/tasks/`
- `archive_local/embeddings/`

These are not guaranteed to be complete, minimal, or redistribution-ready in the same way as the files under `release/`.

## Redistribution Note

The annotation release artifact is anonymized, but it may include task context derived from source conversations. INSQ transcripts are from a public released dataset and can be redistributed in this release. FORA transcript-derived context should be withheld from the public archive until redistribution approval is confirmed.

For INSQ source attribution, cite the ConvoKit Intelligence Squared Debates Corpus (`iq2-corpus`) and Zhang et al. (2016), "Conversational Flow in Oxford-Style Debates." See `../DATA_NOTICE.md`.
