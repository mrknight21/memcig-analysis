# Analysis

This folder contains notebooks and helper scripts used to reproduce paper analyses.

## Public INSQ Inputs

INSQ transcript CSVs and enriched metadata JSON files are available under `data/raw/insq/`. The `*_meta.json` files include segment boundaries, prior summaries, semantic-memory artifacts, GPT-5 ratings under multiple summary/context variants, summary scores, and full-context GPT-5 ratings.

Human labels should be loaded from:

- `analysis/final_labels.csv`: canonical final human consensus labels used by analysis scripts.
- `data/release/integrated_labels_labels_only.csv`: canonical public labels-only release table with human-individual and human-consensus rows.

## Local Archive Inputs

Some notebooks and scripts reproduce analyses that depend on development intermediates, such as feature caches, model-rating runs, WHoW labels, or raw annotation task exports. Those files now live under the ignored `data/archive_local/` folder.

If a notebook references `data/archive_local/...`, it is using local research artifacts rather than the minimal public release surface.

## Runnable Script Defaults

The maintained analysis helper defaults have been updated for the refactored layout:

- `analysis/correlation_analysis.py` defaults to `data/archive_local/feat`, `analysis/final_labels.csv`, and `data/archive_local/ratings`.
- `analysis/calculate_mae.py` rebuilds `analysis/mae_results_detailed.csv` and `results/mae_model_context_table.tex` from the local `run_*.json` rating bundles.
- `analysis/repro_mem_feat.py` defaults to the redistributable INSQ transcript `insq_2228` and writes regenerated features to `data/archive_local/feat/insq/`.
- Debug helpers use `data/raw/insq/*_meta.json` when possible and fall back to `data/archive_local/processed_segments/`.
