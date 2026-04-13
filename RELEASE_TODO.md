# Release TODO

This checklist captures the remaining work before publishing the repository. The goal is to release a useful, reproducible codebase while avoiding accidental redistribution of restricted transcript text, local credentials, or exploratory clutter.

## Must Do Before Public Release

- Confirm the data release scope. Keep FORA transcript text out of the public repository until redistribution approval is confirmed; release only FORA labels and README pointers to the upstream FORA corpus.
- Document the INSQ transcript source, license/terms, and citation because INSQ text is approved for redistribution in this release. Current docs cite the ConvoKit IQ2 page and Zhang et al. (2016); the ConvoKit page does not state a separate corpus-specific license.
- Make clear that `data/raw/insq/*_meta.json` includes enriched pipeline artifacts, including GPT-5 ratings under multiple summary/context variants, not only raw corpus metadata.
- Keep local/generated data in the ignored `data/archive_local/` folder, including previous `data/cache/`, `data/processed_segments/`, `data/ratings/`, `data/tasks/`, `data/whow/`, `data/whow_sequences/`, raw FORA transcript CSVs, and any local full-text integrated releases.
- Verify that public release files contain no direct speaker names, annotator IDs, platform IDs, usernames, API keys, Mongo credentials, or private task-management notes.
- Keep `analysis/final_labels.csv` as the canonical final human-label file and ensure stale alternatives remain deleted from the release.
- Keep `data/release/integrated_labels_labels_only.csv` as the canonical public annotation release; the older `annotations_release_long.csv` raw export overlaps and should remain archived/ignored.
- Stage the refactor intentionally. The working tree currently contains many deletions from moving `static/` and `templates/` into `website/`, plus several new files; do not commit all changes blindly.
- Remove IDE, cache, and deployment artifacts from the release, including `.idea/`, `.DS_Store`, `__pycache__/`, `.ipynb_checkpoints/`, `.elasticbeanstalk/app_versions/`, and local virtual environments.
- Run a final secret scan over the exact staged release contents, not just the current working directory.

## Should Do For Reproducibility

- Re-run the core smoke checks after final staging: import checks, Flask routes, and `py_compile` for maintained scripts.
- Test the documented setup path in a clean environment using `environment.memcig.yml` or `requirements.txt`.
- Validate that `scripts/build_integrated_label_release.py --help` works and that the labels-only release can be regenerated from the approved local inputs.
- Confirm that latency scripts fail clearly when API keys or backends are missing, rather than hanging silently.
- Add a short note for expensive/API-backed scripts explaining expected runtime, required keys, and model settings.
- Confirm that `mem0`, OpenAI, Gemini, Qdrant, and local embedding dependencies are pinned tightly enough for a reader to reproduce the pipeline.

## Should Do For Documentation

- Add citation instructions once the paper metadata is final, including BibTeX and corpus citations for INSQ and FORA.
- Add a data dictionary for the integrated labels-only CSV, including label provenance fields for human individual labels, human consensus labels, and LLM labels.
- Add a short end-to-end example showing how to run CIG scoring on a small user-provided conversation without requiring the full paper dataset.
- Clarify which scripts are stable public entry points and which notebooks/scripts are archival or exploratory.
- Link `results/findings_summary.md` and `results/latency_tables.tex` from any final project webpage or archive landing page.

## Nice To Have

- Add minimal automated tests for the integrated-label builder and anonymization checks.
- Add a small toy transcript under `examples/` so users can run segmentation, memory updates, and rating without private or restricted data.
- Convert key notebooks into deterministic scripts where possible, especially for paper tables that should be reproducible.
- Add a release manifest listing exactly which files are intended to be included in the public artifact.
- Add a Zenodo/GitHub release checklist once the final camera-ready version is accepted.

## Current Release Blockers

- FORA redistribution approval is still pending, so FORA transcript text must remain excluded.
- The current working tree contains many unstaged/untracked refactor changes and local/generated files; final publication needs a careful staging and audit pass.
- Some analysis notebooks and generated intermediates are still exploratory and should be marked archival, cleaned, or excluded.
