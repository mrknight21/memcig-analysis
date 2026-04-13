# memcig

Code and release artifacts for the paper "CIG: Measuring Conversational Information Gain in Deliberative Dialogues with Semantic Memory Dynamics".

This repository is organized into four primary components:

- annotation website in `website/`
- data and release artifacts in `data/`
- analysis notebooks/results in `analysis/` and `results/`
- experiment scripts in `scripts/`

## Release Status

This codebase was originally developed as a working research repository. The current refactor focuses on making the published artifact easier to understand and safer to release:

- direct Mongo credentials were removed from tracked analysis files
- tester-specific blacklist handles were removed from public defaults
- the finalized annotation dump was exported locally and anonymized

Some parts of the repo require external services or model backends, but the sections below describe the intended release surface.

## Repository Layout

- `website/`: Flask annotation interface, templates, static assets, and website-related utilities
- `application.py`: compatibility entrypoint that exposes `website.application`
- `mongodb/`: Mongo helper functions used by the website
- `scripts/dialogue_segmentation.py`: segment dialogue into subtopic units
- `scripts/dialogue_summarisation.py`: generate prior-memory summaries
- `scripts/dialogue_rating.py`: run LLM-based CIG/aspect prediction
- `scripts/aggregate_annotation.py`: aggregate human annotations
- `scripts/README.md`: script-level usage guide
- `analysis/`: paper analysis notebooks and result tables
- `results/`: publication-ready generated tables (e.g., latency appendix LaTeX)
- `data/README.md`: data manifest and release notes

## Data

The public release does not assume that every intermediate dataset can be redistributed directly from Git.

The main included release artifacts are labels-only annotation tables and redistributable INSQ transcripts:

- `data/release/integrated_labels_labels_only.csv`
- `data/release/fora/labels_only.csv`
- `data/release/insq/labels_only.csv`
- `data/raw/insq/`: redistributable INSQ transcripts plus enriched pipeline metadata

See `data/README.md` for field-level details and redistribution caveats.

`data/release/integrated_labels_labels_only.csv` is the canonical public annotation table. It includes anonymized human-individual rows, human-consensus rows, and explicit label provenance. INSQ transcript text is included separately because it can be redistributed for this release. FORA transcript text is not part of the public release while redistribution approval is pending; users should obtain FORA transcripts from the upstream corpus repository. In this repository, the code is released under `Apache-2.0`; the included data artifacts should be understood as research release materials accompanying the paper.

The INSQ `*_meta.json` files include study-generated metadata such as segment boundaries, prior summaries, semantic-memory items, and GPT-5 ratings under multiple summary/context variants. These fields support context-ablation and automated-rating analyses; they should not be confused with human labels, which are released in the integrated labels table.

See `DATA_NOTICE.md` for source-corpus attribution and redistribution notes. INSQ data is derived from the ConvoKit Intelligence Squared Debates Corpus (`iq2-corpus`) and the original Zhang et al. (2016) Oxford-style debates dataset.

## Existing Findings

The paper uses this codebase to study Conversational Information Gain (CIG): how much an utterance advances shared understanding of a deliberative topic. The main empirical findings are:

- CIG annotations over 80 segments from INSQ debates and FORA community discussions achieved moderate-to-high agreement across overall CIG, Novelty, Relevance, and Implication Scope.
- Novelty and Relevance explain most perceived CIG in ordinal regression models; Implication Scope is useful as a descriptive deliberation style dimension, but is not consistently a monotonic driver of perceived information gain.
- GPT-5 ratings under the same memory-summary context shown to annotators are comparable to a human leave-one-out baseline, supporting memory-conditioned automated CIG scoring.
- Retrieved memory summaries closely approximate the rating behavior obtained with full prior transcripts, while short-prior or no-knowledge contexts degrade agreement.
- Memory-dynamics features, especially relevance-gated memory changes, correlate more strongly with human CIG ratings than common heuristics such as length, TF-IDF, entity novelty, or token surprisal.
- Claim-level aggregation suggests a conjunctive bottleneck: utterance-level CIG is often limited by the weakest aspect among Novelty, Relevance, and Implication Scope.

See `results/findings_summary.md` for the compact numeric summary and `results/latency_tables.tex` for the latency appendix tables.

## Environment

Use `.env.example` as the starting point for local configuration.

Common variables:

- `MONGO_URI`
- `DB_NAME`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `ANNOTATOR_BLACKLIST`

## Installation

Create a Python environment and install dependencies from `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For your existing `gpu_env` Conda environment, the following is the tested update path:

```bash
conda activate gpu_env
pip install -U pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm  # optional but recommended
```

For a clean and reproducible setup (recommended), use the dedicated Conda spec:

```bash
conda env create -f environment.memcig.yml
conda activate memcig
python -m spacy download en_core_web_sm
```

Some scripts also require external services or model backends that are not bundled here, such as MongoDB, Qdrant, OpenAI, Gemini, or local OSS model infrastructure.

## Recommended Entry Points

The main reusable pipeline pieces are:

- annotation aggregation: `python scripts/aggregate_annotation.py`
- segmentation and summarisation workflow: `scripts/dialogue_segmentation.py` and `scripts/dialogue_summarisation.py`
- rating workflow: `scripts/dialogue_rating.py`
- integrated label table builder: `python scripts/build_integrated_label_release.py --help`
- latency benchmark scripts: `scripts/benchmark_memory_latency.py` and `scripts/benchmark_rating_latency.py`

For the web interface:

```bash
flask --app application run
```

## Publication Notes

- The annotation dump is anonymized, and its task context is derived from public released transcript datasets used by the project.
- INSQ transcript text can be redistributed in this release; cite the ConvoKit Intelligence Squared Debates Corpus and Zhang et al. (2016) alongside this repository.
- FORA transcript-derived context should remain outside the public archive until redistribution approval is confirmed.
- Notebook outputs may still contain exploratory analysis tables and intermediate results; see `analysis/` and `results/` for existing findings and paper tables.
