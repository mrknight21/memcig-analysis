# Results

This folder contains publication-facing generated outputs.

## Main Findings

- `findings_summary.md`: compact numerical summary of the main empirical findings, including annotation reliability, aspect validation, GPT-based prediction, memory-dynamics correlations, claim-level aggregation, and moderator dynamics.

## Latency Appendix

- `latency_tables.tex`: LaTeX tables reporting memory-operation and rating-inference latency.

The latency tables aggregate measurements over three FORA segments and three INSQ segments. Memory-operation rows report per-utterance timings for claim extraction, retrieval plus matching, memory integration, and total memory update. Rating rows report seconds per utterance in memory mode.

Model settings used for the latency runs:

- ChatGPT series: `reasoning_effort=minimal`
- Gemini series: `thinking_budget=128`

Related scripts:

- `scripts/benchmark_memory_latency_debug.py`
- `scripts/benchmark_rating_latency.py`
