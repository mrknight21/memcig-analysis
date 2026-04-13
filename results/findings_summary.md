# Findings Summary

This file summarizes the main paper-facing results produced from the CIG pipeline. It is intended as a compact guide for readers who want to understand what the released code and data support before inspecting the notebooks or manuscript.

## Study Setup

The study annotated 80 dialogue segments from two moderated deliberative settings:

- `INSQ`: ten Intelligence Squared debate sessions.
- `FORA`: ten community discussion sessions from the FORA corpus.

CIG is defined as how much an utterance advances shared understanding of the topic or goal, given prior knowledge and the preceding dialogue. It is analyzed with three aspect ratings: Novelty, Relevance, and Implication Scope.

## Annotation Reliability

Krippendorff's alpha showed moderate-to-high agreement:

| Corpus | CIG | Novelty | Relevance | Scope |
|---|---:|---:|---:|---:|
| INSQ | 0.589 | 0.506 | 0.669 | 0.597 |
| FORA | 0.567 | 0.583 | 0.566 | 0.510 |

## Aspect Validation

Ordinal regression models show that CIG is mainly explained by Novelty and Relevance.

| Corpus | Baseline AIC | Best aspect model | Best AIC | Interpretation |
|---|---:|---|---:|---|
| INSQ | 472.58 | Base + Novelty + Relevance | 416.01 | Removing Scope slightly improves fit. |
| FORA | 643.25 | Base + all three aspects | 564.06 | Scope adds only a marginal improvement over the Scope ablation. |

The main interpretation is that Scope is useful for describing whether contributions are local, case-based, generalizing, or principle-level, but it does not reliably act as a monotonic driver of perceived information gain.

## Automated CIG Prediction

GPT-5 predictions under the same memory-summary context shown to human annotators are comparable to the human leave-one-out baseline.

| Corpus | Aspect | GPT-5 MAE | Human LOO MAE |
|---|---|---:|---:|
| INSQ | CIG | 0.457 | 0.656 |
| INSQ | Novelty | 0.587 | 0.637 |
| INSQ | Relevance | 0.452 | 0.431 |
| INSQ | Scope | 0.529 | 0.562 |
| FORA | CIG | 0.520 | 0.631 |
| FORA | Novelty | 0.556 | 0.599 |
| FORA | Relevance | 0.414 | 0.446 |
| FORA | Scope | 0.479 | 0.559 |

Context ablations indicate that retrieved memory summaries preserve most of the rating-relevant information from the full prior transcript:

| Context | Overall MAE vs GPT-5 summary reference |
|---|---:|
| GPT-5 memory | 0.213 |
| GPT-5 full transcript | 0.224 |
| GPT-5 short prior | 0.286 |
| GPT-5 no knowledge | 0.333 |

## Memory Dynamics vs Heuristics

Memory-dynamics signals are the strongest correlates of human CIG ratings.

| Feature | Absolute Pearson r with human CIG |
|---|---:|
| Memory changes, relevance-gated | 0.727 |
| Memory changes, any | 0.720 |
| Extracted claim count | 0.713 |
| Memory changes, informativeness-gated | 0.712 |
| Memory changes, novelty-gated | 0.703 |
| TF-IDF sum | 0.701 |
| Length, tokens | 0.675 |
| Token surprisal sum | 0.675 |

This supports the paper's central claim that semantic-memory state changes track perceived informational progress better than many surface-level informativeness proxies.

## Claim-Level Aggregation

An exploratory aggregation analysis suggests a conjunctive bottleneck: utterance-level CIG is often limited by the weakest aspect. The best unsupervised claim-to-utterance aggregation used top-2 mean claim pooling followed by a minimum operator across aspects, reaching MAE 0.583 against human utterance-level CIG.

## Moderator Dynamics Case Study

CIG also supports downstream interaction analysis. In INSQ debates, participant CIG tends to peak immediately after moderator interventions, with confronting acts especially salient. In FORA community discussions, CIG peaks often emerge several turns later, suggesting a more gradual buildup of informativeness; FORA moderators also use more supplementing actions and no confronting acts in the analyzed sample.

## Latency Appendix

Runtime measurements for the memory-update and rating-inference operations are provided in `latency_tables.tex`. These tables standardize the sample unit as utterances and report mean, standard deviation, and sample size for OpenAI and Gemini model families.
