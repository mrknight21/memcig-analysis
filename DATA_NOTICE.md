# Data Notice

This repository contains code and research release artifacts for the CIG study. The repository code is licensed under Apache-2.0. Dataset components may have separate source attribution and redistribution conditions.

## INSQ / Intelligence Squared Debates

The INSQ transcripts included under `data/raw/insq/` are derived from the ConvoKit Intelligence Squared Debates Corpus:

- ConvoKit documentation: https://convokit.cornell.edu/documentation/iq2.html
- ConvoKit download name: `iq2-corpus`
- Corpus description: 108 Intelligence Squared debates held between September 2006 and September 2015.
- Original dataset paper: Zhang, Justine, et al. "Conversational Flow in Oxford-Style Debates." NAACL, 2016.

The ConvoKit documentation page identifies the corpus source and original paper but does not state a separate corpus-specific license on that page. Users should cite both this repository and the original INSQ/ConvoKit source, and should follow any terms associated with the source corpus.

The `data/raw/insq/*_meta.json` files are enriched study artifacts. They include generated summaries, semantic-memory items, segmentation metadata, GPT-5 ratings under multiple context variants, and full-context ratings. Dense embedding vectors were removed from public memory entries to keep the repository lightweight. Local copies can archive those vectors under the ignored `data/archive_local/embeddings/` sidecar, and the released scripts regenerate missing vectors when needed. These generated annotations are not human gold labels.

## FORA

FORA transcript text is not included in this public release while redistribution approval is pending. The release includes FORA-derived labels only, plus README pointers to the upstream source:

- https://github.com/schropes/fora-corpus

Users who need FORA transcript text should obtain it through the upstream FORA corpus repository and its access process.

## Human Annotation Labels

The canonical public annotation table is `data/release/integrated_labels_labels_only.csv`. It contains anonymized human-individual rows and human-consensus rows with explicit label provenance. It excludes utterance text and speaker names.
