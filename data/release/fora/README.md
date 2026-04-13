# FORA Annotation Labels

This folder contains FORA-derived annotation labels only. It does not include FORA conversation transcripts, utterance text, or speaker names.

The upstream FORA corpus is available through:

https://github.com/schropes/fora-corpus

Please obtain transcript data through the upstream corpus repository and its access process. Until redistribution approval is confirmed, this project releases only labels and metadata that can be joined with locally obtained FORA transcripts by `conversation_id` and `utterance_id`.

## Files

- `labels_only.csv`: utterance-level FORA labels with explicit provenance.
- `labels_only.summary.json`: row counts and label-source counts for `labels_only.csv`.

## Label Provenance

- `label_source=human_individual`: anonymized annotator-level labels.
- `label_source=human_consensus`: aggregated human labels.
