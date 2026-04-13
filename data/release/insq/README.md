# INSQ Annotation Labels

This folder contains INSQ-derived annotation labels. The default `labels_only.csv` file excludes utterance text and speaker names so it has the same structure as the FORA labels-only release.

## Files

- `labels_only.csv`: utterance-level INSQ labels with explicit provenance.
- `labels_only.summary.json`: row counts and label-source counts for `labels_only.csv`.

## Label Provenance

- `label_source=human_individual`: anonymized annotator-level labels.
- `label_source=human_consensus`: aggregated human labels.
