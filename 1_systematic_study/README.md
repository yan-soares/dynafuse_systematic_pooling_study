# Systematic Study (Section 3)

This module evaluates 36,288 configurations combining:

- **Hidden-layer selection:** 36 layer ranges × 3 aggregations (SUM, MEAN,
  individual layers).
- **Pooling strategies:** 7 simple operators (CLS, AVG, AVG-NS, SUM, SUM-NS,
  MAX, MAX-NS) plus all pairwise (21) and triple (35) concatenations.
- **Encoders:** BERT-base, RoBERTa-base, DeBERTaV3-base, SBERT (MPNet).

The output is a CSV per (backbone, layer-range, aggregation, pooling)
combination with SentEval accuracy on the 7 classification tasks and STS
correlation on the 7 similarity tasks.

## Quick start

```bash
# Single configuration example
python main_experiments.py \
    --task_type classification \
    --models deberta-base \
    --initial_layer 7 --final_layer 11 \
    --poolings AVG \
    --agg_layers SUM \
    --save_dir example_run
```

## Full reproduction

The scripts in `scripts_sh/` reproduce the 36k grid in 8 sessions per
backbone (`cl_1_sbert_all_1.sh` to `cl_8_roberta_all_2.sh` for
classification; `si_*.sh` for similarity). Each session takes 12–24
hours on an H200.

## Output

Results are written to specific --save_dir.
After all sessions finish, use `evaluate.py` and `join_tables.py` to
consolidate into `tables_processed/`.