# Post-hoc analysis scripts

Three concerns from the reviewer can be partially or fully addressed with
the files you already have on disk. Each concern has a different scaling
strategy: some analyses must be run on all 36 configurations, others only
on the one that appears in the paper. This README explains which is which.

## Directory layout expected

```
../2_dynafuse_training/results_classification_weights/
├── results_google-bert_bert-base-uncased/
│   ├── dynamic_layer_weighted_mean/
│   │   ├── avg/
│   │   │   ├── all_histories.json
│   │   │   ├── dynamic_weights_mean.json
│   │   │   ├── dynamic_weights_std.csv
│   │   │   ├── cl_results_dynamic_layer_weighted_mean-avg.csv
│   │   │   └── ...
│   │   ├── cls/
│   │   └── cls_avg/
│   ├── dynamic_layer_weighted_sum/
│   │   └── ...
│   └── dynamic_layer_cnn/
│       └── ...
├── results_microsoft_deberta-v3-base/
├── results_FacebookAI_roberta-base/
└── results_sentence-transformers_all-mpnet-base-v2/
```

## 1. Figures 9 and 12 (English version) — DO NOT loop over all 36

These figures already exist in the paper and show a specific configuration:
DeBERTa-v3-base, Weighted Average, AVG pooling. No need to regenerate them
for every combination.

Action:
```bash
python regenerate_figures.py \
    --std_csv ../2_dynafuse_training/results_classification_weights//results_microsoft_deberta-v3-base/dynamic_layer_weighted_mean/avg/dynamic_weights_std.csv \
    --mean_json ../2_dynafuse_training/results_classification_weights//results_microsoft_deberta-v3-base/dynamic_layer_weighted_mean/avg/dynamic_weights_mean.json \
    --backbone "DeBERTa-v3-base" \
    --out_dir figures_en
```

If you want the combined 4-backbone heatmap (paper's Figure 12 actual
layout, one row per backbone):

```bash
python build_combined_figure12.py \
    --root ../2_dynafuse_training/results_classification_weights/ \
    --fusion dynamic_layer_weighted_mean \
    --pooling avg \
    --out_path figures_en/fig12_combined_4_backbones.png
```

## 2. Validation--test gap (Major #7) — DO run on all 36

This addresses the reviewer's concern about overfitting on hyperparameter
selection. The argument is "the gap is consistently small/positive across
ALL configurations", which requires running on all 36 and aggregating.

Step 1: orchestrator runs the per-config analysis 36 times.

```bash
python run_all_val_test.py \
    --root ../2_dynafuse_training/results_classification_weights/ \
    --out_root all_val_test_results
```

This will populate `all_val_test_results/<model>__<fusion>__<pooling>/`
with per-config summary CSVs.

Step 2: meta-analysis builds one consolidated table and figure.

```bash
python meta_val_test_analysis.py \
    --results_root all_val_test_results \
    --out_dir meta_val_test
```

Outputs:
- `meta_val_test/meta_overall_stats.txt`: top-level number for the
  response letter ("across N configurations, mean gap is +X.XX +/- Y.YY,
  positive in Z% of cases").
- `meta_val_test/meta_table.tex`: appendix table grouped by backbone and
  fusion.
- `meta_val_test/meta_overall.png`: bar chart per (backbone, fusion).

## 3. Weight evolution (Major #5) — DO run on the 4 backbones

The narrative is "convergence behavior is stable across backbones". One
panel per backbone, keeping fusion and training pooling fixed at the
recommended values (Weighted Average + AVG), is enough.

```bash
python run_weight_evolution_subset.py \
    --root ../2_dynafuse_training/results_classification_weights/ \
    --fusion dynamic_layer_weighted_mean \
    --pooling avg \
    --out_root weight_evolution_per_backbone
```

Produces `weight_evolution_per_backbone/<model>/{convergence_grid.png,
stabilization.png, final_stability.png}` for each of the 4 backbones.

## Summary table

| Analysis | What | Loop over 36? | Output |
|----------|------|---------------|--------|
| Figures 9, 12 | Replace Portuguese figs with English | No (1 config) | 2 figures |
| Combined Fig 12 | Heatmap of 4 backbones | No (4 configs, fixed fusion/pool) | 1 figure |
| Val-test gap | Anti-overfit argument | Yes, all 36 | 1 table + 1 figure |
| Weight evolution | Cross-backbone convergence | Partial (4 configs) | 4 sets of plots |

## Time estimates

All scripts run on CPU in under 5 minutes total for the entire 36-config
sweep. No GPU needed — these are all post-hoc analyses on already-saved
artifacts.
