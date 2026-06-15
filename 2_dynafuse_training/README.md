# DynaFuse Training (Section 4)

This module implements the trainable layer-fusion framework. Two
fusion variants:

- **Weighted Average:** softmax over per-layer scalars (convex
  combination of layers, equivalent to ELMo-style scalar mix without
  the global gamma).
- **Weighted Sum:** free real-valued per-layer scalars, initialized to
  1/L (conic combination, drops the unit-sum constraint).

And two transfer profiles:

- **NLI:** weights trained on a fixed NLI corpus, evaluated on each
  SentEval task without retraining.
- **LOO:** for each held-out task, weights are trained on the union of
  the other six SentEval tasks.

## Training

```bash
# Train DynaFuse on a single backbone, one configuration, all seeds
python train_dynafuse.py \
    --model_name microsoft/deberta-v3-base \
    --pooling_type avg \
    --type_fusion dynamic_layer \
    --mode_fusion weighted_mean \
    --base_path results_classification_weights
```

Arguments:

| Argument | Choices | Description |
|----------|---------|-------------|
| `--model_name` | HuggingFace model ID | Backbone (frozen) |
| `--pooling_type` | `avg`, `cls`, `cls_avg` | Training pooling |
| `--type_fusion` | `dynamic_layer` | Fusion family |
| `--mode_fusion` | `weighted_mean`, `weighted_sum` | Variant |
| `--base_path` | path | Output directory root |

Output: per (backbone, fusion, pooling) directory:

- `dynamic_weights_mean.json`: trained fusion weights, one per task
- `dynamic_weights_std.csv`: cross-seed standard deviation
- `all_histories.json`: per-epoch training metrics
- `final_accuracies.csv`: classification accuracy on each task
- `heatmap_*.png`: weight visualizations

Full grid: see `scripts/run_*_all.sh` (4 backbones × 2 fusions ×
3 poolings = 24 training runs).

## Evaluation

```bash
# Evaluate trained weights under all evaluation poolings
python evaluate_dynafuse.py \
    --task_type classification \
    --models deberta-base \
    --poolings all \
    --agg_layers NLI,LOO-SENTEVAL \
    --dynamic_weights_path \
    <PATH-TO>/dynamic_weights_mean.json
```

Full evaluation grid: see `scripts_sh/evaluate_*.sh`.