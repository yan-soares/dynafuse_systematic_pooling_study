# DynaFuse: Trainable Layer Fusion for Sentence Embeddings

Companion code for the paper:

> **Rethinking Sentence Embeddings in
> Transformer Encoders: A Systematic Study and Trainable Layer Fusion
> Framework**..

This repository contains the code, scripts, and instructions to reproduce
the three blocks of the paper:

1. **Systematic study** of 36,288 layer–pooling combinations on BERT,
   RoBERTa, DeBERTaV3, and SBERT (Section 3).
2. **DynaFuse training**: trainable scalar layer-fusion weights over a
   frozen encoder, with two variants (Weighted Average and Weighted Sum)
   and two transfer profiles (NLI and LOO) (Section 4).
3. **Post-hoc analyses**: inner-CV variability characterization,
   validation–test gap meta-analysis, and weight-evolution plots
   (Appendices B and C).

## Repository layout

dynafuse/
├── README.md
├── environment.yml              # Conda environment specification
├── requirements.txt             # Pip alternative
├── data/                        # SentEval data (download instructions)
├── senteval/                    # SentEval toolkit (vendored)
├── 1_systematic_study/          # Section 3
├── 2_dynafuse_training/         # Section 4
├── 3_post_hoc_analysis/         # Appendices B and C

Each numbered subdirectory has its own README with command examples.

## Setup

### Environment

Tested on Ubuntu 24.04, Python 3.11, CUDA 12.4, single NVIDIA H200 GPU.
Training also works on smaller GPUs (RTX 8000, A100); the systematic
study (Section 3) can be run on CPU but is slow.

**Option A — Conda (recommended):**

```
conda env create -f environment.yml
conda activate dynafuse
\`\`\`

**Option B — Pip:**

```
**Verify the installation:**

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
\`\`\`

Expected output: `2.6.0+cu124 True` on a CUDA-capable machine.

### Data

The `data/` directory uses the SentEval download scripts:

```bash
cd data
bash get_transfer_data.bash
bash get_transfer_data_stsb_and_sick.bash
```

This downloads approximately 90 MB of classification data (MR, CR, SUBJ,
MPQA, SST2, TREC, MRPC) and 90 MB of similarity data (STS12–16, STSB,
SICK).

The NLI training data used by DynaFuse is provided as a separate file
(`nli_optimized_for_layer_search_50k.csv`, 7 MB). The full 1M-row
version is hosted externally; see `2_dynafuse_training/README.md`.

## Reproducing the paper

| Section | Directory | Estimated runtime (H200) |
|---------|-----------|--------------------------|
| 3 (systematic study, 36,288 configs) | `1_systematic_study/` | ~5–7 days |
| 4.5 (DynaFuse training, 4 backbones × 6 configs × 5 seeds) | `2_dynafuse_training/` | ~12 hours |
| 4.6 (DynaFuse evaluation under NLI and LOO) | `2_dynafuse_training/` | ~3 hours |
| 4.6.4 (DeBERTaV3-large and ModernBERT) | `2_dynafuse_training/` | ~4 hours |
| Appendices B, C and Figures 9, 11 | `3_post_hoc_analysis/` | ~5 minutes (CPU) |

If you only want to reproduce the headline numbers (Table 4: 90.33% on
SentEval with DynaFuse-DeBERTaV3-base), download the pretrained weights
and run:

```bash
cd 2_dynafuse_training
bash scripts_sh/evaluate_deberta.sh
```

## License

Code in this repository is released under the MIT License (see
`LICENSE`). The vendored SentEval toolkit in `senteval/` retains its
original BSD-style license from Facebook AI Research; see
`senteval/LICENSE` if present.
