# Antibody Developability Prediction

Predict antibody developability from amino acid sequence using ESM-2 protein language model embeddings.

## Overview

This project classifies antibodies as therapeutically viable based on their VH sequences. It uses pretrained [ESM-2](https://github.com/facebookresearch/esm) embeddings as input to a lightweight classifier (MLP or linear probe), evaluated with stratified k-fold cross-validation.

**Data sources:**
- **Positive class:** Therapeutic antibodies from [TheraSAbDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/) (approved/phase-III)
- **Negative class:** General human antibodies from [OAS](https://opig.stats.ox.ac.uk/webapps/oas/) (CC-BY-4.0)

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Max-Tost/Antibody-Developability-Prediction.git
cd Antibody-Developability-Prediction
uv sync
```

## Pipeline

The project runs in three sequential steps:

### 1. Prepare data

Downloads antibody sequences from TheraSAbDab and OAS, filters and cleans them, and creates stratified train/val/test splits.

```bash
uv run python scripts/prepare_data.py
```

Output: `data/processed/{train,val,test}.csv`

### 2. Extract ESM-2 embeddings

Runs each sequence through ESM-2 (650M parameters by default) and caches the mean-pooled embeddings to disk.

```bash
uv run python scripts/extract_embeddings.py
```

Override the model size via Hydra:
```bash
uv run python scripts/extract_embeddings.py embedder.model_name=facebook/esm2_t12_35M_UR50D
```

Output: `embeddings/{train,val,test}/*.pt`

### 3. Train classifier

Trains an MLP (or linear probe) on the cached embeddings with 5-fold cross-validation. Logs metrics to Weights & Biases.

```bash
uv run python scripts/train.py
```

Switch to linear baseline:
```bash
uv run python scripts/train.py model.type=linear
```

Override any config parameter from the CLI:
```bash
uv run python scripts/train.py training.learning_rate=0.0001 model.hidden_dim=512
```

## Configuration

All settings live in a single Hydra config: [`configs/config.yaml`](configs/config.yaml).

| Section | Key parameters |
|---------|---------------|
| `model` | `type` (mlp/linear), `hidden_dim`, `dropout` |
| `embedder` | `model_name` (ESM-2 variant) |
| `training` | `max_epochs`, `learning_rate`, `n_folds`, `patience` |
| `wandb` | `project` name |

## Project Structure

```
├── configs/config.yaml          # Hydra configuration
├── src/
│   ├── data/
│   │   ├── prepare.py           # Data download and preprocessing
│   │   └── dataset.py           # PyTorch Dataset
│   ├── models/
│   │   ├── embedder.py          # ESM-2 embedding extraction
│   │   └── classifier.py       # MLP and linear classifier heads
│   └── training/
│       └── trainer.py           # K-fold CV training loop with W&B
├── scripts/
│   ├── prepare_data.py          # Step 1: prepare data
│   ├── extract_embeddings.py    # Step 2: extract embeddings
│   └── train.py                 # Step 3: train classifier
```
