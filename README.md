# RACE Reading Comprehension & Quiz Generation System

> **AL2002 — Artificial Intelligence Lab Project**
> BS(CS) Spring 2026 • FAST NUCES Islamabad

## Overview

An AI-powered Reading Comprehension and Quiz Generation System built on the
RACE Dataset. The system integrates two ML pipelines:

- **Model A** — Question & Answer Generator / Verifier (RF, SVM, LR, K-Means,
  Label Propagation, Soft Ensemble)
- **Model B** — Distractor & Hint Generator (OHE cosine similarity, extractive
  hint ranking with LR scoring)
- **UI Layer** — Streamlit 4-screen application wiring both models together

## Project Structure

```
race_rc_project/
├── data/raw/              # RACE CSV files (train.csv, dev.csv, test.csv)
├── models/                # Saved model checkpoints (.pkl)
├── src/
│   ├── preprocessing.py   # Dataset loading & feature engineering
│   ├── model_a_train.py   # Model A training (supervised/unsupervised/ensemble)
│   ├── model_b_train.py   # Model B training (distractor/hint generation)
│   ├── inference.py       # Unified inference API
│   └── evaluate.py        # Metric computation & dashboard plotting
├── ui/app.py              # Streamlit UI (4 screens)
├── notebooks/             # Jupyter notebooks (EDA, experiments)
├── tests/                 # Unit tests
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Data

Place RACE dataset CSVs in `data/raw/`:
- `train.csv`, `dev.csv`, `test.csv`

## Training

Run the full pipeline from the Jupyter notebook:
```bash
jupyter notebook notebooks/experiments.ipynb
```

Or use the modular scripts:
```python
from src.preprocessing import load_all_splits, apply_cleaning, ...
from src.model_a_train import train_random_forest, train_svm, ...
from src.model_b_train import evaluate_distractors, evaluate_hints
```

## Running the UI

```bash
cd race_rc_project
streamlit run ui/app.py
```

## Evaluation Metrics

| Component | Metrics |
|-----------|---------|
| Model A | Accuracy, Macro-F1, Precision, Recall, Exact Match, R² |
| Model B | Distractor P/R/F1, Hint Accuracy/F1/R² |
| Clustering | Silhouette Score |

## License

Academic use only — FAST NUCES AL2002 Lab Project.
