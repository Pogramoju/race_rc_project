# RACE Reading Comprehension & Quiz Generation System

> **AL2002 — Artificial Intelligence Lab Project**
> BS(CS) Spring 2026 • FAST NUCES Islamabad

## Overview

An AI-powered Reading Comprehension and Quiz Generation System built on the
RACE Dataset. The system integrates two ML pipelines:

- **Model A** — Answer Verifier & Question Generator (RF, SVM, LR, K-Means,
  Label Propagation, Soft Ensemble). The Random Forest classifier is the
  strongest individual model.
- **Model B** — Distractor & Hint Generator (OHE cosine similarity, extractive
  hint ranking with weighted sentence scoring)
- **UI Layer** — Streamlit 4-screen application wiring both models together

## Evaluation Approach

Since the core tasks are **text generation** (questions, answers, distractors, hints),
evaluation uses **NLG metrics** rather than traditional classification metrics:

| Metric | What it measures |
|--------|-----------------|
| **BLEU** | N-gram precision (with smoothing) |
| **ROUGE-1 / ROUGE-2 / ROUGE-L** | Recall-oriented overlap (unigram / bigram / LCS) |
| **METEOR** | Alignment with synonym and stem matching |
| **Exact Match** | Full-string correctness (answer selection only) |

## Project Structure

```
race_rc_project/
├── data/raw/                # RACE CSV files (train.csv, dev.csv, test.csv)
├── data/processed/          # Cached preprocessed features (auto-generated)
├── models/
│   ├── model_a/traditional/ # Pickled RF, SVM, LR, vectorizers
│   └── metrics.json         # All NLG evaluation metrics
├── src/
│   ├── preprocessing.py     # Dataset loading & feature engineering
│   ├── model_a_train.py     # Model A training (supervised/unsupervised/ensemble)
│   ├── model_b_train.py     # Model B (distractor/hint generation & NLG eval)
│   ├── inference.py         # Unified inference API
│   └── evaluate.py          # NLG metrics (BLEU/ROUGE/METEOR) & dashboard
├── ui/app.py                # Streamlit UI (4 screens)
├── tests/
│   ├── eda_analysis.py      # Exploratory Data Analysis (10 plots + report)
│   └── test_inference.py    # Unit tests
├── report/
│   ├── generate_report.py   # Word document report generator
│   └── RACE_Project_Report.docx  # Generated report (12 sections)
├── run_pipeline.py          # CLI pipeline runner (5 phases)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt

# NLG metric dependencies
pip install nltk rouge-score python-docx

# Download NLTK data for METEOR
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Data

Place RACE dataset CSVs in `data/raw/`:
- `train.csv`, `dev.csv`, `test.csv`

## Training & Evaluation

### Full Pipeline (recommended)

```bash
cd race_rc_project
python run_pipeline.py
```

This runs 5 phases:
1. **Preprocessing** — load CSVs, clean text, fit vectorizers, cache to `data/processed/`
2. **Model A** — train RF/SVM/LR, K-Means, Label Propagation, ensemble
3. **Model B (NLG)** — evaluate distractors & hints with BLEU/ROUGE/METEOR
4. **Test (NLG)** — evaluate answer selection & question generation on test set
5. **Save** — pickle models, write `metrics.json`

Use `--skip-preprocess` to skip Phase 1 if cached data exists.

### EDA

```bash
python tests/eda_analysis.py          # Full analysis
python tests/eda_analysis.py --quick  # Quick preview (500 rows/split)
```

Outputs 10 plots + summary report to `tests/eda_outputs/`.

## Running the UI

```bash
cd race_rc_project
streamlit run ui/app.py
```

### 4 Screens

| Screen | Name | Description |
|--------|------|-------------|
| 1 | 📝 Article Input | Load RACE passage or paste custom text, generate quiz |
| 2 | ❓ Quiz View | Answer multiple-choice question with confidence scoring |
| 3 | 💡 Hint Panel | Graduated hint reveal (3 levels) |
| 4 | 📊 Analytics | NLG metrics dashboard (BLEU/ROUGE/METEOR for all tasks) |

## Generating the Report

```bash
cd race_rc_project
python report/generate_report.py
```

Generates `report/RACE_Project_Report.docx` with 12 sections:
Abstract, Introduction, Related Work (6 citations), Dataset Analysis,
Model A, Model B, UI Description, Evaluation & Discussion,
Limitations & Future Work, **Ethical Considerations**, Conclusion, References.

The report auto-loads metrics from `models/metrics.json` when available.

## Key Design Decisions

- **Classical ML only** — scikit-learn (no deep learning / transformers)
- **NLG evaluation** — BLEU, ROUGE, METEOR (not accuracy/precision/recall)
- **RF > Ensemble** — Random Forest outperforms the soft-voting ensemble because
  weaker LR dilutes averaged probabilities
- **Modular architecture** — `src/` modules can be imported independently or run
  via `run_pipeline.py`

## License

Academic use only — FAST NUCES AL2002 Lab Project.
