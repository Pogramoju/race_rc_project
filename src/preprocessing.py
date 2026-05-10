"""
preprocessing.py — Dataset loading & feature engineering
Handles RACE CSV loading, text cleaning, tokenization,
OHE/TF-IDF vectorization, and handcrafted feature extraction.
"""

import os
import re
import string
import numpy as np
import pandas as pd
from collections import Counter
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ── Stopwords ────────────────────────────────────────────────────────────────
STOPWORDS = {
    'a','an','the','is','it','in','of','to','and','or','but','was','are',
    'were','be','been','being','have','has','had','do','does','did','will',
    'would','could','should','may','might','shall','can','not','no','nor',
    'so','yet','both','either','neither','for','on','at','by','with','from',
    'up','about','into','through','during','this','that','these','those',
    'he','she','they','we','you','i','me','him','her','us','them','my',
    'your','his','its','our','their','what','which','who','when','where',
    'why','how','all','each','every','few','more','most','other','some',
    'such','than','then','too','very','just','also','as','if','s','t'
}

MAX_FEATURES = 10_000


# ── Text Cleaning ────────────────────────────────────────────────────────────
def clean_text(text, lowercase=True, remove_punct=True, remove_stopwords=False):
    """Clean and normalise a text string."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ''
    text = re.sub(r'\s+', ' ', text).strip()
    if lowercase:
        text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    if remove_punct:
        text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\b\d+\b', '', text)
    if remove_stopwords:
        tokens = text.split()
        tokens = [t for t in tokens if t not in STOPWORDS]
        text = ' '.join(tokens)
    return re.sub(r'\s+', ' ', text).strip()


def tokenize(text):
    """Tokenize text after cleaning and removing stopwords."""
    return clean_text(text, remove_stopwords=True).split()


# ── Data Loading ─────────────────────────────────────────────────────────────
def setup_environment():
    """Checks if running in Colab or a local IDE like PyCharm."""
    try:
        import google.colab
        is_colab = True
    except ImportError:
        is_colab = False

    if is_colab:
        print("Running in Google Colab. Mounting Drive...")
        from google.colab import drive
        drive.mount('content/drive', force_remount=True)
        return 'content/drive/MyDrive/AIProject/data'
    else:
        print("Running in Local Environment (PyCharm/Jupyter).")
        local_path = os.path.join(os.getcwd(), 'data')
        print(f"Expected local data path: {local_path}")
        return local_path


def load_race_csv(path):
    """Load a RACE CSV file and resolve the answer text from option columns."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    option_cols = [c for c in df.columns if c.startswith('option')]
    letter_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    def resolve(row):
        idx = letter_map.get(str(row.get('answer', '')).strip().upper(), -1)
        if idx >= 0 and len(option_cols) > idx:
            return str(row[option_cols[idx]])
        # Fallback: try lowercase a/b/c/d columns
        lbl = str(row.get('answer', '')).strip().upper()
        for col in [lbl, lbl.lower()]:
            if col in row.index and str(row[col]).strip():
                return str(row[col])
        return str(row.get('answer', ''))

    df['answer_text'] = df.apply(resolve, axis=1)
    print(f'Loaded  {os.path.basename(path):15s} → {len(df):,} rows')
    return df


def load_all_splits(data_dir):
    """Load train, dev, and test splits from the data directory."""
    # Try raw/ subdirectory first, then root
    for subdir in ['raw', '']:
        base = os.path.join(data_dir, subdir) if subdir else data_dir
        train_path = os.path.join(base, 'train.csv')
        dev_path = os.path.join(base, 'dev.csv')
        test_path = os.path.join(base, 'test.csv')
        if os.path.exists(train_path):
            train_df = load_race_csv(train_path)
            dev_df = load_race_csv(dev_path)
            test_df = load_race_csv(test_path)
            print(f'\nTotal rows: {len(train_df)+len(dev_df)+len(test_df):,}')
            return train_df, dev_df, test_df
    raise FileNotFoundError(f"CSV files not found in {data_dir} or {data_dir}/raw/")


# ── Feature Engineering ──────────────────────────────────────────────────────
def apply_cleaning(dfs):
    """Apply text cleaning to all dataframes (in-place)."""
    for df in dfs:
        df['article_clean'] = df['article'].fillna('').apply(clean_text)
        df['question_clean'] = df['question'].fillna('').apply(clean_text)
        df['answer_clean'] = df['answer_text'].fillna('').apply(clean_text)
    print('Text cleaning applied to all splits.')


def combined_text(row):
    """Combine cleaned article, question, and answer for vectorization."""
    return ' '.join([
        str(row.get('article_clean', '')),
        str(row.get('question_clean', '')),
        str(row.get('answer_clean', ''))
    ])


def build_corpus(df):
    """Build a text corpus from a dataframe for vectorization."""
    return [combined_text(r) for _, r in df.iterrows()]


def fit_vectorizers(train_corpus, dev_corpus, test_corpus,
                    max_features=MAX_FEATURES):
    """Fit OHE and TF-IDF vectorizers on training data, transform all splits."""
    # One-Hot Encoding (PRIMARY)
    print('Fitting One-Hot Encoder (binary CountVectorizer)...')
    ohe_vec = CountVectorizer(max_features=max_features, binary=True,
                               min_df=2, token_pattern=r'(?u)\b\w+\b')
    X_train_ohe = ohe_vec.fit_transform(train_corpus)
    X_dev_ohe = ohe_vec.transform(dev_corpus)
    X_test_ohe = ohe_vec.transform(test_corpus)
    print(f'OHE matrix — Train: {X_train_ohe.shape}, '
          f'Dev: {X_dev_ohe.shape}, Test: {X_test_ohe.shape}')

    # TF-IDF (SECONDARY / OPTIONAL)
    print('\nFitting TF-IDF (secondary)...')
    tfidf_vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                                 sublinear_tf=True,
                                 token_pattern=r'(?u)\b\w+\b')
    X_train_tfidf = tfidf_vec.fit_transform(train_corpus)
    X_dev_tfidf = tfidf_vec.transform(dev_corpus)
    X_test_tfidf = tfidf_vec.transform(test_corpus)
    print(f'TF-IDF matrix — Train: {X_train_tfidf.shape}')

    return {
        'ohe_vec': ohe_vec,
        'tfidf_vec': tfidf_vec,
        'X_train_ohe': X_train_ohe, 'X_dev_ohe': X_dev_ohe,
        'X_test_ohe': X_test_ohe,
        'X_train_tfidf': X_train_tfidf, 'X_dev_tfidf': X_dev_tfidf,
        'X_test_tfidf': X_test_tfidf,
    }


def keyword_overlap(q, a, article):
    """Compute keyword overlap between question+answer and article."""
    q_tok = set(tokenize(q))
    a_tok = set(tokenize(a))
    art_tok = set(tokenize(article))
    combined = q_tok | a_tok
    return len(combined & art_tok) / max(len(combined), 1)


def handcrafted_features(df):
    """Build handcrafted lexical features for each row."""
    wh = ['what', 'who', 'where', 'when', 'why', 'how', 'which']
    rows = []
    for _, row in df.iterrows():
        q_lower = str(row.get('question', '')).lower()
        feat = [
            len(str(row.get('question', '')).split()),
            len(str(row.get('article', '')).split()),
            len(str(row.get('answer_text', '')).split()),
            keyword_overlap(
                str(row.get('question', '')),
                str(row.get('answer_text', '')),
                str(row.get('article', ''))
            ),
        ] + [int(q_lower.startswith(w)) for w in wh]
        rows.append(feat)
    return np.array(rows, dtype=np.float32)


def build_combined_features(X_ohe, H):
    """Combine OHE features with handcrafted features."""
    return hstack([X_ohe, csr_matrix(H)])


# ── Verification Dataset Expansion ──────────────────────────────────────────
def expand_to_options(df):
    """Expand each question → 4 option rows; label correct option as 1."""
    option_cols = ['a', 'b', 'c', 'd']
    letter_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    rows = []
    for _, row in df.iterrows():
        correct_idx = letter_map.get(
            str(row.get('answer', '')).strip().upper(), -1)
        for i, col_name in enumerate(option_cols[:4]):
            r = row.to_dict()
            r['option_text'] = str(row[col_name])
            r['is_correct'] = int(i == correct_idx)
            rows.append(r)
    return pd.DataFrame(rows).reset_index(drop=True)


def make_verify_corpus(df):
    """Build verification corpus from expanded option dataframe."""
    texts = []
    for _, r in df.iterrows():
        texts.append(' '.join([
            clean_text(str(r.get('article', ''))),
            clean_text(str(r.get('question', ''))),
            clean_text(str(r.get('option_text', '')))
        ]))
    return texts


def build_verification_data(train_df, dev_df, verify_sample=20_000,
                             dev_sample=5_000, max_features=8000):
    """Build the verification dataset with OHE features."""
    train_sample = train_df.sample(
        n=min(verify_sample, len(train_df)), random_state=42)
    dev_sample_df = dev_df.sample(
        n=min(dev_sample, len(dev_df)), random_state=42)

    exp_train = expand_to_options(train_sample)
    exp_dev = expand_to_options(dev_sample_df)

    print(f'Expanded Train: {len(exp_train):,}  |  '
          f'Expanded Dev: {len(exp_dev):,}')
    print(f'Class balance:\n{exp_train["is_correct"].value_counts()}')

    ohe_verify = CountVectorizer(max_features=max_features, binary=True,
                                  min_df=2)
    Xv_train = ohe_verify.fit_transform(make_verify_corpus(exp_train))
    Xv_dev = ohe_verify.transform(make_verify_corpus(exp_dev))
    yv_train = exp_train['is_correct'].values
    yv_dev = exp_dev['is_correct'].values

    print(f'Verify X_train: {Xv_train.shape}  '
          f'y balance: {Counter(yv_train)}')

    return {
        'ohe_verify': ohe_verify,
        'Xv_train': Xv_train, 'Xv_dev': Xv_dev,
        'yv_train': yv_train, 'yv_dev': yv_dev,
        'exp_train': exp_train, 'exp_dev': exp_dev,
    }
