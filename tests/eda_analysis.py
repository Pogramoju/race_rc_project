import sys, os, re, string, argparse, textwrap
from collections import Counter

# ── Ensure project root is importable ────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless — works even without a display
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import (
    load_all_splits, clean_text, tokenize, STOPWORDS
)

# ── Config ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'eda_outputs')
PALETTE = ['#4C78A8', '#F58518', '#54A24B', '#E45756', '#72B7B2',
           '#EECA3B', '#B279A2', '#FF9DA6']

sns.set_theme(style='whitegrid', palette=PALETTE, font_scale=1.05)
plt.rcParams.update({
    'figure.dpi': 130,
    'savefig.bbox': 'tight',
    'axes.titleweight': 'bold',
})



# HELPER UTILITIES

def _save(fig, filename):
    """Save a figure to OUTPUT_DIR and close it."""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'    Saved → {filename}')


def classify_question_type(question):
    """Classify a question by its leading wh-word or fill-in-the-blank."""
    q = str(question).lower().strip()
    if '_' in q or 'blank' in q:
        return 'Fill-in-blank'
    for wh in ['who', 'what', 'where', 'when', 'why', 'how', 'which']:
        if q.startswith(wh):
            return wh.capitalize()
    return 'Other'


def word_count(text):
    """Simple whitespace-based word count."""
    return len(str(text).split())


def char_count(text):
    """Character count."""
    return len(str(text))


def unique_word_count(text):
    """Number of unique words (lowercased)."""
    return len(set(str(text).lower().split()))


def top_ngrams(corpus, n=1, top_k=20):
    """Return the top-k n-grams from a corpus (list of strings)."""
    counter = Counter()
    for doc in corpus:
        tokens = tokenize(doc)
        if n == 1:
            counter.update(tokens)
        else:
            grams = [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
            counter.update(grams)
    return counter.most_common(top_k)



# EDA FUNCTIONS

def eda_split_sizes(train_df, dev_df, test_df):
    """01 — Bar chart of dataset split sizes."""
    fig, ax = plt.subplots(figsize=(7, 4))
    names = ['Train', 'Dev', 'Test']
    sizes = [len(train_df), len(dev_df), len(test_df)]
    bars = ax.bar(names, sizes, color=PALETTE[:3], edgecolor='white', width=0.55)
    for bar, s in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sizes)*0.01,
                f'{s:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.set_title('Dataset Split Sizes')
    ax.set_ylabel('Number of Samples')
    ax.set_ylim(0, max(sizes) * 1.12)
    _save(fig, '01_split_sizes.png')


def eda_answer_distribution(train_df, dev_df, test_df):
    """02 — Answer label distribution (A/B/C/D) per split."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (name, df) in zip(axes, [('Train', train_df), ('Dev', dev_df),
                                      ('Test', test_df)]):
        counts = df['answer'].astype(str).str.strip().str.upper().value_counts()
        counts = counts.reindex(['A', 'B', 'C', 'D'], fill_value=0)
        ax.bar(counts.index, counts.values, color=PALETTE[:4], edgecolor='white')
        for i, v in enumerate(counts.values):
            ax.text(i, v + max(counts.values)*0.01, str(v),
                    ha='center', va='bottom', fontsize=9)
        ax.set_title(f'{name} — Answer Balance')
        ax.set_xlabel('Correct Option')
    axes[0].set_ylabel('Count')
    fig.suptitle('Answer Label Distribution Across Splits',
                 fontsize=13, y=1.03, fontweight='bold')
    plt.tight_layout()
    _save(fig, '02_answer_distribution.png')


def _add_length_cols(df):
    """Add word-length columns if not already present."""
    if 'article_wc' not in df.columns:
        df['article_wc'] = df['article'].fillna('').apply(word_count)
    if 'question_wc' not in df.columns:
        df['question_wc'] = df['question'].fillna('').apply(word_count)
    if 'answer_text' in df.columns:
        if 'answer_wc' not in df.columns:
            df['answer_wc'] = df['answer_text'].fillna('').apply(word_count)
    if 'article_cc' not in df.columns:
        df['article_cc'] = df['article'].fillna('').apply(char_count)
    if 'article_unique' not in df.columns:
        df['article_unique'] = df['article'].fillna('').apply(unique_word_count)


def eda_text_lengths(train_df):
    """03 — Histograms of article, question, and answer word counts."""
    _add_length_cols(train_df)
    cols = [('article_wc', 'Article Words', PALETTE[0]),
            ('question_wc', 'Question Words', PALETTE[1])]
    if 'answer_wc' in train_df.columns:
        cols.append(('answer_wc', 'Answer Words', PALETTE[2]))

    fig, axes = plt.subplots(1, len(cols), figsize=(5*len(cols), 4))
    if len(cols) == 1:
        axes = [axes]
    for ax, (col, label, color) in zip(axes, cols):
        clip_val = train_df[col].quantile(0.98)
        data = train_df[col].clip(upper=clip_val)
        ax.hist(data, bins=50, color=color, edgecolor='white', alpha=0.85)
        med = train_df[col].median()
        ax.axvline(med, color='#E45756', linestyle='--', lw=1.5,
                   label=f'Median = {med:.0f}')
        ax.set_title(label)
        ax.set_xlabel('Word Count')
        ax.legend(fontsize=9)
    axes[0].set_ylabel('Frequency')
    fig.suptitle('Text Length Distributions (Train)',
                 fontsize=13, y=1.03, fontweight='bold')
    plt.tight_layout()
    _save(fig, '03_text_length_distributions.png')


def eda_length_boxplots(train_df, dev_df, test_df):
    """04 — Side-by-side box plots for article lengths across splits."""
    for df in [train_df, dev_df, test_df]:
        _add_length_cols(df)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, title in zip(axes,
                               ['article_wc', 'question_wc', 'answer_wc'],
                               ['Article', 'Question', 'Answer']):
        data_parts = []
        labels_parts = []
        for name, df in [('Train', train_df), ('Dev', dev_df), ('Test', test_df)]:
            if col in df.columns:
                data_parts.append(df[col].dropna().values)
                labels_parts.append(name)
        if not data_parts:
            ax.set_visible(False)
            continue
        bp = ax.boxplot(data_parts, labels=labels_parts, patch_artist=True,
                        showfliers=False, widths=0.5)
        for patch, color in zip(bp['boxes'], PALETTE[:len(data_parts)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(f'{title} Word Count')
        ax.set_ylabel('Words')
    fig.suptitle('Text Length Comparison Across Splits',
                 fontsize=13, y=1.03, fontweight='bold')
    plt.tight_layout()
    _save(fig, '04_text_length_boxplots.png')


def eda_option_lengths(train_df):
    """05 — Compare word-length distributions of the 4 options."""
    option_cols = [c for c in train_df.columns
                   if c.lower() in ('a', 'b', 'c', 'd')]
    if len(option_cols) < 4:
        option_cols = [c for c in train_df.columns
                       if c.lower().startswith('option')][:4]
    if not option_cols:
        print('    ⚠️ Skipping option length plot — option columns not found')
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, col in enumerate(option_cols):
        lengths = train_df[col].fillna('').apply(word_count)
        ax.hist(lengths, bins=40, alpha=0.55, color=PALETTE[i],
                label=f'Option {col.upper()}', edgecolor='white')
    ax.set_title('Option Word-Length Distributions (Train)')
    ax.set_xlabel('Word Count'); ax.set_ylabel('Frequency')
    ax.legend(fontsize=9)
    _save(fig, '05_option_length_comparison.png')


def eda_question_types(train_df):
    """06 — Pie/bar chart of question types (who/what/where/…/fill-blank)."""
    train_df['q_type'] = train_df['question'].fillna('').apply(
        classify_question_type)
    counts = train_df['q_type'].value_counts()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Bar chart
    ax1.barh(counts.index[::-1], counts.values[::-1],
             color=PALETTE[:len(counts)], edgecolor='white')
    for i, v in enumerate(counts.values[::-1]):
        ax1.text(v + max(counts.values)*0.01, i, f'{v:,}',
                 va='center', fontsize=9)
    ax1.set_title('Question Type Counts')
    ax1.set_xlabel('Count')

    # Pie chart
    ax2.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
            colors=PALETTE[:len(counts)], startangle=140,
            textprops={'fontsize': 9})
    ax2.set_title('Question Type Proportions')

    fig.suptitle('Question Type Distribution (Train)',
                 fontsize=13, y=1.02, fontweight='bold')
    plt.tight_layout()
    _save(fig, '06_question_type_distribution.png')


def eda_top_unigrams(train_df):
    """07 — Top 20 unigrams in articles (stopwords removed)."""
    corpus = train_df['article'].fillna('').tolist()
    top = top_ngrams(corpus, n=1, top_k=20)
    words, freqs = zip(*top) if top else ([], [])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(list(words)[::-1], list(freqs)[::-1],
            color=PALETTE[0], edgecolor='white')
    ax.set_title('Top 20 Unigrams in Articles (stopwords removed)')
    ax.set_xlabel('Frequency')
    plt.tight_layout()
    _save(fig, '07_top_unigrams.png')


def eda_top_bigrams(train_df):
    """08 — Top 20 bigrams in articles."""
    corpus = train_df['article'].fillna('').tolist()
    top = top_ngrams(corpus, n=2, top_k=20)
    words, freqs = zip(*top) if top else ([], [])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(list(words)[::-1], list(freqs)[::-1],
            color=PALETTE[1], edgecolor='white')
    ax.set_title('Top 20 Bigrams in Articles')
    ax.set_xlabel('Frequency')
    plt.tight_layout()
    _save(fig, '08_top_bigrams.png')


def eda_overlap_distribution(train_df):
    """09 — Distribution of keyword overlap between question+answer and article."""
    overlaps = []
    for _, row in train_df.iterrows():
        q_tok = set(tokenize(str(row.get('question', ''))))
        a_tok = set(tokenize(str(row.get('answer_text', ''))))
        art_tok = set(tokenize(str(row.get('article', ''))))
        combined = q_tok | a_tok
        overlap = len(combined & art_tok) / max(len(combined), 1)
        overlaps.append(overlap)
    train_df['qa_article_overlap'] = overlaps

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(overlaps, bins=50, color=PALETTE[4], edgecolor='white', alpha=0.85)
    med = np.median(overlaps)
    ax.axvline(med, color='#E45756', ls='--', lw=1.5,
               label=f'Median = {med:.3f}')
    ax.set_title('Question+Answer ↔ Article Keyword Overlap (Train)')
    ax.set_xlabel('Overlap Ratio'); ax.set_ylabel('Frequency')
    ax.legend(fontsize=9)
    _save(fig, '09_article_question_overlap.png')


def eda_correlation_heatmap(train_df):
    """10 — Correlation heatmap of numeric features."""
    _add_length_cols(train_df)
    num_cols = [c for c in ['article_wc', 'question_wc', 'answer_wc',
                            'article_cc', 'article_unique',
                            'qa_article_overlap']
                if c in train_df.columns]
    if len(num_cols) < 2:
        print('    ⚠️ Not enough numeric columns for correlation heatmap')
        return

    corr = train_df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                ax=ax, square=True, linewidths=0.5,
                xticklabels=[c.replace('_', ' ').title() for c in num_cols],
                yticklabels=[c.replace('_', ' ').title() for c in num_cols])
    ax.set_title('Feature Correlation Heatmap (Train)', fontweight='bold')
    plt.tight_layout()
    _save(fig, '10_correlation_heatmap.png')


def generate_text_report(train_df, dev_df, test_df):
    """Generate a plain-text summary report."""
    _add_length_cols(train_df)

    lines = [
        '=' * 70,
        'RACE Dataset — EDA Summary Report',
        '=' * 70,
        '',
        '── Split Sizes ─────────────────────────────────────────',
        f'  Train: {len(train_df):>8,} rows',
        f'  Dev:   {len(dev_df):>8,} rows',
        f'  Test:  {len(test_df):>8,} rows',
        f'  Total: {len(train_df)+len(dev_df)+len(test_df):>8,} rows',
        '',
        '── Columns ─────────────────────────────────────────────',
        f'  {train_df.columns.tolist()}',
        '',
        '── Missing Values (Train) ──────────────────────────────',
    ]
    for col in train_df.columns:
        miss = train_df[col].isna().sum()
        if miss > 0:
            lines.append(f'  {col:25s}: {miss:,} ({miss/len(train_df)*100:.1f}%)')
    if not any(train_df[col].isna().sum() > 0 for col in train_df.columns):
        lines.append('  No missing values ✓')

    lines += [
        '',
        '── Answer Distribution (Train) ─────────────────────────',
    ]
    for label, count in (train_df['answer'].astype(str).str.strip()
                         .str.upper().value_counts().sort_index().items()):
        lines.append(f'  {label}: {count:>7,} ({count/len(train_df)*100:.1f}%)')

    lines += [
        '',
        '── Text Length Statistics (Train — word counts) ────────',
        f'  Article:  min={train_df["article_wc"].min():>4}, '
        f'median={train_df["article_wc"].median():>6.0f}, '
        f'mean={train_df["article_wc"].mean():>6.1f}, '
        f'max={train_df["article_wc"].max():>5}',
        f'  Question: min={train_df["question_wc"].min():>4}, '
        f'median={train_df["question_wc"].median():>6.0f}, '
        f'mean={train_df["question_wc"].mean():>6.1f}, '
        f'max={train_df["question_wc"].max():>5}',
    ]
    if 'answer_wc' in train_df.columns:
        lines.append(
            f'  Answer:   min={train_df["answer_wc"].min():>4}, '
            f'median={train_df["answer_wc"].median():>6.0f}, '
            f'mean={train_df["answer_wc"].mean():>6.1f}, '
            f'max={train_df["answer_wc"].max():>5}')

    if 'q_type' in train_df.columns:
        lines += [
            '',
            '── Question Types (Train) ──────────────────────────────',
        ]
        for qt, cnt in train_df['q_type'].value_counts().items():
            lines.append(f'  {qt:15s}: {cnt:>7,} ({cnt/len(train_df)*100:.1f}%)')

    if 'qa_article_overlap' in train_df.columns:
        ov = train_df['qa_article_overlap']
        lines += [
            '',
            '── Q+A ↔ Article Overlap (Train) ───────────────────────',
            f'  min={ov.min():.4f}, median={ov.median():.4f}, '
            f'mean={ov.mean():.4f}, max={ov.max():.4f}',
        ]

    lines += ['', '=' * 70, '']

    report_path = os.path.join(OUTPUT_DIR, 'eda_summary_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'    Saved → eda_summary_report.txt')
    print()
    print('\n'.join(lines))



# MAIN

def main():
    parser = argparse.ArgumentParser(description='RACE Dataset — EDA')
    parser.add_argument('--quick', action='store_true',
                        help='Run on a small sample (500 rows/split) for speed')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

    print('=' * 60)
    print('RACE Dataset — Exploratory Data Analysis')
    print('=' * 60)

    # ── Load data ────────────────────────────────────────────────────────
    print('\n[1/12] Loading data...')
    train_df, dev_df, test_df = load_all_splits(DATA_DIR)

    if args.quick:
        n = 500
        print(f'  --quick mode: sampling {n} rows per split')
        train_df = train_df.sample(n=min(n, len(train_df)), random_state=42)
        dev_df = dev_df.sample(n=min(n, len(dev_df)), random_state=42)
        test_df = test_df.sample(n=min(n, len(test_df)), random_state=42)

    # Resolve answer_text if missing
    letter_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    option_cols = [c for c in train_df.columns if c.lower() in ('a', 'b', 'c', 'd')]
    if not option_cols:
        option_cols = [c for c in train_df.columns if c.lower().startswith('option')]
    for df in [train_df, dev_df, test_df]:
        if 'answer_text' not in df.columns and option_cols:
            def resolve(row):
                idx = letter_map.get(str(row.get('answer', '')).strip().upper(), -1)
                if 0 <= idx < len(option_cols):
                    return str(row[option_cols[idx]])
                return str(row.get('answer', ''))
            df['answer_text'] = df.apply(resolve, axis=1)

    # ── Run EDA steps ────────────────────────────────────────────────────
    print('\n[2/12] Split sizes...')
    eda_split_sizes(train_df, dev_df, test_df)

    print('[3/12] Answer distribution...')
    eda_answer_distribution(train_df, dev_df, test_df)

    print('[4/12] Text length histograms...')
    eda_text_lengths(train_df)

    print('[5/12] Text length box plots...')
    eda_length_boxplots(train_df, dev_df, test_df)

    print('[6/12] Option length comparison...')
    eda_option_lengths(train_df)

    print('[7/12] Question type distribution...')
    eda_question_types(train_df)

    print('[8/12] Top unigrams...')
    eda_top_unigrams(train_df)

    print('[9/12] Top bigrams...')
    eda_top_bigrams(train_df)

    print('[10/12] Q+A ↔ Article keyword overlap...')
    eda_overlap_distribution(train_df)

    print('[11/12] Correlation heatmap...')
    eda_correlation_heatmap(train_df)

    print('[12/12] Generating text report...')
    generate_text_report(train_df, dev_df, test_df)

    print(f'\n✅ All EDA outputs saved to: {OUTPUT_DIR}/')
    print('=' * 60)


if __name__ == '__main__':
    main()
