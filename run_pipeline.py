"""
run_pipeline.py — Full RACE ML Pipeline (standalone, no notebook required)
Replicates the entire notebook workflow using the modular src/ modules.

Usage:
    cd race_rc_project
    python run_pipeline.py
"""

import os, sys, time, json, pickle, warnings, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import save_npz, load_npz

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
pd.set_option('display.max_colwidth', 80)

# ── Ensure src/ is importable ────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import (
    load_all_splits, apply_cleaning, build_corpus, fit_vectorizers,
    handcrafted_features, build_combined_features,
    build_verification_data, expand_to_options, make_verify_corpus,
    clean_text, tokenize
)
from src.model_a_train import (
    train_random_forest, train_svm, train_logistic_regression,
    train_kmeans, train_label_propagation,
    train_question_ranker, evaluate_ensemble, build_comparison_table,
    soft_vote_predict
)
from src.model_b_train import (
    evaluate_distractors, evaluate_hints, generate_distractors, generate_hints
)
from src.inference import generate_question
from src.evaluate import exact_match, evaluate_test_set, plot_dashboard


def main():
    parser = argparse.ArgumentParser(description='RACE ML Pipeline')
    parser.add_argument('--skip-preprocess', action='store_true',
                        help='Load cached processed data instead of recomputing')
    args = parser.parse_args()

    t_start = time.time()

    # ── Paths ────────────────────────────────────────────────────────────────
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
    MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
    MODEL_A_DIR = os.path.join(MODELS_DIR, 'model_a', 'traditional')
    MODEL_B_DIR = os.path.join(MODELS_DIR, 'model_b', 'traditional')
    PLOT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
    os.makedirs(MODEL_A_DIR, exist_ok=True)
    os.makedirs(MODEL_B_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    
    # PHASE 1 — Data Loading & Preprocessing
    
    print('=' * 70)
    print('PHASE 1 — Data Loading & Preprocessing')
    print('=' * 70)

    cache_exists = os.path.exists(os.path.join(PROCESSED_DIR, 'Xv_train.npz'))

    if args.skip_preprocess and cache_exists:
        # ── Load from cache ──────────────────────────────────────────────────
        print('Loading cached processed data from data/processed/...')
        train_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'train_clean.csv'))
        dev_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'dev_clean.csv'))
        test_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'test_clean.csv'))
        Xv_train = load_npz(os.path.join(PROCESSED_DIR, 'Xv_train.npz'))
        Xv_dev = load_npz(os.path.join(PROCESSED_DIR, 'Xv_dev.npz'))
        yv_train = np.load(os.path.join(PROCESSED_DIR, 'yv_train.npy'))
        yv_dev = np.load(os.path.join(PROCESSED_DIR, 'yv_dev.npy'))
        X_train = load_npz(os.path.join(PROCESSED_DIR, 'X_train.npz'))
        X_dev = load_npz(os.path.join(PROCESSED_DIR, 'X_dev.npz'))
        X_test = load_npz(os.path.join(PROCESSED_DIR, 'X_test.npz'))
        H_train = np.load(os.path.join(PROCESSED_DIR, 'H_train.npy'))
        H_dev = np.load(os.path.join(PROCESSED_DIR, 'H_dev.npy'))
        H_test = np.load(os.path.join(PROCESSED_DIR, 'H_test.npy'))
        with open(os.path.join(PROCESSED_DIR, 'vectorizers.pkl'), 'rb') as f:
            vec_cache = pickle.load(f)
        ohe_vec = vec_cache['ohe_vec']
        tfidf_vec = vec_cache['tfidf_vec']
        ohe_verify = vec_cache['ohe_verify']
        print(f'  Loaded Xv_train={Xv_train.shape}, X_train={X_train.shape}')
    else:
        # ── Build from scratch ───────────────────────────────────────────────
        train_df, dev_df, test_df = load_all_splits(DATA_DIR)
        apply_cleaning([train_df, dev_df, test_df])

        # Add length columns for EDA / dashboard
        train_df['article_len'] = train_df['article'].fillna('').apply(
            lambda x: len(x.split()))
        train_df['question_len'] = train_df['question'].fillna('').apply(
            lambda x: len(x.split()))
        train_df['answer_len'] = train_df['answer_text'].fillna('').apply(
            lambda x: len(x.split()))

        # ── EDA Plots ────────────────────────────────────────────────────────
        print('\nGenerating EDA plots...')
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, (name, df) in zip(axes, [('Train', train_df), ('Dev', dev_df),
                                          ('Test', test_df)]):
            counts = df['answer'].value_counts().sort_index()
            ax.bar(counts.index, counts.values, color='steelblue',
                   edgecolor='white')
            ax.set_title(f'{name} — Answer Balance')
            ax.set_xlabel('Correct Option'); ax.set_ylabel('Count')
        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/eda_answer_balance.png', dpi=120)
        plt.close()

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, col, color, label in zip(
            axes,
            ['article_len', 'question_len', 'answer_len'],
            ['#4C78A8', '#F58518', '#54A24B'],
            ['Article Words', 'Question Words', 'Answer Words']
        ):
            ax.hist(train_df[col].clip(upper=train_df[col].quantile(0.98)),
                    bins=50, color=color, edgecolor='white', alpha=0.85)
            ax.axvline(train_df[col].median(), color='red', linestyle='--',
                       lw=1.5, label=f'Median {train_df[col].median():.0f}')
            ax.set_title(label); ax.legend(fontsize=9)
        plt.suptitle('RACE — Length Distributions (Train)', fontsize=13,
                     y=1.02)
        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/eda_lengths.png', dpi=120)
        plt.close()
        print('  EDA plots saved to outputs/')

        # ── Vectorization ────────────────────────────────────────────────────
        print('\nBuilding feature matrices...')
        train_corpus = build_corpus(train_df)
        dev_corpus = build_corpus(dev_df)
        test_corpus = build_corpus(test_df)
        vecs = fit_vectorizers(train_corpus, dev_corpus, test_corpus)
        ohe_vec = vecs['ohe_vec']
        tfidf_vec = vecs['tfidf_vec']

        # Handcrafted features
        print('\nBuilding handcrafted features...')
        H_train = handcrafted_features(train_df)
        H_dev = handcrafted_features(dev_df)
        H_test = handcrafted_features(test_df)
        print(f'  Shape: {H_train.shape}')
        X_train = build_combined_features(vecs['X_train_ohe'], H_train)
        X_dev = build_combined_features(vecs['X_dev_ohe'], H_dev)
        X_test = build_combined_features(vecs['X_test_ohe'], H_test)
        print(f'  Combined: {X_train.shape}')

        # ── Verification dataset ─────────────────────────────────────────────
        print('\nBuilding verification dataset...')
        verify_data = build_verification_data(train_df, dev_df)
        ohe_verify = verify_data['ohe_verify']
        Xv_train = verify_data['Xv_train']
        Xv_dev = verify_data['Xv_dev']
        yv_train = verify_data['yv_train']
        yv_dev = verify_data['yv_dev']

        # ── Save to data/processed/ ──────────────────────────────────────────
        print(f'\nSaving processed data to {PROCESSED_DIR}/...')
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        train_df.to_csv(os.path.join(PROCESSED_DIR, 'train_clean.csv'),
                        index=False)
        dev_df.to_csv(os.path.join(PROCESSED_DIR, 'dev_clean.csv'),
                      index=False)
        test_df.to_csv(os.path.join(PROCESSED_DIR, 'test_clean.csv'),
                       index=False)
        save_npz(os.path.join(PROCESSED_DIR, 'Xv_train.npz'), Xv_train)
        save_npz(os.path.join(PROCESSED_DIR, 'Xv_dev.npz'), Xv_dev)
        np.save(os.path.join(PROCESSED_DIR, 'yv_train.npy'), yv_train)
        np.save(os.path.join(PROCESSED_DIR, 'yv_dev.npy'), yv_dev)
        save_npz(os.path.join(PROCESSED_DIR, 'X_train.npz'), X_train)
        save_npz(os.path.join(PROCESSED_DIR, 'X_dev.npz'), X_dev)
        save_npz(os.path.join(PROCESSED_DIR, 'X_test.npz'), X_test)
        np.save(os.path.join(PROCESSED_DIR, 'H_train.npy'), H_train)
        np.save(os.path.join(PROCESSED_DIR, 'H_dev.npy'), H_dev)
        np.save(os.path.join(PROCESSED_DIR, 'H_test.npy'), H_test)
        with open(os.path.join(PROCESSED_DIR, 'vectorizers.pkl'), 'wb') as f:
            pickle.dump({'ohe_vec': ohe_vec, 'tfidf_vec': tfidf_vec,
                         'ohe_verify': ohe_verify}, f)
        print('  ✅ All processed data cached.')

    
    # PHASE 2 — Model A Training
    
    print('\n' + '=' * 70)
    print('PHASE 2 — Model A: Supervised Baselines')
    print('=' * 70)

    rf_clf, rf_metrics = train_random_forest(Xv_train, yv_train, Xv_dev, yv_dev)
    svm_clf, svm_metrics = train_svm(Xv_train, yv_train, Xv_dev, yv_dev)
    lr_clf, lr_metrics = train_logistic_regression(Xv_train, yv_train, Xv_dev, yv_dev)

    # ── Unsupervised / Semi-Supervised ────────────────────────────────────────
    print('\n' + '-' * 70)
    print('Model A: Unsupervised & Semi-Supervised')
    print('-' * 70)

    kmeans, Xv_train_norm, cluster_summary = train_kmeans(
        Xv_train, yv_train, save_dir=PLOT_DIR)
    lp_model, lp_metrics = train_label_propagation(Xv_train_norm, yv_train)

    # ── Question Generation Demo ──────────────────────────────────────────────
    print('\n' + '-' * 70)
    print('Question Generation Demo')
    print('-' * 70)
    sample = test_df.iloc[0]
    gen_q = generate_question(
        str(sample['article']), str(sample['answer_text']), lr_clf, ohe_verify)
    print(f'  Article:     {str(sample["article"])[:150]}...')
    print(f'  Gold Q:      {sample["question"]}')
    print(f'  Generated Q: {gen_q}')
    print(f'  Answer:      {sample["answer_text"]}')

    # ── Question Quality Ranker ───────────────────────────────────────────────
    print('\n' + '-' * 70)
    print('Question Quality Ranker')
    print('-' * 70)
    svm_ranker, rf_ranker = train_question_ranker(
        test_df, lr_clf, ohe_verify, generate_question, n=200)

    # ── Ensemble ──────────────────────────────────────────────────────────────
    print('\n' + '-' * 70)
    print('Ensemble (Soft Voting)')
    print('-' * 70)
    ens_metrics, ens_preds, ens_proba = evaluate_ensemble(
        rf_clf, svm_clf, lr_clf, Xv_dev, yv_dev, save_dir=PLOT_DIR)

    # ── Comparison Table ──────────────────────────────────────────────────────
    print('\n' + '-' * 70)
    print('Supervised vs Unsupervised vs Semi-Supervised')
    print('-' * 70)
    all_metrics_a = {**rf_metrics, **svm_metrics, **lr_metrics,
                     **ens_metrics, **lp_metrics}
    comp_df, sil_score = build_comparison_table(
        all_metrics_a, Xv_train_norm, kmeans)

    
    # PHASE 3 — Model B Training & Evaluation
    
    print('\n' + '=' * 70)
    print('PHASE 3 — Model B: Distractor & Hint Evaluation')
    print('=' * 70)

    dist_metrics, dist_cm = evaluate_distractors(test_df, ohe_verify, n_eval=100)

    train_sample = train_df.sample(n=min(20000, len(train_df)), random_state=42)
    hint_metrics, hint_lr = evaluate_hints(train_sample, n_eval=200)

    # ── Distractor Demo ───────────────────────────────────────────────────────
    print('\nDistractor Generation Demo:')
    sample = test_df.iloc[2]
    dists = generate_distractors(
        str(sample['article']), str(sample['answer_text']), ohe_verify)
    print(f'  Answer: {sample["answer_text"]}')
    for i, d in enumerate(dists, 1):
        print(f'  Distractor {i}: {d}')

    
    # PHASE 4 — Test Set Evaluation
    
    print('\n' + '=' * 70)
    print('PHASE 4 — Full Test Set Evaluation')
    print('=' * 70)

    test_metrics, test_preds, test_proba, yv_test = evaluate_test_set(
        rf_clf, svm_clf, lr_clf, ohe_verify, test_df)

    # ── Dashboard Plot ────────────────────────────────────────────────────────
    print('\nGenerating analytics dashboard...')
    plot_dashboard(
        all_metrics_a,
        rf_metrics['rf_acc'], svm_metrics['svm_acc'],
        lr_metrics['lr_acc'], ens_metrics['ens_acc'],
        rf_metrics['rf_f1'], svm_metrics['svm_f1'],
        lr_metrics['lr_f1'], ens_metrics['ens_f1'],
        ens_metrics['ens_prec'], ens_metrics['ens_rec'],
        test_metrics, test_preds, test_proba, yv_test,
        cluster_summary, train_df, save_dir=PLOT_DIR
    )
    print(f'  Dashboard saved to {PLOT_DIR}/analytics_dashboard.png')

    
    # PHASE 5 — Save Models & Metrics
    
    print('\n' + '=' * 70)
    print('PHASE 5 — Saving Models & Metrics')
    print('=' * 70)

    # ── Model A artifacts → models/model_a/traditional/ ──────────────────
    model_a_artifacts = {
        'rf_clf': rf_clf,
        'svm_clf': svm_clf,
        'lr_clf': lr_clf,
        'ohe_verify': ohe_verify,
        'ohe_vec': ohe_vec,
        'tfidf_vec': tfidf_vec,
    }
    for name, obj in model_a_artifacts.items():
        path = os.path.join(MODEL_A_DIR, f'{name}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        print(f'  Saved {name} → {path}')

    # ── Model B artifacts → models/model_b/traditional/ ──────────────────
    model_b_artifacts = {}
    if hint_lr is not None:
        model_b_artifacts['hint_lr'] = hint_lr
    for name, obj in model_b_artifacts.items():
        path = os.path.join(MODEL_B_DIR, f'{name}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        print(f'  Saved {name} → {path}')
    if not model_b_artifacts:
        print('  (Model B uses heuristic methods — no fitted models to save)')

    # ── Metrics JSON → models/ ───────────────────────────────────────────
    metrics_json = {
        'model_a': {**all_metrics_a},
        'model_b': {**dist_metrics, **hint_metrics},
        'silhouette': sil_score,
        'test': test_metrics,
    }
    metrics_path = os.path.join(MODELS_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2, default=str)
    print(f'  Saved metrics → {metrics_path}')

    
    elapsed = time.time() - t_start
    print('\n' + '=' * 70)
    print(f'✅ Pipeline complete in {elapsed:.1f}s')
    print(f'   Model A saved to: {MODEL_A_DIR}/')
    print(f'   Model B saved to: {MODEL_B_DIR}/')
    print(f'   Metrics saved to: {MODELS_DIR}/metrics.json')
    print(f'   Plots saved to:   {PLOT_DIR}/')
    print(f'   Now run:  streamlit run ui/app.py')
    print('=' * 70)


if __name__ == '__main__':
    main()
