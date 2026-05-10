"""
evaluate.py — NLG Metric computation (BLEU, ROUGE, METEOR)
Evaluates generated questions and answers against gold references.
Also retains exact-match and the dashboard plot for compatibility.
"""
import numpy as np, pandas as pd, warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from src.preprocessing import clean_text, expand_to_options, make_verify_corpus
from src.model_a_train import soft_vote_predict

# ── NLG metric imports ──────────────────────────────────────────────────────
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
try:
    from nltk.translate.meteor_score import meteor_score as _nltk_meteor
    import nltk
    # Ensure wordnet data is available
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    _HAS_METEOR = True
except Exception:
    _HAS_METEOR = False

warnings.filterwarnings('ignore')

# ── Core NLG metrics ────────────────────────────────────────────────────────
def compute_bleu(reference: str, hypothesis: str) -> float:
    """Compute sentence-level BLEU score between reference and hypothesis."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    smooth = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth)


def compute_rouge(reference: str, hypothesis: str) -> dict:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'],
                                       use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure,
    }


def compute_meteor(reference: str, hypothesis: str) -> float:
    """Compute METEOR score between reference and hypothesis."""
    if not _HAS_METEOR:
        return 0.0
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    try:
        return _nltk_meteor([ref_tokens], hyp_tokens)
    except Exception:
        return 0.0


def compute_all_nlg_metrics(reference: str, hypothesis: str) -> dict:
    """Compute all NLG metrics (BLEU, ROUGE-1/2/L, METEOR) for a pair."""
    bleu = compute_bleu(reference, hypothesis)
    rouge = compute_rouge(reference, hypothesis)
    meteor = compute_meteor(reference, hypothesis)
    return {
        'bleu': bleu,
        'rouge1': rouge['rouge1'],
        'rouge2': rouge['rouge2'],
        'rougeL': rouge['rougeL'],
        'meteor': meteor,
    }


# ── Batch evaluation ────────────────────────────────────────────────────────
def evaluate_generation_nlg(references, hypotheses, task_name='Generation'):
    """Evaluate a batch of generated texts against references.

    Args:
        references:  list of gold reference strings
        hypotheses:  list of generated hypothesis strings
        task_name:   label for printing (e.g. 'Question Gen', 'Answer Sel')

    Returns:
        dict with averaged BLEU, ROUGE-1, ROUGE-2, ROUGE-L, METEOR
    """
    n = min(len(references), len(hypotheses))
    bleu_scores, r1, r2, rL, meteor_scores = [], [], [], [], []

    for i in range(n):
        ref = str(references[i])
        hyp = str(hypotheses[i])
        if not ref.strip() or not hyp.strip():
            continue
        m = compute_all_nlg_metrics(ref, hyp)
        bleu_scores.append(m['bleu'])
        r1.append(m['rouge1'])
        r2.append(m['rouge2'])
        rL.append(m['rougeL'])
        meteor_scores.append(m['meteor'])

    results = {
        f'{task_name}_bleu': np.mean(bleu_scores) if bleu_scores else 0.0,
        f'{task_name}_rouge1': np.mean(r1) if r1 else 0.0,
        f'{task_name}_rouge2': np.mean(r2) if r2 else 0.0,
        f'{task_name}_rougeL': np.mean(rL) if rL else 0.0,
        f'{task_name}_meteor': np.mean(meteor_scores) if meteor_scores else 0.0,
    }

    print(f'\n  {task_name} NLG Metrics (n={len(bleu_scores)}):')
    print(f'    BLEU:    {results[f"{task_name}_bleu"]:.4f}')
    print(f'    ROUGE-1: {results[f"{task_name}_rouge1"]:.4f}')
    print(f'    ROUGE-2: {results[f"{task_name}_rouge2"]:.4f}')
    print(f'    ROUGE-L: {results[f"{task_name}_rougeL"]:.4f}')
    print(f'    METEOR:  {results[f"{task_name}_meteor"]:.4f}')

    return results


# ── Exact match (kept for compatibility) ────────────────────────────────────
def exact_match(predictions, ground_truths):
    """Exact match: 1 if predicted == ground truth (after normalising)."""
    def norm(s): return clean_text(str(s), lowercase=True, remove_punct=True)
    correct = sum(norm(p)==norm(g) for p,g in zip(predictions, ground_truths))
    return correct / max(len(predictions), 1)


# ── Test-set evaluation (updated with NLG metrics) ──────────────────────────
def evaluate_test_set(rf_clf, svm_clf, lr_clf, ohe_verify, test_df,
                      generate_question_fn=None, n=2000):
    """Full evaluation on test set using NLG metrics.

    Computes BLEU/ROUGE/METEOR for:
      1. Answer selection (selected answer text vs gold answer text)
      2. Question generation (generated question vs gold question) — if gen fn provided
    """
    sample_df = test_df.sample(n=min(n, len(test_df)), random_state=42)
    exp_test = expand_to_options(sample_df)
    Xv_test = ohe_verify.transform(make_verify_corpus(exp_test))
    yv_test = exp_test['is_correct'].values
    preds, proba = soft_vote_predict([rf_clf, svm_clf, lr_clf], Xv_test)

    # ── Answer selection: pick best option per question ───────────────────
    n_q = len(exp_test) // 4
    selected_answers, gold_answers = [], []
    for i in range(n_q):
        best = np.argmax(proba[i*4:(i+1)*4, 1])
        selected_answers.append(str(exp_test.iloc[i*4+best].get('option_text', '')))
        gold_answers.append(str(exp_test.iloc[i*4].get('answer_text', '')))

    em = exact_match(selected_answers, gold_answers)
    answer_nlg = evaluate_generation_nlg(gold_answers, selected_answers,
                                          task_name='answer')

    # ── Question generation evaluation (if generator provided) ────────────
    qgen_nlg = {}
    if generate_question_fn is not None:
        eval_n = min(200, len(sample_df))
        gold_questions, gen_questions = [], []
        for _, row in sample_df.head(eval_n).iterrows():
            article = str(row.get('article', ''))
            answer = str(row.get('answer_text', ''))
            gold_q = str(row.get('question', ''))
            gen_q = generate_question_fn(article, answer, lr_clf, ohe_verify)
            gold_questions.append(gold_q)
            gen_questions.append(gen_q)
        qgen_nlg = evaluate_generation_nlg(gold_questions, gen_questions,
                                            task_name='qgen')

    print(f'\n  Exact Match: {em:.4f}')

    metrics = {'exact_match': em, **answer_nlg, **qgen_nlg}
    return metrics, preds, proba, yv_test


# ── Dashboard plot (updated for NLG metrics) ────────────────────────────────
def plot_dashboard(nlg_metrics, cluster_summary, train_df, save_dir='content'):
    """Plot the analytics dashboard with NLG metrics."""
    fig = plt.figure(figsize=(16, 10))

    # 1 — Answer NLG metrics bar chart
    ax1 = fig.add_subplot(2, 3, 1)
    ans_keys = ['answer_bleu', 'answer_rouge1', 'answer_rouge2',
                'answer_rougeL', 'answer_meteor']
    ans_labels = ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'METEOR']
    ans_vals = [nlg_metrics.get(k, 0) for k in ans_keys]
    colors = ['#4C78A8', '#F58518', '#54A24B', '#E45756', '#72B7B2']
    bars = ax1.bar(ans_labels, ans_vals, color=colors, edgecolor='white')
    for bar, v in zip(bars, ans_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    ax1.set_ylim(0, 1.05)
    ax1.set_title('Answer Selection — NLG Metrics')
    ax1.set_ylabel('Score')

    # 2 — Question Gen NLG metrics (if available)
    ax2 = fig.add_subplot(2, 3, 2)
    qgen_keys = ['qgen_bleu', 'qgen_rouge1', 'qgen_rouge2',
                 'qgen_rougeL', 'qgen_meteor']
    qgen_vals = [nlg_metrics.get(k, 0) for k in qgen_keys]
    if any(v > 0 for v in qgen_vals):
        bars = ax2.bar(ans_labels, qgen_vals, color=colors, edgecolor='white')
        for bar, v in zip(bars, qgen_vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        ax2.set_ylim(0, 1.05)
    ax2.set_title('Question Generation — NLG Metrics')
    ax2.set_ylabel('Score')

    # 3 — K-Means Clusters
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.bar(cluster_summary.index, cluster_summary['correct_rate'],
            color='#54A24B', edgecolor='white')
    ax3.axhline(0.25, color='red', ls='--')
    ax3.set_title('K-Means Clusters')

    # 4 — NLG Metrics Summary Table
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.axis('off')
    td = [['Metric', 'Answer Sel.', 'Question Gen.']]
    for label, ak, qk in zip(ans_labels, ans_keys, qgen_keys):
        av = f'{nlg_metrics.get(ak, 0):.4f}'
        qv = f'{nlg_metrics.get(qk, 0):.4f}' if nlg_metrics.get(qk) else '—'
        td.append([label, av, qv])
    td.append(['Exact Match', f'{nlg_metrics.get("exact_match", 0):.4f}', '—'])
    tbl = ax4.table(cellText=td, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.6)
    ax4.set_title('NLG Summary', pad=20)

    # 5 — Distractor NLG metrics (if available)
    ax5 = fig.add_subplot(2, 3, 5)
    dist_keys = ['dist_bleu', 'dist_rouge1', 'dist_rougeL', 'dist_meteor']
    dist_labels = ['BLEU', 'ROUGE-1', 'ROUGE-L', 'METEOR']
    dist_vals = [nlg_metrics.get(k, 0) for k in dist_keys]
    if any(v > 0 for v in dist_vals):
        bars = ax5.bar(dist_labels, dist_vals,
                       color=['#B279A2', '#FF9DA6', '#EECA3B', '#4C78A8'],
                       edgecolor='white')
        for bar, v in zip(bars, dist_vals):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        ax5.set_ylim(0, 1.05)
    ax5.set_title('Distractor — NLG Metrics')

    # 6 — Answer length distribution
    ax6 = fig.add_subplot(2, 3, 6)
    if 'answer_len' in train_df.columns:
        train_df['answer_len'].hist(bins=30, ax=ax6, color='#72B7B2',
                                     edgecolor='white')
    ax6.set_title('Answer Length')

    plt.suptitle('RACE ML Pipeline — NLG Evaluation Dashboard',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/analytics_dashboard.png', dpi=130,
                bbox_inches='tight')
    plt.close()
