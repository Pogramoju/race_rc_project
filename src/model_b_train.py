"""
model_b_train.py — Training script for Model B
(Distractor & Hint Generator)
Extracted from notebook cells 40, 42, 44.
"""
import re, numpy as np
from sklearn.preprocessing import normalize as sk_normalize
from src.preprocessing import clean_text, tokenize

# ── Helpers ──────────────────────────────────────────────────────────────────
def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if len(s.strip())>10]

def sentence_keyword_overlap(sentence, answer):
    s_tok = set(tokenize(sentence)); a_tok = set(tokenize(answer))
    return len(s_tok & a_tok) / max(len(a_tok), 1)

def cosine_sim(a, b):
    a_n = sk_normalize(a, norm='l2'); b_n = sk_normalize(b, norm='l2')
    return float(a_n.dot(b_n.T).toarray())

# ── Distractor Generation ───────────────────────────────────────────────────
def extract_candidate_phrases(article, min_len=2, max_len=8):
    patterns = [r'"([^"]{5,60})"', r"'([^']{5,60})'",
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})',
                r'([a-z]+(?:\s+[a-z]+){1,3})']
    candidates = set()
    for pat in patterns:
        for m in re.finditer(pat, article):
            span = m.group(1).strip()
            if min_len <= len(span.split()) <= max_len:
                candidates.add(span)
    return list(candidates)

def generate_distractors(article, answer, vectorizer, top_n=3):
    candidates = extract_candidate_phrases(article)
    candidates = [c for c in candidates if c.lower().strip() != answer.lower().strip()]
    if not candidates:
        return ['None of the above','Not mentioned','Cannot be determined'][:top_n]
    ans_vec = vectorizer.transform([clean_text(answer)])
    cand_vecs = vectorizer.transform([clean_text(c) for c in candidates])
    scores = []
    for i,c in enumerate(candidates):
        sim = cosine_sim(cand_vecs[i], ans_vec)
        scores.append((c, 1.0 - abs(sim - 0.3), sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    chosen = []
    for c,_,_ in scores:
        if len(chosen)>=top_n: break
        if not any(c.lower() in d.lower() or d.lower() in c.lower() for d in chosen):
            chosen.append(c)
    fallbacks = ['Not applicable','None of the above','Cannot be determined']
    while len(chosen)<top_n: chosen.append(fallbacks[len(chosen)%len(fallbacks)])
    return chosen[:top_n]

# ── Hint Generation ─────────────────────────────────────────────────────────
def build_sentence_features(sentences, question, answer):
    n = len(sentences); feats = []
    for i,sent in enumerate(sentences):
        feats.append([
            sentence_keyword_overlap(sent, question),
            sentence_keyword_overlap(sent, answer),
            i / max(n-1, 1),
            len(sent.split())
        ])
    return np.array(feats, dtype=np.float32)

def generate_hints(article, question, answer, n_hints=3):
    sentences = split_sentences(article)
    if not sentences: return [article[:200]]
    feats = build_sentence_features(sentences, question, answer)
    scores = (0.50*feats[:,1] + 0.30*feats[:,0] +
              0.10*(1-feats[:,2]) + 0.10*np.clip(feats[:,3]/50,0,1))
    top_idx = sorted(np.argsort(scores)[::-1][:n_hints])
    return [sentences[i] for i in top_idx]

# ── NLG Evaluation ──────────────────────────────────────────────────────────
def evaluate_distractors(test_df, ohe_verify, n_eval=100):
    """Evaluate distractor quality using BLEU, ROUGE, METEOR.
    Compares generated distractors against the gold wrong options."""
    from src.evaluate import compute_all_nlg_metrics
    all_bleu, all_r1, all_rL, all_meteor = [], [], [], []
    eval_rows = test_df.head(n_eval)
    option_cols = [c for c in test_df.columns if c.lower() in ('a','b','c','d')]
    if not option_cols:
        option_cols = [c for c in test_df.columns if c.lower().startswith('option')]

    for _, row in eval_rows.iterrows():
        article = str(row.get('article', ''))
        answer = str(row.get('answer_text', ''))
        dists = generate_distractors(article, answer, ohe_verify, top_n=3)
        # Gold wrong options
        gold_wrong = []
        for c in option_cols:
            opt = str(row.get(c, ''))
            if clean_text(opt) != clean_text(answer):
                gold_wrong.append(opt)
        # Compare each generated distractor against best-matching gold option
        for d in dists:
            if not gold_wrong:
                continue
            best = max(gold_wrong,
                       key=lambda g: compute_all_nlg_metrics(g, d)['rougeL'])
            m = compute_all_nlg_metrics(best, d)
            all_bleu.append(m['bleu'])
            all_r1.append(m['rouge1'])
            all_rL.append(m['rougeL'])
            all_meteor.append(m['meteor'])

    results = {
        'dist_bleu':   np.mean(all_bleu)   if all_bleu else 0.0,
        'dist_rouge1': np.mean(all_r1)     if all_r1 else 0.0,
        'dist_rougeL': np.mean(all_rL)     if all_rL else 0.0,
        'dist_meteor': np.mean(all_meteor) if all_meteor else 0.0,
    }
    print(f'Distractor NLG (n={len(all_bleu)}):')
    print(f'  BLEU={results["dist_bleu"]:.4f}  ROUGE-1={results["dist_rouge1"]:.4f}  '
          f'ROUGE-L={results["dist_rougeL"]:.4f}  METEOR={results["dist_meteor"]:.4f}')
    return results

def evaluate_hints(train_sample, n_eval=200):
    """Evaluate hint quality using BLEU/ROUGE/METEOR.
    Compares generated hints against gold answer text."""
    from src.evaluate import evaluate_generation_nlg
    gold_refs, gen_hints_flat = [], []
    for _, row in train_sample.head(n_eval).iterrows():
        art = str(row.get('article', ''))
        q = str(row.get('question', ''))
        ans = str(row.get('answer_text', ''))
        sents = split_sentences(art)
        if not sents:
            continue
        hints = generate_hints(art, q, ans, n_hints=3)
        combined_hint = ' '.join(hints)
        gold_refs.append(ans)
        gen_hints_flat.append(combined_hint)

    if not gold_refs:
        print('Warning: no hints generated; skipping.')
        return {}
    results = evaluate_generation_nlg(gold_refs, gen_hints_flat, task_name='hint')
    return results
