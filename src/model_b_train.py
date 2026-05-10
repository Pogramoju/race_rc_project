"""
model_b_train.py — Training script for Model B
(Distractor & Hint Generator)
Extracted from notebook cells 40, 42, 44.
"""
import re, numpy as np
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, r2_score,
    classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
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

# ── Evaluation ──────────────────────────────────────────────────────────────
def evaluate_distractors(test_df, ohe_verify, n_eval=100):
    """Evaluate distractor quality: Precision/Recall/F1 + confusion matrix."""
    dist_correct = dist_total = dist_relevant = 0
    eval_rows = test_df.head(n_eval)
    for _,row in eval_rows.iterrows():
        article = str(row.get('article','')); answer = str(row.get('answer_text',''))
        dists = generate_distractors(article, answer, ohe_verify, top_n=3)
        for d in dists:
            dist_total += 1
            if clean_text(d) != clean_text(answer): dist_correct += 1
            d_tok = set(tokenize(d)); a_tok = set(tokenize(article))
            if len(d_tok & a_tok) > 0: dist_relevant += 1
    p = dist_correct/max(dist_total,1)
    r = dist_relevant/max(dist_total,1)
    f = 2*p*r/max(p+r,1e-9)
    print(f'Distractor P={p:.4f} R={r:.4f} F1={f:.4f} (n={dist_total})')
    # Confusion matrix
    preds, golds = [], []
    for _,row in eval_rows.iterrows():
        art = str(row.get('article','')); ans = str(row.get('answer_text',''))
        for d in generate_distractors(art,ans,ohe_verify,top_n=3):
            preds.append(1 if clean_text(d)!=clean_text(ans) else 0)
            golds.append(1)
    cm = confusion_matrix(golds, preds)
    print('Distractor CM:'); print(cm)
    return {'dist_precision':p,'dist_recall':r,'dist_f1':f}, cm

def evaluate_hints(train_sample, n_eval=200):
    """Train LR on sentence features and evaluate hint quality."""
    hint_X, hint_y = [], []
    for _,row in train_sample.head(n_eval).iterrows():
        art = str(row.get('article','')); q = str(row.get('question',''))
        ans = str(row.get('answer_text',''))
        sents = split_sentences(art)
        if not sents: continue
        feats = build_sentence_features(sents, q, ans)
        for i,s in enumerate(sents):
            hint_X.append(feats[i])
            hint_y.append(1 if sentence_keyword_overlap(s,ans)>0.3 else 0)
    hint_X, hint_y = np.array(hint_X), np.array(hint_y)
    if len(set(hint_y)) < 2:
        print('Warning: single class in hint labels; skipping.')
        return {}, None
    Xtr,Xte,ytr,yte = train_test_split(hint_X,hint_y,test_size=0.3,random_state=42,stratify=hint_y)
    lr = LogisticRegression(max_iter=1000,random_state=42)
    lr.fit(Xtr,ytr); p = lr.predict(Xte)
    a = accuracy_score(yte,p); f = f1_score(yte,p,average='macro')
    r2 = r2_score(yte, lr.predict_proba(Xte)[:,1])
    print(f'Hint LR Acc={a:.4f} F1={f:.4f} R2={r2:.4f}')
    print(classification_report(yte,p,target_names=['Non-Hint','Hint']))
    return {'hint_acc':a,'hint_f1':f,'hint_r2':r2}, lr
