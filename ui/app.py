import streamlit as st
import pandas as pd
import numpy as np
import os, sys, time, pickle, re, string, random
from collections import Counter

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RACE Quiz System",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths (search multiple locations) ────────────────────────────────────────
_file_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_project_base = r"C:\Users\Ahmed Mujtaba\PycharmProjects\PythonProject\AI lab Project"

def _find_dir(name):
    for base in [_file_base, os.getcwd(), _project_base]:
        p = os.path.join(base, name)
        if os.path.isdir(p):
            return os.path.abspath(p)
    return os.path.join(_file_base, name)

BASE_DIR = _file_base
MODEL_DIR = _find_dir("models")
MODEL_A_DIR = os.path.join(MODEL_DIR, 'model_a', 'traditional')
MODEL_B_DIR = os.path.join(MODEL_DIR, 'model_b', 'traditional')
DATA_DIR = _find_dir("data")

# ── Stopwords ────────────────────────────────────────────────────────────────
STOPWORDS = set("i me my myself we our ours ourselves you your yours yourself "
    "yourselves he him his himself she her hers herself it its itself they them "
    "their theirs themselves what which who whom this that these those am is are "
    "was were be been being have has had having do does did doing a an the and "
    "but if or because as until while of at by for with about against between "
    "through during before after above below to from up down in out on off over "
    "under again further then once here there when where why how all both each "
    "few more most other some such no nor not only own same so than too very s t "
    "can will just don should now d ll m o re ve y ain aren couldn didn doesn "
    "hadn hasn haven isn ma mightn mustn needn shan shouldn wasn weren won wouldn".split())

# ── Text utilities (self-contained copies from notebook) ─────────────────────
def clean_text(text, remove_stopwords=False):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    if remove_stopwords:
        text = ' '.join(t for t in text.split() if t not in STOPWORDS)
    return re.sub(r'\s+', ' ', text).strip()

def tokenize(text):
    return clean_text(text, remove_stopwords=True).split()

def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if len(s.strip()) > 10]

def sentence_keyword_overlap(sentence, answer):
    s_tok = set(tokenize(sentence))
    a_tok = set(tokenize(answer))
    return len(s_tok & a_tok) / max(len(a_tok), 1)

def cosine_sim(a, b):
    from sklearn.preprocessing import normalize as sk_normalize
    a_n = sk_normalize(a, norm='l2')
    b_n = sk_normalize(b, norm='l2')
    return float(a_n.dot(b_n.T).toarray())

# ── Answer Extraction (for custom articles without a gold answer) ────────────
def extract_answer_from_article(article):
    """Extract a meaningful key phrase from the article to use as the answer.
    Uses regex only — no NLP libraries. Returns the best candidate."""
    # Priority 1: Quoted phrases
    quoted = re.findall(r'["\u201c]([^"\u201d]{3,40})["\u201d]', article)
    if quoted:
        return max(quoted, key=len)
    # Priority 2: Proper noun phrases (2-4 capitalized words)
    proper = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', article)
    if proper:
        return max(proper, key=len)
    # Priority 3: Frequent meaningful noun chunks (skip first 3 words of sentences)
    words = re.findall(r'\b[a-z]{4,}\b', article.lower())
    word_freq = Counter(w for w in words if w not in STOPWORDS)
    if word_freq:
        top_word = word_freq.most_common(1)[0][0]
        # Find a 2-3 word phrase containing this word
        pattern = re.compile(r'\b(\w+\s+' + re.escape(top_word) + r'(?:\s+\w+)?)\b', re.IGNORECASE)
        phrases = pattern.findall(article)
        if phrases:
            return phrases[0].strip()
        return top_word
    # Fallback: first noun-like word over 4 chars
    fallback = [w for w in article.split() if len(w) > 4 and w[0].isupper()]
    return fallback[0] if fallback else article.split('.')[0].split()[-1]

# ── Question Generation ─────────────────────────────────────────────────────
def wh_template(sentence, answer):
    ans_lower = answer.lower().strip()
    person_ind = ['mr','mrs','dr','president','minister','he','she','his','her']
    place_ind = ['city','country','town','village','river','mountain','street']
    if any(ans_lower.startswith(p) for p in person_ind): wh = 'Who'
    elif any(p in ans_lower for p in place_ind): wh = 'Where'
    elif re.match(r'^\d', ans_lower) or any(t in ans_lower for t in ['year','month','day','time','ago']): wh = 'When'
    elif ans_lower.startswith('because') or 'reason' in ans_lower: wh = 'Why'
    elif ans_lower.startswith('by') or 'method' in ans_lower: wh = 'How'
    else: wh = 'What'
    # Use word boundaries to avoid replacing inside other words
    pattern = re.compile(r'\b' + re.escape(ans_lower) + r'\b', re.IGNORECASE)
    if pattern.search(sentence):
        question = pattern.sub(wh, sentence, count=1)
    else:
        # Answer not in this sentence — ask about the sentence's topic
        sent_short = sentence[:80].rstrip()
        question = f"{wh} does the passage say about {ans_lower}"
    question = question.strip().rstrip('.!?')
    question = question[0].upper() + question[1:] if question else ''
    return question + '?'

def generate_question(article, answer, verifier_model, vectorizer, top_k=3):
    sentences = split_sentences(article)
    if not sentences: return 'What does the passage discuss?'
    scored = sorted(
        [(s, sentence_keyword_overlap(s, answer)) for s in sentences],
        key=lambda x: x[1], reverse=True
    )
    candidates = []
    for sent, score in scored[:top_k]:
        q = wh_template(sent, answer)
        candidates.append((q, sent, score))
    if not candidates: return 'What is the main idea?'
    best_q, best_score = candidates[0][0], -1
    for q, sent, _ in candidates:
        text = clean_text(article + ' ' + q + ' ' + answer)
        feat = vectorizer.transform([text])
        try:
            prob = verifier_model.predict_proba(feat)[0][1]
        except Exception:
            prob = 0.5
        if prob > best_score:
            best_score = prob
            best_q = q
    return best_q

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
        return ['None of the above', 'Not mentioned', 'Cannot be determined'][:top_n]
    ans_vec = vectorizer.transform([clean_text(answer)])
    cand_vecs = vectorizer.transform([clean_text(c) for c in candidates])
    scores = []
    for i, c in enumerate(candidates):
        sim = cosine_sim(cand_vecs[i], ans_vec)
        plausibility = 1.0 - abs(sim - 0.3)
        scores.append((c, plausibility, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    chosen = []
    for c, plaus, sim in scores:
        if len(chosen) >= top_n: break
        if not any(c.lower() in d.lower() or d.lower() in c.lower() for d in chosen):
            chosen.append(c)
    fallbacks = ['Not applicable', 'None of the above', 'Cannot be determined']
    while len(chosen) < top_n:
        chosen.append(fallbacks[len(chosen) % len(fallbacks)])
    return chosen[:top_n]

# ── Hint Generation ─────────────────────────────────────────────────────────
def build_sentence_features(sentences, question, answer):
    n = len(sentences)
    feats = []
    for i, sent in enumerate(sentences):
        q_overlap = sentence_keyword_overlap(sent, question)
        a_overlap = sentence_keyword_overlap(sent, answer)
        pos = i / max(n - 1, 1)
        length = len(sent.split())
        feats.append([q_overlap, a_overlap, pos, length])
    return np.array(feats, dtype=np.float32)

def generate_hints(article, question, answer, n_hints=3):
    sentences = split_sentences(article)
    if not sentences: return [article[:200]]
    feats = build_sentence_features(sentences, question, answer)
    scores = (0.50 * feats[:, 1] + 0.30 * feats[:, 0] +
              0.10 * (1 - feats[:, 2]) + 0.10 * np.clip(feats[:, 3] / 50, 0, 1))
    top_idx = np.argsort(scores)[::-1][:n_hints]
    top_idx = sorted(top_idx)
    return [sentences[i] for i in top_idx]

# ── Model Loading ────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    for name in ['rf_clf', 'svm_clf', 'lr_clf', 'ohe_verify', 'ohe_vec', 'tfidf_vec']:
        path = os.path.join(MODEL_A_DIR, f'{name}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
        else:
            models[name] = None
    # Also load Model B artifacts if available
    hint_lr_path = os.path.join(MODEL_B_DIR, 'hint_lr.pkl')
    if os.path.exists(hint_lr_path):
        with open(hint_lr_path, 'rb') as f:
            models['hint_lr'] = pickle.load(f)
    return models

@st.cache_data
def load_dataset():
    for subdir in ['', 'raw']:
        base = os.path.join(DATA_DIR, subdir) if subdir else DATA_DIR
        test_path = os.path.join(base, 'test.csv')
        if os.path.exists(test_path):
            df = pd.read_csv(test_path)
            # Resolve answer text: map letter label → actual option text
            if 'answer_text' not in df.columns:
                def resolve(row):
                    lbl = str(row.get('answer', '')).strip().upper()
                    # Try both uppercase and lowercase column names
                    for col in [lbl, lbl.lower()]:
                        if col in row.index and str(row[col]).strip():
                            return str(row[col])
                    return lbl  # fallback to letter
                df['answer_text'] = df.apply(resolve, axis=1)
            return df
    return pd.DataFrame()

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
    .stApp { color: #e0e0e0; }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e, #16213e);
        border-right: 1px solid #0f3460;
    }
    .quiz-option { padding: 12px 20px; margin: 8px 0; border-radius: 10px;
        border: 1px solid #333; cursor: pointer; transition: all 0.3s; }
    .quiz-option:hover { border-color: #4361ee; background: rgba(67,97,238,0.1); }
    .correct { background: rgba(0,200,83,0.2) !important; border-color: #00c853 !important; }
    .incorrect { background: rgba(255,23,68,0.2) !important; border-color: #ff1744 !important; }
    .hint-box { padding: 15px; margin: 10px 0; border-radius: 10px;
        background: rgba(255,193,7,0.08); border-left: 4px solid #ffc107; }
    .metric-card { padding: 20px; border-radius: 12px;
        background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
        text-align: center; margin: 5px; }
    .metric-value { font-size: 2em; font-weight: 700; color: #4361ee; }
    .metric-label { font-size: 0.9em; color: #aaa; margin-top: 5px; }
    h1, h2, h3 { color: #e0e0e0 !important; }
    .stButton>button { border-radius: 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Load resources ───────────────────────────────────────────────────────────
models = load_models()
test_df = load_dataset()
models_loaded = all(v is not None for k, v in models.items() if k in ['lr_clf', 'ohe_verify'])

# ── Session State ────────────────────────────────────────────────────────────
for key in ['article', 'question', 'answer', 'distractors', 'hints',
            'hints_used', 'user_answer', 'checked', 'inference_time',
            'session_log', 'quiz_ready']:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ('distractors', 'hints', 'session_log') else (
            0 if key == 'hints_used' else (False if key in ('checked', 'quiz_ready') else (
                0.0 if key == 'inference_time' else '')))

# ── Sidebar Navigation ──────────────────────────────────────────────────────
st.sidebar.markdown("# 📖 RACE Quiz System")
st.sidebar.markdown("---")
screen = st.sidebar.radio("Navigate", [
    "📝 Article Input",
    "❓ Quiz View",
    "💡 Hint Panel",
    "📊 Analytics Dashboard"
], label_visibility="collapsed")

if not models_loaded:
    st.sidebar.warning("⚠️ Models not found. Run the pipeline first to save models to `models/`.")
st.sidebar.markdown("---")
st.sidebar.caption("AL2002 Lab Project — FAST NUCES")

# ════════════════════════════════════════════════════════════════════════════
# SCREEN 1 — Article Input
# ════════════════════════════════════════════════════════════════════════════
if screen == "📝 Article Input":
    st.markdown("## 📝 Screen 1 — Article Input")
    st.markdown("Load a random reading comprehension from the **RACE dataset** to generate a quiz.")

    # ── Load Random RACE Sample ──────────────────────────────────────────
    col_btn, col_opt = st.columns([1, 1])
    with col_btn:
        if st.button("🎲 Load Random RACE Passage", use_container_width=True, type="secondary"):
            if len(test_df) > 0:
                row = test_df.sample(1).iloc[0]
                st.session_state.article = str(row.get('article', ''))
                st.session_state['_gold_question'] = str(row.get('question', ''))
                st.session_state['_gold_answer'] = str(row.get('answer_text', ''))
                # Store all 4 original options
                opts = []
                for c in ['A', 'B', 'C', 'D', 'a', 'b', 'c', 'd']:
                    if c in row.index and str(row[c]).strip():
                        opts.append(str(row[c]))
                st.session_state['_gold_options'] = opts[:4] if len(opts) >= 4 else opts
                st.session_state.quiz_ready = False
                st.session_state.checked = False
                st.rerun()
            else:
                st.error("❌ RACE dataset not found. Place CSV files in the `data/` directory.")
    with col_opt:
        use_original = st.checkbox("Use original RACE question & options",
                                    value=True)

    # ── Display loaded article (read-only) ───────────────────────────────
    article = st.session_state.article
    if article:
        st.markdown("---")
        st.markdown("#### 📄 Loaded RACE Passage")
        st.text_area("Reading Passage", value=article, height=250,
                     disabled=True, key="article_display")

        st.markdown("")
        if st.button("🚀 Generate Quiz from this Passage", use_container_width=True, type="primary"):
            if not models_loaded:
                st.error("❌ Models not loaded. Run the pipeline first to save models.")
            else:
                with st.spinner("Generating question, distractors, and hints..."):
                    t0 = time.time()
                    lr = models['lr_clf']
                    vec = models['ohe_verify']
                    answer = st.session_state.get('_gold_answer', '')
                    gold_q = st.session_state.get('_gold_question', '')
                    gold_opts = st.session_state.get('_gold_options', [])

                    # Decide: use original RACE question or AI-generated
                    if use_original and gold_q and gold_opts and len(gold_opts) >= 4:
                        q = gold_q
                        options = gold_opts[:]
                    else:
                        # Generate from scratch: extract a key phrase as answer
                        if not answer:
                            answer = extract_answer_from_article(article)
                        q = generate_question(article, answer, lr, vec)
                        distractors = generate_distractors(article, answer, vec)
                        options = [answer] + distractors[:3]

                    random.shuffle(options)
                    hints = generate_hints(article, q, answer)
                    elapsed = time.time() - t0

                    st.session_state.article = article
                    st.session_state.question = q
                    st.session_state.answer = answer
                    st.session_state.distractors = options
                    st.session_state.hints = hints
                    st.session_state.hints_used = 0
                    st.session_state.checked = False
                    st.session_state.user_answer = ''
                    st.session_state.inference_time = elapsed
                    st.session_state.quiz_ready = True
                    mode = "RACE original" if (use_original and gold_q) else "AI-generated"
                    st.success(f"✅ Quiz ready ({mode}) in {elapsed:.2f}s! Go to **Quiz View**.")
    else:
        st.info("👆 Click **Load Random RACE Passage** to get started.")

# ════════════════════════════════════════════════════════════════════════════
# SCREEN 2 — Quiz View
# ════════════════════════════════════════════════════════════════════════════
elif screen == "❓ Quiz View":
    st.markdown("## ❓ Screen 2 — Question & Answer Quiz")

    if not st.session_state.quiz_ready:
        st.info("💡 Go to **Article Input** first to generate a quiz.")
    else:
        st.markdown(f"**Article excerpt:** {st.session_state.article[:300]}...")
        st.markdown("---")
        st.markdown(f"### {st.session_state.question}")

        options = st.session_state.distractors
        labels = ['A', 'B', 'C', 'D']

        selected = st.radio("Select your answer:",
                            [f"{labels[i]}. {opt}" for i, opt in enumerate(options)],
                            index=None, key="quiz_radio")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Check Answer", type="primary", use_container_width=True):
                if selected is None:
                    st.warning("⚠️ Please select an answer first.")
                else:
                    chosen_text = selected.split('. ', 1)[1]
                    correct = st.session_state.answer
                    st.session_state.checked = True
                    st.session_state.user_answer = chosen_text

                    is_correct = clean_text(chosen_text) == clean_text(correct)

                    # Model A verifier inference
                    verifier_prob = None
                    if models.get('lr_clf') and models.get('ohe_verify'):
                        try:
                            text = clean_text(st.session_state.article + ' ' +
                                              st.session_state.question + ' ' + chosen_text)
                            feat = models['ohe_verify'].transform([text])
                            verifier_prob = float(models['lr_clf'].predict_proba(feat)[0][1])
                        except Exception:
                            verifier_prob = None
                    st.session_state['verifier_prob'] = verifier_prob

                    # Log result
                    st.session_state.session_log.append({
                        'question': st.session_state.question[:60],
                        'selected': chosen_text[:40],
                        'correct_answer': correct[:40],
                        'is_correct': is_correct,
                        'verifier_conf': verifier_prob,
                        'time': st.session_state.inference_time,
                        'timestamp': time.strftime('%H:%M:%S')
                    })

        with col2:
            if st.button("🔄 New Question", use_container_width=True):
                st.session_state.quiz_ready = False
                st.session_state.checked = False
                st.rerun()

        if st.session_state.checked:
            chosen_text = st.session_state.user_answer
            correct = st.session_state.answer
            vp = st.session_state.get('verifier_prob')
            verifier_tag = (f' | Model A confidence: <b>{vp*100:.1f}%</b>'
                            if vp is not None else '')
            if clean_text(chosen_text) == clean_text(correct):
                st.markdown(f'<div class="correct" style="padding:15px;border-radius:10px;'
                            f'background:rgba(0,200,83,0.2);border:2px solid #00c853;margin-top:15px;">'
                            f'✅ <b>Correct!</b> Great job!{verifier_tag}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="incorrect" style="padding:15px;border-radius:10px;'
                            f'background:rgba(255,23,68,0.2);border:2px solid #ff1744;margin-top:15px;">'
                            f'❌ <b>Incorrect.</b> The correct answer is: <b>{correct}</b>'
                            f'{verifier_tag}</div>',
                            unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SCREEN 3 — Hint Panel
# ════════════════════════════════════════════════════════════════════════════
elif screen == "💡 Hint Panel":
    st.markdown("## 💡 Screen 3 — Graduated Hints")

    if not st.session_state.quiz_ready:
        st.info("💡 Go to **Article Input** first to generate a quiz.")
    else:
        st.markdown(f"**Question:** {st.session_state.question}")
        st.markdown("---")

        hints = st.session_state.hints
        used = st.session_state.hints_used
        total = len(hints) if hints else 3
        hint_labels = ["🔍 Hint 1 — General Clue", "🔎 Hint 2 — More Specific",
                       "🎯 Hint 3 — Near-Explicit"]

        for i in range(total):
            if i < used:
                label = hint_labels[i] if i < len(hint_labels) else f"Hint {i+1}"
                hint_text = hints[i] if i < len(hints) else "No hint available."
                with st.expander(label, expanded=True):
                    st.markdown(f'<div class="hint-box">{hint_text}</div>',
                                unsafe_allow_html=True)

        if used < total:
            if st.button(f"💡 Reveal Hint {used + 1}", use_container_width=True):
                st.session_state.hints_used = used + 1
                st.rerun()
        else:
            st.markdown("---")
            st.success("All hints revealed!")
            if st.button("🔓 Reveal Answer", type="primary", use_container_width=True):
                st.markdown(f"### ✅ Answer: **{st.session_state.answer}**")

        st.markdown("---")
        st.progress(min(used / max(total, 1), 1.0), text=f"Hints used: {used}/{total}")

# ════════════════════════════════════════════════════════════════════════════
# SCREEN 4 — Analytics Dashboard
# ════════════════════════════════════════════════════════════════════════════
elif screen == "📊 Analytics Dashboard":
    st.markdown("## 📊 Screen 4 — NLG Evaluation Dashboard")

    # Load persisted metrics if available
    metrics_path = os.path.join(MODEL_DIR, 'metrics.json')
    metrics = None
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

    tab1, tab2, tab3 = st.tabs(["📈 Answer & Question NLG",
                                 "📉 Distractor & Hint NLG",
                                 "📋 Session Log"])

    def _fmt(v):
        """Format a metric value as a string."""
        if v is None or v == 0:
            return '—'
        return f"{v:.4f}"

    # --- Tab 1: Answer selection & Question generation NLG ---
    with tab1:
        st.markdown("### Answer Selection — NLG Metrics")
        te = metrics.get('test', {}) if metrics else {}

        c1, c2, c3, c4, c5 = st.columns(5)
        nlg_cards = [
            ('BLEU', te.get('answer_bleu')),
            ('ROUGE-1', te.get('answer_rouge1')),
            ('ROUGE-2', te.get('answer_rouge2')),
            ('ROUGE-L', te.get('answer_rougeL')),
            ('METEOR', te.get('answer_meteor')),
        ]
        for col, (label, val) in zip([c1, c2, c3, c4, c5], nlg_cards):
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-value">'
                            f'{_fmt(val)}</div>'
                            f'<div class="metric-label">{label}</div></div>',
                            unsafe_allow_html=True)

        em = te.get('exact_match')
        if em is not None:
            st.markdown("---")
            st.metric("Exact Match Score", f"{em:.4f}")

        # Question Generation NLG
        if te.get('qgen_bleu'):
            st.markdown("---")
            st.markdown("### Question Generation — NLG Metrics")
            c1, c2, c3, c4, c5 = st.columns(5)
            qgen_cards = [
                ('BLEU', te.get('qgen_bleu')),
                ('ROUGE-1', te.get('qgen_rouge1')),
                ('ROUGE-2', te.get('qgen_rouge2')),
                ('ROUGE-L', te.get('qgen_rougeL')),
                ('METEOR', te.get('qgen_meteor')),
            ]
            for col, (label, val) in zip([c1, c2, c3, c4, c5], qgen_cards):
                with col:
                    st.markdown(f'<div class="metric-card">'
                                f'<div class="metric-value">{_fmt(val)}</div>'
                                f'<div class="metric-label">QGen {label}</div></div>',
                                unsafe_allow_html=True)

        # Summary table
        st.markdown("---")
        st.markdown("#### NLG Metrics Summary")
        summary_data = {
            'Metric': ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'METEOR', 'Exact Match'],
            'Answer Selection': [_fmt(te.get('answer_bleu')),
                                  _fmt(te.get('answer_rouge1')),
                                  _fmt(te.get('answer_rouge2')),
                                  _fmt(te.get('answer_rougeL')),
                                  _fmt(te.get('answer_meteor')),
                                  _fmt(te.get('exact_match'))],
            'Question Gen.': [_fmt(te.get('qgen_bleu')),
                              _fmt(te.get('qgen_rouge1')),
                              _fmt(te.get('qgen_rouge2')),
                              _fmt(te.get('qgen_rougeL')),
                              _fmt(te.get('qgen_meteor')),
                              '—'],
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True,
                     hide_index=True)

        if not metrics:
            st.caption("⚠️ Run the pipeline to compute and persist NLG metrics.")

    # --- Tab 2: Distractor & Hint NLG ---
    with tab2:
        st.markdown("### Distractor Generation — NLG Metrics")
        mb = metrics.get('model_b', {}) if metrics else {}

        c1, c2, c3, c4 = st.columns(4)
        dist_cards = [
            ('BLEU', mb.get('dist_bleu')),
            ('ROUGE-1', mb.get('dist_rouge1')),
            ('ROUGE-L', mb.get('dist_rougeL')),
            ('METEOR', mb.get('dist_meteor')),
        ]
        for col, (label, val) in zip([c1, c2, c3, c4], dist_cards):
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-value">'
                            f'{_fmt(val)}</div>'
                            f'<div class="metric-label">Dist. {label}</div></div>',
                            unsafe_allow_html=True)

        # Hint NLG metrics
        if mb.get('hint_bleu') or mb.get('hint_rouge1'):
            st.markdown("---")
            st.markdown("### Hint Generation — NLG Metrics")
            c1, c2, c3, c4, c5 = st.columns(5)
            hint_cards = [
                ('BLEU', mb.get('hint_bleu')),
                ('ROUGE-1', mb.get('hint_rouge1')),
                ('ROUGE-2', mb.get('hint_rouge2')),
                ('ROUGE-L', mb.get('hint_rougeL')),
                ('METEOR', mb.get('hint_meteor')),
            ]
            for col, (label, val) in zip([c1, c2, c3, c4, c5], hint_cards):
                with col:
                    st.markdown(f'<div class="metric-card">'
                                f'<div class="metric-value">{_fmt(val)}</div>'
                                f'<div class="metric-label">Hint {label}</div></div>',
                                unsafe_allow_html=True)

        if not metrics:
            st.caption("⚠️ Run the pipeline to compute NLG metrics.")

    # --- Tab 3: Session log ---
    with tab3:
        st.markdown("### Session Results Log")
        log = st.session_state.session_log
        if log:
            log_df = pd.DataFrame(log)
            st.dataframe(log_df, use_container_width=True, hide_index=True)
            correct_count = sum(1 for r in log if r['is_correct'])
            total_count = len(log)
            avg_time = np.mean([r['time'] for r in log])

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Session Accuracy", f"{correct_count}/{total_count}")
            with c2:
                st.metric("Avg Inference Time", f"{avg_time:.2f}s")
            with c3:
                if st.button("📥 Export to CSV"):
                    csv = log_df.to_csv(index=False)
                    st.download_button("Download CSV", csv, "session_results.csv",
                                       "text/csv", use_container_width=True)
        else:
            st.info("No quiz attempts yet. Complete a quiz to see results here.")

