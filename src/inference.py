"""
inference.py — Unified inference API
Provides question generation, distractor generation, and hint extraction.
Extracted from notebook cells 30 (question gen) and re-exports from model_b.
"""
import re, numpy as np
from src.preprocessing import clean_text, tokenize
from src.model_b_train import (split_sentences, sentence_keyword_overlap,
    generate_distractors, generate_hints, cosine_sim)

# ── Wh-word Template Question Generation ────────────────────────────────────
def wh_template(sentence, answer):
    """Apply Wh-word template to generate a question from a sentence."""
    ans_lower = answer.lower().strip()
    person_ind = ['mr','mrs','dr','president','minister','he','she','his','her']
    place_ind = ['city','country','town','village','river','mountain','street']
    if any(ans_lower.startswith(p) for p in person_ind): wh = 'Who'
    elif any(p in ans_lower for p in place_ind): wh = 'Where'
    elif re.match(r'^\d', ans_lower) or any(t in ans_lower for t in ['year','month','day','time','ago']): wh = 'When'
    elif ans_lower.startswith('because') or 'reason' in ans_lower: wh = 'Why'
    elif ans_lower.startswith('by') or 'method' in ans_lower: wh = 'How'
    else: wh = 'What'
    pattern = re.compile(re.escape(ans_lower), re.IGNORECASE)
    question = pattern.sub(wh, sentence, count=1)
    question = question.strip().rstrip('.!?')
    question = question[0].upper() + question[1:] if question else ''
    return question + '?'

def generate_question(article, answer, verifier_model, vectorizer, top_k=3):
    """Full question generation pipeline with ML-based ranking."""
    sentences = split_sentences(article)
    if not sentences: return 'What does the passage discuss?'
    scored = sorted(
        [(s, sentence_keyword_overlap(s, answer)) for s in sentences],
        key=lambda x: x[1], reverse=True)
    candidates = [(wh_template(s,answer), s, sc) for s,sc in scored[:top_k]]
    if not candidates: return 'What is the main idea?'
    best_q, best_score = candidates[0][0], -1
    for q, sent, _ in candidates:
        text = clean_text(article + ' ' + q + ' ' + answer)
        feat = vectorizer.transform([text])
        try: prob = verifier_model.predict_proba(feat)[0][1]
        except: prob = 0.5
        if prob > best_score: best_score = prob; best_q = q
    return best_q

# Re-export model_b functions for unified access
__all__ = ['generate_question', 'generate_distractors', 'generate_hints',
           'wh_template', 'split_sentences', 'sentence_keyword_overlap',
           'cosine_sim']
