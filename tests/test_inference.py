import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import clean_text, tokenize, STOPWORDS
from src.model_b_train import split_sentences, sentence_keyword_overlap
from src.inference import wh_template, generate_question


def test_clean_text():
    assert clean_text("Hello, World! 123") == "hello world"
    assert clean_text("  Multiple   spaces  ") == "multiple spaces"
    assert clean_text("URL http://example.com here") == "url here"
    print("✅ clean_text passed")


def test_tokenize():
    tokens = tokenize("The quick brown fox is very fast")
    assert 'the' not in tokens
    assert 'quick' in tokens
    assert 'brown' in tokens
    print("✅ tokenize passed")


def test_split_sentences():
    text = "First sentence here. Second sentence here. Third one right here."
    sents = split_sentences(text)
    assert len(sents) == 3
    print("✅ split_sentences passed")


def test_keyword_overlap():
    score = sentence_keyword_overlap("The cat sat on the mat", "cat")
    assert score > 0
    score_zero = sentence_keyword_overlap("The dog ran away", "cat")
    assert score_zero == 0.0
    print("✅ sentence_keyword_overlap passed")


def test_wh_template():
    q = wh_template("Mr. Smith went to the store.", "Mr. Smith")
    assert q.startswith("Who")
    assert q.endswith("?")
    print("✅ wh_template passed")


if __name__ == '__main__':
    test_clean_text()
    test_tokenize()
    test_split_sentences()
    test_keyword_overlap()
    test_wh_template()
    print("\n🎉 All tests passed!")
