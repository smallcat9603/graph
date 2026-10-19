"""spaCy keyword extraction for the On-the-fly graph construction mode."""

from collections import Counter
from functools import lru_cache

import spacy

MODELS = {
    ("en", "Small"): "en_core_web_sm",
    ("en", "Medium"): "en_core_web_md",
    ("en", "Large"): "en_core_web_lg",
    ("ja", "Small"): "ja_core_news_sm",
    ("ja", "Medium"): "ja_core_news_md",
    ("ja", "Large"): "ja_core_news_lg",
}


@lru_cache(maxsize=None)
def get_nlp(language, pipeline_size):
    return spacy.load(MODELS[(language, pipeline_size)])


def extract_keywords(nlp, text, word_class, n):
    """Top-n (keyword, count) pairs among non-stop, non-punct tokens of the given POS."""
    doc = nlp(text)
    keywords = [token.text for token in doc if not token.is_stop and not token.is_punct and token.pos_ in word_class]
    return Counter(keywords).most_common(n)
