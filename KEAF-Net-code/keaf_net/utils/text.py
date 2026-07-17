"""Answer normalisation following the standard VQA evaluation rules.

This mirrors the punctuation / article / contraction handling from the official
VQA v2.0 evaluation code so that predicted and ground-truth answers are compared
on the same footing (Section 5.4).
"""
from __future__ import annotations

import re

_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldnt": "couldn't",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hasnt": "hasn't",
    "havent": "haven't", "hes": "he's", "im": "i'm", "isnt": "isn't", "its": "it's",
    "shouldnt": "shouldn't", "thats": "that's", "theres": "there's", "theyre": "they're",
    "wasnt": "wasn't", "werent": "weren't", "whats": "what's", "wheres": "where's",
    "wont": "won't", "wouldnt": "wouldn't", "youre": "you're", "youll": "you'll",
}
_MANUAL_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_ARTICLES = {"a", "an", "the"}
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")
_PUNCT = ";/[]\"{}()=+\\_-><@`,?!"


def _strip_punctuation(text: str) -> str:
    for p in _PUNCT:
        if (p + " " in text or " " + p in text) or (re.search(_COMMA_STRIP, text) is None):
            text = text.replace(p, "")
        else:
            text = text.replace(p, " ")
    return _PERIOD_STRIP.sub("", text, re.UNICODE)


def normalize_answer(answer: str) -> str:
    """Lower-case, strip punctuation/articles and canonicalise number words."""
    text = answer.replace("\n", " ").replace("\t", " ").strip().lower()
    text = _strip_punctuation(text)
    words = []
    for word in text.split():
        word = _MANUAL_MAP.get(word, word)
        if word not in _ARTICLES:
            words.append(_CONTRACTIONS.get(word, word))
    return " ".join(words)
