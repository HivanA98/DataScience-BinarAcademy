"""
cleansing.py — refactored text-cleansing for the Gold challenge.

Fixes vs the original Gold notebook:
  * changealay() no longer rebuilds the 15k-entry dict on every call — it is
    built once at import time.
  * Slang/word replacement operates on whole WORDS. The original Platinum
    removechars used re.sub('di','',t) which deleted 'di' from inside words
    ('sedih'->'seh', 'modifikasi'->'mofikasi'). That class of bug is gone.
  * Stopword removal is OPTIONAL and negation-aware. The supplied stopword list
    contains 'tidak/bukan/jangan/kurang'; removing them flips sentiment, so by
    default we keep them (set remove_stopwords=True, keep_negation=True).
  * Sastrawi stemming is OPTIONAL (off by default): it is ~0.2 s/row (~35 min for
    11k rows) and, on this dataset, does not improve sentiment accuracy — see
    benchmark results in the README.
"""
from __future__ import annotations
import re, functools
import pandas as pd

# ---- resources loaded ONCE ----
_alay = pd.read_csv("data/new_kamusalay.csv", header=None, encoding="latin-1",
                    names=["slang", "formal"])
ALAY = dict(zip(_alay["slang"].astype(str), _alay["formal"].astype(str)))

with open("data/stopwordbahasa.csv", encoding="latin-1") as _fh:
    STOPWORDS = {w.strip() for w in _fh if w.strip()}

# negations must survive stopword removal for sentiment to make sense
NEGATION = {"tidak", "tak", "bukan", "jangan", "belum", "tanpa", "kurang",
            "gak", "ga", "nggak", "enggak", "jangankan", "jgn", "gk", "tdk"}

_URL   = re.compile(r"(https?://\S+|www\.\S+)")
_NONAZ = re.compile(r"[^a-z\s]")
_MULTI = re.compile(r"\s+")
_ELONG = re.compile(r"(.)\1{2,}")

# optional stemmer (lazy — only built if requested)
@functools.lru_cache(maxsize=1)
def _get_stemmer():
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    return StemmerFactory().create_stemmer()


def clean(text: str,
          remove_stopwords: bool = False,
          keep_negation: bool = True,
          stem: bool = False) -> str:
    """Return a cleaned, space-joined token string."""
    text = str(text).lower()
    text = _URL.sub(" ", text)
    text = _ELONG.sub(r"\1", text)          # baguuuus -> bagus
    text = _NONAZ.sub(" ", text)            # drop digits/punct/emoji (byte-safe)
    text = _MULTI.sub(" ", text).strip()

    tokens = [ALAY.get(w, w) for w in text.split() if len(w) > 1]

    if stem:
        tokens = _get_stemmer().stem(" ".join(tokens)).split()

    if remove_stopwords:
        drop = STOPWORDS - NEGATION if keep_negation else STOPWORDS
        tokens = [w for w in tokens if w not in drop]

    return " ".join(tokens)


if __name__ == "__main__":
    demo = [
        "Makanannya ENAK bgt tp pelayanannya tidak ramah!!! 😤",
        "modifikasi mobilnya keren sekaliii",
        "saya sedih dengan kondisi ini",
    ]
    for d in demo:
        print(repr(d), "->", repr(clean(d)))
