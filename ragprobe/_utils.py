"""
_utils.py — Shared text-processing primitives for ragprobe.
"""
from __future__ import annotations
import re

_STOPWORDS: frozenset[str] = frozenset({
    "a","an","the","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could",
    "should","may","might","shall","can","to","of","in","for",
    "on","with","at","by","from","as","or","and","but","if",
    "this","that","it","its","not","no","so","what","which",
    "who","how","when","where","why","i","we","you","he","she",
    "they","me","us","him","her","them","my","your","his","their",
    "our","than","then","there","here","up","about","any","all",
    "each","few","more","most","some","such","other","both","nor",
    "neither","either","these","those","also","into","over","after",
    "between","during","before","under","while","per","only","just",
    "very","too","yet","still","already","even",
})

def tokenize(text: str) -> set[str]:
    """Return significant lowercase tokens (len>=2, not stopwords)."""
    return {
        t for t in re.findall(r"[a-z0-9]+", text.lower())
        if len(t) >= 2 and t not in _STOPWORDS
    }

def tokenize_ordered(text: str) -> list[str]:
    """Same as tokenize() but order-preserving and deduplicated."""
    return list(dict.fromkeys(
        t for t in re.findall(r"[a-z0-9]+", text.lower())
        if len(t) >= 2 and t not in _STOPWORDS
    ))
