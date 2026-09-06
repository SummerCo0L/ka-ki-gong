#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A small, dependency-free BM25 implementation for the lexical half of hybrid search.

The corpus is tiny (a few hundred schemes at most), so a pure NumPy BM25Okapi
is simpler and more reproducible than pulling in ``rank_bm25`` or a full-text
search engine.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-only tokenization."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """BM25Okapi scoring over a fixed corpus of documents.

    Scores are min-max normalized to ``[0, 1]`` per query via
    :meth:`score_normalized` so they can be linearly combined with dense
    cosine-similarity scores in hybrid retrieval.
    """

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b = b
        self._doc_tokens: list[list[str]] = [tokenize(doc) for doc in documents]
        self._doc_lens = np.array([len(toks) for toks in self._doc_tokens], dtype=np.float64)
        self._avg_doc_len = float(self._doc_lens.mean()) if len(self._doc_lens) else 0.0
        self._n_docs = len(self._doc_tokens)

        self._term_freqs: list[Counter] = [Counter(toks) for toks in self._doc_tokens]

        doc_freq: Counter = Counter()
        for tf in self._term_freqs:
            doc_freq.update(tf.keys())

        self._idf: dict[str, float] = {}
        for term, df in doc_freq.items():
            # BM25 idf with a +1 floor so common terms never go negative.
            self._idf[term] = math.log(1 + (self._n_docs - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> np.ndarray:
        """Raw BM25 scores for ``query`` against every document, in corpus order."""
        if self._n_docs == 0:
            return np.zeros(0, dtype=np.float64)

        query_terms = tokenize(query)
        scores = np.zeros(self._n_docs, dtype=np.float64)
        if not query_terms:
            return scores

        for i in range(self._n_docs):
            tf = self._term_freqs[i]
            doc_len = self._doc_lens[i]
            denom_norm = self._k1 * (1 - self._b + self._b * doc_len / (self._avg_doc_len or 1.0))
            total = 0.0
            for term in query_terms:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                idf = self._idf.get(term, 0.0)
                total += idf * (freq * (self._k1 + 1)) / (freq + denom_norm)
            scores[i] = total

        return scores

    def score_normalized(self, query: str) -> np.ndarray:
        """BM25 scores min-max normalized to ``[0, 1]`` for this query."""
        raw = self.score(query)
        if raw.size == 0:
            return raw
        max_score = raw.max()
        if max_score <= 0:
            return np.zeros_like(raw)
        return raw / max_score
