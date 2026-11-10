"""Label-free lexical relevance shared across all retrieval methods."""

from typing import Sequence
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from .models import Paper

VECTORIZER = HashingVectorizer(n_features=2**18, alternate_sign=False, norm="l2",
                              stop_words="english", ngram_range=(1, 2))


def relevance(query: str, papers: Sequence[Paper]) -> dict[str, float]:
    """Cosine similarity to ORIGINAL question; no labels or corpus fit required."""
    if not papers:
        return {}
    matrix = VECTORIZER.transform([p.text for p in papers])
    scores = (matrix @ VECTORIZER.transform([query]).T).toarray().ravel()
    return {p.id: float(np.clip(s, 0, 1)) for p, s in zip(papers, scores)}
