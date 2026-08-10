"""TF-IDF variants (Task A-3)."""
import math, numpy as np
from typing import List, Dict, Tuple

def tfidf_variants(
        docs: List[List[str]],
        tf_mode: str = "len",
        k: float = 1.2
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Build a TF-IDF document-term matrix.
    Args:
        docs: One token list per document.
        tf_mode: Term-frequency mode, one of 'len', 'log', or 'bm25'.
            The 'log' mode uses the natural logarithm.
        k: BM25 term-frequency saturation parameter.
    Returns:
        A pair (X, vocab). X has one row per document and one column per
        vocabulary term. vocab contains every unique input term and assigns
        columns in alphabetical term order.
    """
    pass
