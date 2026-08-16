"""TF-IDF variants (Task A-3)."""
from collections import Counter
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

        Given a corpus, get the mathematical set (unique) words, sort by alpha then assign indexes. 
        e.g. d1 = ["cat", "dad", "glad"], d2 = ["dog", "dog", "mad"] = ["cat", "dad", "dog", "glad", "mad"] with {"cat": 0, "dad": 1, etc.}
    """
    if tf_mode not in {"len", "log", "bm25"}:
        raise ValueError("tf_mode must be 'len', 'log', or 'bm25'")

    terms = set()

    for doc in docs:
        for token in doc:
            terms.add(token)

    # sort unique terms in alpha order
    terms = sorted(terms)

    # build vocab index
    vocab = {}
    for index, term in enumerate(terms):
        vocab[term] = index

    # create X, a matrix with rows = number of docs and columns = number of vocab
    rows = len(docs)
    cols = len(vocab)
    X = np.zeros((rows, cols), dtype=float)

    if rows == 0 or cols == 0:
        return X, vocab

    # Document Frequency 
    df = {}
    for term in terms: 
        df[term] = 0

    for doc in docs:
        for term in set(doc):
            df[term] += 1

    # unsmoothed IDF
    idf = {}

    for term in terms:
        idf[term] = math.log(rows/df[term])

    # TF-IDF
    for doc_index, doc in enumerate(docs):
        counts = Counter(doc)
        doc_length = len(doc)

        for term, count in counts.items():
            if tf_mode == "len":
                tf = count/doc_length

            elif tf_mode == "log":
                tf = 1 + math.log(count)

            elif tf_mode == "bm25":
                tf = ((k+1) * count / (k + count))

            column = vocab[term]

            X[doc_index, column] = (tf*idf[term])

    return X, vocab