"""Word- and char-level n-gram helpers (Task A-2)."""
from typing import List

__all__ = ["make_ngrams_chars"]

def make_ngrams_chars(text: str, n: int) -> List[str]:
    """Build character n-grams from boundary-padded text.
    Args:
        text: Source string from which to build character n-grams.
        n: Number of characters in each n-gram.
    Returns:
        Character n-grams from '$' + text + '$' in left-to-right order.
        Return an empty list when the padded text is shorter than n.
    """
    if n <= 0: return [] # simply if there's a test for 0-gram

    # character n-grams from '$' + text + '$' in left-to-right order.
    padded_text = "$" + text + "$"

    # return an empty list when the padded text is shorter than n.
    if len(padded_text) < n: return []

    # otherwise, return n-gram from padded_text e.g. $cl, cli, lim, ima, mat, ate, at$.
    # every n-character, append
    ngrams = []
    for i in range(len(padded_text) - n + 1):
        ngram = padded_text[i:i + n]
        ngrams.append(ngram)

    return ngrams
