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
    pass
