"""Task 1 preprocessing starter."""

from typing import List

from nltk.tokenize import word_tokenize

__all__ = ["preprocess"]


def preprocess(raw_html_list: List[str]) -> List[List[str]]:
    """Clean and tokenize a batch of noisy documents.
    Args:
        raw_html_list: One raw HTML or noisy web-text string per document.
    Returns:
        One token list per input document, preserving input order and length.
        Produce the final tokens with nltk.word_tokenize after cleaning. Return
        an empty list for a document with no usable text.
    TODO:
      Replace this section with your own brief, code-consistent procedure.
      Use observations from both public dev profiles, justify the main
      operation order, and explain why the choices should generalise beyond
      individual examples.
    """
    pass
