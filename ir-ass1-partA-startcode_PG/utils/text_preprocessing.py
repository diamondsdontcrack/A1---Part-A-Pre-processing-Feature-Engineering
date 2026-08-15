"""Task 1 preprocessing starter."""

import html
from typing import List
import unicodedata

from bs4 import BeautifulSoup
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
    Procedure:
        the checker essentially turns documents.jsonl into a list of strings, with each string containing dirty text
        Profile A's doc.jsonl mainly contains numeric HTML entity decoding inside words + regular HTML. sol&#117;tions -> solutions
        Profile B's doc.jsonl mainly contains the HTML itself escaped + invisible Unicode characters inserted inside words. &lt;div&gt; hello &lt;/div&gt; -> <div> hello </div>

        Simply removing html entities won't work since in cases like sol&#117;tions, the html entity is useful. So the first step is to decode it first.
        1. If it becomes a real character, it is likely part of the word, otherwise it becomes real HTML markup which bs4 can recognize (and therefore parsed) then lowercased.
        2. NLTK to tokenize the real words and output a list of list of tokens.
        3. After using repr(), unicode \u200b is found in profile b and belongs to Cf category, which belongs to formatting/control characters that are not meaningful visible text.
            - Remove all unicode belonging to Cf category as other unicode may be useful, such as symbols or foreign languages.
    """
    final_doc: List[List[str]] = []

    for raw_text in raw_html_list:
        if raw_text is None:
            final_doc.append([])
            continue

        # decode html entity. e.g. &lt;div&gt;Machine <span>learning</span> is useful!&lt;/div&gt; -> <div>Machine <span>learning</span> is useful!</div>
        decoded_text = html.unescape(raw_text)

        # parse using bs4 then extract human-legible text and separate. <div>Machine <span>learning</span> is useful!</div> -> "machine learning is useful!"
        cleaned_text = BeautifulSoup(decoded_text, "html.parser").get_text().lower()

        cleaned_text = "".join(char for char in cleaned_text if unicodedata.category(char) != "Cf")
        # "machine learning is useful!" -> ["machine", "learning", "is", "useful", "!"]
        tokens = word_tokenize(cleaned_text)

        final_doc.append(tokens)

    return final_doc

        




