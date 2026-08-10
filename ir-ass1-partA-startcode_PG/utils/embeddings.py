"""Light-weight GloVe loader & semantic aggregation (Task A-4)."""
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np

# ----------------------------------------------------------------------
# 1.  Load the provided local GloVe subset
# ----------------------------------------------------------------------
_GLOVE_PATH = Path(__file__).resolve().parents[1] / "data" / "glove" / "glove_subset_200d.pkl"


def _ensure_glove() -> Dict[str, np.ndarray]:
    if not _GLOVE_PATH.exists():
        raise FileNotFoundError(
            f"Missing GloVe subset: {_GLOVE_PATH}. "
            "Keep data/glove/glove_subset_200d.pkl in your submission folder."
        )

    with _GLOVE_PATH.open("rb") as f:
        loaded = pickle.load(f)

    if not isinstance(loaded, dict):
        raise TypeError(f"GloVe subset must be a dict, got {type(loaded)}")

    vocab: Dict[str, np.ndarray] = {
        str(word): np.asarray(vec, dtype=float)
        for word, vec in loaded.items()
    }

    if not vocab:
        raise ValueError(f"GloVe subset is empty: {_GLOVE_PATH}")

    # deterministic random <unk>
    if "<unk>" not in vocab:
        rng = np.random.default_rng(seed=42)
        dim = len(next(iter(vocab.values())))
        vocab["<unk>"] = rng.normal(0.0, 0.05, size=dim)

    return vocab


_WORD_VEC: Dict[str, np.ndarray] = _ensure_glove()
_DIM: int = len(next(iter(_WORD_VEC.values())))


# ----------------------------------------------------------------------
# 2.  Public helper
# ----------------------------------------------------------------------
def _key(tok: str) -> str:
    """Return the exact vocabulary key, or '<unk>' for an OOV token."""
    return tok if tok in _WORD_VEC else "<unk>"


# ----------------------------------------------------------------------
# 3.  Main entry
# ----------------------------------------------------------------------
def semantic_vector(docs: List[List[str]], method: str = "meanmax") -> np.ndarray:
    """Build one semantic vector per document.
    Args:
        docs: One token list per document.
        method: Aggregation method, either 'tfidf_weighted' or 'meanmax'.
    Returns:
        A matrix with shape (N, D) for 'tfidf_weighted' or (N, 2D) for
        'meanmax', where N is the number of documents and D is the embedding
        dimension.
    """
    pass
