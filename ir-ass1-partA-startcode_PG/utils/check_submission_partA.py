#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal public sanity checks (Part A):
- Environment checks: Python version + dependency versions (numpy / nltk / bs4)
- NLTK tokenizer resource check (punkt / punkt_tab): if missing, prints the download command
  and tries to auto-download once.
- Task 1: checks both public PG dev profiles and reports per-profile and combined Hit@K.
- Task 2: ONLY tests the public examples shown in the spec/LaTeX (necessary, not sufficient)
- Task 3: ONLY checks “format + runnable + reasonable shapes” (no strict numeric correctness)
- Task 4: IMPORTANT: checks whether you *really use* `_WORD_VEC` and `_DIM` from utils.embeddings
  (via monkey-patching these globals and verifying outputs change accordingly)

Task 1 note (per course interface):
- preprocess(...) MUST return List[List[str]] (one token list per document).
- If you return a flat List[str], this script will FAIL and show feedback.

Note:
- Passing this script does NOT guarantee full marks. Hidden tests include more edge cases.
"""

from __future__ import annotations

import sys
import traceback
import json
import os
import tempfile
import threading
import time
from collections import defaultdict
from importlib import metadata as importlib_metadata
from typing import Any, Dict, List, Optional, Sequence, Tuple
from pathlib import Path
import importlib.util


# ============================================================
# Config
# ============================================================
SUPPORTED_PY_MAJOR = 3
SUPPORTED_PY_MIN_MINOR = 10
SUPPORTED_PY_MAX_MINOR = 14
SANITY_TIMEOUT_SECONDS = 180

REQUIRED_IMPORTS = ["numpy", "nltk", "bs4"]

# Warning only: having these installed locally doesn't mean you used them,
# but importing/depending on them may fail in the marking environment.
DISALLOWED_IMPORTS_WARN = ["sklearn", "pandas", "scipy", "gensim", "torch", "tensorflow"]

RESULTS: List[Tuple[str, str, str]] = []  # (name, status, msg)  status in {"PASS","FAIL","WARN","SKIP"}
_START_TIME = time.perf_counter()
_WATCHDOG: Optional[threading.Timer] = None

# Keep generated NLTK data out of the submitted repository. Python selects a
# writable per-user system temp directory on Windows, macOS, and Linux. The
# repository fallback preserves runnability in unusually restricted setups.
REPO_ROOT_FOR_NLTK = Path(__file__).resolve().parent.parent
TMP_DIR_FOR_NLTK = REPO_ROOT_FOR_NLTK / "utils" / "_tmp"
try:
    NLTK_DATA_DIR = Path(tempfile.gettempdir()) / "ir_ass1_partA_nltk_data"
    NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    NLTK_DATA_DIR = TMP_DIR_FOR_NLTK / "nltk_data"
    NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
os.environ["NLTK_DATA"] = str(NLTK_DATA_DIR)


# ============================================================
# Small helpers
# ============================================================
def record(name: str, status: str, msg: str = "") -> None:
    RESULTS.append((name, status, msg))
    print(f"[{status}] {name}" + (f": {msg}" if msg else ""))


def _timeout_exit() -> None:
    print(
        "\n[FAIL] Sanity checker runtime limit "
        f": exceeded {SANITY_TIMEOUT_SECONDS} seconds. "
        "If this happens, your implementation is likely too slow or stuck.",
        flush=True,
    )
    os._exit(1)


def _fail_if_not(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _exc(e: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(e), e)).strip()


def _get_dist_version(dist_name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(dist_name)
    except importlib_metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _get_pkg_version(import_name: str) -> Optional[str]:
    dist_candidates = {
        "numpy": ["numpy"],
        "nltk": ["nltk"],
        "bs4": ["beautifulsoup4", "bs4"],
        "sklearn": ["scikit-learn", "sklearn"],
        "pandas": ["pandas"],
        "scipy": ["scipy"],
        "gensim": ["gensim"],
        "torch": ["torch"],
        "tensorflow": ["tensorflow"],
    }.get(import_name, [import_name])

    for dist in dist_candidates:
        v = _get_dist_version(dist)
        if v is not None:
            return v

    # fallback: module.__version__
    try:
        mod = __import__(import_name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def import_or_fail(module_path: str):
    try:
        return __import__(module_path, fromlist=["*"])
    except Exception as e:
        raise ImportError(f"Import failed for '{module_path}': {e}") from e


def import_first(paths: List[str]):
    last = None
    for p in paths:
        try:
            return import_or_fail(p)
        except Exception as e:
            last = e
    raise last if last is not None else ImportError("No module path candidates provided")


def _is_2d_float_ndarray(x: Any) -> bool:
    try:
        import numpy as np
        return isinstance(x, np.ndarray) and x.ndim == 2 and x.dtype.kind in {"f", "i"}
    except Exception:
        return False


def bag_f1(pred_tokens: List[str], gold_tokens: List[str]) -> float:
    """
    Bag-of-words F1 on token multisets (counts).
    - True positives = sum over tokens of min(pred_count, gold_count)
    - Precision = tp / len(pred_tokens)
    - Recall    = tp / len(gold_tokens)
    """
    from collections import Counter

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    p = Counter(pred_tokens)
    g = Counter(gold_tokens)
    tp = sum(min(p[t], g[t]) for t in p.keys() | g.keys())

    precision = tp / max(1, len(pred_tokens))
    recall = tp / max(1, len(gold_tokens))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def flatten_docs(docs_tokens: List[List[str]]) -> List[str]:
    flat: List[str] = []
    for d in docs_tokens:
        flat.extend(d)
    return flat


def _ensure_repo_root_on_syspath(repo_root: Path) -> None:
    """
    Ensure repo root is at the FRONT of sys.path so imports resolve to the submission code.
    Also tries to avoid 'utils' name clashes with any third-party module named 'utils'.
    """
    rr = str(repo_root)

    # Put repo root at sys.path[0]
    if rr in sys.path:
        sys.path.remove(rr)
    sys.path.insert(0, rr)

    # If 'utils' was already imported from elsewhere, drop it so we can re-import ours.
    u = sys.modules.get("utils")
    if u is not None:
        u_file = getattr(u, "__file__", "") or ""
        # If utils resolves outside the repo root, it is likely a name clash.
        if u_file and (rr not in u_file):
            del sys.modules["utils"]


# ============================================================
# 0) Environment gate
# ============================================================
def step_environment_gate() -> None:
    name = "Environment: Python must be 3.10 to 3.14"
    vi = sys.version_info
    supported = (
        vi.major == SUPPORTED_PY_MAJOR
        and SUPPORTED_PY_MIN_MINOR <= vi.minor <= SUPPORTED_PY_MAX_MINOR
    )
    if not supported:
        msg = (
            f"Detected Python {vi.major}.{vi.minor}.{vi.micro}. "
            f"Supported versions are Python {SUPPORTED_PY_MAJOR}.{SUPPORTED_PY_MIN_MINOR} "
            f"to {SUPPORTED_PY_MAJOR}.{SUPPORTED_PY_MAX_MINOR} inclusive.\n"
            "Fix: select a supported Python environment and re-run."
        )
        record(name, "FAIL", msg)
        sys.exit(1)

    record(name, "PASS", f"Python {vi.major}.{vi.minor}.{vi.micro}")


def step_required_packages() -> None:
    name = "Environment: required packages import (numpy, nltk, bs4)"
    try:
        for pkg in REQUIRED_IMPORTS:
            __import__(pkg)

        numpy_v = _get_pkg_version("numpy")
        nltk_v = _get_pkg_version("nltk")
        bs4_v = _get_pkg_version("bs4")
        record(name, "PASS", f"numpy={numpy_v}; nltk={nltk_v}; beautifulsoup4={bs4_v}")
    except Exception as e:
        record(
            name,
            "FAIL",
            f"Missing/broken required package import. Details: {_exc(e)}\n"
            f"Fix: install dependencies in a supported Python "
            f"{SUPPORTED_PY_MAJOR}.{SUPPORTED_PY_MIN_MINOR} to "
            f"{SUPPORTED_PY_MAJOR}.{SUPPORTED_PY_MAX_MINOR} environment."
        )
        raise


def step_disallowed_packages_warning() -> None:
    name = "Environment: disallowed package scan"
    found = []
    for pkg in DISALLOWED_IMPORTS_WARN:
        try:
            if importlib.util.find_spec(pkg) is not None:
                v = _get_pkg_version(pkg)
                found.append(f"{pkg}={v}" if v else pkg)
        except Exception:
            pass

    if found:
        record(
            name,
            "WARN",
            "Found: " + ", ".join(found) + ". "
            "These packages are not permitted / not guaranteed in the marking environment. "
            "Having them installed is OK, but your submission must not import/depend on them."
        )
    else:
        record(name, "PASS", "No common disallowed packages detected")


# ============================================================
# 0.5) NLTK resource check (punkt / punkt_tab)
# ============================================================
def step_nltk_tokenizers() -> None:
    name = "NLTK: tokenizer resources (punkt / punkt_tab)"
    try:
        import nltk
        from nltk.tokenize import word_tokenize

        if str(NLTK_DATA_DIR) not in nltk.data.path:
            nltk.data.path.insert(0, str(NLTK_DATA_DIR))

        try:
            _ = word_tokenize("Hello world. This is a test!")
            record(name, "PASS", "word_tokenize() works")
            return
        except LookupError:
            ok1 = nltk.download("punkt", download_dir=str(NLTK_DATA_DIR), quiet=True)
            ok2 = nltk.download("punkt_tab", download_dir=str(NLTK_DATA_DIR), quiet=True)
            _ = word_tokenize("Hello world. This is a test!")
            record(name, "PASS", f"Downloaded missing resources (punkt={ok1}, punkt_tab={ok2})")
            return

    except LookupError as e:
        record(
            name,
            "FAIL",
            "NLTK tokenizer resource missing.\n"
            "Run:\n"
            "  python -c \"import nltk; nltk.download('punkt'); nltk.download('punkt_tab')\"\n"
            f"Details: {_exc(e)}"
        )
    except Exception as e:
        record(name, "FAIL", _exc(e))


# ============================================================
# Task 1: preprocess() on both PG dev profiles
# ============================================================
def _load_dev_documents(data_dir: Path) -> List[Dict[str, Any]]:
    docs_path = data_dir / "documents.jsonl"
    with docs_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_dev_queries(data_dir: Path) -> List[Dict[str, Any]]:
    queries_path = data_dir / "queries.json"
    with queries_path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_dev_qrels(data_dir: Path) -> Dict[str, List[int]]:
    qrels_path = data_dir / "qrels.txt"
    qrels: Dict[str, List[int]] = defaultdict(list)
    with qrels_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4 and int(parts[3]) > 0:
                qrels[parts[0]].append(int(parts[2]))
    return dict(qrels)


def _score_overlap(query_toks: Sequence[str], doc_toks: Sequence[str]) -> float:
    if not query_toks or not doc_toks:
        return 0.0
    query_set = {str(token) for token in query_toks if str(token).strip()}
    doc_set = {str(token) for token in doc_toks if str(token).strip()}
    return float(len(query_set & doc_set))


def _dev_hit_metrics(
    doc_ids: List[int],
    doc_tokens: List[List[str]],
    queries: List[Dict[str, Any]],
    qrels: Dict[str, List[int]],
    preprocess,
) -> Tuple[float, float]:
    hit1 = 0
    hit5 = 0
    evaluated = [
        (item, set(qrels.get(item["qid"], [])))
        for item in queries
        if qrels.get(item["qid"], [])
    ]
    query_tokens = preprocess([item["query"] for item, _ in evaluated])
    _fail_if_not(
        isinstance(query_tokens, list),
        "preprocess(queries) must return List[List[str]].",
    )
    _fail_if_not(
        len(query_tokens) == len(evaluated),
        "preprocess(queries) must return one token list per query.",
    )
    _fail_if_not(
        all(isinstance(tokens, list) for tokens in query_tokens),
        "preprocess(queries) must return List[List[str]].",
    )
    _fail_if_not(
        all(all(isinstance(token, str) for token in tokens) for tokens in query_tokens),
        "preprocess(queries) must return string tokens.",
    )

    for (item, gold), q_toks in zip(evaluated, query_tokens):
        scored = []
        for doc_id, toks in zip(doc_ids, doc_tokens):
            score = _score_overlap(q_toks, toks)
            if score > 0:
                scored.append((doc_id, score))
        scored.sort(key=lambda x: (-x[1], x[0]))
        ranked = [doc_id for doc_id, _ in scored]
        hit1 += int(bool(ranked[:1] and ranked[0] in gold))
        hit5 += int(bool(gold & set(ranked[:5])))

    used = len(evaluated)
    if used == 0:
        return 0.0, 0.0
    return hit1 / used, hit5 / used


def step_task1_preprocess_dev_data() -> None:
    """
    Strict interface check:
    - preprocess(raw_html_list) MUST return List[List[str]] with one list per input document.
    Dev sanity check:
    - Runs the same token-overlap retrieval baseline on both public PG profiles.
    """
    name = "Task1: preprocess() two-profile dev retrieval report"
    try:
        m = import_first(["utils.text_preprocessing", "text_preprocessing"])
        _fail_if_not(hasattr(m, "preprocess"), "Missing function: preprocess(...)")
        preprocess = getattr(m, "preprocess")

        length_check_inputs = ["", "   ", "<p>alpha beta</p>"]
        length_check_outputs = preprocess(length_check_inputs)
        _fail_if_not(isinstance(length_check_outputs, list),
                     "preprocess must return a list for every input batch.")
        _fail_if_not(len(length_check_outputs) == len(length_check_inputs),
                     "preprocess must preserve input length: return one token list per input document. "
                     "If a document has no usable text, return an empty token list for that document.")
        _fail_if_not(all(isinstance(d, list) for d in length_check_outputs),
                     "preprocess must return List[List[str]], not drop or flatten documents.")

        repo_root = Path(__file__).resolve().parents[1]
        profile_results = []
        for label, directory in (
            ("A", repo_root / "data" / "dev_profile_a"),
            ("B", repo_root / "data" / "dev_profile_b"),
        ):
            docs = _load_dev_documents(directory)
            queries = _load_dev_queries(directory)
            qrels = _load_dev_qrels(directory)
            raw = [str(d["text"]) for d in docs]
            preds = preprocess(raw)

            _fail_if_not(isinstance(preds, list),
                         f"Profile {label}: preprocess must return a list. Got: {type(preds)}")
            _fail_if_not(len(preds) == len(raw),
                         f"Profile {label}: expected one token list per document "
                         f"(expected {len(raw)}, got {len(preds)}).")
            if preds and isinstance(preds[0], str):
                raise AssertionError(
                    f"Profile {label}: preprocess returned List[str]. "
                    "Expected List[List[str]], one token list per document."
                )
            _fail_if_not(all(isinstance(d, list) for d in preds),
                         f"Profile {label}: each document output must be List[str].")
            _fail_if_not(all(all(isinstance(t, str) for t in d) for d in preds),
                         f"Profile {label}: every output token must be a string.")

            total_tokens = sum(len(d) for d in preds)
            _fail_if_not(total_tokens > 0,
                         f"Profile {label}: preprocess returned no tokens.")
            doc_ids = [int(d["id"]) for d in docs]
            hit1, hit5 = _dev_hit_metrics(
                doc_ids,
                preds,
                queries,
                qrels,
                preprocess,
            )
            profile_results.append((label, len(queries), hit1, hit5))

        total_queries = sum(count for _, count, _, _ in profile_results)
        combined_hit1 = sum(count * hit1 for _, count, hit1, _ in profile_results) / total_queries
        combined_hit5 = sum(count * hit5 for _, count, _, hit5 in profile_results) / total_queries
        combined_avg = (combined_hit1 + combined_hit5) / 2
        details = "; ".join(
            f"Profile {label}: Hit@1={hit1:.3f}, Hit@5={hit5:.3f}"
            for label, _, hit1, hit5 in profile_results
        )
        record(name, "PASS", f"{details}; combined Average Hit@K={combined_avg:.3f}")
    except Exception as e:
        record(name, "FAIL", _exc(e))


# ============================================================
# Task 2: make_ngrams_chars / make_positions (public examples)
# ============================================================
def step_task2_examples() -> None:
    name = "Task2: make_ngrams_chars('hello', 3) public example"
    try:
        m = import_first(["utils.ngram", "ngram"])
        _fail_if_not(hasattr(m, "make_ngrams_chars"), "Missing function: make_ngrams_chars(text, n)")
        make_ngrams_chars = getattr(m, "make_ngrams_chars")
        out = make_ngrams_chars("hello", 3)
        expected = ["$he", "hel", "ell", "llo", "lo$"]
        _fail_if_not(out == expected, f"Expected {expected}, got {out}")
        record(name, "PASS")
    except Exception as e:
        record(name, "FAIL", _exc(e))

    name = "Task2: make_positions(...) public example"
    try:
        m = import_first(["utils.positions", "positions"])
        _fail_if_not(hasattr(m, "make_positions"), "Missing function: make_positions(tokens)")
        make_positions = getattr(m, "make_positions")

        tokens = ["the", "quick", "brown", "fox", "jumps", "over", "the", "dog"]
        out = make_positions(tokens)

        expected = {
            "the": [0, 6],
            "quick": [1],
            "brown": [2],
            "fox": [3],
            "jumps": [4],
            "over": [5],
            "dog": [7],
        }
        _fail_if_not(isinstance(out, dict), f"Expected dict, got {type(out)}")
        _fail_if_not(out == expected, f"Expected {expected}, got {out}")
        record(name, "PASS")
    except Exception as e:
        record(name, "FAIL", _exc(e))


# ============================================================
# Task 3: tfidf_variants (format + runnability only)
# ============================================================
def step_task3_tfidf_runnable() -> None:
    name = "Task3: tfidf_variants(..., tf_mode='log'/'bm25') runs + returns correct shapes"
    try:
        import numpy as np

        m = import_first(["utils.tfidf", "tfidf"])
        _fail_if_not(hasattr(m, "tfidf_variants"), "Missing function: tfidf_variants(docs, tf_mode=...)")
        tfidf_variants = getattr(m, "tfidf_variants")

        docs = [["cat", "sat", "mat"], ["cat", "cat", "dog"]]

        for mode in ("log", "bm25"):
            tfidf_matrix, vocab = tfidf_variants(docs, tf_mode=mode)

            _fail_if_not(_is_2d_float_ndarray(tfidf_matrix), f"[{mode}] tfidf_matrix must be a 2D numpy array, got {type(tfidf_matrix)}")
            _fail_if_not(isinstance(vocab, dict), f"[{mode}] vocab must be dict, got {type(vocab)}")
            _fail_if_not(tfidf_matrix.shape[0] == len(docs), f"[{mode}] shape mismatch: rows {tfidf_matrix.shape[0]} != n_docs {len(docs)}")
            _fail_if_not(tfidf_matrix.shape[1] == len(vocab), f"[{mode}] shape mismatch: cols {tfidf_matrix.shape[1]} != vocab_size {len(vocab)}")
            _fail_if_not(all(isinstance(k, str) for k in vocab.keys()), f"[{mode}] vocab keys must be str")
            _fail_if_not(all(isinstance(v, int) for v in vocab.values()), f"[{mode}] vocab values must be int")

            idxs = sorted(vocab.values())
            _fail_if_not(idxs == list(range(len(vocab))), f"[{mode}] vocab indices must be contiguous 0..V-1, got {idxs[:10]}...")

            expected_tokens = sorted({token for doc in docs for token in doc})
            _fail_if_not(
                set(vocab.keys()) == set(expected_tokens),
                f"[{mode}] vocab must contain every unique term in docs. "
                f"Expected {expected_tokens}, got {sorted(vocab.keys())}.",
            )
            mismatches = []
            for i, tok in enumerate(expected_tokens):
                if vocab.get(tok) != i:
                    mismatches.append(f"{tok}->{vocab.get(tok)} (expected {i})")
            if mismatches:
                preview = "; ".join(mismatches[:8])
                _fail_if_not(False, f"[{mode}] vocab indices must follow alphabetical token order. Mismatches: {preview}")

            _fail_if_not(np.isfinite(tfidf_matrix).all(), f"[{mode}] tfidf_matrix contains NaN/Inf")

        record(name, "PASS", "log/bm25 produced finite matrices with consistent shapes")
    except Exception as e:
        record(name, "FAIL", _exc(e))


# ============================================================
# Task 4: semantic_vector (key: did they use _WORD_VEC / _DIM?)
# ============================================================
def step_task4_semantic_vector_uses_globals() -> None:
    name = "Task4: semantic_vector uses utils.embeddings._WORD_VEC and _DIM (monkey-patch test)"
    try:
        import numpy as np

        # IMPORTANT: do NOT fall back to 'embeddings' (top-level) because it breaks relative imports.
        # We want to enforce the course repo structure: utils/embeddings.py imported as utils.embeddings.
        try:
            emb = import_or_fail("utils.embeddings")
        except Exception as e:
            # Provide actionable debug info (where Python is finding 'utils')
            spec_utils = importlib.util.find_spec("utils")
            origin = getattr(spec_utils, "origin", None) if spec_utils else None
            record(
                name,
                "FAIL",
                "Cannot import utils.embeddings.\n"
                "This usually happens when:\n"
                "  1) you are not running from the repo root, so 'utils' cannot be found, or\n"
                "  2) your folder structure is incorrect (embeddings.py is not under utils/), or\n"
                "  3) a different third-party module named 'utils' is being imported.\n"
                f"Python resolved 'utils' origin as: {origin}\n"
                f"Details: {_exc(e)}"
            )
            return

        _fail_if_not(hasattr(emb, "semantic_vector"), "Missing function: semantic_vector(docs, method=...)")
        _fail_if_not(hasattr(emb, "_WORD_VEC"), "Missing global: _WORD_VEC")
        _fail_if_not(hasattr(emb, "_DIM"), "Missing global: _DIM")

        semantic_vector = getattr(emb, "semantic_vector")

        setup_name = "Setup: provided GloVe data"
        dim = getattr(emb, "_DIM")
        word_vec = getattr(emb, "_WORD_VEC")

        if not isinstance(dim, int):
            record(setup_name, "FAIL", f"_DIM must be int, got {type(dim)}")
            record(name, "SKIP", "Not run because the provided GloVe setup check failed")
            return
        if not isinstance(word_vec, dict):
            record(setup_name, "FAIL", f"_WORD_VEC must be dict, got {type(word_vec)}")
            record(name, "SKIP", "Not run because the provided GloVe setup check failed")
            return
        if dim != 200:
            record(setup_name, "WARN", f"_DIM={dim}; len(_WORD_VEC)={len(word_vec)}")
        else:
            record(setup_name, "PASS", f"_DIM=200; len(_WORD_VEC)={len(word_vec)}")

        orig_dim = dim
        orig_word_vec = word_vec

        try:
            test_dim = 8
            dummy = {
                "cat": np.arange(test_dim, dtype=float),
                "dog": np.arange(test_dim, dtype=float) + 100.0,
                "<unk>": np.zeros(test_dim, dtype=float),
            }

            setattr(emb, "_DIM", test_dim)
            setattr(emb, "_WORD_VEC", dummy)

            # If the student has a loader, neutralize it to avoid side effects (e.g., downloading)
            if hasattr(emb, "_ensure_glove"):
                try:
                    setattr(emb, "_ensure_glove", lambda: dummy)  # type: ignore
                except Exception:
                    pass

            docs_tfidf = [["cat"], ["dog"]]
            out1 = semantic_vector(docs_tfidf, method="tfidf_weighted")
            _fail_if_not(out1.shape == (2, test_dim), f"tfidf_weighted shape should be (2,{test_dim}), got {out1.shape}")

            dummy2 = dict(dummy)
            dummy2["cat"] = dummy2["cat"] + 9999.0
            setattr(emb, "_WORD_VEC", dummy2)

            out2 = semantic_vector(docs_tfidf, method="tfidf_weighted")
            _fail_if_not(out2.shape == out1.shape, "shape changed unexpectedly after patch")
            _fail_if_not(
                not np.allclose(out1[0], out2[0]),
                "Patching _WORD_VEC['cat'] did NOT change tfidf_weighted output for doc0 "
                "(this suggests you are not using _WORD_VEC)"
            )

            repeat_dummy = {
                "cat": np.eye(test_dim, dtype=float)[0],
                "dog": np.eye(test_dim, dtype=float)[1],
                "<unk>": np.eye(test_dim, dtype=float)[2],
            }
            setattr(emb, "_WORD_VEC", repeat_dummy)
            if hasattr(emb, "_ensure_glove"):
                setattr(emb, "_ensure_glove", lambda: repeat_dummy)  # type: ignore

            repeated = semantic_vector(
                [
                    ["cat", "cat", "cat", "dog"],
                    ["cat", "dog", "dog", "dog"],
                    ["unknown_word"],
                ],
                method="tfidf_weighted",
            )
            _fail_if_not(
                repeated.shape == (3, test_dim),
                f"tfidf_weighted repeated-token shape should be (3,{test_dim}), "
                f"got {repeated.shape}",
            )
            _fail_if_not(
                np.allclose(repeated[0, :2], [0.9, 0.1], atol=1e-6)
                and np.allclose(repeated[1, :2], [0.1, 0.9], atol=1e-6),
                "Repeated-token TF-IDF weighting must include each token occurrence "
                "in the weighted average.",
            )

            setattr(emb, "_WORD_VEC", dummy)
            out3 = semantic_vector([["cat"]], method="meanmax")
            _fail_if_not(out3.shape == (1, 2 * test_dim), f"meanmax shape should be (1,{2*test_dim}), got {out3.shape}")

            setattr(emb, "_WORD_VEC", dummy2)
            out4 = semantic_vector([["cat"]], method="meanmax")
            _fail_if_not(
                not np.allclose(out3, out4),
                "Patching _WORD_VEC['cat'] did NOT change meanmax output "
                "(this suggests you are not using _WORD_VEC)"
            )

        finally:
            try:
                setattr(emb, "_DIM", orig_dim)
                setattr(emb, "_WORD_VEC", orig_word_vec)
            except Exception:
                pass

        record(name, "PASS", "semantic_vector reacts to patched _DIM/_WORD_VEC (good)")
    except Exception as e:
        record(name, "FAIL", _exc(e))


# ============================================================
# Summary
# ============================================================
def _is_environment_or_setup_check(name: str) -> bool:
    return name.startswith(("Environment:", "NLTK:", "Setup:", "Sanity checker:"))


def _print_result_group(heading: str, rows: List[Tuple[str, str, str]]) -> None:
    print(f"\n--- {heading} ---")
    if not rows:
        print("No checks were run.")
        return

    for name, status, msg in rows:
        print(f"{status:4} - {name}")
        if msg:
            for line in msg.splitlines():
                print(f"       └─ {line}")

    passed = sum(1 for _, status, _ in rows if status == "PASS")
    warned = sum(1 for _, status, _ in rows if status == "WARN")
    skipped = sum(1 for _, status, _ in rows if status == "SKIP")
    failed = sum(1 for _, status, _ in rows if status == "FAIL")
    print(f"{heading}: {passed} passed, {warned} warned, {skipped} skipped, {failed} failed")


def _print_summary_and_exit(exit_code: int, title: str = "Summary") -> None:
    if _WATCHDOG is not None:
        _WATCHDOG.cancel()

    print(f"\n=== {title} ===")
    setup_rows = [row for row in RESULTS if _is_environment_or_setup_check(row[0])]
    task_rows = [row for row in RESULTS if not _is_environment_or_setup_check(row[0])]
    _print_result_group("Environment/setup checks", setup_rows)
    _print_result_group("Assignment task checks", task_rows)
    print(f"Runtime: {time.perf_counter() - _START_TIME:.1f} seconds")

    if exit_code == 0:
        print("\n" + "="*60)
        print("  SUCCESS! All public setup and assignment checks passed.")
        print("  IMPORTANT: This script ONLY checks for crash-safety and")
        print("  basic interface compliance. It does NOT check for")
        print("  mathematical correctness on hidden test cases.")
        print("  Passing this check does NOT guarantee full marks!")
        print("="*60)

    sys.exit(exit_code)


def main() -> None:
    global _WATCHDOG
    _WATCHDOG = threading.Timer(SANITY_TIMEOUT_SECONDS, _timeout_exit)
    _WATCHDOG.daemon = True
    _WATCHDOG.start()

    print("=== Part A Sanity Check: starting ===")
    print(f"Runtime limit: {SANITY_TIMEOUT_SECONDS} seconds")

    # Ensure repo root is on sys.path BEFORE any task imports, to avoid 'utils' name clashes.
    repo_root = Path(__file__).resolve().parents[1]
    _ensure_repo_root_on_syspath(repo_root)

    try:
        step_environment_gate()
        step_required_packages()
        step_disallowed_packages_warning()
    except Exception:
        _print_summary_and_exit(1, title="Summary (stopped due to environment failure)")

    step_nltk_tokenizers()

    step_task1_preprocess_dev_data()
    step_task2_examples()
    step_task3_tfidf_runnable()
    step_task4_semantic_vector_uses_globals()

    failed = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    _print_summary_and_exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        record(
            "Sanity checker: unexpected crash",
            "FAIL",
            "The checker stopped unexpectedly before it could finish all checks.\n"
            f"Details: {_exc(e)}"
        )
        _print_summary_and_exit(1, title="Summary (stopped due to unexpected crash)")
