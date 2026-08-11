"""Token-position mapping (Task A-2)."""
from typing import Dict, List

def make_positions(tokens: List[str]) -> Dict[str, List[int]]:
    """Map each token to its positions in a token sequence.
    Args:
        tokens: Token sequence in document order.
    Returns:
        A mapping from each token to its 0-based positions in tokens.
    """
    """ e.g. ["to", "be", "or", "not", "to", "be"] -> 
        {
            "to": [0, 4],
            "be": [1, 5],
            "or": [2],
            "not": [3]
        }
    """
    # init dict strongly typed to be same as required output type
    positions: Dict[str, List[int]] = {}

    # for each token, if its not already in the dict, add it to the dict and an empty list. Then, append its index to the list
    for index, token in enumerate(tokens):
        if token not in positions:
            positions[token] = []

        positions[token].append(index)

    return positions

