"""Trie (prefix tree) for function names in the constrained decoder.

Built dynamically from the names in ``functions_definition.json``: every
node is one character, terminal nodes carry the full name and
``is_end = True``. Zero hardcoding — new functions in the JSON extend
the trie automatically.

Public API (used by the token filter and the decoder facade):
  - build_trie(names)         -> trie root
  - find_node(prefix)         -> node for the prefix, or None
  - valid_next_chars(prefix)  -> chars that keep at least one name viable
  - is_complete_name(prefix)  -> True if the prefix IS exactly a name
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TrieNode:
    """One node of a character trie.

    ``slots=True`` for the same reason as ``state.py``: the filter walks
    this trie for every candidate token, and ``__slots__`` drops
    ``__dict__`` (less memory, faster attribute access).
    """

    children: dict[str, TrieNode] = field(default_factory=dict)
    function_name: str | None = None
    is_end: bool = False


def build_trie(function_names: list[str]) -> TrieNode:
    """Build a trie from a list of function names.

    Each character adds one node level; when a name ends its node is
    marked terminal (``is_end = True``) and stores the full name.
    """
    root = TrieNode()
    for name in function_names:
        node = root
        for char in name:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.function_name = name
        node.is_end = True
    return root


def find_node(root: TrieNode, prefix: str) -> TrieNode | None:
    """Walk the trie by *prefix* and return the resulting node.

    Returns ``None`` when some character of the prefix is missing (the
    path does not exist).
    """
    node = root
    for char in prefix:
        if char not in node.children:
            return None
        node = node.children[char]
    return node


def valid_next_chars(root: TrieNode, prefix: str) -> set[str]:
    """Characters that, appended to *prefix*, keep at least one name viable.

    An unknown path returns an empty set — the model is generating an
    invalid name.
    """
    node = find_node(root, prefix)
    if node is None:
        return set()
    return set(node.children.keys())


def is_complete_name(root: TrieNode, prefix: str) -> bool:
    """True when *prefix* matches a function name exactly."""
    node = find_node(root, prefix)
    return node is not None and node.is_end
