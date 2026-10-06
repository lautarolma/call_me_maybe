"""Constrained decoder: state machine, trie, schema and token filter.

Re-exports the decoder's public API as a facade (same pattern as
``src.loader``); the individual modules stay the implementation.
"""

from src.decoder.schema_validator import SchemaContext
from src.decoder.state import DecoderPhase, DecoderState
from src.decoder.token_filter import compute_allowed_ids
from src.decoder.trie import TrieNode, build_trie, find_node, is_complete_name, valid_next_chars

__all__ = [
    "DecoderPhase",
    "DecoderState",
    "SchemaContext",
    "TrieNode",
    "build_trie",
    "compute_allowed_ids",
    "find_node",
    "is_complete_name",
    "valid_next_chars",
]
