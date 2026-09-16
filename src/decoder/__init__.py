"""Decoder constrained: state machine + trie (futuro schema/token_filter).

ROL DE ESTE __init__.py (por dentro):
- Facade de imports (mismo patrón que src/loader/__init__.py): los
  consumidores hablan con el package, no con los submódulos internos.
- state.py + trie.py ya existen; schema_validator.py llegó con la Task 3.3
  y token_filter.py cerrará con la 3.4: amplían este __all__.
"""

from src.decoder.schema_validator import SchemaContext
from src.decoder.state import DecoderPhase, DecoderState
from src.decoder.trie import TrieNode, build_trie, find_node, is_complete_name, valid_next_chars

__all__ = [
    "DecoderPhase",
    "DecoderState",
    "SchemaContext",
    "TrieNode",
    "build_trie",
    "find_node",
    "is_complete_name",
    "valid_next_chars",
]
