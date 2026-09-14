"""Decoder constrained: state machine + (futuro) trie/schema/validators.

ROL DE ESTE __init__.py (por dentro):
- Facade de imports (mismo patrón que src/loader/__init__.py): los
  consumidores hablan con el package, no con los submódulos internos.
- En este punto del proyecto SOLO existe state.py; trie.py, schema_validator.py
  y token_filter.py llegan con las Tasks 3.2-3.4 y amplían este __all__.
"""

from src.decoder.state import DecoderPhase, DecoderState

__all__ = ["DecoderPhase", "DecoderState"]
