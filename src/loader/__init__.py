"""Loaders for input data and model vocabulary.

ROL DE ESTE __init__.py (por dentro):
- Un package de Python es un directorio con __init__.py; ese archivo corre
  AL IMPORTAR el package. Aprovechamos eso para RE-EXPORTAR la API pública:
  `from src.loader import load_vocab` funciona sin que el consumidor sepa
  (o le importe) en qué submódulo vive cada función.
- Patrón "facade de imports": los consumidores hablan con el package, no
  con los módulos internos. Si mañana movés load_vocab a otro file, solo
  cambiás acá, no en todos lados.
"""

from src.loader.function_loader import load_functions
from src.loader.input_loader import load_prompts
from src.loader.vocab_loader import BYTE_CATEGORY, Vocab, load_vocab

# `__all__` define la lista explícita de nombres exportados cuando alguien
# hace `from src.loader import *`. No afecta los imports nombrados normales,
# pero documenta la API pública y silencia warnings de linters (ruff F401)
# sobre imports no usados directamente.
__all__ = ["BYTE_CATEGORY", "Vocab", "load_functions", "load_prompts", "load_vocab"]
