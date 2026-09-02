"""call_me_maybe — LLM function calling with constrained decoding.

Run with ``uv run python -m src``.

ROL DE ESTE __init__.py:
- Convierte el directorio `src/` en un PACKAGE importable. Sin este archivo,
  `python -m src` y los imports `from src.cli import ...` no funcionarían.
- Al ejecutarse en cada import del package, debe permanecer LIVIANO: solo
  metadata (versión). Poner lógica acá ralentizaría cada import y crearía
  dependencias circulares potenciales.
"""

__version__ = "0.1.0"
