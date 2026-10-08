"""call_me_maybe — LLM function calling with constrained decoding.

Run with ``uv run python -m src``.

ROLE OF THIS __init__.py:
- It turns the `src/` directory into an importable PACKAGE. Without this file,
  `python -m src` and the `from src.cli import ...` imports would not work.
- Since it runs on every import of the package, it must stay LIGHT: only
  metadata (version). Putting logic here would slow down every import and
  create potential circular dependencies.
"""

__version__ = "0.1.0"
