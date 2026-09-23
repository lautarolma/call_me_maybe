"""Pydantic models for the I/O layer (no generation logic here)."""

from src.models.function_definition import FunctionDef, ParameterDef
from src.models.output import FunctionCall

# Mismo patrón facade que src/loader/__init__.py: re-exportar la API pública
# del subpackage para que `from src.models import FunctionDef` ande sin
# acoplarse a la estructura interna de files.
__all__ = ["FunctionCall", "FunctionDef", "ParameterDef"]
