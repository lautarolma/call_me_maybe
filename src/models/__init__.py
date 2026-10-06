"""Pydantic models for the I/O layer (no generation logic here).

Re-exports the three models as a facade, as in ``src.loader``: consumers
depend on ``src.models``, not on individual modules.
"""

from src.models.function_definition import FunctionDef, ParameterDef
from src.models.output import FunctionCall

# Export list for 'from src.models import *'; named imports work regardless.
__all__ = ["FunctionCall", "FunctionDef", "ParameterDef"]
