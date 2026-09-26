"""Output validation layer (I/O boundary).

Public API:
    parse_output         — string crudo del decoder -> dict
    build_function_call  — (prompt, dict) -> FunctionCall
    validate_output      — string crudo + definiciones -> FunctionCall | str
    build_results        — (prompts, generaciones) -> list[FunctionCall]
"""

from __future__ import annotations

from src.validator.output_validator import (
    build_function_call,
    build_results,
    parse_output,
    validate_output,
)

__all__ = [
    "build_function_call",
    "build_results",
    "parse_output",
    "validate_output",
]
