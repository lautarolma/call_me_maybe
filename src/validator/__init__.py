"""Output validation layer (I/O boundary).

Public API:
    parse_output            — raw decoder string -> dict
    build_function_call     — (prompt, dict) -> FunctionCall
    build_results           — (prompts, generations) -> list[FunctionCall]
    find_unsupported_prompts — prompts with no value backed by their text
"""

from __future__ import annotations

from src.validator.output_validator import (
    build_function_call,
    build_results,
    find_unsupported_prompts,
    parse_output,
)

__all__ = [
    "build_function_call",
    "build_results",
    "find_unsupported_prompts",
    "parse_output",
]
