"""Load and validate function definitions from the JSON file."""

from __future__ import annotations

import json
from pathlib import Path

from src.models.function_definition import FunctionDef


def load_functions(path: Path) -> list[FunctionDef]:
    """Load and validate function definitions.

    Every failure mode — missing file, JSON syntax, payload shape, entry
    schema, duplicate name — is re-raised as a single ``ValueError``
    carrying the path, so callers only ever catch one exception type.

    Args:
        path: Path to the functions definition JSON file.

    Returns:
        Validated ``FunctionDef`` models, in file order.

    Raises:
        ValueError: File missing, JSON malformed, payload not a non-empty
            array, an entry failing the ``FunctionDef`` schema, or a
            duplicate function name.
    """
    try:
        # utf-8: open() would otherwise use the system locale (cp1252 on
        # Windows) and corrupt non-ASCII JSON.
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        # 'from exc' keeps the root cause in __cause__ for the traceback.
        raise ValueError(f"Functions file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        # JSONDecodeError is a ValueError subclass; re-raised only to
        # normalize the message and add the path.
        raise ValueError(f"Invalid JSON in functions file {path}: {exc}") from exc

    # Shape before content: a dict or empty payload fails here, before any
    # model is built (fail fast, same contract as load_prompts).
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Expected a non-empty JSON array of function definitions in {path}")

    # One pass builds and validates: FunctionDef(**item) runs pydantic
    # validation on every field. The 'seen' set makes duplicate detection
    # O(1) per name, aborting at the first repeat.
    seen: set[str] = set()
    functions: list[FunctionDef] = []
    for item in data:
        fn = FunctionDef(**item)
        if fn.name in seen:
            raise ValueError(f"Duplicate function name: {fn.name}")
        seen.add(fn.name)
        functions.append(fn)
    return functions
