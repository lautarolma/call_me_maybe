"""Load and validate user prompts from the input JSON file."""

from __future__ import annotations

import json
from pathlib import Path


def load_prompts(path: Path) -> list[str]:
    """Load prompts from a JSON file.

    Accepts an array of plain strings or of objects with a ``prompt`` key.
    Each entry is validated with manual ``isinstance`` checks, not with
    pydantic, and that is deliberate: the payload is trivial (only
    strings), so wrapping it in a model would be over-engineering.

    Args:
        path: Path to the input JSON file.

    Returns:
        Prompt strings, in file order.

    Raises:
        ValueError: File missing, JSON malformed, payload not a non-empty
            array, or an item matching neither accepted shape.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise ValueError(f"Prompts file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in prompts file {path}: {exc}") from exc

    # Shape before content, as in load_functions: fail fast before any
    # per-item validation runs.
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Expected a non-empty JSON list in {path}")

    prompts: list[str] = []
    for item in data:
        # Two accepted forms: a plain string ["a"], or an object with a
        # string "prompt" key [{"prompt": "a"}]. The str check on
        # item["prompt"] rejects {"prompt": 42} here, at the data boundary,
        # instead of breaking much later in the pipeline (fail fast).
        if isinstance(item, str):
            prompts.append(item)
        elif (
            isinstance(item, dict)
            and "prompt" in item
            and isinstance(item["prompt"], str)
        ):
            prompts.append(item["prompt"])
        else:
            # !r quotes and escapes the value, so invisible whitespace in a
            # rejected prompt stays visible.
            raise ValueError(f"Invalid prompt format in {path}: {item!r}")
    return prompts
