"""Load and validate user prompts from the input JSON file."""

from __future__ import annotations

import json
from pathlib import Path


def load_prompts(path: Path) -> list[str]:
    """Load prompts from a JSON file.

    Supports a JSON array of plain strings or of objects with a ``prompt``
    key. Raises ValueError for missing/malformed files, empty arrays or
    invalid entries.

    Args:
        path: Path to the input JSON file.

    Returns:
        List of prompt strings.

    Raises:
        ValueError: If the file is missing, malformed, empty or contains
            invalid entries.

    CÓMO FUNCIONA (por dentro):
    - Mismo patrón defensivo que function_loader.py: todo error se traduce
      a ValueError con contexto (el path) y exception chaining (`from exc`).
    - La diferencia clave: acá NO hay modelo pydantic porque la forma de los
      datos es trivial (strings). Usar pydantic para validar "es un string"
      sería sobre-ingeniería; el isinstance manual es más directo.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise ValueError(f"Prompts file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in prompts file {path}: {exc}") from exc

    # Doble condición en una línea: primero TIPO (list), luego CONTENIDO
    # (no vacío). Python evalúa con cortocircuito: si `isinstance` falla,
    # `len(data)` nunca se ejecuta (evitaría un TypeError sobre un objeto
    # sin __len__). Igual que `data or []`, pero explícito.
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Expected a non-empty JSON list in {path}")

    prompts: list[str] = []
    for item in data:
        # Duck typing explícito vía isinstance: aceptamos DOS formas.
        # Caso 1: string plano. ["hola", "chau"]
        if isinstance(item, str):
            prompts.append(item)
        # Caso 2: objeto con key "prompt" cuyo valor sea string.
        # [{"prompt": "hola"}]
        #
        # Detalle sutil: `"prompt" in item` chequea existencia de la KEY en
        # el dict (O(1), lookup de hash table). El tercer check
        # `isinstance(item["prompt"], str)` rechaza {"prompt": 42}. Sin él,
        # un int se colaría silenciosamente y rompería el pipeline mucho
        # después, con un error desconectado de la causa real. Fallar temprano
        # y cerca del origen del dato (fail fast) es la filosofía acá.
        elif (
            isinstance(item, dict)
            and "prompt" in item
            and isinstance(item["prompt"], str)
        ):
            prompts.append(item["prompt"])
        else:
            # `{item!r}` usa repr() en vez de str(): muestra el valor ENTRE
            # COMILLAS y con caracteres escapados ('{"a": 1}' vs {'a': 1}).
            # Para debugging eso es oro: distinguis "" de '' de espacios
            # invisibles, y el output es copy-pasteable como literal Python.
            raise ValueError(f"Invalid prompt format in {path}: {item!r}")
    return prompts
