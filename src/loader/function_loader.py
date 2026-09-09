"""Load and validate function definitions from the JSON file."""

from __future__ import annotations

# `json` es la librería estándar de serialización JSON. `json.load` lee un
# file object ya abierto; `json.loads` (con "s" de string) parsea texto que
# ya tenés en memoria. Por dentro: tokeniza el texto (llaves, corchetes,
# strings, números) y lo convierte a objetos Python con este mapeo:
#   JSON object {} -> dict | JSON array [] -> list
#   JSON string    -> str | JSON number  -> int/float | true/false/null -> True/False/None
import json
from pathlib import Path

from src.models.function_definition import FunctionDef


def load_functions(path: Path) -> list[FunctionDef]:
    """Load and validate function definitions.

    Raises ValueError for missing/malformed files, empty arrays,
    non-list payloads, invalid entries or duplicate function names.

    Args:
        path: Path to the functions definition JSON file.

    Returns:
        List of validated FunctionDef models.

    Raises:
        ValueError: If the file is missing, malformed, or the definitions are
            invalid or contain duplicate names.

    CÓMO FUNCIONA (por dentro):
    - Patrón general: traducir TODAS las fallas posibles a ValueError con
      mensaje claro. Es una decisión de diseño deliberada: el caller
      (`__main__.py`) no necesita conocer 4 tipos de excepción distintas,
      solo atrapar ValueError.
    """
    try:
        # El `with` es un context manager: garantiza que el file descriptor
        # se cierre AL SALIR del bloque, incluso si lanza una excepción en
        # el medio. Internamente llama a f.__enter__()/f.__exit__().
        # `encoding="utf-8"` es CRÍTICO: sin él, open() usa la codificación
        # del locale del sistema (en Windows cp1252) y un JSON con
        # caracteres no-ASCII explota o se corrompe silenciosamente.
        with open(path, encoding="utf-8") as f:
            # json.load consume el stream completo y parsea. Si el texto no
            # es JSON válido, lanza json.JSONDecodeError indicando línea y
            # columna exacta del error de sintaxis.
            data = json.load(f)
    except FileNotFoundError as exc:
        # `raise ... from exc` = EXCEPTION CHAINING. Guarda la excepción
        # original en `exc.__cause__` para que el traceback muestre ambas:
        # "The above exception was the direct cause of...". Sin el `from`,
        # perderías el rastro de POR QUÉ no se encontró el archivo.
        raise ValueError(f"Functions file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        # JSONDecodeError ES subclase de ValueError, pero la re-lanzamos
        # igual para normalizar el mensaje y agregar el path al contexto.
        raise ValueError(f"Invalid JSON in functions file {path}: {exc}") from exc

    # Validación de forma ANTES de validar contenido: esperamos un array
    # JSON (que json.load convirtió a list) y NO vacío. Un dict suelto, un
    # string o una lista vacía fallan acá, con mensaje claro, ANTES de
    # construir modelos (fail fast) — cierra la asimetría con input_loader.py,
    # ver BUG-002 en BITACORA_BUGS.md.
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Expected a non-empty JSON array of function definitions in {path}")

    # Una sola pasada: construye Y valida a la vez.
    # - `FunctionDef(**item)` desempaqueta cada dict como keyword arguments.
    #   Pydantic reescribe el __init__ para VALIDAR cada campo contra sus
    #   anotaciones (tipos, Literal, campos requeridos). Si algo no cumple,
    #   lanza pydantic.ValidationError — subclase de ValueError, así el
    #   contrato del método se cumple sin código extra.
    # - `seen` es un set (hash table): consultar/registrar un nombre es O(1);
    #   el loop total es O(n). Detectar el duplicado ACÁ corta el procesamiento
    #   en la primera reincidencia (fail-fast real), en vez de construir toda
    #   la lista y contar después con names.count() (O(n²) de la versión
    #   anterior).
    seen: set[str] = set()
    functions: list[FunctionDef] = []
    for item in data:
        fn = FunctionDef(**item)
        if fn.name in seen:
            raise ValueError(f"Duplicate function name: {fn.name}")
        seen.add(fn.name)
        functions.append(fn)
    return functions
