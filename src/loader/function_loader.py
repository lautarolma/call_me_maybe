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

    Raises ValueError for missing/malformed files, non-list payloads,
    invalid entries or duplicate function names.

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
    # JSON (que json.load convirtió a list). Un dict suelto o un string
    # fallarían después de forma críptica al hacer **item.
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array of function definitions in {path}")

    # `FunctionDef(**item)` desempaqueta cada dict como keyword arguments:
    # {"name": "fn_x", ...} -> FunctionDef(name="fn_x", ...).
    # El __init__ NO es el generado por Python: pydantic lo reescribe en la
    # definición de la clase para VALIDAR cada campo contra sus anotaciones
    # (tipos, Literal, campos requeridos). Si algo no cumple, lanza
    # pydantic.ValidationError — que también es subclase de ValueError,
    # así que este método cumple su contrato sin código extra.
    functions = [FunctionDef(**item) for item in data]

    # Detección de duplicados. OJO con la complejidad: `names.count(name)`
    # recorre la lista entera por CADA nombre -> O(n²). Para ~10 funciones
    # es irrelevante; para miles usarías collections.Counter:
    #   from collections import Counter
    #   dup_counts = Counter(names); duplicates = sorted(n for n, c in dup_counts.items() if c > 1)
    #
    # Anatomía de la comprehension:
    #   {name for name in names if names.count(name) > 1}  -> set comprehension
    #   (el set elimina repetidos del resultado) -> sorted(...) devuelve lista ordenada
    names = [fn.name for fn in functions]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate function names: {duplicates}")
    return functions
