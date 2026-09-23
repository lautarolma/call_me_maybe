"""Command-line interface parsing for call_me_maybe."""

# `from __future__ import annotations` (PEP 563) pospone la evaluación de las
# type hints: en vez de evaluarse al importar el módulo, se guardan como
# strings dentro de `__annotations__`. Esto permite usar sintaxis moderna
# como `list[str] | None` (PEP 604) incluso en versiones de Python que no la
# soportan en runtime (el operador `|` entre tipos existe desde 3.10).
# Es "gratis" y hace el código más portable.
from __future__ import annotations

import argparse
from pathlib import Path

# Los defaults se evalúan UNA sola vez, en tiempo de importación del módulo.
# Son objetos `Path` RELATIVOS al directorio de trabajo actual (CWD), no a la
# ubicación del archivo: si ejecutás el programa desde otra carpeta, estos
# paths apuntan a otro lado. `Path` es preferible a strings crudos porque
# ofrece operaciones portables (`/` para unir, `.exists()`, `.read_text()`...)
# y abstrae las diferencias Windows (`\`) vs Unix (`/`).
DEFAULT_FUNCTIONS_DEFINITION = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calls.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; ``None`` means use ``sys.argv[1:]``.

    Returns:
        Parsed namespace with ``functions_definition``, ``input`` and
        ``output`` path attributes.

    CÓMO FUNCIONA (por dentro):
    - `ArgumentParser` construye un registro interno ("registry") de
      argumentos declarados. Cada `add_argument` agrega una entrada con:
      nombre del flag, tipo, default y help.
    - `parse_args(None)` lee `sys.argv[1:]` (los args reales del proceso,
      sin incluir el nombre del script). Si le pasás una lista, usa esa.
    - El parser tokeniza los argv: empareja cada `--flag valor`, IGNORA los
      guiones dobles y mapea `--functions_definition` al atributo
      `functions_definition` (los `-` se convierten en `_`).
    - El parámetro `type=Path` NO es solo tipado: es un CALLABLE que se
      ejecuta sobre el string crudo del argv. O sea, internamente hace
      `Path(valor_del_argv)`. Si el callable lanza excepción, argparse aborta
      con un error amigable y exit code 2 (nunca llega a nuestro código).
    - Si falta un argumento requerido o hay uno desconocido, argparse
      imprime el error + usage en stderr y llama a `sys.exit(2)` por su
      cuenta. Por eso acá no hay validación manual de nada de eso.
    - Devuelve un `argparse.Namespace`: un objeto-bolsa (como un dict pero
      con acceso por atributo). `args.input` equivale a `args.__dict__["input"]`.
    """
    # `description` aparece en el header del texto de ayuda (`-h/--help`),
    # que argparse genera automáticamente a partir de todos los add_argument.
    parser = argparse.ArgumentParser(
        description="LLM function calling with constrained decoding"
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        # El default se asigna tal cual cuando el flag NO está presente en
        # argv. Nota: es el MISMO objeto Path para toda la vida del proceso;
        # como Path es inmutable, compartirlo es seguro.
        default=DEFAULT_FUNCTIONS_DEFINITION,
        help="Path to functions definition JSON file",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to input prompts JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to output JSON file",
    )
    return parser.parse_args(argv)
