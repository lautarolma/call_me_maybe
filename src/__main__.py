"""Entry point for call_me_maybe.

Usage:
    uv run python -m src [--functions_definition PATH] [--input PATH] [--output PATH]

Exit code is 0 on success and 1 on any error; errors are printed to stderr
without an unhandled traceback.

CÓMO SE EJECUTA ESTE ARCHIVO:
- `python -m src` le dice al intérprete: "buscá el paquete `src` en sys.path
  y ejecutá su `__main__.py` como script". Python lo importa con
  `__name__ == "__main__"`, por eso el bloque de abajo se ejecuta.
- Este patrón separa el punto de entrada (este archivo, que orquesta y
  maneja errores) de la lógica (cli.py y pipeline.py), dejando ambos
  testeables sin ejecutar nada al importarlos.
"""

from __future__ import annotations

# `sys` da acceso a cosas del intérprete: `sys.argv` (args del proceso),
# `sys.stderr` (stream de errores, NO bufferizado igual que stdout),
# `sys.exit()` (termina el proceso con un código).
import sys

from src.cli import parse_args
from src.pipeline import run


def main() -> int:
    """Parse arguments and run the pipeline."""
    # Sin argumento, parse_args lee sys.argv[1:] automáticamente.
    args = parse_args()
    # Convención POSIX: un proceso devuelve un int al SO; 0 = éxito,
    # cualquier otro valor = fallo. Devolver el int (en vez de llamar
    # sys.exit() acá adentro) mantiene a `main` pura y testeable.
    return run(args)


if __name__ == "__main__":
    try:
        # `main()` devuelve el exit code; sys.exit(int) lo propaga al SO.
        # Si el int es 0, el shell lo interpreta como éxito ($? == 0).
        sys.exit(main())
    except Exception as exc:
        # Catch-amortiguador de último recurso: atrapa CUALQUIER excepción
        # no manejada (FileNotFoundError, ValidationError de pydantic,
        # errores de red del Hub...) para que el usuario final vea un
        # mensaje limpio en stderr en lugar de un traceback completo.
        #
        # Tradeoff aceptado acá: perdemos el stack trace (útil para debug)
        # a cambio de una UX limpia. En desarrollo conviene comentar este
        # except temporalmente para ver el traceback completo.
        #
        # Detalle: se imprime en STDERR, no en STDOUT. Los streams existen
        # para eso: stdout es para DATOS (redirigibles a archivos/pipes),
        # stderr para DIAGNÓSTICO. Así `python -m src > salida.txt` nunca
        # contaminaría los datos con mensajes de error.
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
